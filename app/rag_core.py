"""Core RAG logic shared by the API — refactored from the notebooks.

The embedding model is loaded once at import time (the server imports this
module at startup), so requests do not pay the model-loading cost.

Retrieval embedder is intfloat/multilingual-e5-base (繁中+英文皆強，跨語言檢索佳)，
與 embed_folder.py / test_query.py 一致，取代原本對中文較弱且只看前 77 個 token
的 CLIP 文字編碼器。E5 需要前綴：文件用 "passage: "，查詢用 "query: "。
"""
import os
import re
import shutil
import base64
import hashlib

import torch
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from huggingface_hub import HfApi, snapshot_download
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- API keys (two Gemini keys to spread the usage limit) -------------------
for _key in ("GEMINI_API_KEY_PRIMARY", "GEMINI_API_KEY_SECONDARY"):
    if not os.environ.get(_key):
        raise RuntimeError(f"缺少環境變數 {_key}，請在 .env 檔中設定")

gemini_api_key_primary = os.environ["GEMINI_API_KEY_PRIMARY"]
gemini_api_key_secondary = os.environ["GEMINI_API_KEY_SECONDARY"]

# Each operation prefers one key and automatically fails over to the other when
# its usage limit (quota / rate limit) is exhausted.
# Answer generation prefers the primary key; vision prefers the secondary key.
_LLM_KEY_ORDER = [gemini_api_key_primary, gemini_api_key_secondary]
_VISION_KEY_ORDER = [gemini_api_key_secondary, gemini_api_key_primary]

_vision_clients = [genai.Client(api_key=k) for k in _VISION_KEY_ORDER]
_llms = [
    ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        max_retries=2,
        google_api_key=k,
    )
    for k in _LLM_KEY_ORDER
]


def _is_quota_error(exc: Exception) -> bool:
    """True if the error looks like an exceeded usage / rate limit."""
    text = str(exc).upper()
    return any(
        s in text
        for s in ("RESOURCE_EXHAUSTED", "QUOTA", "RATE LIMIT", "RATE_LIMIT", "429", "USAGE LIMIT")
    )


def _vision_generate(contents, config):
    """generate_content with automatic fail-over to the backup key on quota errors."""
    last_exc = None
    for i, client in enumerate(_vision_clients):
        try:
            return client.models.generate_content(
                model="gemini-3.1-flash-lite", contents=contents, config=config
            )
        except Exception as e:
            last_exc = e
            if _is_quota_error(e) and i < len(_vision_clients) - 1:
                print(f"[rag_core] vision 金鑰 #{i} 額度用盡，改用備援金鑰…")
                continue
            raise
    raise last_exc


def _content_to_text(content) -> str:
    """Flatten an LLM response's .content to plain text.

    Newer Gemini models (e.g. gemini-3.1-flash-lite) may return content as a list
    of blocks (each {"type": "text", "text": ...}) rather than a bare string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _llm_invoke(prompt: str) -> str:
    """llm.invoke with automatic fail-over to the backup key on quota errors."""
    last_exc = None
    for i, model in enumerate(_llms):
        try:
            return _content_to_text(model.invoke(prompt).content)
        except Exception as e:
            last_exc = e
            if _is_quota_error(e) and i < len(_llms) - 1:
                print(f"[rag_core] LLM 金鑰 #{i} 額度用盡，改用備援金鑰…")
                continue
            raise
    raise last_exc

# --- embedding model (loaded once) -----------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "intfloat/multilingual-e5-base"
print(f"[rag_core] loading {EMBED_MODEL} on {device} ...")
embedder = SentenceTransformer(EMBED_MODEL, device=device)
print("[rag_core] embedder ready.")

VECTORSTORE_ROOT = os.environ.get("VECTORSTORE_ROOT", "./vectorstores")


# --- embeddings -------------------------------------------------------------
def embed_text(texts: list[str]) -> list[list[float]]:
    """Embed note passages. E5 expects a 'passage: ' prefix for documents."""
    inputs = [f"passage: {t}" for t in texts]
    return embedder.encode(inputs, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a question. E5 expects a 'query: ' prefix for queries."""
    return embedder.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()


# --- per-user vector store --------------------------------------------------
def _safe_name(user_id: str) -> str:
    """Map an arbitrary user id (e.g. a Google email like a@b.com) to a valid Chroma
    collection / directory name: 3–512 chars from [a-zA-Z0-9._-], starting and ending
    with an alphanumeric. Plain ids such as 'alice' pass through unchanged, so existing
    stores keep working; invalid chars (e.g. '@') become '_'."""
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", (user_id or "").strip()).strip("._-")
    if len(name) < 3:  # too short / empty after cleaning -> stable fallback name
        name = "u" + hashlib.md5((user_id or "").encode("utf-8")).hexdigest()[:8]
    return name[:512]


def get_user_collection(user_id: str):
    name = _safe_name(user_id)
    chroma_client = chromadb.PersistentClient(path=f"{VECTORSTORE_ROOT}/{name}")
    return chroma_client.get_or_create_collection(name)


# --- persistence: sync vectorstores to a private HF Dataset --------------------------
# Space storage is ephemeral, so without this, runtime uploads are lost on every
# restart/rebuild. We back the stores with a Dataset repo: restore on startup, push
# the changed user's collection after each add/delete. Needs HF_TOKEN with WRITE access.
HF_TOKEN = os.environ.get("HF_TOKEN")
VECTORSTORE_DATASET = os.environ.get("VECTORSTORE_DATASET", "timhelloyee/notehandler-store")
_hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN else None


def _restore_from_dataset():
    """Pull saved vectorstores from the backing Dataset on startup so uploads survive
    restarts/rebuilds. First run (empty dataset) seeds it from the image's baked store.
    Any failure (no/invalid token, offline) is logged and ignored — the app still runs
    on the baked seed, just without persistence."""
    if not _hf_api:
        print("[rag_core] HF_TOKEN not set — vectorstore persistence disabled (ephemeral).")
        return
    try:
        try:  # best-effort: needs write access; a read token can still restore below
            _hf_api.create_repo(VECTORSTORE_DATASET, repo_type="dataset", private=True, exist_ok=True)
        except Exception as e:
            print(f"[rag_core] create_repo skipped ({e!r}); assuming dataset exists.")
        local = snapshot_download(VECTORSTORE_DATASET, repo_type="dataset", token=HF_TOKEN)
        entries = [e for e in os.listdir(local) if not e.startswith(".")]
        if entries:
            os.makedirs(VECTORSTORE_ROOT, exist_ok=True)
            for e in entries:
                src = os.path.join(local, e)
                if os.path.isdir(src):
                    shutil.copytree(src, os.path.join(VECTORSTORE_ROOT, e), dirs_exist_ok=True)
            print(f"[rag_core] restored vectorstores from {VECTORSTORE_DATASET}: {entries}")
        else:
            _persist_all()  # first run: initialize the dataset from the baked seed
            print(f"[rag_core] initialized {VECTORSTORE_DATASET} from baked seed.")
    except Exception as e:
        print(f"[rag_core] vectorstore restore skipped ({e!r}); using local/baked store.")


def _persist_all():
    """Push the whole vectorstores tree to the Dataset (used to seed it on first run)."""
    if not _hf_api or not os.path.isdir(VECTORSTORE_ROOT):
        return
    try:
        _hf_api.upload_folder(folder_path=VECTORSTORE_ROOT, path_in_repo=".",
                              repo_id=VECTORSTORE_DATASET, repo_type="dataset")
    except Exception as e:
        print(f"[rag_core] persist-all failed ({e!r}).")


def _persist_user(user_id: str):
    """Push one user's collection back to the Dataset after a change."""
    if not _hf_api:
        return
    name = _safe_name(user_id)
    folder = f"{VECTORSTORE_ROOT}/{name}"
    if not os.path.isdir(folder):
        return
    try:
        _hf_api.upload_folder(folder_path=folder, path_in_repo=name,
                              repo_id=VECTORSTORE_DATASET, repo_type="dataset")
    except Exception as e:
        print(f"[rag_core] persist '{name}' failed ({e!r}).")


# --- image description (Gemini vision) -------------------------------------
def describe_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    response = _vision_generate(
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            types.Part.from_text(text="""請以繁體中文，完整且忠實地擷取並整理這張筆記圖片中的所有內容，不要遺漏。

請遵守以下規則：
1. 在最開頭先判斷這是什麼科目（國文、英文、數學、物理、化學、地球科學、生物、地理、公民、歷史擇一），格式為：這是一份【科目】科的筆記。
2. 逐項記錄圖片中「實際出現的內容」，包括：定義、公式、定理、重要名詞、例題與其完整條件（已知數值、單位、所求）。不要只寫「涉及……的計算」這類概括性描述，而要寫出真正的內容與細節。
3. 數學式請使用 LaTeX 表示（行內以 $...$，獨立公式以 $$...$$ 包住，例如 $F = \\dfrac{G m_1 m_2}{r^2}$），以忠實保留上下標與符號。
4. 只描述圖片中實際出現的內容，不要編造；無法辨識處請標註為（無法辨識）。
5. 盡量保留原文的關鍵用語與數據，以便後續能據此回答細節問題。
"""),
        ],
        config={"temperature": 0, "top_p": 0.95, "top_k": 20},
    )
    return response.text


# --- PDF description (Gemini document understanding) ------------------------
def describe_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    response = _vision_generate(
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            types.Part.from_text(text="""請以繁體中文，完整且忠實地擷取並整理這份 PDF 筆記的所有內容，逐頁處理，不要遺漏任何一頁。

請遵守以下規則：
1. 在最開頭先判斷這是什麼科目（國文、英文、數學、物理、化學、地球科學、生物、地理、公民、歷史擇一），格式為：這是一份【科目】科的筆記。
2. 依文件原本的章節、標題、條列順序，逐項記錄「實際出現的內容」，包括：定義、公式、定理、重要名詞、例題與其完整條件（已知數值、單位、所求）。不要只寫「涉及……的計算」這類概括性描述，而要寫出真正的內容與細節。
3. 數學式請使用 LaTeX 表示（行內以 $...$，獨立公式以 $$...$$ 包住，例如 $F = \\dfrac{G m_1 m_2}{r^2}$），以忠實保留上下標與符號。
4. 只根據文件中實際出現的內容描述，不要自行補充或編造文件中沒有的資訊；若某處無法辨識，請標註為（無法辨識）。
5. 盡量保留原文的關鍵用語與數據，以便後續能據此回答細節問題。
"""),
        ],
        config={"temperature": 0, "top_p": 0.95, "top_k": 20},
    )
    return response.text


# --- ingestion --------------------------------------------------------------
def add_text_note(user_id: str, file_id: str, content: str, source: str) -> bool:
    """Returns True if added, False if it already existed."""
    collection = get_user_collection(user_id)
    if collection.get(ids=[file_id])["ids"]:
        return False
    collection.add(
        ids=[file_id],
        embeddings=[embed_text([content])[0]],
        documents=[content],
        metadatas=[{"type": "text", "source": source}],
    )
    _persist_user(user_id)
    return True


def add_image_note(user_id: str, file_id: str, image_path: str, source: str) -> str:
    """Describes the image with Gemini, embeds the description, stores it.

    Returns the generated description (or "" if the note already existed).
    """
    collection = get_user_collection(user_id)
    if collection.get(ids=[file_id])["ids"]:
        return ""
    description = describe_image(image_path)
    collection.add(
        ids=[file_id],
        embeddings=[embed_text([description])[0]],
        documents=[description],
        metadatas=[{"type": "image", "source": source}],
    )
    _persist_user(user_id)
    return description


def add_pdf_note(user_id: str, file_id: str, pdf_path: str, source: str) -> str:
    """Describes the PDF with Gemini, embeds the description, stores it.

    Returns the generated description (or "" if the note already existed).
    """
    collection = get_user_collection(user_id)
    if collection.get(ids=[file_id])["ids"]:
        return ""
    description = describe_pdf(pdf_path)
    collection.add(
        ids=[file_id],
        embeddings=[embed_text([description])[0]],
        documents=[description],
        metadatas=[{"type": "pdf", "source": source}],
    )
    _persist_user(user_id)
    return description


def list_notes(user_id: str) -> list[dict]:
    collection = get_user_collection(user_id)
    results = collection.get(include=["documents", "metadatas"])
    return [
        {
            "id": id_,
            "type": (meta or {}).get("type"),
            "source": (meta or {}).get("source"),
            "preview": (doc or "")[:240],
        }
        for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
    ]


def delete_note(user_id: str, file_id: str) -> None:
    get_user_collection(user_id).delete(ids=[file_id])
    _persist_user(user_id)


# --- query ------------------------------------------------------------------
PROMPT_TEMPLATE = """你是一個專業的學習筆記問答助手。請「僅」根據下列【參考資料】回答問題，並盡量完整、詳細、有條理。
如果參考資料中沒有相關資訊，請直接回答「參考資料中沒有記載相關訊息」，絕對不要自行編造或補充資料以外的內容。

回答時請遵守：
1. 介面支援 Markdown 與 LaTeX，請善用排版讓答案更易讀：可使用粗體（**重點**）、斜體（*強調*）、條列、標題等 Markdown 語法；數學式請使用 LaTeX，行內公式以單個 $ 包住（例如 $F = \\dfrac{G m_1 m_2}{r^2}$），獨立公式以 $$ 包住。
2. 參考資料是對原始筆記（文字、圖片或 PDF）逐字擷取後的敘述，請先理解其內容，再依使用者問題作答。
3. 若使用者詢問的項目散落在多段資料中，請將相關內容整合後再條列回答，並盡量保留原始的關鍵數據與用語，不要省略重要細節。
4. 請以繁體中文回答。

【參考資料】
{context}

【使用者問題】
{question}

【回答】"""


# Drop retrieved chunks whose cosine distance exceeds this cutoff. With normalized
# E5 embeddings, relevant chunks sit around 0.2–0.3 while unrelated ones are 0.4+.
# Filtering keeps recall (top_k stays high) but removes noise that distracts the
# lighter LLM (gemini-3.1-flash-lite) into falsely answering "no info" when the
# relevant snippet is short and buried among many irrelevant chunks.
MAX_DISTANCE = 0.35


def rag_query(user_id: str, question: str, top_k: int = 10) -> str:
    collection = get_user_collection(user_id)
    q_embedding = embed_query(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    docs = results["documents"][0] if results["documents"] else []
    dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
    if not docs:
        return "參考資料中沒有記載相關訊息"
    # Keep only relevant chunks, but always keep the single best one as a fallback.
    kept = [d for d, dist in zip(docs, dists) if dist <= MAX_DISTANCE]
    docs = kept or docs[:1]
    context = "\n---\n".join(docs)
    # Use replace (not str.format) so literal { } in LaTeX examples don't break.
    prompt = PROMPT_TEMPLATE.replace("{context}", context).replace("{question}", question)
    return _llm_invoke(prompt)


# Restore any previously-saved vectorstores from the backing Dataset (runs once at
# import / server startup, before requests are served).
_restore_from_dataset()

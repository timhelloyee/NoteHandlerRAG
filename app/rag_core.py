"""Core RAG logic shared by the API — refactored from the notebooks.

The CLIP model is loaded once at import time (the server imports this module
at startup), so requests do not pay the model-loading cost.
"""
import os
import base64

import torch
import open_clip
import chromadb
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- API keys (two Gemini keys to spread the usage limit) -------------------
for _key in ("GEMINI_API_KEY_PRIMARY", "GEMINI_API_KEY_SECONDARY"):
    if not os.environ.get(_key):
        raise RuntimeError(f"缺少環境變數 {_key}，請在 .env 檔中設定")

gemini_api_key_primary = os.environ["GEMINI_API_KEY_PRIMARY"]
gemini_api_key_secondary = os.environ["GEMINI_API_KEY_SECONDARY"]

# Vision client for image description (secondary key, as in Input.ipynb)
vision_client = genai.Client(api_key=gemini_api_key_secondary)

# Answer-generation model (primary key)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2,
    google_api_key=gemini_api_key_primary,
)

# --- CLIP model (loaded once) ----------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[rag_core] loading CLIP on {device} ...")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()
print("[rag_core] CLIP ready.")

VECTORSTORE_ROOT = os.environ.get("VECTORSTORE_ROOT", "./vectorstores")


# --- embeddings -------------------------------------------------------------
def embed_text(texts: list[str]) -> list[list[float]]:
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()


def embed_image(image_path: str) -> list[float]:
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy().tolist()


# --- per-user vector store --------------------------------------------------
def get_user_collection(user_id: str):
    chroma_client = chromadb.PersistentClient(path=f"{VECTORSTORE_ROOT}/{user_id}")
    return chroma_client.get_or_create_collection(user_id)


# --- image description (Gemini vision) -------------------------------------
def describe_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    response = vision_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            types.Part.from_text(text="""請詳細描述這張圖片中有什麼內容。注意: 1.不要使用粗體 2.圖片應該以左右反轉的方式描述 3. 在開頭輸出
            你判斷這是什麼科目(國文、英文、數學、物理、化學、地球科學、生物、地理、公民、歷史擇一)的筆記，格式為: 這是一份【科目】科的筆記。
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
    return description


def list_notes(user_id: str) -> list[dict]:
    collection = get_user_collection(user_id)
    results = collection.get(include=["documents", "metadatas"])
    return [
        {
            "id": id_,
            "type": (meta or {}).get("type"),
            "source": (meta or {}).get("source"),
            "preview": (doc or "")[:120],
        }
        for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
    ]


def delete_note(user_id: str, file_id: str) -> None:
    get_user_collection(user_id).delete(ids=[file_id])


# --- query ------------------------------------------------------------------
PROMPT_TEMPLATE = """你是一個問答助手，請根據以下提供的資料盡量完整且詳細的回答問題。
如果資料中沒有相關資訊，請回答「參考資料中沒有記載相關訊息」，不要自行編造。
注意: 1.請勿使用粗體 2.請勿套用latex格式 3.你得到的資料是基於原始資料的敘述，若敘述中含有使用者提問的項目，請將其統整後再回答。

【參考資料】
{context}

【使用者問題】
{question}

【回答】"""


def rag_query(user_id: str, question: str, top_k: int = 10) -> str:
    collection = get_user_collection(user_id)
    q_embedding = embed_text([question])[0]
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    docs = results["documents"][0] if results["documents"] else []
    if not docs:
        return "參考資料中沒有記載相關訊息"
    context = "\n---\n".join(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return llm.invoke(prompt).content

"""Core RAG logic shared by the API — refactored from the notebooks.

The embedding model is loaded once at import time (the server imports this
module at startup), so requests do not pay the model-loading cost.

Retrieval embedder is intfloat/multilingual-e5-base (繁中+英文皆強，跨語言檢索佳)，
與 embed_folder.py / test_query.py 一致，取代原本對中文較弱且只看前 77 個 token
的 CLIP 文字編碼器。E5 需要前綴：文件用 "passage: "，查詢用 "query: "。
"""
import os
import re
import io
import json
import time
import uuid
import shutil
import base64
import gc
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import chromadb
from chromadb.api.client import SharedSystemClient
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

# Without a timeout a request the server never answers blocks forever and the other
# key is never tried. A slow vision page takes well under this.
REQUEST_TIMEOUT_S = 120

_vision_clients = [genai.Client(api_key=k, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000))
                   for k in _VISION_KEY_ORDER]

# --- selectable models --------------------------------------------------------
# Ids verified against the live models.list() endpoint — note 3 Flash exists only as
# a -preview id. The UI offers exactly this list (served by /api/models) and requests
# are validated against it, so a caller can't push an arbitrary string to the API.
AVAILABLE_MODELS = [
    {"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash"},
    {"id": "gemini-3.7-flash", "label": "Gemini 3.7 Flash"},
    {"id": "gemini-3.6-flash", "label": "Gemini 3.6 Flash"},
    {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
    {"id": "gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash Lite"},
    {"id": "gemini-3.1-flash-lite", "label": "Gemini 3.1 Flash Lite（預設）"},
    {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash（preview）"},
    {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
]
_MODEL_IDS = {m["id"] for m in AVAILABLE_MODELS}
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")


def resolve_model(name: str | None) -> str:
    """Validate a requested model id against the allowlist, falling back to the
    default. Never let an unvetted string reach the Gemini API."""
    return name if name in _MODEL_IDS else DEFAULT_MODEL


# Chat clients are cached per (model, key): building one is cheap but not free, and a
# user flipping the dropdown back and forth shouldn't re-create them every request.
_llm_cache: dict[tuple[str, int], ChatGoogleGenerativeAI] = {}


def _get_llm(model: str, key_index: int) -> ChatGoogleGenerativeAI:
    cached = _llm_cache.get((model, key_index))
    if cached is None:
        cached = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            max_retries=2,
            timeout=REQUEST_TIMEOUT_S,
            google_api_key=_LLM_KEY_ORDER[key_index],
        )
        _llm_cache[(model, key_index)] = cached
    return cached


class ModelUnavailableError(RuntimeError):
    """Every key was tried and the model is over quota or temporarily overloaded.
    The API layer turns this into a 503 telling the user to retry or pick another
    model — which the dropdown now makes a real option."""


def _is_quota_error(exc: Exception) -> bool:
    """True if the error looks like an exceeded usage / rate limit."""
    text = str(exc).upper()
    return any(
        s in text
        for s in ("RESOURCE_EXHAUSTED", "QUOTA", "RATE LIMIT", "RATE_LIMIT", "429", "USAGE LIMIT")
    )


def _is_transient_error(exc: Exception) -> bool:
    """True for upstream hiccups worth retrying: 503 overload, 500 internal, timeouts.
    Popular models return 503 'high demand' at busy times — that is not our bug.
    The type name is included: a client timeout (httpx ReadTimeout) says only 'timed out'."""
    text = f"{type(exc).__name__} {exc}".upper()
    return any(
        s in text
        for s in ("UNAVAILABLE", "503", "OVERLOADED", "INTERNAL", "500 ", "DEADLINE", "TIMEOUT")
    )


def _is_retryable(exc: Exception) -> bool:
    return _is_quota_error(exc) or _is_transient_error(exc)


def _vision_generate(contents, config, model: str | None = None):
    """generate_content with automatic fail-over to the backup key on quota errors."""
    model = resolve_model(model)
    last_exc = None
    for i, client in enumerate(_vision_clients):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as e:
            last_exc = e
            if _is_retryable(e) and i < len(_vision_clients) - 1:
                print(f"[rag_core] vision 金鑰 #{i} 失敗（{type(e).__name__}），改用備援金鑰…")
                continue
            if _is_retryable(e):
                raise ModelUnavailableError(model) from e
            raise
    raise ModelUnavailableError(model) from last_exc


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


def _llm_invoke(prompt: str, model: str | None = None) -> str:
    """llm.invoke with automatic fail-over to the backup key on quota errors."""
    model = resolve_model(model)
    last_exc = None
    for i in range(len(_LLM_KEY_ORDER)):
        try:
            return _content_to_text(_get_llm(model, i).invoke(prompt).content)
        except Exception as e:
            last_exc = e
            if _is_retryable(e) and i < len(_LLM_KEY_ORDER) - 1:
                print(f"[rag_core] LLM 金鑰 #{i} 失敗（{type(e).__name__}），改用備援金鑰…")
                continue
            if _is_retryable(e):
                raise ModelUnavailableError(model) from e
            raise
    raise ModelUnavailableError(model) from last_exc

# --- embedding model (loaded once) -----------------------------------------
# Swappable via EMBED_MODEL so the eval harness can compare encoders without a code
# edit. Changing it changes the vector DIMENSION, so an existing collection embedded
# with another model must be rebuilt — get_user_collection refuses to mix them.
device = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")

# Prompt conventions are part of the model, not decoration: E5 was trained with
# "passage: "/"query: " prefixes, Qwen3-Embedding with a bare document and an
# instruction-prefixed query. Feeding a model the other family's format silently
# costs retrieval quality, so key the prefixes off the model name.
_QWEN_TASK = ("Given a question about a student's study notes, "
              "retrieve the note passage that answers it")
if "qwen3-embedding" in EMBED_MODEL.lower():
    DOC_PREFIX, QUERY_PREFIX = "", f"Instruct: {_QWEN_TASK}{chr(10)}Query: "
    # 0.6B in fp32 would not fit alongside Chroma on a 4 GB card; fp16 is what the
    # model card recommends anyway. Qwen3 pools the LAST token, so padding must be
    # on the left or short texts pool a pad token instead of their own content.
    # fp16 halves VRAM and is what the model card uses, but at 1024 dims its rounding is
    # enough to reorder near-ties, and which way it rounds depends on the cuBLAS kernel
    # the driver picks — so identical runs can disagree on a borderline chunk. Set
    # EMBED_DTYPE=float32 for measurement runs, where reproducibility outranks memory.
    _dtype = getattr(torch, os.environ.get("EMBED_DTYPE", "float16"))
    _LOAD_KW = {"model_kwargs": {"dtype": _dtype} if device == "cuda" else {},
                "processor_kwargs": {"padding_side": "left"}}
else:
    DOC_PREFIX, QUERY_PREFIX = "passage: ", "query: "
    _LOAD_KW = {}

print(f"[rag_core] loading {EMBED_MODEL} on {device} ...")
try:
    embedder = SentenceTransformer(EMBED_MODEL, device=device, **_LOAD_KW)
except TypeError:  # sentence-transformers < 5.4 spells it tokenizer_kwargs
    _LOAD_KW["tokenizer_kwargs"] = _LOAD_KW.pop("processor_kwargs")
    embedder = SentenceTransformer(EMBED_MODEL, device=device, **_LOAD_KW)
EMBED_DIM = (embedder.get_embedding_dimension()  # renamed in sentence-transformers 5.x
             if hasattr(embedder, "get_embedding_dimension")
             else embedder.get_sentence_embedding_dimension())
print(f"[rag_core] embedder ready ({EMBED_DIM} dims).")

VECTORSTORE_ROOT = os.environ.get("VECTORSTORE_ROOT", "./vectorstores")


# --- embeddings -------------------------------------------------------------
def embed_text(texts: list[str]) -> list[list[float]]:
    """Embed note passages, with whatever document prefix the model was trained on."""
    inputs = [DOC_PREFIX + t for t in texts]
    return embedder.encode(inputs, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a question, with whatever query prefix the model was trained on."""
    return embedder.encode([QUERY_PREFIX + text], normalize_embeddings=True)[0].tolist()


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


# --- categories -----------------------------------------------------------------------
# A user's notes are grouped into flat, independent categories. Each category is its own
# folder — its own Chroma store plus its own originals — under the user's folder:
#
#   <VECTORSTORE_ROOT>/<user>/                       uncategorized notes (the original store)
#   <VECTORSTORE_ROOT>/<user>/categories/<id>/       one category
#       category.json                                {"name": ..., "created": ...}
#
# Categories never nest: there is no way to create a category inside another. The folder
# is named by a stable id, not the display name, so renaming only rewrites category.json
# and never has to move a folder that Chroma holds open. Asking inside a category opens
# that one store and nothing else; only a question with no category reads every store.
CATEGORIES_DIR = "categories"
CATEGORY_FILE = "category.json"
CATEGORY_NAME_MAX = 30
_CATEGORY_ID = re.compile(r"^c[0-9a-f]{12}$")
_RESERVED_NAMES = {"未分類", "全部", "全部類別"}   # labels the page already uses
_store_lock = threading.RLock()


class CategoryError(ValueError):
    """A category request the user can fix (bad or duplicate name). Message is shown as-is."""
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class CategoryNotFoundError(LookupError):
    """The category id does not exist for this user."""


def _user_dir(user_id: str) -> str:
    return os.path.join(VECTORSTORE_ROOT, _safe_name(user_id))


def _store_dir(user_id: str, category: str | None = None) -> str:
    """Folder holding one store: the uncategorized store when category is None."""
    if category is None:
        return _user_dir(user_id)
    if not isinstance(category, str) or not _CATEGORY_ID.match(category):
        raise CategoryNotFoundError(category)
    return os.path.join(_user_dir(user_id), CATEGORIES_DIR, category)


def _read_category(user_id: str, category: str) -> dict | None:
    try:
        with open(os.path.join(_store_dir(user_id, category), CATEGORY_FILE), encoding="utf-8") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) and meta.get("name") else None
    except (OSError, ValueError, CategoryNotFoundError):
        return None


def _require_category(user_id: str, category: str | None) -> None:
    if category is not None and _read_category(user_id, category) is None:
        raise CategoryNotFoundError(category)


_dim_checked: set[str] = set()


def get_user_collection(user_id: str, category: str | None = None):
    """The Chroma collection for one store — uncategorized by default, or one category."""
    _require_category(user_id, category)
    name = _safe_name(user_id)
    path = _store_dir(user_id, category)
    chroma_client = chromadb.PersistentClient(path=path)
    col = chroma_client.get_or_create_collection(name)
    # A store embedded by another model has vectors of a different width. Chroma would
    # accept the write and fail later with an opaque dimension error, so say it here,
    # once per collection, in terms of the fix (re-upload the notes).
    if path not in _dim_checked:
        _dim_checked.add(path)
        got = col.peek(1).get("embeddings")
        if got is not None and len(got) and len(got[0]) != EMBED_DIM:
            raise RuntimeError(
                f"vectorstore '{path}' holds {len(got[0])}-dim vectors but {EMBED_MODEL} "
                f"produces {EMBED_DIM}. Delete {path} and re-add the "
                f"notes, or unset EMBED_MODEL.")
    return col


def list_categories(user_id: str) -> list[dict]:
    """The user's categories, oldest first."""
    base = os.path.join(_user_dir(user_id), CATEGORIES_DIR)
    if not os.path.isdir(base):
        return []
    found = []
    for entry in os.listdir(base):
        if not _CATEGORY_ID.match(entry):
            continue
        meta = _read_category(user_id, entry)
        if meta:
            found.append({"id": entry, "name": meta["name"], "created": meta.get("created", 0)})
    found.sort(key=lambda c: (c["created"], c["name"]))
    return [{"id": c["id"], "name": c["name"]} for c in found]


def _stores(user_id: str) -> list[str | None]:
    """Every store of this user: uncategorized first, then each category."""
    return [None] + [c["id"] for c in list_categories(user_id)]


def _clean_category_name(user_id: str, name: str, keep: str | None = None) -> str:
    name = re.sub(r"\s+", " ", name or "").strip()
    if not name:
        raise CategoryError("類別名稱不可為空")
    if len(name) > CATEGORY_NAME_MAX:
        raise CategoryError(f"類別名稱最多 {CATEGORY_NAME_MAX} 個字")
    # A slash would read as a sub-category, which this system deliberately does not have.
    if re.search(r"[/\\\x00-\x1f]", name):
        raise CategoryError("類別名稱不可包含 / 或 \\（類別不能有子類別）")
    if name in _RESERVED_NAMES:
        raise CategoryError(f"「{name}」是保留名稱，請換一個")
    if any(c["name"].casefold() == name.casefold() and c["id"] != keep
           for c in list_categories(user_id)):
        raise CategoryError(f"已經有名為「{name}」的類別", status=409)
    return name


def _write_category(user_id: str, category: str, meta: dict) -> None:
    with open(os.path.join(_store_dir(user_id, category), CATEGORY_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def create_category(user_id: str, name: str) -> dict:
    require_loaded()
    with _store_lock:
        name = _clean_category_name(user_id, name)
        category = "c" + uuid.uuid4().hex[:12]
        os.makedirs(_store_dir(user_id, category))
        _write_category(user_id, category, {"name": name, "created": time.time()})
    _persist_user(user_id)
    return {"id": category, "name": name}


def rename_category(user_id: str, category: str, name: str) -> dict:
    require_loaded()
    with _store_lock:
        meta = _read_category(user_id, category)
        if meta is None:
            raise CategoryNotFoundError(category)
        meta["name"] = _clean_category_name(user_id, name, keep=category)
        _write_category(user_id, category, meta)
    _persist_user(user_id)
    return {"id": category, "name": meta["name"]}


def delete_category(user_id: str, category: str) -> int:
    """Remove a category. Its notes are moved to uncategorized first, so deleting a
    category never deletes a note. Returns how many notes were moved."""
    require_loaded()
    with _store_lock:
        _require_category(user_id, category)
        col = get_user_collection(user_id, category)
        note_ids = sorted({_base_id(i) for i in col.get()["ids"]})
        for file_id in note_ids:
            _transfer_note(user_id, file_id, category, None)
        del col
        _dim_checked.discard(_store_dir(user_id, category))
        _remove_store_folder(_store_dir(user_id, category))
    _persist_user(user_id)
    return len(note_ids)


def _remove_store_folder(path: str) -> None:
    """Delete a category folder that Chroma may still hold open.

    Chroma keeps its SQLite file open for as long as a client for that path lives, and
    Windows refuses to delete an open file (Linux, where the Space runs, does not care).
    Closing every cached client is safe — the next request simply reopens its store — but
    the handles are released only once the Python objects are collected, so collect and
    retry briefly before giving up."""
    for attempt in range(10):
        gc.collect()
        SharedSystemClient.clear_system_cache()
        gc.collect()
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


# --- persistence: sync vectorstores to a private HF Dataset --------------------------
# Space storage is ephemeral, so without this, runtime uploads are lost on every
# restart/rebuild. We back the stores with a Dataset repo: restore on startup, push
# the changed user's collection after each add/delete. Needs HF_TOKEN with WRITE access.
HF_TOKEN = os.environ.get("HF_TOKEN")
VECTORSTORE_DATASET = os.environ.get("VECTORSTORE_DATASET", "timhelloyee/notehandler-store")
_hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

# Whether sync is expected at all. On a Hugging Face Space (which sets SPACE_ID) the
# container disk is wiped on every restart, so running without sync means silently
# losing whatever users upload. Locally, no token just means development.
SYNC_REQUIRED = bool(os.environ.get("SPACE_ID") or os.environ.get("NOTEHANDLER_REQUIRE_SYNC"))


# --- sync health ----------------------------------------------------------------------
# The app used to shrug off a failed restore and keep serving an empty store. A user saw
# "尚無筆記" — indistinguishable from genuinely having no notes — and anything they
# uploaded lived only on the container disk until the next restart erased it. That is
# what happened on 2026-09-14 when the Space's HF_TOKEN stopped being valid.
#
#   ok           restored from the Dataset; changes are synced
#   disabled     no token and none expected (local development)
#   unavailable  the user's notes could not be loaded. Writes are REFUSED: syncing now
#                would push an empty or partial store over the complete one in the Dataset
#   unsaved      loaded fine, but a later sync failed. The local store is still complete,
#                so the next write's sync repairs it — writes stay allowed
_sync = {"state": "disabled", "message": ""}
_unsynced_users: set[str] = set()


class SyncUnavailableError(RuntimeError):
    """Raised instead of writing when the store was never loaded from the Dataset."""


def _set_sync(state: str, message: str = "") -> None:
    _sync["state"], _sync["message"] = state, message


def sync_status() -> dict:
    return {"state": _sync["state"], "message": _sync["message"],
            "writable": _sync["state"] != "unavailable"}


def require_loaded() -> None:
    """Refuse to act on a store that was never restored — see _sync above."""
    if _sync["state"] == "unavailable":
        raise SyncUnavailableError(_sync["message"])


def _describe_sync_error(e: Exception) -> str:
    """A reason a user can act on. The raw exception goes to the log, not the page: it
    carries request ids and URLs that mean nothing to someone looking at their notes."""
    text = repr(e)
    if any(s in text for s in ("401", "403", "Invalid username or password",
                               "RepositoryNotFound", "Unauthorized")):
        return "雲端筆記庫拒絕了存取（HF_TOKEN 無效、已撤銷，或沒有這個 Dataset 的權限）"
    return "暫時無法連線到雲端筆記庫"


def _restore_from_dataset():
    """Pull saved vectorstores from the backing Dataset on startup so uploads survive
    restarts/rebuilds. First run (empty dataset) seeds it from the image's baked store.

    Failure no longer passes silently: it sets the sync state, which blocks writes and
    tells the user why their notes are missing."""
    if not _hf_api:
        if SYNC_REQUIRED:
            _set_sync("unavailable", "伺服器沒有設定 HF_TOKEN，無法載入雲端筆記庫")
            print("[rag_core] HF_TOKEN not set on a Space — notes unavailable, writes refused.")
        else:
            _set_sync("disabled", "本機模式，筆記不會同步到雲端")
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
        _set_sync("ok")
    except Exception as e:
        _set_sync("unavailable", _describe_sync_error(e))
        print(f"[rag_core] vectorstore restore FAILED ({e!r}); notes unavailable, writes refused.")


def _persist_all():
    """Push the whole vectorstores tree to the Dataset (used to seed it on first run)."""
    if not _hf_api or not os.path.isdir(VECTORSTORE_ROOT):
        return
    try:
        _hf_api.upload_folder(folder_path=VECTORSTORE_ROOT, path_in_repo=".",
                              repo_id=VECTORSTORE_DATASET, repo_type="dataset")
    except Exception as e:
        print(f"[rag_core] persist-all failed ({e!r}).")


def _persist_user(user_id: str) -> bool:
    """Push one user's collection back to the Dataset after a change. Returns whether the
    change is now safe in the Dataset (True when sync is disabled — there is nothing to
    lose that the user was promised)."""
    if not _hf_api:
        return True
    name = _safe_name(user_id)
    folder = f"{VECTORSTORE_ROOT}/{name}"
    if not os.path.isdir(folder):
        return True
    try:
        # delete_patterns makes the Dataset mirror the folder. Without it a deleted
        # category (or a note's original moved to another category) would stay in the
        # Dataset and be copied back on the next restart. Safe only because writes are
        # refused unless the folder was fully restored first (require_loaded).
        _hf_api.upload_folder(folder_path=folder, path_in_repo=name,
                              repo_id=VECTORSTORE_DATASET, repo_type="dataset",
                              delete_patterns="*")
    except Exception as e:
        _unsynced_users.add(name)
        if _sync["state"] != "unavailable":
            _set_sync("unsaved", "最近的變更還沒有同步到雲端：" + _describe_sync_error(e))
        print(f"[rag_core] persist '{name}' failed ({e!r}).")
        return False
    # The whole folder was uploaded, so this also carries any change an earlier failed
    # sync left behind for this user.
    _unsynced_users.discard(name)
    if _sync["state"] == "unsaved" and not _unsynced_users:
        _set_sync("ok")
    return True


# --- original files -----------------------------------------------------------
# Ingestion stores Gemini's *description*, not the note itself, so without this the
# source image is gone the moment the upload's temp file is deleted. Anything that later
# wants the pixels — re-running extraction with a better model, layout/geometry work, or
# simply showing the user their own page next to an answer — would need every note
# re-collected. Originals live inside the user's store folder, so the existing Dataset
# sync persists them with no extra plumbing.
ORIGINALS_DIR = "originals"


def _originals_path(user_id: str, file_id: str = "", category: str | None = None) -> str:
    base = os.path.join(_store_dir(user_id, category), ORIGINALS_DIR)
    return os.path.join(base, os.path.basename(file_id)) if file_id else base


def save_original(user_id: str, file_id: str, src_path: str,
                  category: str | None = None) -> str | None:
    """Keep the uploaded file alongside the vectors of the store it belongs to."""
    try:
        folder = _originals_path(user_id, category=category)
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, os.path.basename(file_id))
        shutil.copyfile(src_path, dest)
        return dest
    except Exception as e:
        print(f"[rag_core] could not keep original for '{file_id}' ({e!r}).")
        return None


def get_original(user_id: str, file_id: str, category: str | None = None) -> str | None:
    """Path to a note's stored source file in the given store, or None if it wasn't kept."""
    path = _originals_path(user_id, file_id, category)
    return path if os.path.isfile(path) else None


def find_original(user_id: str, file_id: str) -> str | None:
    """Like get_original, for a caller that does not know which category the note is in."""
    found, category = _find_note(user_id, file_id)
    return get_original(user_id, file_id, category) if found else None


def _delete_original(user_id: str, file_id: str, category: str | None = None) -> None:
    path = _originals_path(user_id, file_id, category)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError as e:
            print(f"[rag_core] could not remove original for '{file_id}' ({e!r}).")


# --- description cleanup ----------------------------------------------------
# Gemini occasionally emits placeholder/box glyphs for symbols it can't transcribe
# (□ ▢ ☐ �, etc.) and wraps bold around a stray space (** x ** which isn't valid
# Markdown). Normalize both at ingest so stored notes render cleanly.
_NOISE_GLYPHS = re.compile(r"[□▢▫◻◽☐⬜�][ \t]?")


def _clean_description(text: str) -> str:
    text = _NOISE_GLYPHS.sub("", text or "")
    text = re.sub(r"\*\*\s*([^*\n]+?)\s*\*\*", r"**\1**", text)  # ** x ** -> **x**
    return text


# --- chunking ----------------------------------------------------------------
# Long notes (especially PDF extractions) stored as one giant vector dilute retrieval:
# the embedding averages many topics, and e5-base truncates past its 512-token input
# anyway. Split into sentence-aligned chunks near the embedder's effective window.
CHUNK_CHARS = 450
CHUNK_OVERLAP = 50
_SENTENCE_END = re.compile(r"[。！？；!?\n]")


def _chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into <=size-char chunks ending on sentence boundaries (hard-cut for
    run-ons), with `overlap` chars carried across the seam. Same algorithm as
    embed_folder.py's chunk_text, sized for e5-base instead of NoteMind."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    boundaries = [m.end() for m in _SENTENCE_END.finditer(text)]
    n = len(text)
    chunks, start = [], 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = max((b for b in boundaries if start < b <= end), default=end)
            # Only snap back to a boundary that leaves a substantial chunk. Snapping to
            # one that sits just after `start` used to emit a sliver, and since the next
            # window then snapped back to the SAME boundary, `start` crawled forward one
            # character at a time: one 1,132-char section of 一下地科三段.pdf#p06 produced
            # 52 chunks of 414, 236, 48, 47, 46, 45 ... 1 chars. The overlap guard below
            # stopped it looping forever but not from filling the index with near
            # duplicates. Past the halfway point, hard-cut instead — which is what the
            # docstring always claimed happened for run-ons.
            if cut - start >= size // 2:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# --- structure-aware chunking (experimental, config B4) -------------------------
# The sliding window above cuts on sentence punctuation, which has nothing to do with a
# note's own layout: a sub-item can be severed from its heading and then never ranks,
# because e5's separation between "in my notes" and "not in my notes" is only ~0.02 and
# an orphaned fragment carries neither topic words nor keyword anchors. When the
# extraction preserves Markdown headings, split on those instead and prefix every chunk
# with its ancestor path ("地理 › 氣候 › 季風成因 ›"), which puts the missing topic words
# back into the embedded text.
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
ANCESTOR_DEPTH = 3  # deeper paths eat the 512-token budget for little gain


def _ancestor_prefix(path: list[str]) -> str:
    kept = [p for p in path if p][-ANCESTOR_DEPTH:]
    return (" › ".join(kept) + " ›\n") if kept else ""


def _common_path(a: list[str], b: list[str]) -> list[str]:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


def _merge_small_sections(sections: list[tuple[list[str], str]],
                          target: int) -> list[tuple[list[str], str]]:
    """Merge consecutive sections into subtree blocks of roughly `target` chars.

    Structure chunking splits on every heading, so a page whose headings sit close
    together yields many tiny chunks — measured on the 62-page eval corpus, B0's median
    chunk is 111 chars against B1's 421, and 139 of its 300 chunks are under 100. Chunk
    length is a confound: comparing a 111-char config with a 421-char one measures
    length at least as much as it measures structure, and neither can be attributed.

    Merging picks a COARSER LEVEL OF THE SAME TREE rather than abandoning structure.
    Sections merge only in document order, the block is labelled with the deepest
    ancestor path common to everything in it, and every merged section's own heading is
    written into the body — so no text is lost, the heading simply moves from the prefix
    into the content. Merging siblings alone is not enough: most adjacent sections are
    parent-and-child, so a sibling-only rule plateaus at a median of 224 chars.
    """
    out: list[tuple[list[str], str]] = []
    for path, body in sections:
        block = f"{path[-1]}\n{body}" if path else body   # own heading joins the body
        parent = path[:-1]
        if out:
            cur_path, cur_body = out[-1]
            merged = _common_path(cur_path, parent)
            room = max(target - len(_ancestor_prefix(merged)), 120)
            if len(cur_body) + 1 + len(block) <= room:
                out[-1] = (merged, cur_body + "\n" + block)
                continue
        out.append((parent, block))
    return out


def chunk_by_structure(text: str, size: int = CHUNK_CHARS,
                       overlap: int = CHUNK_OVERLAP,
                       merge_target: int = 0) -> list[str]:
    """Chunk on Markdown heading hierarchy, prefixing each chunk with its ancestor path.

    Falls back to the sliding window when the text carries no headings, so this is safe
    to run over prose extractions too.

    `merge_target` > 0 first merges adjacent sibling sections up to that many chars
    (see _merge_small_sections). It defaults to 0 — off — so existing stores and the
    already-measured B4/B0 configurations keep byte-identical behaviour.
    """
    text = (text or "").strip()
    if not text:
        return []
    sections: list[tuple[list[str], str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((list(stack), body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            del stack[level - 1:]
            stack.append(m.group(2).strip())
        else:
            buf.append(line)
    flush()

    if not any(path for path, _ in sections):
        return _chunk_text(text, size, overlap)   # no headings — nothing to preserve

    if merge_target:
        sections = _merge_small_sections(sections, merge_target)

    chunks: list[str] = []
    for path, body in sections:
        prefix = _ancestor_prefix(path)
        room = max(size - len(prefix), 120)       # keep the prefix from starving the body
        for part in _chunk_text(body, room, overlap):
            chunks.append(prefix + part)
    return chunks


# --- keywords -----------------------------------------------------------------
# The Gemini describe prompts end the description with a "關鍵字：…" line (free — same
# API call). Stored per chunk in metadata, the keywords give exact-match anchors that
# pure cosine similarity lacks; rag_query uses them as a small ranking boost.
KEYWORDS_PER_NOTE = 10   # the describe prompts ask for 5-10; the single-page and
                         # assembled-PDF paths must not disagree on the cap.
_KEYWORD_LINE = re.compile(r"關鍵字[:：]\s*(.+)")


def _keywords_of(description: str) -> str:
    """Extract the trailing 關鍵字 line as a 、-joined string ('' if absent)."""
    m = _KEYWORD_LINE.search(description or "")
    if not m:
        return ""
    kws = [k.strip() for k in re.split(r"[、,，;；/]\s*", m.group(1)) if k.strip()]
    return "、".join(kws[:KEYWORDS_PER_NOTE])


# Config B4 of the structure experiment: ask the model to emit the note's hierarchy as
# nested Markdown instead of prose, so chunk_by_structure has real headings to split on.
# This is the cheap alternative that must be ruled out before any layout-detection work.
STRUCTURED_RULE = """
7. 請用巢狀 Markdown 還原筆記本身的階層：整頁主題用 #，大標題用 ##，次標題用 ###，條列項目用 - 並以縮排表示層級。標題必須採用筆記上實際出現的文字，不要自行發明分類；若某區塊沒有標題，可用該區第一行作為標題。
"""


# --- image description (Gemini vision) -------------------------------------
def describe_image(image_path: str, model: str | None = None,
                   structured: bool = False) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    prompt = """請以繁體中文，完整且忠實地擷取並整理這張筆記圖片中的所有內容，不要遺漏。

請遵守以下規則：
1. 在最開頭先判斷這是什麼科目（國文、英文、數學、物理、化學、地球科學、生物、地理、公民、歷史、程式設計擇一），格式為：這是一份【科目】科的筆記。整頁沒有任何可辨識的內容時，科目寫「無法辨識」。
2. 逐項記錄圖片中「實際出現的內容」，包括：定義、公式、定理、重要名詞、例題與其完整條件（已知數值、單位、所求）。不要只寫「涉及……的計算」這類概括性描述，而要寫出真正的內容與細節。
3. 數學式請使用 LaTeX 表示（行內以 $...$，獨立公式以 $$...$$ 包住，例如 $F = \\dfrac{G m_1 m_2}{r^2}$），以忠實保留上下標與符號。
4. 只描述圖片中實際出現的內容，不要編造；無法辨識處請標註為（無法辨識）。
5. 盡量保留原文的關鍵用語與數據，以便後續能據此回答細節問題。
6. 在最後另起一行，以「關鍵字：」開頭，列出 5 到 10 個此筆記最重要的關鍵字（以「、」分隔），優先選擇專有名詞與術語。
"""
    if structured:
        prompt += STRUCTURED_RULE
    response = _vision_generate(
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            types.Part.from_text(text=prompt),
        ],
        config={"temperature": 0, "top_p": 0.95, "top_k": 20},
        model=model,
    )
    return _clean_description(response.text)


# Subject tag Gemini is asked to emit at the top of every extraction.
_SUBJECT_TAG = re.compile(r"這是一份[【\[]?([^】\]\s]+?)[】\]]?科的筆記")


def subject_of(text: str) -> str:
    """The 科目 Gemini assigned to a note ('?' when the tag is missing)."""
    m = _SUBJECT_TAG.search(text or "")
    return m.group(1) if m else "?"


# Per-page variant of the PDF prompt: one page at a time, tables transcribed not skipped.
PDF_PAGE_PROMPT = """請以繁體中文，完整且忠實地擷取並整理這一頁筆記的所有內容，不要遺漏。

請遵守以下規則：
1. 在最開頭先判斷這是什麼科目（國文、英文、數學、物理、化學、地球科學、生物、地理、公民、歷史、程式設計擇一），格式為：這是一份【科目】科的筆記。整頁沒有任何可辨識的內容時，科目寫「無法辨識」。
2. 逐項記錄這一頁「實際出現的內容」，包括：定義、公式、定理、重要名詞、例題與其完整條件（已知數值、單位、所求）。不要只寫「涉及……的計算」這類概括性描述，而要寫出真正的內容與細節。表格請逐列轉成文字，不可略過。
3. 數學式請使用 LaTeX 表示（行內以 $...$，獨立公式以 $$...$$ 包住），以忠實保留上下標與符號。程式碼請用 ``` 圍起來並保留縮排。
4. 只描述這一頁實際出現的內容，不要編造，也不要補充其他頁的內容；無法辨識處請標註為（無法辨識）。
5. 這是手寫筆記，字跡可能潦草；請盡力辨識並保留原文的關鍵用語與數據。
6. 在最後另起一行，以「關鍵字：」開頭，列出 5 到 10 個此頁最重要的關鍵字（以「、」分隔），優先選擇專有名詞與術語。
"""

# --- PDF description (Gemini document understanding) ------------------------
# PDFs are described one page at a time. A whole multi-page document in a single call is
# both lossy (a 4-page exam came back with its table replaced by "（表略）") and, past
# roughly 10 MB, impossible: base64-encoded it exceeds the inline request limit, so large
# scans failed outright. Pages are split with pypdf — no rasterising, nothing re-encoded —
# and described concurrently so a 23-page upload stays a normal request rather than a
# two-minute stall.
PDF_PAGE_WORKERS = int(os.environ.get("PDF_PAGE_WORKERS", 4))
_BLANK_MARKERS = ("沒有任何文字", "空白", "無內容")


def is_blank_page(text: str) -> bool:
    """True for a photographed blank page. Gemini reports these honestly; indexing them
    would put 'this page is empty' into the store as a retrievable passage."""
    t = text or ""
    return subject_of(t) == "無法辨識" and len(t) < 160 and any(m in t for m in _BLANK_MARKERS)


def _pdf_page_bytes(pdf_path: str) -> list[bytes]:
    """Each page as a standalone one-page PDF."""
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(pdf_path)
    out = []
    for page in reader.pages:
        writer = PdfWriter()
        writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        out.append(buf.getvalue())
    return out


def describe_pdf_page(data: bytes, model: str | None = None,
                      structured: bool = False) -> str:
    """Describe one page of a PDF."""
    prompt = PDF_PAGE_PROMPT + (STRUCTURED_RULE if structured else "")
    response = _vision_generate(
        contents=[types.Part.from_bytes(data=data, mime_type="application/pdf"),
                  types.Part.from_text(text=prompt)],
        config={"temperature": 0, "top_p": 0.95, "top_k": 20},
        model=model,
    )
    return _clean_description(response.text or "")


def _strip_page_furniture(text: str) -> str:
    """Drop the per-page subject tag and keyword line; the assembled note carries its own."""
    out = _SUBJECT_TAG.sub("", text or "", count=1)
    out = _KEYWORD_LINE.sub("", out)
    return out.strip()


def describe_pdf(pdf_path: str, model: str | None = None,
                 structured: bool = False) -> str:
    """Describe a PDF page by page and assemble one note.

    Pages become '## 第 N 頁' sections, which gives the structure chunker real headings
    and keeps page provenance in the stored text. Blank pages are skipped.
    """
    pages = _pdf_page_bytes(pdf_path)
    if not pages:
        return ""
    workers = max(1, min(PDF_PAGE_WORKERS, len(pages)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        texts = list(pool.map(
            lambda d: describe_pdf_page(d, model=model, structured=structured), pages))

    kept = [(i + 1, t) for i, t in enumerate(texts) if t and not is_blank_page(t)]
    if not kept:
        return ""

    subject = next((subject_of(t) for _, t in kept if subject_of(t) != "?"), "?")
    keywords: list[str] = []
    for _, t in kept:
        for k in (_keywords_of(t) or "").split("、"):
            if k and k not in keywords:
                keywords.append(k)

    nl = chr(10)
    body = (nl + nl).join(
        f'## 第 {n} 頁' + nl + _strip_page_furniture(t) for n, t in kept)
    head = f'這是一份【{subject}】科的筆記。' + nl + nl if subject != '?' else ''
    tail = (nl + nl + '關鍵字：' + '、'.join(keywords[:KEYWORDS_PER_NOTE])) if keywords else ''
    return head + body + tail


# --- ingestion --------------------------------------------------------------
def _base_id(chunk_id: str) -> str:
    """'note.pdf#3' -> 'note.pdf'; ids without a numeric #suffix pass through."""
    base, sep, suffix = chunk_id.rpartition("#")
    return base if sep and suffix.isdigit() else chunk_id


def _note_exists(collection, file_id: str) -> bool:
    # Whole-doc notes use the bare id; chunked notes start at '<id>#0'.
    return bool(collection.get(ids=[file_id, f"{file_id}#0"])["ids"])


def _find_note(user_id: str, file_id: str) -> tuple[bool, str | None]:
    """(found, category) for a note id. Ids are unique across all of a user's categories,
    so managing a note — open, edit, move, delete — needs only its id. This looks in
    every store; answering a question never goes through here."""
    for category in _stores(user_id):
        if _note_exists(get_user_collection(user_id, category), file_id):
            return True, category
    return False, None


def _add_note(user_id: str, file_id: str, text: str, kind: str, source: str,
              structured: bool = False, merge_target: int = 0,
              chunks: list[str] | None = None, category: str | None = None) -> bool:
    """Chunk, embed, and store one note (one vector per chunk) in the given category.

    Returns True if added, False if a note with this id already exists in ANY category.
    Pre-chunking data (one whole document under the bare id) coexists fine:
    _base_id/list/delete handle both shapes.

    `chunks` lets a caller supply its own chunking and reuse the embed/store path — the
    eval harness needs it to build control conditions (e.g. the same chunks with their
    ancestor prefixes removed) without that logic living here.
    """
    require_loaded()
    collection = get_user_collection(user_id, category)
    if _find_note(user_id, file_id)[0]:
        return False
    if chunks is None:
        chunks = (chunk_by_structure(text, merge_target=merge_target) if structured
                  else _chunk_text(text))
    if not chunks:
        return False
    keywords = _keywords_of(text)
    ids = [file_id] if len(chunks) == 1 else [f"{file_id}#{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embed_text(chunks),
        documents=chunks,
        metadatas=[
            {"type": kind, "source": source, "chunk": i, "n_chunks": len(chunks),
             "keywords": keywords}
            for i in range(len(chunks))
        ],
    )
    _persist_user(user_id)
    return True


def add_text_note(user_id: str, file_id: str, content: str, source: str,
                  category: str | None = None) -> bool:
    """Returns True if added, False if it already existed."""
    return _add_note(user_id, file_id, content, "text", source, category=category)


def _add_described_note(user_id: str, file_id: str, src_path: str, description: str,
                        kind: str, source: str, category: str | None = None) -> None:
    """Store a Gemini description together with the file it came from.

    The original is written BEFORE _add_note runs, because _add_note is what syncs the
    user's folder. main.py used to save the original afterwards, so the upload that went
    to the Dataset never contained it — the file only reached the Dataset on the user's
    next change, and a restart before then lost it for good.
    """
    if not description:
        return
    save_original(user_id, file_id, src_path, category)
    try:
        added = _add_note(user_id, file_id, description, kind, source, category=category)
    except Exception:
        _delete_original(user_id, file_id, category)
        raise
    if not added:
        _delete_original(user_id, file_id, category)


def add_image_note(user_id: str, file_id: str, image_path: str, source: str,
                   model: str | None = None, category: str | None = None) -> str:
    """Describes the image with Gemini, embeds the description, stores it with the image.

    Returns the generated description (or "" if the note already existed).
    """
    require_loaded()   # before the Gemini call: no point spending quota on a refused write
    _require_category(user_id, category)
    if _find_note(user_id, file_id)[0]:
        return ""
    description = describe_image(image_path, model=model)
    _add_described_note(user_id, file_id, image_path, description, "image", source, category)
    return description


def add_pdf_note(user_id: str, file_id: str, pdf_path: str, source: str,
                 model: str | None = None, category: str | None = None) -> str:
    """Describes the PDF with Gemini, embeds the description, stores it with the PDF.

    Returns the generated description (or "" if the note already existed).
    """
    require_loaded()
    _require_category(user_id, category)
    if _find_note(user_id, file_id)[0]:
        return ""
    description = describe_pdf(pdf_path, model=model)
    _add_described_note(user_id, file_id, pdf_path, description, "pdf", source, category)
    return description


def _join_chunks(rows: list[tuple[str, dict]]) -> str:
    """Rejoin a note's chunks into its stored text, undoing CHUNK_OVERLAP at each seam.

    Chunks are stored with an overlap so retrieval keeps context across boundaries; when
    showing or editing the note we want the text back, not the duplicated seams.
    """
    rows = sorted(rows, key=lambda r: int((r[1] or {}).get("chunk") or 0))
    text = ""
    for doc, _ in rows:
        doc = doc or ""
        if not text:
            text = doc
            continue
        # Longest suffix of what we have that is also a prefix of the next chunk.
        window = min(len(text), len(doc), CHUNK_OVERLAP + 40)
        overlap = next((k for k in range(window, 0, -1) if text.endswith(doc[:k])), 0)
        text += doc[overlap:]
    return text


def get_note_text(user_id: str, file_id: str) -> str | None:
    """Full stored text of a note (None if it doesn't exist)."""
    found, category = _find_note(user_id, file_id)
    if not found:
        return None
    got = get_user_collection(user_id, category).get(include=["documents", "metadatas"])
    rows = [(d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
            if _base_id(i) == file_id]
    return _join_chunks(rows) if rows else None


def update_note(user_id: str, file_id: str, new_text: str) -> bool:
    """Replace a note's text with a user-corrected version: re-chunk and re-embed, keeping
    its type, source and category. No Gemini call — the description was already generated;
    this is the human fixing what the model misread. False if the note doesn't exist.
    """
    require_loaded()   # checked before the delete below, not just inside _add_note
    found, category = _find_note(user_id, file_id)
    if not found:
        return False
    collection = get_user_collection(user_id, category)
    got = collection.get(include=["metadatas"])
    ids = [i for i in got["ids"] if _base_id(i) == file_id]
    meta = next((m for i, m in zip(got["ids"], got["metadatas"])
                 if _base_id(i) == file_id), {}) or {}
    collection.delete(ids=ids)
    return _add_note(user_id, file_id, new_text,
                     meta.get("type", "text"), meta.get("source", file_id), category=category)


def _transfer_note(user_id: str, file_id: str, src: str | None, dst: str | None) -> None:
    """Move a note's chunks and original from one store to another. The stored vectors are
    copied as they are, so moving costs no embedding and no Gemini call."""
    source = get_user_collection(user_id, src)
    got = source.get(include=["embeddings", "documents", "metadatas"])
    rows = [k for k, i in enumerate(got["ids"]) if _base_id(i) == file_id]
    if not rows:
        return
    embeddings = got["embeddings"]
    get_user_collection(user_id, dst).add(
        ids=[got["ids"][k] for k in rows],
        embeddings=[[float(x) for x in embeddings[k]] for k in rows],
        documents=[got["documents"][k] for k in rows],
        metadatas=[got["metadatas"][k] for k in rows],
    )
    # Added to the destination before removing from the source: a failure in between
    # leaves a duplicate the user can see, never a note that is gone.
    source.delete(ids=[got["ids"][k] for k in rows])
    original = get_original(user_id, file_id, src)
    if original:
        folder = _originals_path(user_id, category=dst)
        os.makedirs(folder, exist_ok=True)
        shutil.move(original, os.path.join(folder, os.path.basename(file_id)))


def move_note(user_id: str, file_id: str, category: str | None) -> bool:
    """Put a note into a category (None = uncategorized). False if the note doesn't exist."""
    require_loaded()
    with _store_lock:
        _require_category(user_id, category)
        found, current = _find_note(user_id, file_id)
        if not found:
            return False
        if current == category:
            return True
        _transfer_note(user_id, file_id, current, category)
    _persist_user(user_id)
    return True


def list_notes(user_id: str) -> list[dict]:
    """One entry per note across every category, aggregating chunked notes
    (preview = first chunk). `category` is the category id, or None for uncategorized."""
    out: list[dict] = []
    for category in _stores(user_id):
        results = get_user_collection(user_id, category).get(include=["documents", "metadatas"])
        notes: dict[str, dict] = {}
        for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            meta = meta or {}
            base = _base_id(id_)
            entry = notes.setdefault(base, {
                "id": base, "type": meta.get("type"), "source": meta.get("source"),
                "category": category, "preview": "", "_chunks": int(meta.get("n_chunks") or 1),
            })
            if meta.get("chunk", 0) == 0:  # whole-doc notes have no 'chunk' key -> 0
                doc = doc or ""
                truncated = len(doc) > 240 or entry["_chunks"] > 1
                entry["preview"] = doc[:240] + ("…" if truncated else "")
        # has_original tells the editor whether to draw the side-by-side original pane.
        # Reported here rather than letting the UI probe /original per note, which would
        # cost a 404 round trip for every text note (they never have one) and log noise.
        for base, entry in notes.items():
            entry["has_original"] = get_original(user_id, base, category) is not None
        out.extend({k: v for k, v in e.items() if not k.startswith("_")} for e in notes.values())
    return out


def delete_note(user_id: str, file_id: str) -> bool:
    """Deletes a note and all its chunks. Returns False if it wasn't found."""
    require_loaded()
    found, category = _find_note(user_id, file_id)
    if not found:
        return False
    collection = get_user_collection(user_id, category)
    ids = [i for i in collection.get()["ids"] if _base_id(i) == file_id]
    collection.delete(ids=ids)
    _delete_original(user_id, file_id, category)
    _persist_user(user_id)
    return True


# --- query ------------------------------------------------------------------
PROMPT_TEMPLATE = """你是一個專業的學習筆記問答助手。請「僅」根據下列【參考資料】回答問題，並盡量完整、詳細、有條理。
如果參考資料中沒有相關資訊，請直接回答「參考資料中沒有記載相關訊息」，絕對不要自行編造或補充資料以外的內容。

回答時請遵守：
1. 介面支援 Markdown 與 LaTeX，請善用排版讓答案更易讀：可使用粗體（**重點**）、斜體（*強調*）、條列、標題等 Markdown 語法；數學式請使用 LaTeX，行內公式以單個 $ 包住（例如 $F = \\dfrac{G m_1 m_2}{r^2}$），獨立公式以 $$ 包住。
2. 參考資料是對原始筆記（文字、圖片或 PDF）逐字擷取後的敘述，請先理解其內容，再依使用者問題作答。
3. 若使用者詢問的項目散落在多段資料中，請將相關內容整合後再條列回答，並盡量保留原始的關鍵數據與用語，不要省略重要細節。
4. 請以繁體中文回答。
5. 每一個重點的句尾都要標註出處，格式為〔來源：檔名 第 N 頁〕；若該筆記沒有「## 第 N 頁」標記，寫〔來源：檔名〕。檔名必須與參考資料中【來源：…】的檔名完全相同，頁碼必須是該筆記中實際出現的「## 第 N 頁」，不可推測。一句話有多個出處時，分別標註。回答「參考資料中沒有記載相關訊息」時不需要標註出處。

【參考資料】
{context}

【使用者問題】
{question}

【回答】"""


# Chunk selection takes two steps, because one number cannot do both jobs.
#
# Chroma reports squared L2 over normalized vectors — exactly 2-2·cos — and e5
# similarities are range-compressed (unrelated Traditional Chinese still scores ~0.77
# cosine). Measured against real notes: a question whose answer IS present has a best
# distance of 0.20-0.36, while a question about an absent topic bottoms out at 0.37+.
#
# 1. RELEVANCE_CEILING gates the *query*. If even the best chunk is past it, nothing in
#    the notes is on topic, so hand over only that chunk and let the prompt's
#    "沒有記載相關訊息" rule fire rather than burying the model in unrelated notes.
# 2. RELATIVE_MARGIN then decides *how many* chunks to keep, measured from the best hit.
#    This adapts to the query — a clear winner stays tight, a broad question matching
#    many notes evenly keeps them all — and it survives changing the embedding model,
#    which a fixed per-chunk cutoff does not.
#
# The previous single MAX_DISTANCE=0.35 conflated the two: as a per-chunk filter it sat
# below almost every genuine answer, so nearly every query collapsed to the single-best
# fallback and broad questions ("涵蓋哪些科目") saw exactly one note.
#
# Both numbers are raw distances, so both belong to the EMBEDDER, not to the app: e5
# packs unrelated Traditional Chinese around 0.77 cosine, Qwen3 pushes it to ~0.31, and
# the same 0.38 ceiling that admits every real answer under e5 rejects all of them under
# Qwen3 (measured worst present distance 1.125). Calibrate a new model before adding it
# here (the calibration script, eval/calibrate_gate.py, was retired with the retrieval
# research and is in git history); the env vars override for experiments.
_GATE_BY_MODEL = {                     # model: (ceiling, margin)
    "intfloat/multilingual-e5-base": (0.38, 0.10),
    # ceiling 1.00 = best single split of present (max 1.125) vs absent (min 0.830);
    # margin 0.30 keeps ~5 chunks, matching what 0.10 kept under e5's compressed scale.
    "Qwen/Qwen3-Embedding-0.6B": (1.00, 0.30),
}
if EMBED_MODEL not in _GATE_BY_MODEL:
    print(f"[rag_core] WARNING: no calibrated gate for {EMBED_MODEL}; using e5's values. "
          f"Calibrate its gate and add it to _GATE_BY_MODEL.")
_ceiling, _margin = _GATE_BY_MODEL.get(EMBED_MODEL, (0.38, 0.10))
RELEVANCE_CEILING = float(os.environ.get("RELEVANCE_CEILING", _ceiling))
RELATIVE_MARGIN = float(os.environ.get("RELATIVE_MARGIN", _margin))


# Keyword-overlap boost: stored keywords that literally appear in the question lower
# the effective distance. Substring matching works well for Chinese (no word spacing),
# and exact term hits are precisely the signal cosine similarity dilutes. Kept small so
# it reorders near-ties instead of overriding semantic ranking. Unlike the gate above it
# is NOT scale-bound in practice: swept over the 29 chunk probes it is flat from 0.03 to
# 0.15 on both encoders (e5 0.813 -> 0.819, Qwen3 0.897 -> 0.913 chunk MRR, then no
# further change), so one constant serves both and the smallest one that gets the gain
# wins.
KEYWORD_BONUS = float(os.environ.get("KEYWORD_BONUS", 0.03))
KEYWORD_BONUS_CAP = 3


def _effective_distance(dist: float, meta: dict, question: str) -> float:
    kws = ((meta or {}).get("keywords") or "").split("、")
    hits = sum(1 for k in kws if len(k) >= 2 and k in question)
    return dist - min(hits, KEYWORD_BONUS_CAP) * KEYWORD_BONUS


def _retrieve_rows(user_id: str, question: str, top_k: int = 10,
                   stores: list[str | None] | None = None) -> list[dict]:
    """Top-k chunks for a question, keyword-boosted and distance-filtered, each with the
    id, metadata and store it came from (needed to say where an answer came from).

    `stores` limits the search to those categories (None in the list = uncategorized).
    Left as None it searches every store — only for a question asked with no category."""
    if stores is None:
        stores = _stores(user_id)
    q_embedding = embed_query(question)
    pooled: list[dict] = []
    for store in stores:
        collection = get_user_collection(user_id, store)
        if collection.count() == 0:
            continue
        results = collection.query(
            query_embeddings=[q_embedding], n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        docs = results["documents"][0] if results["documents"] else []
        ids = results["ids"][0] if results.get("ids") else [""] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        metas = (results.get("metadatas") or [[]])[0] or [{}] * len(docs)
        for id_, doc, dist, meta in zip(ids, docs, dists, metas):
            pooled.append({"id": id_, "doc": doc, "meta": meta or {}, "store": store,
                           "eff": _effective_distance(dist, meta, question)})
    if not pooled:
        return []
    # Every store uses the same embedder, so distances are comparable across them.
    scored = sorted(pooled, key=lambda r: r["eff"])[:top_k]
    best = scored[0]["eff"]
    if best > RELEVANCE_CEILING:
        return scored[:1]  # nothing on topic — one chunk is context enough to refuse
    return [r for r in scored if r["eff"] <= best + RELATIVE_MARGIN] or scored[:1]


def _retrieve(user_id: str, question: str, top_k: int = 10,
              stores: list[str | None] | None = None) -> list[str]:
    """The retrieved chunk texts alone (the eval harness measures these)."""
    return [r["doc"] for r in _retrieve_rows(user_id, question, top_k, stores)]


# Whole-corpus mode. When every note fits comfortably in the model's context window,
# retrieval can only lose information: any ranking mistake drops a note that would have
# fitted anyway. Gemini's window is ~1M tokens, so this cap (~20k tokens of Chinese) is
# far below it and keeps a query cheap, while covering a student's notes for a long time.
# Above the cap we fall back to retrieval, which is what makes the app keep scaling.
# Set WHOLE_CORPUS_CHARS=0 to force the retrieval path — required when measuring
# retrieval quality, since otherwise a small corpus never exercises the retriever and
# every chunking configuration scores identically.
WHOLE_CORPUS_CHARS = int(os.environ.get("WHOLE_CORPUS_CHARS", 60_000))


def _whole_corpus(collections) -> list[str] | None:
    """The note texts of _whole_corpus_notes, without their ids."""
    notes = _whole_corpus_notes(collections)
    return None if notes is None else [text for _, text in notes]


def _whole_corpus_notes(collections) -> list[tuple[str, str]] | None:
    """(note id, text) per note — chunks rejoined, overlaps removed — or None when the
    corpus is too big to send whole. Takes one collection or a list of them (one per
    category being asked); the size limit applies to what would actually be sent.

    Deliberately *not* the raw chunk list: chunks overlap by CHUNK_OVERLAP, so feeding
    them verbatim would repeat text at every seam and split each note across arbitrary
    450-char boundaries. Rejoining restores the note as it was extracted, which keeps the
    document's own structure (headings, numbered questions) intact for the model.
    """
    if WHOLE_CORPUS_CHARS <= 0:
        return None
    if not isinstance(collections, (list, tuple)):
        collections = [collections]
    by_note: dict[str, list[tuple[str, dict]]] = {}
    total = 0
    for collection in collections:
        got = collection.get(include=["documents", "metadatas"])
        docs = got["documents"] or []
        metas = got["metadatas"] or [{}] * len(docs)
        total += sum(len(d or "") for d in docs)
        for id_, doc, meta in zip(got["ids"], docs, metas):
            by_note.setdefault(_base_id(id_), []).append((doc, meta or {}))
    if not by_note or total > WHOLE_CORPUS_CHARS:
        return None
    return [(note_id, text) for note_id, rows in sorted(by_note.items())
            if (text := _join_chunks(rows))]


# --- sources ------------------------------------------------------------------------
# An answer the user cannot trace back to their own page is an answer they have to take
# on trust — and the transcription it rests on is a model's guess at their handwriting.
# So every note sent to the model is labelled with its file name (PDF pages already carry
# "## 第 N 頁"), the prompt asks for 〔來源：檔名 第 N 頁〕 after each point, and the server
# checks every citation against what was actually sent: a file or page that was not in
# the context is removed rather than shown, since a made-up source is worse than none.
REFUSAL = "參考資料中沒有記載相關訊息"
_PAGE_HEADING = re.compile(r"^##\s*第\s*(\d+)\s*頁", re.MULTILINE)
_CITATION = re.compile(r"[〔【\[]\s*來源\s*[:：]\s*([^〔〕【】\[\]\n]+?)\s*[〕】\]]")
_PAGE_REF = re.compile(r"第\s*([0-9][0-9\s、,，~～\-–]*)\s*頁?")
# Split a citation into one part per file: at ；/; always, and at 、/, only when what
# follows is not a digit, so "第 3、5 頁" stays one part.
_CITATION_SPLIT = re.compile(r"[；;]|[、,，](?=\s*[^\d\s])")


def source_marker(note_id: str, page: int | None) -> str:
    """The one citation spelling the page and every client rely on."""
    return f"〔來源：{note_id} 第 {page} 頁〕" if page else f"〔來源：{note_id}〕"


def _page_before_chunk(collection, note_id: str, index: int) -> int | None:
    """The page a retrieved chunk starts on: the last page heading in the chunks before it."""
    if index <= 0:
        return None
    got = collection.get(ids=[f"{note_id}#{i}" for i in range(index)],
                         include=["documents", "metadatas"])
    text = _join_chunks(list(zip(got["documents"] or [], got["metadatas"] or [])))
    pages = _PAGE_HEADING.findall(text)
    return int(pages[-1]) if pages else None


def _retrieval_notes(user_id: str, rows: list[dict]) -> list[tuple[str, str]]:
    """Group retrieved chunks by note, in reading order. A chunk that starts mid-page is
    given its page heading back, so the model can cite the page it came from."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        note_id = _base_id(r["id"])
        index = int(r["meta"].get("chunk") or 0)
        doc = r["doc"] or ""
        if index and not _PAGE_HEADING.match(doc.lstrip()):
            page = _page_before_chunk(get_user_collection(user_id, r["store"]), note_id, index)
            if page:
                doc = f"## 第 {page} 頁（續）\n{doc}"
        grouped.setdefault(note_id, []).append((index, doc))
    return [(note_id, "\n…\n".join(doc for _, doc in sorted(parts)))
            for note_id, parts in grouped.items()]


def _format_context(notes: list[tuple[str, str]]) -> str:
    return "\n---\n".join(f"【來源：{note_id}】\n{text}" for note_id, text in notes)


def _expand_pages(text: str) -> list[list[int]]:
    """Page references as groups: [[3], [5]] for 第 3、5 頁, [[3, 4, 5]] for 第 3-5 頁.
    A range is one reference — it is valid if any page in it exists."""
    groups: list[list[int]] = []
    for ref in _PAGE_REF.findall(text):
        for token in re.split(r"[、,，\s]+", ref.strip()):
            span = re.split(r"[~～\-–]", token)
            if len(span) == 2 and span[0].isdigit() and span[1].isdigit():
                lo, hi = int(span[0]), int(span[1])
                groups.append(list(range(lo, hi + 1)) if 0 <= hi - lo <= 50 else [lo, hi])
            elif token.isdigit():
                groups.append([int(token)])
    return groups


def _resolve_citations(answer: str, notes: list[tuple[str, str]]) -> tuple[str, list[dict], int]:
    """Check every citation in the answer against the notes that were sent.

    Returns the answer with each valid citation rewritten to source_marker() form and
    each invalid one removed, the distinct sources in order of first appearance, and how
    many citations were dropped."""
    pages = {note_id: {int(p) for p in _PAGE_HEADING.findall(text)} for note_id, text in notes}
    ids_longest_first = sorted(pages, key=len, reverse=True)
    sources: list[dict] = []
    seen: set[tuple[str, int | None]] = set()
    dropped = 0

    def cited_id(part: str) -> str | None:
        return next((i for i in ids_longest_first if i in part), None)

    def replace(match: re.Match) -> str:
        nonlocal dropped
        refs: list[tuple[str, int | None]] = []
        current = None
        for part in _CITATION_SPLIT.split(match.group(1)):
            if not part.strip():
                continue
            note_id = cited_id(part)
            # A part naming no known file is either more pages of the file before it
            # ("第 3 頁；第 5 頁") or a file that was not in the context.
            if note_id is None and (current is None or _PAGE_REF.sub("", part).strip()):
                dropped += 1
                current = None
                continue
            if note_id is not None:
                current = note_id
                part = part.replace(note_id, " ")
            groups = _expand_pages(part)
            if not groups:
                if note_id is not None:
                    refs.append((current, None))
                continue
            for group in groups:
                if not pages[current]:
                    refs.append((current, None))   # a note without pages: cite the note
                    continue
                found = [p for p in group if p in pages[current]]
                refs.extend((current, p) for p in found)
                if not found:
                    dropped += 1        # a page that was not in the context
        out = []
        for ref in dict.fromkeys(refs):
            out.append(source_marker(*ref))
            if ref not in seen:
                seen.add(ref)
                sources.append({"id": ref[0], "page": ref[1], "marker": source_marker(*ref)})
        return "".join(out)

    return _CITATION.sub(replace, answer), sources, dropped


def rag_answer(user_id: str, question: str, top_k: int = 10, model: str | None = None,
               category: str | None = None) -> dict:
    """Answer from one category's notes, or from every category when none is chosen, with
    the sources the answer cites: {"answer", "sources": [{id, page, marker}], "dropped"}.

    With a category, only that category's folder is opened: notes in other categories
    are never read, retrieved or sent to the model."""
    # An unloaded store would answer every question with 「沒有記載相關訊息」, which reads
    # as a statement about the user's notes rather than about the server.
    require_loaded()
    _require_category(user_id, category)
    stores = [category] if category is not None else _stores(user_id)
    notes = _whole_corpus_notes([get_user_collection(user_id, s) for s in stores])
    if notes is None:
        notes = _retrieval_notes(user_id, _retrieve_rows(user_id, question, top_k, stores))
    if not notes:
        return {"answer": REFUSAL, "sources": [], "dropped": 0}
    # Use replace (not str.format) so literal { } in LaTeX examples don't break.
    prompt = (PROMPT_TEMPLATE.replace("{context}", _format_context(notes))
              .replace("{question}", question))
    answer, sources, dropped = _resolve_citations(_llm_invoke(prompt, model=model), notes)
    return {"answer": answer, "sources": sources, "dropped": dropped}


def rag_query(user_id: str, question: str, top_k: int = 10, model: str | None = None,
              category: str | None = None) -> str:
    """rag_answer's text alone, citations included."""
    return rag_answer(user_id, question, top_k, model, category)["answer"]


# Restore any previously-saved vectorstores from the backing Dataset (runs once at
# import / server startup, before requests are served).
_restore_from_dataset()

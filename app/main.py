"""FastAPI server exposing the RAG system over HTTP.

Run locally:
    uvicorn app.main:app --reload

Endpoints (all /api/notes*, /api/categories*, /api/query and /api/upload need a
signed-in user):
    GET  /                          -> the web UI (static/index.html)
    POST /api/query                 -> ask a question, optionally inside one category
    POST /api/upload                -> upload a text, image, or PDF note into a category
    GET  /api/notes                 -> list your stored notes (each with its category)
    DELETE /api/notes/{id}          -> delete a note
    POST /api/notes/{id}/move       -> put a note into another category
    GET/POST /api/categories        -> list / create categories
    PUT/DELETE /api/categories/{id} -> rename / delete a category (its notes are kept)
    GET  /api/health                -> liveness plus cloud-sync state (no sign-in needed)
"""
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from pydantic import BaseModel

from . import rag_core

app = FastAPI(title="NoteHandlerRAG")

# CORS: the GitHub Pages frontend calls this backend cross-origin. Override with the
# ALLOWED_ORIGINS env var (comma-separated) when the Pages / Space URL changes.
# PUT was missing, so saving an edited note from GitHub Pages failed its preflight.
# allow_headers=["*"] makes Starlette echo the requested headers, which is what lets the
# Authorization header through (a literal "*" in the response would not cover it).
_DEFAULT_ORIGINS = "https://timhelloyee.github.io,http://localhost:8000,http://127.0.0.1:8000"
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Local development mode: mounts the UI editor and accepts X-Dev-User in place of a
# Google sign-in. dev_server.py sets NOTEHANDLER_DEV. It is refused outright on a
# Hugging Face Space (which always sets SPACE_ID): a single stray variable there would
# otherwise let any visitor act as any user.
DEV_MODE = bool(os.environ.get("NOTEHANDLER_DEV")) and not os.environ.get("SPACE_ID")
if DEV_MODE:
    print("[main] DEV MODE: X-Dev-User is accepted instead of Google sign-in. Never deploy this.")


# --- authentication -----------------------------------------------------------------
# The browser used to decode the Google ID token itself and send only the email, which
# the API trusted as-is: anyone who knew an address could list, read and delete that
# person's notes. Now the ID token itself is sent, and the server checks Google's
# signature, the audience (this app's client id) and expiry before believing the email.
GOOGLE_CLIENT_ID = os.environ.get(
    "GOOGLE_CLIENT_ID",
    "1068367879973-ps73rd0aoqr4u03cri4vomho72mh4qkm.apps.googleusercontent.com")

_google_request = google_requests.Request()     # reused: keeps the HTTPS connection warm
# verify_oauth2_token fetches Google's signing certificates on every call. Caching each
# verified token until it expires means one verification per sign-in (~1 hour), not one
# certificate download per API request.
_verified_tokens: dict[str, tuple[str, float]] = {}


def _verify_google_token(token: str) -> str:
    now = time.time()
    hit = _verified_tokens.get(token)
    if hit and hit[1] > now:
        return hit[0]
    try:
        claims = id_token.verify_oauth2_token(token, _google_request, GOOGLE_CLIENT_ID,
                                              clock_skew_in_seconds=10)
    except ValueError:
        # Bad signature, wrong audience, wrong issuer, or expired.
        raise HTTPException(status_code=401, detail="登入已過期或無效，請重新登入")
    except google_auth_exceptions.TransportError:
        raise HTTPException(status_code=503, detail="暫時無法向 Google 驗證登入，請稍後再試")
    if not claims.get("email") or not claims.get("email_verified"):
        raise HTTPException(status_code=401, detail="這個 Google 帳號沒有已驗證的 Email")
    email = claims["email"]
    _verified_tokens[token] = (email, float(claims.get("exp", now + 300)))
    if len(_verified_tokens) > 2000:        # drop expired entries; bounded memory
        for t, (_, exp) in list(_verified_tokens.items()):
            if exp <= now:
                _verified_tokens.pop(t, None)
    return email


def current_user(request: Request) -> str:
    """The verified email of whoever is calling. Every per-user endpoint depends on it,
    so there is no longer any way to name a user in the request itself."""
    if DEV_MODE:
        dev_user = request.headers.get("x-dev-user", "").strip()
        if dev_user:
            return dev_user
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="請先使用 Google 登入")
    return _verify_google_token(token.strip())


@app.exception_handler(rag_core.SyncUnavailableError)
def _sync_unavailable(_: Request, exc: rag_core.SyncUnavailableError):
    return JSONResponse(status_code=503, content={
        "detail": f"雲端筆記庫目前無法載入（{exc}）。為了避免覆蓋雲端上的筆記，暫停上傳、修改、刪除與問答。"})

@app.exception_handler(rag_core.CategoryNotFoundError)
def _category_not_found(_: Request, exc: rag_core.CategoryNotFoundError):
    return JSONResponse(status_code=404, content={"detail": "找不到這個類別，可能已被刪除，請重新整理"})


@app.exception_handler(rag_core.CategoryError)
def _category_error(_: Request, exc: rag_core.CategoryError):
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


def _category_param(value: str | None) -> str | None:
    """'' and None both mean uncategorized / no category chosen."""
    value = (value or "").strip()
    return value or None


STATIC_DIR = Path(__file__).parent / "static"

TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
PDF_EXTS = {".pdf"}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    model: str | None = None
    category: str | None = None     # None: search every category


class QueryResponse(BaseModel):
    answer: str
    model: str


@app.get("/api/models")
def models():
    """Models the UI dropdown offers, plus the default. Single source of truth so the
    frontend list can't drift from what the backend actually accepts."""
    return {"models": rag_core.AVAILABLE_MODELS, "default": rag_core.DEFAULT_MODEL}


def _busy_detail(model: str) -> str:
    return f"「{model}」目前忙碌或額度已用盡，請稍後再試，或在上方切換其他模型。"


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest, user: str = Depends(current_user)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="問題不可為空")
    model = rag_core.resolve_model(req.model)
    try:
        answer = rag_core.rag_query(user, req.question, top_k=req.top_k, model=model,
                                    category=_category_param(req.category))
    except rag_core.ModelUnavailableError:
        raise HTTPException(status_code=503, detail=_busy_detail(model))
    return QueryResponse(answer=answer, model=model)


def _decode_text(raw: bytes) -> str:
    """Decode an uploaded text note. Try UTF-8 (incl. BOM) first, then the common
    Chinese encodings (Big5/CP950 for Taiwan, GB18030 as a superset fallback), so a
    note saved by Windows Notepad in a legacy encoding doesn't 500 the upload."""
    for enc in ("utf-8-sig", "cp950", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=415, detail="無法辨識文字檔編碼，請以 UTF-8 儲存後重新上傳")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), model: str = Form(None),
                 category: str = Form(None), user: str = Depends(current_user)):
    rag_core.require_loaded()   # refuse before reading the upload or spending Gemini quota
    user_id = user
    model = rag_core.resolve_model(model)
    category = _category_param(category)
    rag_core._require_category(user_id, category)   # a 404 now, not after a Gemini call

    file_id = os.path.basename(file.filename or "")
    if not file_id:
        raise HTTPException(status_code=400, detail="檔名無效")
    ext = Path(file_id).suffix.lower()

    # Save the upload to a temp file so the embedder/Gemini can read it from disk.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # `sync` tells the page whether this change actually reached the cloud store, so
        # a failed sync is reported on the upload itself rather than discovered after a
        # restart. The originals are now saved inside add_*_note, before its sync runs.
        if ext in TEXT_EXTS:
            content = _decode_text(Path(tmp_path).read_bytes())
            added = rag_core.add_text_note(user_id, file_id, content, source=f"upload/{file_id}",
                                           category=category)
            return {"id": file_id, "type": "text", "added": added, "category": category,
                    "sync": rag_core.sync_status()["state"]}
        elif ext in IMAGE_EXTS:
            description = rag_core.add_image_note(user_id, file_id, tmp_path,
                                                  source=f"upload/{file_id}", model=model,
                                                  category=category)
            return {"id": file_id, "type": "image", "added": bool(description),
                    "description": description, "model": model, "category": category,
                    "sync": rag_core.sync_status()["state"]}
        elif ext in PDF_EXTS:
            description = rag_core.add_pdf_note(user_id, file_id, tmp_path,
                                                source=f"upload/{file_id}", model=model,
                                                category=category)
            return {"id": file_id, "type": "pdf", "added": bool(description),
                    "description": description, "model": model, "category": category,
                    "sync": rag_core.sync_status()["state"]}
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
    except rag_core.ModelUnavailableError:
        raise HTTPException(status_code=503, detail=_busy_detail(model))
    finally:
        os.unlink(tmp_path)


@app.get("/api/notes")
def notes(user: str = Depends(current_user)):
    return rag_core.list_notes(user)


class NoteUpdate(BaseModel):
    content: str


@app.get("/api/notes/{file_id}")
def note_text(file_id: str, user: str = Depends(current_user)):
    """Full stored text of one note, for review and editing."""
    text = rag_core.get_note_text(user, file_id)
    if text is None:
        raise HTTPException(status_code=404, detail=f"找不到筆記：{file_id}")
    return {"id": file_id, "text": text}


@app.get("/api/notes/{file_id}/original")
def note_original(file_id: str, request: Request, user: str = Depends(current_user)):
    """The source image/PDF a note was extracted from, when it was kept.

    Served inline, not as a download: the editor shows this next to the extracted text
    so the user can check what the model actually read. FileResponse's default
    `Content-Disposition: attachment` makes a browser download a PDF instead of
    rendering it in the <iframe>, which would defeat the whole comparison.

    Cached deliberately. Jumping to a PDF page reloads the <iframe> (the embedded viewer
    only reads #page= on a fresh navigation), and FileResponse sets an ETag but does not
    honour If-None-Match — so without this every page click re-sent the whole file, which
    is 14 MB for lv.0_compressed.pdf. max-age serves repeat views from cache outright;
    the 304 covers revalidation after it expires.

    The page fetches this with the Authorization header and hands the result to the
    <img>/<iframe> as a blob: URL, because those elements cannot send headers — and
    putting a token or an email in their src would leak it into logs and history.
    """
    path = rag_core.find_original(user, file_id)
    if not path:
        raise HTTPException(status_code=404,
                            detail=f"沒有保留 {file_id} 的原始檔（可能是在此功能上線前上傳的）")
    # stat_result must be passed for FileResponse to fill in etag/last-modified NOW;
    # without it those headers are only computed while the response is being sent, so
    # reading them here would give None — and comparing None to a missing If-None-Match
    # would 304 every first request, serving an empty body for every image.
    resp = FileResponse(path, filename=file_id, content_disposition_type="inline",
                        stat_result=os.stat(path))
    resp.headers["Cache-Control"] = "private, max-age=3600"
    # The URL is the same for every user, and the response is cached for an hour — so
    # without Vary, whoever signs in next on the same browser could be served the
    # previous user's cached file for a note with the same name.
    resp.headers["Vary"] = "Authorization, X-Dev-User"
    etag = resp.headers.get("etag")
    inm = request.headers.get("if-none-match")
    if etag and inm and etag in [t.strip() for t in inm.split(",")]:
        return Response(status_code=304, headers={
            k: v for k, v in resp.headers.items()
            if k.lower() in ("etag", "cache-control", "last-modified", "vary")})
    return resp


@app.put("/api/notes/{file_id}")
def edit_note(file_id: str, req: NoteUpdate, user: str = Depends(current_user)):
    """Save a corrected extraction. Re-chunks and re-embeds; no Gemini call."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="筆記內容不可為空")
    if not rag_core.update_note(user, file_id, req.content):
        raise HTTPException(status_code=404, detail=f"找不到筆記：{file_id}")
    return {"id": file_id, "updated": True, "sync": rag_core.sync_status()["state"]}


@app.delete("/api/notes/{file_id}")
def remove_note(file_id: str, user: str = Depends(current_user)):
    if not rag_core.delete_note(user, file_id):
        raise HTTPException(status_code=404, detail=f"找不到筆記：{file_id}")
    return {"deleted": file_id, "sync": rag_core.sync_status()["state"]}


class NoteMove(BaseModel):
    category: str | None = None     # None: uncategorized


@app.post("/api/notes/{file_id}/move")
def move_note(file_id: str, req: NoteMove, user: str = Depends(current_user)):
    """Drop a note into another category. Moves the stored vectors and original as they
    are — no re-embedding, no Gemini call."""
    category = _category_param(req.category)
    if not rag_core.move_note(user, file_id, category):
        raise HTTPException(status_code=404, detail=f"找不到筆記：{file_id}")
    return {"id": file_id, "category": category, "sync": rag_core.sync_status()["state"]}


class CategoryBody(BaseModel):
    name: str


@app.get("/api/categories")
def categories(user: str = Depends(current_user)):
    return rag_core.list_categories(user)


@app.post("/api/categories")
def create_category(req: CategoryBody, user: str = Depends(current_user)):
    return {**rag_core.create_category(user, req.name), "sync": rag_core.sync_status()["state"]}


@app.put("/api/categories/{category}")
def rename_category(category: str, req: CategoryBody, user: str = Depends(current_user)):
    return {**rag_core.rename_category(user, category, req.name),
            "sync": rag_core.sync_status()["state"]}


@app.delete("/api/categories/{category}")
def delete_category(category: str, user: str = Depends(current_user)):
    moved = rag_core.delete_category(user, category)
    return {"deleted": category, "moved": moved, "sync": rag_core.sync_status()["state"]}


@app.get("/api/health")
def health():
    """Unauthenticated on purpose: the page needs the sync state before anyone signs in,
    to explain an empty list. It says nothing about any particular user."""
    return {"status": "ok", "sync": rag_core.sync_status()}


# --- UI editor (local development only) --------------------------------------
# A tiny visual editor for the design tokens in index.html's :root block, so styling
# can be changed without editing HTML. It WRITES TO THE SOURCE TREE, so it is mounted
# only when NOTEHANDLER_DEV is set — dev_server.py sets it, the Dockerfile does not.
# On the deployed Space these routes simply do not exist; if they did, any visitor
# could rewrite the interface. Gated by the single DEV_MODE defined near the top, which
# also refuses to turn on when SPACE_ID is present.

# Only these may be written, and only with values that cannot escape the CSS rule.
# Without the value check, a token could carry "</style><script>…" and the editor would
# become a way to inject script into the page it saves.
_TOKEN_NAMES = {
    "--card", "--ink", "--muted", "--border", "--accent", "--accent-2", "--danger",
    "--radius", "--shadow", "--page-width", "--bg-top", "--bg-body", "--font-size",
    "--line-height", "--font-family", "--font-heading", "--font-mono",
}
_TOKEN_VALUE = re.compile(r"^[A-Za-z0-9 ,.\-#%()'\"/_]{1,200}$")
_ROOT_BLOCK = re.compile(r"(:root\s*\{)(.*?)(\n    \})", re.DOTALL)
_TOKEN_LINE = re.compile(r"^\s*(--[a-z0-9-]+)\s*:\s*(.+?);\s*$", re.MULTILINE)

STATIC_INDEX = STATIC_DIR / "index.html"
DOCS_INDEX = Path(__file__).parent.parent / "docs" / "index.html"


def _read_tokens(html: str) -> dict:
    m = _ROOT_BLOCK.search(html)
    if not m:
        raise HTTPException(status_code=500, detail="找不到 :root 區塊")
    return {k: v.strip() for k, v in _TOKEN_LINE.findall(m.group(2))}


if DEV_MODE:
    @app.get("/ui-editor")
    def ui_editor():
        return FileResponse(STATIC_DIR / "ui_editor.html")

    @app.get("/api/ui-tokens")
    def get_ui_tokens():
        return {"tokens": _read_tokens(STATIC_INDEX.read_text(encoding="utf-8"))}

    @app.post("/api/ui-tokens")
    def put_ui_tokens(payload: dict):
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict) or not tokens:
            raise HTTPException(status_code=400, detail="tokens 不可為空")
        for name, value in tokens.items():
            if name not in _TOKEN_NAMES:
                raise HTTPException(status_code=400, detail=f"不允許的變數：{name}")
            if not isinstance(value, str) or not _TOKEN_VALUE.match(value.strip()):
                raise HTTPException(status_code=400, detail=f"{name} 的值含有不允許的字元")

        html = STATIC_INDEX.read_text(encoding="utf-8")
        current = _read_tokens(html)
        merged = {**current, **{k: v.strip() for k, v in tokens.items()}}
        body = "".join(f"\n      {k}: {v};" for k, v in merged.items())
        body += "\n      font-family: var(--font-family);\n      font-size: var(--font-size);"
        new_html = _ROOT_BLOCK.sub(lambda m: m.group(1) + body + m.group(3), html, count=1)

        # Keep one undo step. Losing hand-written CSS to a bad save would be worse than
        # any styling mistake the editor can make.
        backup = STATIC_INDEX.with_suffix(".html.bak")
        backup.write_text(html, encoding="utf-8")
        STATIC_INDEX.write_text(new_html, encoding="utf-8")
        DOCS_INDEX.write_text(new_html, encoding="utf-8")   # Pages copy must not drift
        return {"saved": True, "tokens": merged, "backup": backup.name}

    @app.post("/api/ui-tokens/revert")
    def revert_ui_tokens():
        backup = STATIC_INDEX.with_suffix(".html.bak")
        if not backup.is_file():
            raise HTTPException(status_code=404, detail="沒有可還原的備份")
        html = backup.read_text(encoding="utf-8")
        STATIC_INDEX.write_text(html, encoding="utf-8")
        DOCS_INDEX.write_text(html, encoding="utf-8")
        return {"reverted": True, "tokens": _read_tokens(html)}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

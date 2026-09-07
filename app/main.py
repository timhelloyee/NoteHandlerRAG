"""FastAPI server exposing the RAG system over HTTP.

Run locally:
    uvicorn app.main:app --reload

Endpoints:
    GET  /                -> the web UI (static/index.html)
    POST /api/query       -> ask a question for a given user
    POST /api/upload      -> upload a text, image, or PDF note
    GET  /api/notes       -> list a user's stored notes
    DELETE /api/notes/{id}-> delete a note
"""
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import rag_core

app = FastAPI(title="NoteHandlerRAG")

# CORS: the GitHub Pages frontend calls this backend cross-origin. Override with the
# ALLOWED_ORIGINS env var (comma-separated) when the Pages / Space URL changes.
_DEFAULT_ORIGINS = "https://timhelloyee.github.io,http://localhost:8000,http://127.0.0.1:8000"
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
PDF_EXTS = {".pdf"}


class QueryRequest(BaseModel):
    user_id: str
    question: str
    top_k: int = 10
    model: str | None = None


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
def query(req: QueryRequest):
    if not req.user_id or not req.question.strip():
        raise HTTPException(status_code=400, detail="user_id and question are required")
    model = rag_core.resolve_model(req.model)
    try:
        answer = rag_core.rag_query(req.user_id, req.question, top_k=req.top_k, model=model)
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
async def upload(user_id: str = Form(...), file: UploadFile = File(...),
                 model: str = Form(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    model = rag_core.resolve_model(model)

    file_id = os.path.basename(file.filename or "")
    if not file_id:
        raise HTTPException(status_code=400, detail="檔名無效")
    ext = Path(file_id).suffix.lower()

    # Save the upload to a temp file so the embedder/Gemini can read it from disk.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if ext in TEXT_EXTS:
            content = _decode_text(Path(tmp_path).read_bytes())
            added = rag_core.add_text_note(user_id, file_id, content, source=f"upload/{file_id}")
            return {"id": file_id, "type": "text", "added": added}
        elif ext in IMAGE_EXTS:
            description = rag_core.add_image_note(user_id, file_id, tmp_path,
                                                  source=f"upload/{file_id}", model=model)
            if description:
                # Keep the pixels: the store only holds Gemini's description of them.
                rag_core.save_original(user_id, file_id, tmp_path)
            return {"id": file_id, "type": "image", "added": bool(description),
                    "description": description, "model": model}
        elif ext in PDF_EXTS:
            description = rag_core.add_pdf_note(user_id, file_id, tmp_path,
                                                source=f"upload/{file_id}", model=model)
            if description:
                rag_core.save_original(user_id, file_id, tmp_path)
            return {"id": file_id, "type": "pdf", "added": bool(description),
                    "description": description, "model": model}
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
    except rag_core.ModelUnavailableError:
        raise HTTPException(status_code=503, detail=_busy_detail(model))
    finally:
        os.unlink(tmp_path)


@app.get("/api/notes")
def notes(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    return rag_core.list_notes(user_id)


class NoteUpdate(BaseModel):
    user_id: str
    content: str


@app.get("/api/notes/{file_id}")
def note_text(file_id: str, user_id: str):
    """Full stored text of one note, for review and editing."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    text = rag_core.get_note_text(user_id, file_id)
    if text is None:
        raise HTTPException(status_code=404, detail=f"找不到筆記：{file_id}")
    return {"id": file_id, "text": text}


@app.get("/api/notes/{file_id}/original")
def note_original(file_id: str, user_id: str, request: Request):
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
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    path = rag_core.get_original(user_id, file_id)
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
    etag = resp.headers.get("etag")
    inm = request.headers.get("if-none-match")
    if etag and inm and etag in [t.strip() for t in inm.split(",")]:
        return Response(status_code=304, headers={
            k: v for k, v in resp.headers.items()
            if k.lower() in ("etag", "cache-control", "last-modified")})
    return resp


@app.put("/api/notes/{file_id}")
def edit_note(file_id: str, req: NoteUpdate):
    """Save a corrected extraction. Re-chunks and re-embeds; no Gemini call."""
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="筆記內容不可為空")
    if not rag_core.update_note(req.user_id, file_id, req.content):
        raise HTTPException(status_code=404, detail=f"找不到筆記：{file_id}")
    return {"id": file_id, "updated": True}


@app.delete("/api/notes/{file_id}")
def remove_note(file_id: str, user_id: str):
    if not rag_core.delete_note(user_id, file_id):
        raise HTTPException(status_code=404, detail=f"找不到筆記：{file_id}")
    return {"deleted": file_id}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

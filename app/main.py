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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


class QueryResponse(BaseModel):
    answer: str


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.user_id or not req.question.strip():
        raise HTTPException(status_code=400, detail="user_id and question are required")
    answer = rag_core.rag_query(req.user_id, req.question, top_k=req.top_k)
    return QueryResponse(answer=answer)


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
async def upload(user_id: str = Form(...), file: UploadFile = File(...)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

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
            description = rag_core.add_image_note(user_id, file_id, tmp_path, source=f"upload/{file_id}")
            return {"id": file_id, "type": "image", "added": bool(description), "description": description}
        elif ext in PDF_EXTS:
            description = rag_core.add_pdf_note(user_id, file_id, tmp_path, source=f"upload/{file_id}")
            return {"id": file_id, "type": "pdf", "added": bool(description), "description": description}
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
    finally:
        os.unlink(tmp_path)


@app.get("/api/notes")
def notes(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    return rag_core.list_notes(user_id)


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

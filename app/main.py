"""FastAPI server exposing the RAG system over HTTP.

Run locally:
    uvicorn app.main:app --reload

Endpoints:
    GET  /                -> the web UI (static/index.html)
    POST /api/query       -> ask a question for a given user
    POST /api/upload      -> upload a text or image note
    GET  /api/notes       -> list a user's stored notes
    DELETE /api/notes/{id}-> delete a note
"""
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import rag_core

app = FastAPI(title="NoteHandlerRAG")

STATIC_DIR = Path(__file__).parent / "static"

TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


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


@app.post("/api/upload")
async def upload(user_id: str = Form(...), file: UploadFile = File(...)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    file_id = os.path.basename(file.filename)
    ext = Path(file_id).suffix.lower()

    # Save the upload to a temp file so CLIP/Gemini can read it from disk.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if ext in TEXT_EXTS:
            content = Path(tmp_path).read_text(encoding="utf-8")
            added = rag_core.add_text_note(user_id, file_id, content, source=f"upload/{file_id}")
            return {"id": file_id, "type": "text", "added": added}
        elif ext in IMAGE_EXTS:
            description = rag_core.add_image_note(user_id, file_id, tmp_path, source=f"upload/{file_id}")
            return {"id": file_id, "type": "image", "added": bool(description), "description": description}
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
    rag_core.delete_note(user_id, file_id)
    return {"deleted": file_id}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

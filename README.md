---
title: NoteHandlerRAG
emoji: 📒
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
license: mit
pinned: false
---

# NoteHandlerRAG

A multimodal Retrieval-Augmented Generation (RAG) web app over personal study notes
(text, images, and PDFs). Sign in with Google, upload notes, and ask questions — answers
are generated only from your own notes, rendered with Markdown + LaTeX.

## Architecture

- **Frontend** — a single static page ([app/static/index.html](app/static/index.html)),
  served by the backend and also publishable to GitHub Pages (copy in `docs/`). Google
  Identity Services supplies the user id (account email); `API_BASE` switches between
  same-origin (local / Space) and the Space URL (Pages).
- **Backend** — FastAPI ([app/main.py](app/main.py)) + core RAG logic
  ([app/rag_core.py](app/rag_core.py)), deployed as a **Docker Space** on Hugging Face
  (port 7860).
- **Retrieval** — `intfloat/multilingual-e5-base` sentence embeddings (strong for
  Traditional Chinese + English; `passage:` / `query:` prefixes) in per-user **ChromaDB**
  collections under `vectorstores/<safe_user_id>/`.
- **Generation & vision** — Google **Gemini** (`gemini-3.1-flash-lite`) describes uploaded
  images/PDFs into searchable text and answers questions from retrieved context. Two API
  keys with automatic quota fail-over.
- **Persistence** — Space storage is ephemeral, so vectorstores are synced to a **private
  HF Dataset** (`VECTORSTORE_DATASET`): restored at startup, pushed back after every
  upload/delete. No note content or user emails are committed to this repo.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/query` | Ask a question (`user_id`, `question`, `top_k`) |
| `POST /api/upload` | Upload a `.txt`/`.md`/image/PDF note (multipart) |
| `GET /api/notes?user_id=` | List a user's notes (with previews) |
| `DELETE /api/notes/{id}?user_id=` | Delete a note |
| `GET /api/health` | Liveness check |

## Configuration (env vars / Space secrets)

Copy `.env.example` to `.env` for local runs; on the Space set these as **secrets**:

- `GEMINI_API_KEY_PRIMARY` / `GEMINI_API_KEY_SECONDARY` — required; the app fails over
  between them on quota errors.
- `HF_TOKEN` — token with access to the private Dataset (write access enables saving
  uploads; without it the app runs but data is ephemeral).
- `VECTORSTORE_DATASET` (optional) — backing Dataset repo id.
- `ALLOWED_ORIGINS` (optional) — comma-separated CORS origins for the Pages frontend.
- `VECTORSTORE_ROOT` (optional) — local store directory (default `./vectorstores`).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000   # open http://localhost:8000
```

Google sign-in requires the page origin to be listed under the OAuth client's
**Authorized JavaScript origins** (e.g. `http://localhost:8000`), and does not work
inside the Hugging Face Space iframe — open the Space in its own tab.

## Repo layout notes

- `docs/index.html` is a copy of `app/static/index.html` for GitHub Pages — keep them in
  sync when editing the frontend.
- [embed_folder.py](embed_folder.py) and the notebooks are local experiments; the CLI
  chunks text (ids like `a.txt#0`) which is **incompatible** with the app's whole-document
  ids — don't point them at the same store the app uses.
- `vectorstores/` and `.env` are git-ignored by design (privacy / secrets).

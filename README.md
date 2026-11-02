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
python scripts/dev_server.py --port 8000   # open http://localhost:8000
```

`scripts/dev_server.py` ignores `HF_TOKEN` and keeps notes in `.devstore/`, so a local run
never reads or overwrites the private Dataset. (Plain `uvicorn app.main:app` is what the
Space runs; on a laptop with `.env` it would sync with the real notes.) Locally you can
skip Google sign-in: set `localStorage.nh_dev_user` in the browser to any name.

Google sign-in requires the page origin to be listed under the OAuth client's
**Authorized JavaScript origins** (e.g. `http://localhost:8000`), and does not work
inside the Hugging Face Space iframe — open the Space in its own tab.

## Repo layout

```
app/          the web app: FastAPI backend (main.py, rag_core.py) and static/ pages
docs/         GitHub Pages copy of the frontend (index.html only)
scripts/      command-line tools: dev_server.py, notehandler_input.py,
              notehandler_output.py, embed_folder.py
tests/        standalone test scripts (python tests/test_*.py)
eval/         research measurements (R0 NotebookLM baseline; G0 gold transcripts, kept local)
handoff/      project documents: 專題總覽.md (overview, progress, roadmap, technical
              reference), 實驗架構.md (experiment design), 文獻整理.md
samples/      small example notes (text/ and images/) for trying the scripts
```

Files that stay in the root do so because something requires it there: `README.md`
(its front matter configures the Hugging Face Space), `Dockerfile`, `requirements.txt`,
`LICENSE`, `.env.example`, `.gitignore`, `.gitattributes`.

## Repo layout notes

- `docs/index.html` is a copy of `app/static/index.html` for GitHub Pages — keep them in
  sync when editing the frontend.
- [scripts/notehandler_input.py](scripts/notehandler_input.py) (add / delete notes) and
  [scripts/notehandler_output.py](scripts/notehandler_output.py) (list notes / ask a question) are the
  command-line versions of the app. They call `app/rag_core.py`, so notes are stored
  exactly as a web upload would store them, and like `scripts/dev_server.py` they use `.devstore/`
  and never touch the private Dataset:

  ```bash
  python scripts/notehandler_input.py add samples/text/a.txt --user alice --category 生物
  python scripts/notehandler_output.py ask "粒線體的功能是什麼？" --user alice
  ```
- [scripts/embed_folder.py](scripts/embed_folder.py) is an older local experiment; it chunks text with a
  different id scheme (`a.txt#0`) that is **incompatible** with the app's store — don't
  point it at a store the app uses.
- `vectorstores/` and `.env` are git-ignored by design (privacy / secrets).

# NoteHandlerRAG — Project Overview

> Handoff context for another Claude Code session. Read this first to understand
> what the project is, how it's structured, and the non-obvious details that
> aren't visible from a quick file scan.

## What this project is

A **multimodal Retrieval-Augmented Generation (RAG) system over personal study
notes** (text + images). It targets students: prompts are written in Traditional
Chinese and image notes are auto-classified by school subject (國文 / 英文 / 數學 /
物理 / 化學 / 地球科學 / 生物 / 地理 / 公民 / 歷史).

The user uploads notes, the system embeds and stores them in a per-user vector
database, and then answers questions grounded **only** in the retrieved notes
(explicitly instructed not to hallucinate).

## The pipeline

1. **Embedding** — OpenCLIP `ViT-B-32` (OpenAI pretrained) embeds both text and
   query strings into a shared vector space (`embed_text`, `embed_image`).
2. **Image notes** — each image is sent to **Google Gemini 2.5 Flash** (vision),
   which writes a detailed text description *and* tags the subject. That
   description is what gets embedded and stored (not the raw image vector).
3. **Vector store** — **ChromaDB**, one persistent collection per user under
   `vectorstores/<user_id>/` (git-ignored).
4. **Querying** — the question is embedded, top-k nearest notes are retrieved,
   and Gemini Flash generates an answer from that context only.

## Repository layout

```
NoteHandlerRAG/
├── README.md                  # user-facing setup/usage
├── PROJECT_OVERVIEW.md        # this file (model handoff context)
├── requirements.txt           # open_clip_torch, torch, chromadb,
│                              #   langchain-google-genai, google-genai,
│                              #   python-dotenv, Pillow, fastapi, uvicorn, ...
├── .env.example               # required keys (copy to .env, git-ignored)
├── Input.ipynb                # NOTEBOOK: build/update store (add_txt, add_image)
├── NoteHandler_opus.ipynb     # NOTEBOOK: query store, get Gemini answer
├── app/                       # FastAPI refactor of the notebook logic
│   ├── __init__.py
│   ├── rag_core.py            # all RAG logic (CLIP, Chroma, Gemini)
│   ├── main.py                # FastAPI server + REST endpoints
│   └── static/index.html      # web UI
├── docs/                      # sample text notes (a.txt, b.txt)
└── images/                    # image-note input directory
```

## Two ways to run it

| Surface | Files | Purpose |
| --- | --- | --- |
| **Notebooks** (original) | `Input.ipynb`, `NoteHandler_opus.ipynb` | Interactive: build the store, then query. |
| **FastAPI app** (refactor) | `app/` | Web service + static UI exposing the same logic over HTTP. |

### FastAPI endpoints (`app/main.py`)
- `GET  /` — serves `static/index.html`
- `POST /api/query` — `{user_id, question, top_k=10}` → `{answer}`
- `POST /api/upload` — multipart `user_id` + `file` (text `.txt/.md` or image
  `.jpg/.jpeg/.png/.webp/.gif`); uploads are written to a temp file so CLIP/Gemini
  can read them from disk, then unlinked.
- `GET  /api/notes?user_id=...` — list stored notes (id, type, source, preview)
- `DELETE /api/notes/{file_id}?user_id=...` — delete a note

### Core functions (`app/rag_core.py`)
`embed_text`, `embed_image`, `get_user_collection`, `describe_image`,
`add_text_note`, `add_image_note`, `list_notes`, `delete_note`, `rag_query`.

## Setup / environment

```bash
pip install -r requirements.txt   # heavy: torch + CLIP weights download on first use
cp .env.example .env              # then fill in keys
uvicorn app.main:app --reload
```

Required env vars (read via `python-dotenv`):
- `GEMINI_API_KEY_PRIMARY` — used for answer generation (`ChatGoogleGenerativeAI`)
- `GEMINI_API_KEY_SECONDARY` — used for image description (`genai.Client`)
- `OPENAI_API_KEY`, `LANGCHAIN_API_KEY` — present in `.env.example`/requirements
- `VECTORSTORE_ROOT` — optional, defaults to `./vectorstores`

Two Gemini keys are intentional: they split usage across the Gemini Flash rate
limit (primary = generation, secondary = vision).

## Non-obvious details / gotchas

- **CLIP loads once at import time** in `rag_core.py` (CUDA if available, else CPU),
  so the server pays the model-load cost at startup, not per request.
- **Hard import-time requirement:** `rag_core.py` raises `RuntimeError` (message in
  Chinese) if either Gemini key is missing. Because `main.py` does
  `from . import rag_core` at module load, **the server will not boot without both
  keys** — there is no lazy/degraded mode.
- **Image description prompt quirk:** it instructs Gemini to describe the image
  "以左右反轉的方式" (left–right flipped) and to prefix output with the detected
  subject line. If image retrieval seems mirrored/odd, this is why.
- **LangChain is barely used:** `LANGCHAIN_API_KEY` is in `.env.example` and
  requirements, but the code only uses LangChain's `ChatGoogleGenerativeAI`
  wrapper. No LangSmith tracing is actually wired up.
- **Per-user isolation** is purely by directory/collection name (`user_id`); there
  is no auth.
- `.env` and `vectorstores/` are git-ignored. Requires Python 3.10+ (`list[str]`
  type hints).

## Current state (as of this handoff)

- Active development branch: `claude/repository-overview-jHCkj`.
- The repo has **not** been run end-to-end in CI/containers here: dependencies are
  not installed by default and no Gemini keys are present, so the server can't boot
  without setup.
- Web tools in this kind of environment: `WebSearch` works; `WebFetch` may be
  blocked by the network policy (returns 403).

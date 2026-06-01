# NoteHandlerRAG

A multimodal Retrieval-Augmented Generation (RAG) system over personal study notes
(text and images). It uses CLIP to embed content, ChromaDB as the vector store, and
Google Gemini Flash for image description and answer generation.

## How it works

- **Embedding** — `ViT-B-32` (OpenAI pretrained) via `open_clip` embeds both query text
  and note content into a shared vector space.
- **Vector store** — per-user ChromaDB collections under `vectorstores/<user>/`.
- **Image notes** — Gemini Flash generates a text description of each image (including
  the detected school subject), and that description is embedded and stored.
- **Querying** — a question is embedded, the top-k relevant notes are retrieved, and
  Gemini Flash answers based only on the retrieved context.

## Notebooks

| File | Purpose |
| --- | --- |
| `Input.ipynb` | Build/update the vector store: add text (`add_txt`) and image (`add_image`) notes. |
| `NoteHandler_opus.ipynb` | Query the vector store and get a Gemini-generated answer. |

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=...
GEMINI_API_KEY_PRIMARY=...      # default key
GEMINI_API_KEY_SECONDARY=...    # backup key, used when the primary hits its quota
LANGCHAIN_API_KEY=...
```

Two Gemini keys are supported to spread usage across the Gemini Flash rate limit.
Notebooks read keys from `.env` via `python-dotenv` — no keys are hardcoded.

## Usage

1. Place note files in `docs/` (text) and `images/` (images).
2. Run `Input.ipynb` to embed and store them.
3. Run `NoteHandler_opus.ipynb` and enter a question when prompted.

## Notes

- `.env` and `vectorstores/` are git-ignored.
- Requires Python 3.10+ (uses `list[str]` type hints).

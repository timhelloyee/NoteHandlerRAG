# NoteHandlerRAG — Project Handoff

> **Audience:** another AI assistant picking this project up cold.
> **Written:** 2026-08-31. **Repo HEAD at writing:** `e51a3cb`.
> Verify claims against the code before acting — this document ages, the repo doesn't.

---

## 1. What this is

A multimodal **RAG (Retrieval-Augmented Generation) web app over personal study notes**,
built by a Taiwanese high-school student (`timhelloyee`). The user uploads notes as text,
images (photos of handwritten notes), or PDFs; asks questions in Traditional Chinese; and
gets answers generated **only** from their own notes, rendered with Markdown + LaTeX.

The notes are real high-school material: 公民 (civics), 地理 (geography), 生物 (biology),
物理 (physics). Answers must preserve formulas, so LaTeX fidelity matters throughout.

**Live:** https://timhelloyee-notehandler.hf.space

---

## 2. Where everything lives

| Thing | Location | Notes |
|---|---|---|
| Working copy | `C:\Users\user\Desktop\ClaudeProjects\NoteHandler` | **Moved** from `Desktop\PythonProjects` around 2026-08 |
| GitHub | `github.com/timhelloyee/NoteHandlerRAG` (remote `origin`) | **Still private** |
| HF Space | `huggingface.co/spaces/timhelloyee/NoteHandler` | Docker SDK, port 7860 |
| Note data | HF Dataset `timhelloyee/notehandler-store` | **Private. Never committed to git.** |
| Python env | `C:/Users/user/anaconda3/envs/ragtraining/python.exe` | Has torch+CUDA locally; the Space is CPU-only |

### ⚠️ Deploying means pushing to TWO remotes

```bash
git push origin main                                          # GitHub
git push https://<user>:<HF_WRITE_TOKEN>@huggingface.co/spaces/timhelloyee/NoteHandler main:main
```

A Space push triggers a rebuild. When Docker layers are cached (Python-only change) the
swap can be nearly invisible — the old container keeps serving and health never drops, so
**do not poll for downtime to detect a new build**; test the behaviour you changed instead.
A dependency change takes ~2–3 minutes before `/api/health` returns 200.

---

## 3. Architecture

```
Browser (app/static/index.html — also copied to docs/index.html for GitHub Pages)
   │  Google Identity Services sign-in → account email is the user_id
   │  fetch() → API_BASE ("" same-origin, or the hf.space URL when served from github.io)
   ▼
FastAPI (app/main.py)  ── validates, routes, converts upstream failures to clean HTTP codes
   ▼
RAG core (app/rag_core.py)
   ├─ embeddings : EMBED_MODEL env, default intfloat/multilingual-e5-base (768d)
   │                alternative Qwen/Qwen3-Embedding-0.6B (1024d, better separated — see below)
   ├─ vector DB  : ChromaDB, one collection per user, HNSW index, squared-L2 space
   ├─ vision+LLM : Google Gemini (selectable model, 2 API keys with fail-over)
   └─ persistence: private HF Dataset — restore on startup, push after every write
```

### The persistence model is the single most important thing to understand

HF Space storage is **ephemeral**. So:

- **Nothing** is baked into the Docker image and **nothing** is committed to git.
- On startup, `_restore_from_dataset()` downloads the whole `vectorstores/` tree from the
  private Dataset.
- After every add/delete, `_persist_user()` re-uploads that user's folder.

**Consequence:** if the Space's `HF_TOKEN` secret is missing, invalid, or lacks access to
the private Dataset, **the app boots with an empty note list.** There is no baked-in seed
to fall back on. This is the #1 cause of "my notes disappeared".

This design was deliberate: `vectorstores/<user>` is named after the user's email and the
Chroma SQLite contains their note text. The repo is destined to be public, so the data was
purged from git history with `git filter-repo` and force-pushed.

---

## 4. Request flows

### `POST /api/query`
1. `resolve_model(req.model)` → allowlist check, falls back to default.
2. `_retrieve()`: embed question → Chroma `query()` (`documents`, `distances`, `metadatas`)
   → keyword boost → relevance gate → relative-margin selection. **See §5.**
3. Chunks joined with `\n---\n` into `PROMPT_TEMPLATE`, which is filled with `.replace()`
   **not** `str.format()` — the template contains literal `{}` from LaTeX examples.
4. `_llm_invoke()` with key fail-over; `ModelUnavailableError` → HTTP 503.
5. Returns `{answer, model}` — the echoed model is what actually ran.

### `POST /api/upload` (multipart: `user_id`, `file`, `model`)
- `.txt/.md` → `_decode_text()` tries `utf-8-sig`, `cp950`, `gb18030` (Windows Notepad in
  Taiwan writes Big5); undecodable → 415.
- images / PDF → Gemini vision describes the file into Traditional Chinese text (subject
  tag + faithful content + LaTeX + a trailing `關鍵字：` line), then `_clean_description()`,
  then chunk + embed + store.
- Duplicate detection checks **both** the bare id and `<id>#0` (whole-doc and chunked forms).

### Other endpoints
`GET /api/models` (dropdown source of truth + default) · `GET /api/notes` (aggregated,
one row per note) · `DELETE /api/notes/{id}` (deletes all chunks, 404 if absent) ·
`GET /api/health` · `GET /` (serves the UI).

---

## 5. Retrieval tuning — read this before touching the numbers

These constants were **measured**, not guessed. Re-measure before changing them.

```python
CHUNK_CHARS = 450 ; CHUNK_OVERLAP = 50     # sentence-aligned
KEYWORD_BONUS = 0.03 ; KEYWORD_BONUS_CAP = 3
RELEVANCE_CEILING = 0.38                    # gates the QUERY
RELATIVE_MARGIN  = 0.10                     # selects HOW MANY chunks
```

**Chroma reports squared L2 over normalized vectors — exactly `2 − 2·cos`.** Confirmed
empirically (`configuration_json` shows `space: 'l2'`; distance equals `2-2cos` to 4 dp).
So a distance of 0.35 means cosine similarity 0.825.

Measured on the real corpus:

| Query type | Best (min) distance |
|---|---|
| Answer **is** in the notes | 0.20 – 0.36 |
| Topic **absent** from the notes | 0.37 + |

e5 similarities are range-compressed: unrelated Traditional Chinese still scores ~0.77
cosine. So the two populations are close, and the separation is narrow (~0.02).

**This table belongs to the encoder, not to the app.** `EMBED_MODEL` makes the embedder
swappable, and every raw-distance constant below must be recalibrated when it changes —
`eval/calibrate_gate.py` does the measurement. Under `Qwen/Qwen3-Embedding-0.6B` the same
two populations sit at 0.41–1.13 (present) and 0.83–1.37 (absent): unrelated Chinese
falls from ~0.79 to ~0.31 cosine, and the mean question-to-answer margin goes 0.101 →
0.498. e5's 0.38 ceiling is *below Qwen3's smallest genuine answer distance*, so reusing
it would refuse every question. Per-model values live in `rag_core._GATE_BY_MODEL`.

Retrieval quality on the 29-probe set (see `eval/RESULTS_gating.txt`): swapping e5 →
Qwen3-0.6B moves B1 chunk-level MRR 0.813 → 0.897 and R@1 0.69 → 0.83, paired diff
+0.083, 95% CI [-0.009, +0.175], 9W/2L. The largest effect this corpus has produced and
the first CI that nearly excludes zero — but still not separated at n=29. Config ordering
(B1 ≥ B4 > B0 at chunk level) is unchanged, so the fidelity-vs-retrievability finding is
not an artefact of a weak encoder. Measurement runs must set `EMBED_DTYPE=float32`:
Qwen3 at fp16 is not run-to-run reproducible on a 3050 (one probe swung rank 1 ↔ absent,
worth 0.035 MRR; HNSW was ruled out — exact search agreed on 29/29).

Switching the production default is a **data migration**, not a config change: the stored
vectors are 768-wide and every note must be re-embedded. `get_user_collection` refuses a
dimension mismatch with an explicit error rather than letting Chroma fail obscurely.

### 2026-09-07: three defects that invalidated the earlier gate numbers

Every gate result before this date is superseded. Details in `eval/RESULTS_gating.txt`.

1. **`_chunk_text` emitted near-duplicate slivers.** One 1,132-char section produced 52
   chunks of 414, 236, 48, 47, 46 ... 1 chars: the window snapped back to a boundary just
   after `start`, the next window snapped to the same one, and `start` crawled forward a
   character at a time. Fixed by only snapping to a boundary at least `size//2` in —
   hard-cutting otherwise, which the docstring always claimed happened. It hit **B0
   only** (300 → 252 chunks), so the bug inflated the human oracle alone.
2. **Chunk length was confounded with chunking strategy.** B0's median chunk was 142
   chars against B1's 421, so "B0 vs B1" was largely a length comparison.
   `_merge_small_sections` (see `chunk_by_structure(merge_target=…)`) merges consecutive
   sections into subtree blocks, giving length-matched arms `B4L` (318) and `B0L` (348).
   **Every result table now carries chunk median/IQR/count** — without them no
   comparison here is interpretable.
3. **Six of 29 probes contained their own gold span in the question.** They measured
   literal string overlap. Rewriting them **flipped the headline result**: chunk-level
   B4 − B1 went −0.045 → **+0.027** under Qwen3, and B0 − B1 went −0.131 (raw p = 0.041)
   → −0.075 (p = 0.279). Six bad probes were carrying the project's main finding.

**Nothing survives Holm–Bonferroni at chunk level under either encoder.** The earlier
claim that B0 − B1 was significant was uncorrected; the project's own statistics section
requires the correction, and applying it removes the result.

**The positive control fails.** `B0L-strip` deletes the ancestor path from every chunk
with length, text and chunk count held constant — a deliberately large manipulation.
Under e5 it *improves* retrieval (0.810 → 0.862); under Qwen3 it degrades it slightly
(0.844 → 0.821); neither is significant and only 3 and 2 probes move. The two encoders
disagree on the sign. This probe set cannot detect the mechanism M2 is built on, so
expanding the corpus before rebuilding it would measure zero more precisely.

`probes.md` is now generated from `chunk_cases.json` by `eval/sync_probes_md.py` — the
two had already drifted (the 2026-08-31 contamination fix landed only in the JSON).

### 2026-09-07: the structure line is secondary, and why

The project's goal is that a student can **organise and query their own notes by subject,
and understand what the model read**. Judged against that, the retrieval research was
optimising something users never touch:

- **Real users never reach the retrieval path.** `rag_query` calls `_whole_corpus` first;
  under `WHOLE_CORPUS_CHARS` (default 60,000) the entire note set goes to Gemini and
  `_retrieve` is skipped. The measured corpus is 52,875 chars. `eval/gating_experiment.py`
  has to force `WHOLE_CORPUS_CHARS=0` or every configuration scores identically.
- **`STRUCTURED_RULE`, `chunk_by_structure` and `merge_target` are production-dead.**
  `add_image_note` / `add_pdf_note` never pass `structured=True`. B4/M2's mechanism has
  never shipped.

The code stays (`merge_target` defaults to 0), the results stay in
`eval/RESULTS_gating.txt` for the write-up, and no further work goes into that line.
Whole-corpus mode is now the *intended* architecture for a student-sized note collection,
not an accident.

### 2026-09-14: sign-in is verified by the server, and sync failures are visible

**Authentication.** The page used to decode the Google ID token in the browser and send
only the email as `user_id`, which the API trusted — anyone who knew an address could list,
read and delete that person's notes (demonstrated against the live Space). Now:

- The page sends the ID token as `Authorization: Bearer`. `main.current_user` verifies it
  with `google-auth` (`id_token.verify_oauth2_token`: Google signature, `aud` = this app's
  client id, expiry, `email_verified`). No endpoint accepts `user_id` any more.
- Verified tokens are cached until expiry, so certificates are not fetched per request.
- Originals are fetched with the header and shown via `blob:` URLs — `<img>`/`<iframe>`
  cannot send headers, and an email or token in their `src` would land in server logs
  (the Space's access log previously showed `?user_id=timhelloyee%40gmail.com`).
  `/original` now sends `Vary: Authorization` because its URL is identical across users.
- CORS now allows `PUT`; editing a note from GitHub Pages had always failed its preflight.
- ID tokens last ~1 hour. The page restores a stored token only while valid, uses
  `auto_select` + One Tap to sign a returning user back in, and signs out visibly on 401.
- **Local development:** `dev_server.py` accepts `X-Dev-User` instead of Google
  (`localStorage.nh_dev_user` on localhost). `DEV_MODE` refuses to turn on when `SPACE_ID`
  is set, so a stray `NOTEHANDLER_DEV` on the Space cannot enable impersonation.

**Sync health** (`rag_core.sync_status()`, exposed at `/api/health`):
`ok` / `disabled` (local, no token) / `unavailable` (restore failed, or no token on a Space —
**all writes and queries refused**, since syncing an unloaded store would overwrite the
Dataset) / `unsaved` (a later sync failed; writes allowed, the next sync repairs it). The
page shows a banner for each and no longer says 「尚無筆記」 when the store never loaded.

Also fixed: originals were saved *after* `_add_note` had already synced the folder, so they
reached the Dataset only on the user's next change. They are now saved first
(`_add_described_note`), and removed if storing the note fails.

`tests/test_auth_and_sync.py` (18 checks, standalone) covers all of the above, including a
forged token carrying a real address being rejected by the real Google verifier.

**Deploying:** the backend and `docs/index.html` must ship together. A new backend rejects
the old page's email-only requests (401), and a new page cannot talk to an old backend.
Use the Space's own URL if GitHub Pages lags behind.

### Known product gaps (recorded, not yet fixed)

- **Subject is not a field.** It exists only as rule #1 in the extraction prompts and the
  `_SUBJECT_TAG` regex — never written to Chroma metadata, never filtered on, never shown
  as a label, and never applied to `.txt`/`.md` notes (they get no Gemini call at all).
  "以科目別整理" is currently not implemented.
- **Answers have no provenance.** `QueryResponse` is `{answer, model}`; chunk ids are
  discarded in `_retrieve`; the context is an undelimited `"
---
".join(docs)`, so the
  model could not cite a source even if asked to.
- ~~Auth on `user_id`~~ and ~~silent sync failure~~ — fixed 2026-09-14, see above.
- **Refusal is not a distinct state** — 「參考資料中沒有記載相關訊息」 renders exactly like
  a short answer, and the API does not expose whether the relevance gate fired.
- **Editing silently drops keyword anchors**: delete the trailing `關鍵字：` line and
  `_keywords_of` finds nothing, disabling that note's keyword boost.
- **`is_blank_page` relies on Gemini breaking its own rule** — it compares against
  `無法辨識`, which is not one of the 11 permitted subject values.
- `eval/run_eval.py`'s local store has no physics note, so 7 of its 28 cases cannot pass
  under `--local`.

**A bug that was fixed here — do not reintroduce it.** The original code had a single
`MAX_DISTANCE = 0.35` applied as a **per-chunk filter**. Because genuine answers sit at
0.33–0.36, almost nothing ever cleared it, and the `kept or [best]` fallback silently
reduced the system to **top-1 retrieval**. Symptom: broad questions like
「我的筆記涵蓋哪些科目？」 returned **1 of 4 subjects**, and *raising `top_k` changed
nothing* (identical byte-for-byte answers at k=10/30/60) because the filter, not `top_k`,
was binding.

The fix splits the one number into the two jobs it was conflating:

1. **`RELEVANCE_CEILING` gates the query.** If even the best chunk exceeds it, nothing is
   on topic → return just that chunk and let the prompt's 「沒有記載相關訊息」 rule fire.
2. **`RELATIVE_MARGIN` picks how many.** Keep everything within `best + 0.10`. This adapts:
   a clear winner stays tight (粒線體 → 1 chunk), a broad question keeps everything it
   matched evenly (→ 5–6 chunks). It also survives an embedding-model change, which an
   absolute per-chunk cutoff does not.

Result: the global question now covers **4/4 subjects**; specific questions return 1–3
chunks; absent topics still correctly refuse.

### Chunk id scheme (both shapes coexist)
- Single-chunk note → bare id, e.g. `a.txt`
- Multi-chunk note → `<file>#0`, `<file>#1`, … ; `_base_id()` maps back
- Notes ingested **before** chunking are still stored whole. `list_notes`/`delete_note`
  handle both. Don't assume every note is chunked.

---

## 6. Model selection

`AVAILABLE_MODELS` in `rag_core.py` — **ids verified against the live `models.list()`
endpoint**, not guessed:

`gemini-3.7-flash` · `gemini-3.6-flash` · `gemini-3.5-flash` · `gemini-3.5-flash-lite` ·
`gemini-3.1-flash-lite` *(default)* · `gemini-3-flash-preview` · `gemini-2.5-flash`

⚠️ **"Gemini 3 Flash" exists only as `gemini-3-flash-preview`** — there is no plain
`gemini-3-flash`. Do not "correct" it.

The dropdown chooses the model for **both** answering and image/PDF parsing. The frontend
populates from `/api/models` (with a hardcoded fallback list if the backend is asleep);
`resolve_model()` validates server-side, so an unknown id silently becomes the default and
an unvetted string never reaches Google. Chat clients are cached per `(model, key)`.

Popular models intermittently return **503 "high demand"** — this is normal upstream
behaviour, not a bug. `_is_retryable()` covers quota *and* transient errors; exhausting
both keys raises `ModelUnavailableError` → HTTP 503 with a message telling the user to
retry or switch model.

---

## 7. Frontend notes (`app/static/index.html`)

Single self-contained page; `docs/index.html` is a **manual copy** for GitHub Pages —
**after editing one, `cp app/static/index.html docs/index.html`.** They drift silently.

- **Auth:** Google Identity Services. The account email becomes `user_id`, remembered in
  `localStorage` (`nh_user`) with a 登出 link. Model choice persists as `nh_model`.
- **⚠️ Google sign-in cannot complete inside the HF Spaces iframe** — the popup at
  `accounts.google.com/gsi/transform` hangs blank because FedCM is blocked in cross-origin
  iframes and HF controls the wrapper. **The Space must be opened in its own tab.** This is
  not fixable from our side.
- OAuth Client ID is public and in the source (safe). Origins must be registered in Google
  Cloud Console — `origin_mismatch` (HTTP 400) means the current origin is missing.
- **Rendering:** `renderRich()` for answers (full Markdown + MathJax), `renderInline()` for
  note previews (inline-only; headings→bold, rules dropped, dangling `$…` trimmed). LaTeX
  spans are stashed before `marked` runs so `$m_1$` isn't mangled by underscore handling,
  then restored. DOMPurify sanitizes.
- `errText()` extracts a readable message from JSON error bodies — previously showed the
  useless `[object Object]`.

---

## 8. Configuration

Local: `.env` (git-ignored; see `.env.example`). Space: **Settings → Secrets**.

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY_PRIMARY` / `_SECONDARY` | **Yes** — import fails without them | Two keys, automatic fail-over |
| `HF_TOKEN` | Effectively yes | Must read/write the private Dataset, else the app starts empty |
| `VECTORSTORE_DATASET` | No | Default `timhelloyee/notehandler-store` |
| `VECTORSTORE_ROOT` | No | Default `./vectorstores` |
| `ALLOWED_ORIGINS` | No | CORS; defaults include `https://timhelloyee.github.io` |
| `GEMINI_MODEL` | No | Overrides the default model |

`requirements.txt` is **deliberately bounded**, notably `chromadb>=1.5,<2` — a major
upgrade on a Space rebuild could make the persisted store unreadable. Don't unpin casually.

---

## 9. Testing approach

There is **no test suite in the repo**. Verification has been done with throwaway scripts
in the scratchpad, using `fastapi.testclient.TestClient` against the local store with
`HF_TOKEN=""` (which disables Dataset sync so tests never touch production data).

Patterns that work:
- `os.environ["HF_TOKEN"] = ""` **before** importing `app.main`, plus `sys.path.insert` and
  `os.chdir` to the project root.
- `load_dotenv()` fails when a script is piped via stdin (`find_dotenv` assertion) — write
  the script to a file instead of using a heredoc.
- The Bash tool's cwd is `ClaudeProjects`; use forward-slash paths.
- Set `PYTHONIOENCODING=utf-8` or Chinese output becomes mojibake in the terminal.

Suites that have passed: 17 model/allowlist/caching checks; 18 chunking/keyword/aggregation
checks; 7 end-to-end filter checks (present / global / absent); live Space checks.

---

## 10. History — what was done and why

Roughly chronological. Details are in `git log`, which has real messages.

1. **Notebooks → web app.** Originally Jupyter notebooks using CLIP for embeddings.
   Replaced with e5-base (CLIP is weak on Chinese and truncates at 77 tokens) and
   refactored into `app/` (FastAPI + static page).
2. **Model IDs** updated to `gemini-3.1-flash-lite` throughout.
3. **Google sign-in** added; the manual user-id box was removed, so the email is the only
   identity. `_safe_name()` sanitizes it for Chroma (`@`→`_`), since collection names allow
   only `[a-zA-Z0-9._-]`. `timhelloyee@gmail.com` → `timhelloyee_gmail.com`.
4. **Deployed** to HF Spaces (Docker) + GitHub, frontend copy in `docs/` for Pages.
5. **Privacy fix (important).** The seeded vectorstore had been committed — it exposed the
   user's email in a directory name and their note text in SQLite. Removed from the working
   tree, purged from history with `git filter-repo`, force-pushed to both remotes, and
   replaced with the private-Dataset persistence described in §3.
6. **Formatting cleanup**, iteratively, driven by screenshots: bold/LaTeX in previews →
   headings and `---` normalized → box glyphs (`□`) and `** x **` stripped at ingest →
   `…` appended when a preview is truncated.
7. **Hardening pass:** Big5/GB18030 upload decoding, dependency bounds, model baked into
   the image for fast cold starts, sign-in persistence, 404 on missing delete,
   `/api/health`, rewritten README (it had gone stale describing the CLIP era, and it is
   also the Space's landing card), `.env.example`, MIT `LICENSE`.
8. **RAG quality pass** — see §11 for the source analysis: chunking + keyword anchors.
9. **Model selector** — dropdown, `/api/models`, allowlist, 503 handling.
10. **Retrieval filter fix** — the top-1 collapse described in §5.

---

## 11. Research the user has evaluated (and the conclusions reached)

The user is exploring RAG techniques, likely for a science-fair / 專題 project, and has
asked for analysis of external sources. Conclusions so far — **don't re-litigate these
without new evidence**:

**Source A — NPHSF science-fair paper, "結合 LSH 及知識圖譜改善 RAG".**
- *Adopted:* **chunking** (the paper's own weakest metric, recall 82%, is explicitly blamed
  on not chunking) and **LLM keyword extraction** (reused as free ranking anchors).
- *Rejected:* **LSH** — Chroma already indexes with HNSW, a strictly better ANN algorithm,
  so this would rebuild a weaker version of what's already running; its "95% faster" claim
  is measured against GraphRAG, not against plain vector search. **Knowledge graph** — the
  paper's own citations report ~10× tokens and $11 to index one 30k-word book. Its
  **metrics** are LLM-judged on n=10 with a non-standard precision definition.

**Source B — iThome article on Microsoft GraphRAG.** Same family as A, heavier.
- *Verdict:* **right diagnosis, wrong scale.** It correctly identifies that top-k retrieval
  fails "global" questions — which was **verified true here** (1/4 subjects). But community
  summaries exist to compress a corpus that *can't* fit in context. This corpus is ~15–25k
  characters and Gemini takes ~1M tokens, so **putting every note in the prompt beats any
  summary hierarchy** — no information loss, one call, no indexing pipeline.
- Investigating this is what surfaced the retrieval-filter bug, which was the real cause.

**Source C — advice to research HNSW's incremental-insertion degradation / tombstone
deletion, or filtered ANN search.**
- The advice does **not** claim HNSW is unsuitable here; it proposes research topics.
- Measured: NoteHandler **already uses HNSW** (Chroma default). Exact brute force over
  10k×768 vectors takes **1.3 ms**, 100k takes 12 ms — versus 1–5 s for the Gemini call. At
  a realistic ceiling of ~10k chunks after years of use, indexing is irrelevant either way,
  and tombstones are a non-issue given rare deletes plus a natural whole-store rebuild
  point in the Dataset sync cycle.
- *Verdict:* a legitimate and genuinely hard **research topic** (filtered ANN is an open
  problem, and it's a stronger topic than Source A) — but it would use NoteHandler only as
  a motivating story, needing 100k+ synthetic vectors and a recall benchmark. It is **not**
  an improvement to the app.

**Recurring theme:** every one of these was settled by *measuring*, not by reasoning from
priors. Model IDs were checked against `models.list()`; the distance metric was confirmed
to be `2-2cos`; the filter bug was found by noticing `top_k` had no effect. Keep doing that.

---

## 12. Open items

**User-side (cannot be done from code):**
1. **Rotate the exposed credentials.** Multiple HF tokens and the Gemini/LangChain/OpenAI
   keys were pasted into chat during development. ⚠️ If `HF_TOKEN` is rotated, update the
   **Space secret** too or the app boots with no notes.
2. **Repo is still private and GitHub Pages is not enabled.** To publish: make it public →
   Settings → Pages → `main` `/docs` → add `https://timhelloyee.github.io` to the Google
   OAuth authorized JavaScript origins. CORS already allows that origin.

**Known limitations / candidate work:**
3. **No real authorization.** The backend trusts whatever `user_id` it receives — anyone
   can read anyone's notes by supplying the email. Fix would be verifying the Google ID
   token server-side rather than decoding it in the browser.
4. **Frontend duplication** — `app/static/index.html` and `docs/index.html` are kept in sync
   by hand.
5. **Whole-corpus mode for aggregative questions** — the remaining idea from the GraphRAG
   analysis: when a user's notes fit in context, skip retrieval entirely. Less urgent now
   that the filter is fixed.
6. **Persistence is coarse** — the whole user folder is re-uploaded after each write, and
   concurrent uploads could race. Fine for one user, not for many.
7. **Free-tier scaling** — CPU-only, and every write is a Dataset round trip.
8. **`embed_folder.py` is a local CLI, not used by the app**, and chunks with a *different*
   id scheme (`a.txt#0` at 140 chars). Pointing it at the app's store would duplicate
   content. The old notebooks were removed on 2026-09-14 and replaced by
   `notehandler_input.py` / `notehandler_output.py`, which call `app/rag_core.py` and use
   the isolated `.devstore/` (the repo now ignores `*.ipynb`).

---

## 13. Working style the user expects

- **Measure before claiming.** They have corrected wrong assumptions before (e.g. insisting
  `gemini-3.1-flash-lite` existed when it was dismissed as a typo — they were right).
- **Don't push without being asked** when they've said to pause; they will say "continue".
- They review screenshots and report visual bugs precisely — expect UI polish iterations.
- They care about privacy of their own data (this drove the history purge).
- Mixed English/Chinese; UI strings and prompts are **Traditional Chinese** — keep them so.
- Deploy = **two remotes**. Forgetting the HF push means the live app silently lags GitHub.

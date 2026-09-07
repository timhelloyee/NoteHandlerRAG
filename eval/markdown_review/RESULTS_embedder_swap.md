# NoteHandler — Embedder Swap Experiment (e5-base → Qwen3-Embedding-0.6B)

**Date:** 2026-09-07 · **Corpus:** `pdf/`, 62 pages · **Probes:** 29 · **Status:** complete, production default unchanged

Handoff note for another model. Everything below is measured, not estimated. Where a
result is null or a caveat applies it is stated as such — do not repair the numbers into
a cleaner story.

---

## 1. What was tested

NoteHandler is a multimodal RAG system over a student's handwritten Traditional Chinese
study notes. Prior work (the "structure-preservation gate") compared three ingestion
configurations and returned a null result. The hypothesis tested here is different: that
the binding constraint was never chunking but the **encoder** — specifically that
`intfloat/multilingual-e5-base` range-compresses Traditional Chinese so severely
(unrelated pairs still ~0.77 cosine) that retrieval had almost no signal to rank on.

The swap candidate is `Qwen/Qwen3-Embedding-0.6B`, chosen from a 2025 literature sweep
(MMTEB #1 among open models at time of publication, ~70.58 vs multilingual-E5 ≈63).

### Configurations (unchanged from the prior experiment)

| Config | Ingestion | Chunks |
|---|---|---|
| **B1** | prose extraction (Gemini) + `_chunk_text(450, 50)` sliding window | 161 |
| **B4** | nested-Markdown extraction + `chunk_by_structure` + ancestor prefixes | 301 |
| **B0** | human oracle: manually corrected text + human structure tree | 300 |

### Metrics

- **chunk-level MRR / R@k** — rank of the first retrieved chunk *containing the gold
  answer span*. Config-independent (chunk boundaries differ between configs, so gold is
  text, not a chunk id). This is the metric chunking and encoding can actually move.
- **page-level MRR / R@k** — rank of the first chunk belonging to the gold page.
  Saturates on a small corpus; reported for continuity, not as the headline.
- **Paired bootstrap**, 10 000 resamples, 95% CI. Paired because all configurations
  answer the same probes; pairing removes per-question difficulty, the dominant variance
  source at n=29.

### Probe set

29 probes over 6 subjects (程式設計, 公民, 地科, 歷史, 生物, 物理): 19 `fact`, 10
`hierarchy`. Every gold span was verified to appear in the body of exactly one page.

---

## 2. Headline result

| Config | Encoder | chunk R@1 | chunk R@3 | chunk MRR | page R@1 | page MRR |
|---|---|---|---|---|---|---|
| B1 | e5-base (768d) | 0.69 | 0.93 | 0.813 | 0.79 | 0.865 |
| **B1** | **Qwen3-0.6B (1024d)** | **0.83** | **0.97** | **0.897** | **0.83** | **0.897** |
| B4 | e5-base | 0.62 | 0.97 | 0.783 | 0.79 | 0.882 |
| B4 | Qwen3-0.6B | 0.76 | 0.93 | 0.852 | 0.83 | 0.891 |
| B0 | e5-base | 0.62 | 0.83 | 0.738 | 0.86 | 0.915 |
| B0 | Qwen3-0.6B | 0.66 | 0.86 | 0.766 | 0.76 | 0.859 |

### Cross-encoder paired bootstrap (Qwen3 − e5), chunk level

| Config | e5 MRR | Qwen3 MRR | diff | 95% CI | W/L/T |
|---|---|---|---|---|---|
| B1 | 0.813 | 0.897 | **+0.083** | [−0.009, +0.175] | 9W/2L/18T |
| B4 | 0.783 | 0.852 | +0.069 | [−0.046, +0.176] | 9W/3L/17T |
| B0 | 0.738 | 0.766 | +0.028 | [−0.074, +0.121] | 6W/3L/20T |

Page level: B1 +0.032 [−0.046, +0.109]; B4 +0.009 [−0.083, +0.098]; B0 −0.056 [−0.152, +0.029].

**Interpretation — read this carefully.** The B1 interval grazes zero and therefore does
**not** establish a difference at the 95% level. It is nonetheless the largest effect
this corpus has produced across the whole project and the first CI that comes close to
excluding zero. The 9W/2L sign pattern is p≈0.065 on its own. Correct summary: *promising,
directionally consistent, not statistically separated at n=29.* Do not upgrade this to
"significant".

---

## 3. Secondary result: the prior finding survives

Within-run pairwise comparisons, **chunk level**:

| Comparison | e5 | Qwen3 |
|---|---|---|
| B4 − B1 | −0.030 [−0.149, +0.084] (6W/7L) | −0.045 [−0.163, +0.075] (3W/6L) |
| B0 − B1 | −0.075 [−0.179, +0.023] (3W/8L) | **−0.131 [−0.268, −0.006] (3W/8L) → B1 better** |
| B0 − B4 | −0.045 [−0.131, +0.041] (4W/7L) | −0.086 [−0.195, +0.008] (1W/6L) |

Ordering is **B1 ≥ B4 > B0 at chunk level under both encoders**. The
fidelity-vs-retrievability finding — the human oracle retrieves *worse* than Gemini's
looser extraction, because stripping invented headings and padding leaves terser chunks
with less lexical surface — is therefore not an artefact of a weak encoder. Under Qwen3
it becomes the only comparison in the project whose CI excludes zero.

Page level under Qwen3 reverses sign relative to e5 (B0 − B1 goes +0.050 → −0.037), but
every page-level CI spans zero under both encoders, so this is not a finding.

---

## 4. Distance-scale findings (these generalise beyond this project)

Chroma reports squared L2 over normalised vectors — **exactly `2 − 2·cos`**, verified
empirically to 4 dp.

### 4.1 Separation improved substantially

Best (smallest) distance per question, over the B1 index:

| Encoder | present (n=29) min/median/max | absent (n=8) min/median/max |
|---|---|---|
| e5-base | 0.211 / 0.285 / **0.372** | **0.283** / 0.387 / 0.418 |
| Qwen3-0.6B | 0.408 / 0.662 / **1.125** | **0.830** / 1.169 / 1.368 |

Spot check on 3 question/passage pairs against 2 distractors each — margin between the
true passage and the nearest wrong one:

| Encoder | per-pair gaps (cosine) | mean |
|---|---|---|
| e5-base | +0.122, +0.080, +0.101 | **0.101** |
| Qwen3-0.6B | +0.487, +0.420, +0.586 | **0.498** |

Unrelated Traditional Chinese falls from ~0.79 cosine to ~0.31.

### 4.2 Raw-distance constants belong to the ENCODER, not the app

This is the most transferable lesson. `RELEVANCE_CEILING = 0.38` was calibrated for e5.
Qwen3's **smallest genuine answer distance is 0.408** — above the ceiling. Swapping the
model without re-calibrating would have made the app refuse *every* question while
appearing to work.

Per-model values now live in `rag_core._GATE_BY_MODEL`; `eval/calibrate_gate.py`
measures them for a new model, and an uncalibrated model triggers a startup warning.

| Encoder | RELEVANCE_CEILING | RELATIVE_MARGIN |
|---|---|---|
| intfloat/multilingual-e5-base | 0.38 | 0.10 |
| Qwen/Qwen3-Embedding-0.6B | 1.00 | 0.30 |

### 4.3 The present/absent populations OVERLAP under both encoders

No single ceiling separates them cleanly. Best achievable split:

| Encoder | split | accepts present | rejects absent |
|---|---|---|---|
| e5-base | 0.341 | 28/29 | 5/8 |
| Qwen3-0.6B | 1.003 | 28/29 | **6/8** |

Marginal improvement, not a solution. The refusal path remains the weakest part of the
system and is the obvious next target.

### 4.4 `RELATIVE_MARGIN` recalibration

Mean chunks kept out of top-10 (this is what gets handed to the LLM):

| margin | e5-base | Qwen3-0.6B |
|---|---|---|
| 0.05 | 5.72 | 1.28 |
| 0.10 | **9.21** | 1.69 |
| 0.15 | 9.83 | 2.31 |
| 0.20 | 10.00 | 2.97 |
| 0.26 | 10.00 | 4.17 |
| 0.35 | 10.00 | **5.97** |
| 0.50 | 10.00 | 8.45 |

Note what this exposes: under e5, `RELATIVE_MARGIN = 0.10` kept 9.2 of 10 chunks — the
"adaptive" filter was barely filtering, because e5's distances were all bunched together.
Under Qwen3 the same mechanism actually discriminates. 0.30 was chosen to keep ~5 chunks,
comparable to prior behaviour. **This value is not validated against answer quality**
(see §7).

### 4.5 `KEYWORD_BONUS` — measured for the first time

The harness scores raw retrieval; the app additionally applies a keyword-overlap bonus
that lowers effective distance. That path had never been evaluated. Sweep over the same
29 probes (chunk MRR):

| KEYWORD_BONUS | e5-base | Qwen3-0.6B |
|---|---|---|
| 0.00 | 0.813 | 0.897 |
| **0.03** | **0.819** | **0.913** |
| 0.06 | 0.813 | 0.910 |
| 0.09 | 0.813 | 0.910 |
| 0.15 | 0.813 | 0.910 |

Flat above 0.03 on both encoders, so — unlike the gate — this constant is *not*
scale-bound in practice and one value serves both. Kept at 0.03. An earlier
scale-relative formula (`0.3 × RELATIVE_MARGIN`) was written and then reverted, because
the data did not support the extra machinery.

---

## 5. Reproducibility hazard: Qwen3 at fp16 is non-deterministic

Two otherwise identical fp16 runs disagreed. Probe `cs-argv` moved between **rank 1 and
absent from top-10**, worth 0.035 MRR on a 29-probe set.

Diagnosis performed, in order:

1. **HNSW approximation ruled out.** Compared Chroma's ANN ranking against exact
   brute-force cosine over the full 161×1024 matrix: **0/29 mismatches**. The index is
   not losing hits at this corpus size.
2. **float32 is bit-identical across runs** (verified 2× on B1, and 3× at fp16 to
   establish the fp16 value was itself stable *within* a session).
3. Conclusion: fp16 rounding at 1024 dims flips near-ties, and which way it rounds
   depends on the cuBLAS kernel the driver selects — which can vary with GPU memory
   state between sessions.

**All numbers in this document are float32.** Measurement runs must set
`EMBED_DTYPE=float32`. fp16 remains the default for serving, where the memory saving
matters and a one-probe reordering does not.

---

## 6. Per-probe chunk ranks (float32, `None` = not in top-10)

| probe | type | B1 e5 | B1 Qwen | B4 e5 | B4 Qwen | B0 e5 | B0 Qwen |
|---|---|---|---|---|---|---|---|
| bio-acid-growth | fact | 2 | 1 | 1 | 1 | 2 | 2 |
| bio-darwin-oat | fact | 1 | 1 | 3 | 2 | 2 | 2 |
| bio-zeatin | hierarchy | 1 | 1 | 1 | 1 | 1 | 1 |
| bio-triple | fact | 1 | 1 | 1 | 1 | 1 | 1 |
| bio-hsp | fact | 1 | 1 | 1 | 1 | 1 | 1 |
| bio-aril | hierarchy | 1 | 1 | 1 | 1 | 1 | 1 |
| bio-hypogeal | hierarchy | 1 | 1 | 2 | 2 | 2 | 2 |
| bio-stone-cell | fact | 1 | 1 | 3 | 2 | 2 | 1 |
| hist-aigun | fact | 1 | 2 | 1 | 1 | 1 | 1 |
| hist-triple-intervention | fact | 1 | 1 | 1 | 1 | 1 | 1 |
| hist-northeast | fact | 1 | 1 | 1 | 2 | 1 | 2 |
| hist-xian | fact | **None** | **6** | 3 | 1 | 5 | 2 |
| hist-leap | hierarchy | 2 | 1 | 2 | 1 | None | None |
| hist-meiji | hierarchy | 3 | 2 | 2 | 5 | None | 9 |
| cs-setprecision | fact | 2 | 1 | 2 | 1 | 3 | 1 |
| cs-bubble | hierarchy | 2 | 1 | 1 | 1 | 1 | 1 |
| cs-pointer | fact | 1 | 1 | 2 | 1 | 6 | 7 |
| cs-structured-binding | fact | 1 | 1 | 1 | 1 | 1 | 1 |
| cs-argv | fact | 1 | 1 | 1 | 1 | 1 | 1 |
| cs-undefined | hierarchy | 4 | 3 | 5 | 3 | 5 | 3 |
| es-younger-dryas | fact | 1 | 1 | 1 | 1 | 1 | 1 |
| es-kyoto | hierarchy | 2 | 1 | 1 | 1 | 2 | 1 |
| es-treering | hierarchy | 1 | 1 | 1 | 1 | 1 | 1 |
| es-precession | fact | 1 | 1 | 2 | 1 | 1 | 1 |
| civ-appeal | fact | 1 | 1 | 1 | 1 | 1 | 1 |
| civ-legality | hierarchy | 1 | 2 | 1 | 1 | 1 | 1 |
| phys-lagrange | fact | 1 | 1 | 2 | 1 | 1 | 1 |
| phys-areal | fact | 2 | 1 | 1 | 6 | 1 | 8 |
| phys-atomic | fact | 1 | 1 | 1 | 1 | 1 | 1 |

**Ceiling effect persists.** Under Qwen3, B1 already ranks the answer #1 on 24 of 29
probes (up from 20 under e5). Only 5 probes retain headroom. This makes the current
probe set *less* able to discriminate future changes, not more — which is an argument
for building the harder pre-registered probe set before running further comparisons.

---

## 7. Limitations — do not overstate this result

1. **No answer-quality measurement.** Every number is retrieval rank. Whether better
   retrieval yields better generated answers is untested. The app-level harness
   (`eval/run_eval.py`) still points at an obsolete demo corpus and would need Gemini
   quota.
2. **n = 29.** The headline CI includes zero. A ceiling effect (24/29 at rank 1) caps
   what this probe set can ever show.
3. **`RELATIVE_MARGIN = 0.30` for Qwen3 is a behaviour-matching choice**, not an
   optimised one — it reproduces the number of chunks e5 was passing to the LLM. It has
   no evidence behind it beyond that.
4. **The absent-question set is n = 8 and hand-written.** §4.3's accept/reject counts are
   suggestive, not a calibration.
5. **Single corpus, single language pair.** Handwritten Traditional Chinese school notes.
   No claim beyond that.

---

## 8. Production status and migration cost

**The default remains `intfloat/multilingual-e5-base`.** Switching is a data migration,
not a config change:

- Stored vectors are 768-wide; every note must be re-embedded at 1024. The extracted text
  is already stored, so this needs **no Gemini quota** — a re-embed script plus a re-upload
  to the private HF Dataset.
- `get_user_collection` now refuses a dimension mismatch with an explicit, actionable
  error rather than letting Chroma fail obscurely at query time.
- The HF Space free tier is **CPU-only**. Qwen3-0.6B is ~5.5× the parameters of e5-base,
  so query embedding latency rises — still small relative to the Gemini call, but real.
  fp32 weights are ~2.4 GB.

---

## 9. Code changes

| File | Change |
|---|---|
| `app/rag_core.py` | `EMBED_MODEL` / `EMBED_DTYPE` env vars; per-family prompt prefixes; `_GATE_BY_MODEL`; `EMBED_DIM`; dimension guard in `get_user_collection` |
| `eval/gating_experiment.py` | writes per-probe rank dumps to `eval/runs/<model>.json` |
| `eval/compare_runs.py` | **new** — paired bootstrap across two runs from different encoders |
| `eval/calibrate_gate.py` | **new** — measures present/absent distance populations, suggests a ceiling |
| `eval/RESULTS_gating.txt` | appended full run log |
| `PROJECT_OVERVIEW.md` | distance table annotated as encoder-specific; migration note |

### Prompt formats are part of the model, not decoration

```
e5-base     documents: "passage: " + text        queries: "query: " + text
Qwen3       documents: text (no prefix)          queries: "Instruct: {task}\nQuery: " + text
```

Feeding one family the other's format degrades retrieval silently. `DOC_PREFIX` /
`QUERY_PREFIX` are keyed off the model name.

### Reproducing

```bash
# baseline
python eval/gating_experiment.py

# candidate — float32 is mandatory for reproducible numbers
EMBED_DTYPE=float32 EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B python eval/gating_experiment.py

# cross-encoder comparison (loads no model)
python eval/compare_runs.py intfloat_multilingual-e5-base Qwen_Qwen3-Embedding-0.6B

# gate calibration for a new encoder
EMBED_DTYPE=float32 EMBED_MODEL=<hf-id> python eval/calibrate_gate.py
```

Corpora are disk-cached JSON (`eval/corpus/corpus_b{0,1,4}.json`), so re-runs cost **no
API calls**. `gating_experiment.py` forces `HF_TOKEN=''` and `WHOLE_CORPUS_CHARS=0`, so it
never touches the private Dataset and always exercises the retrieval path.

**Environment:** torch 2.5.1+cu121, transformers 5.8.1, sentence-transformers 5.4.1,
RTX 3050 Laptop (4.3 GB). Use the `ragtraining` conda interpreter — a bare `python` lacks
torch.

---

## 10. Suggested next steps, ranked

1. **Harder pre-registered probe set** (~40–50 questions targeting pages where B1 ranks
   below #1, weighted to hierarchy/cross-block). The ceiling effect now blocks every
   further comparison, and this was already agreed before the swap. It should come first —
   a better encoder raises the ceiling and makes the current set *less* informative.
2. **Answer-quality evaluation.** Point `run_eval.py` at the `pdf/` corpus. Retrieval rank
   is a proxy; nothing here measures the product.
3. **Attack the refusal path** (§4.3). Overlapping populations mean the gate is guessing on
   ~25% of absent questions under both encoders. Candidates: a small cross-encoder reranker
   (Qwen3-Reranker-0.6B), or an explicit LLM-side abstention check.
4. **Late chunking** (arXiv 2409.04701) as a "B5" config — obtains B4's document context
   without depending on Gemini reconstructing a heading tree, which is precisely where B4
   lost fidelity to B0.
5. Only then reconsider the production migration in §8.

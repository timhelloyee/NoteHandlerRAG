# eval/

Research measurements for the current line: **handwriting recognition and answers, measured
without user correction, against NotebookLM** (see `handoff/實驗架構.md` and `handoff/專題總覽.md` §9, §14).

| Path | What it is | In git |
|---|---|---|
| `r0/手寫筆記稿.md` | Note pages to copy by hand for the R0 NotebookLM baseline | yes |
| `r0/題組與標準答案.md` | R0 questions, gold answers and the scoring sheet | no (answer key) |
| `r0/標準辨識稿.md` | Hand-typed transcript of the 8 handwritten pages, with each diagram's nodes and arrows — the gold for recognition | no (answer key) |
| `r0/關鍵事實.md` / `關鍵事實.sha256.txt` | Facts each answer must contain, words it must not, evidence pages; the hash taken before any NoteHandler answer existed, and the one change since | no (answer key) |
| `r0/NotebookLM_提問單.md` | NotebookLM setup, the 23 questions as asked, and its answers | no |
| `r0/run_baseline.py` | Uploads the 8 pages and asks the 23 questions through the app's own functions, into a scratch store | yes |
| `r0/score.py` | Answer verdicts, character error rate, and three deterministic "unsupported content" rules with their error-injection check | yes |
| `r0/compare_runs.py` | Several runs side by side, with every character each one misread | yes |
| `r0/run_vote.py` | Reads each page N times with sampling on and votes per character; scores the vote and the "reads disagree = suspect" flag | yes |
| `r0/*.jpg` / `r0/results/<run>/` | Scans of the handwritten pages; per run: transcripts, answers, `meta.json` (model, prompts, commit), `score.md` | no |
| `g0/transcripts/` | 66 pages of hand-corrected transcripts of the author's real notes — the gold standard for character error rate (G0/G1) | no (note content) |
| `g0/baseline_gemini_whole_page.json` | The same pages as Gemini read them whole-page, before any correction — the starting point of the G1 ablation | no (note content) |
| `g0/lv.0_校正問題.md` | Notes written while correcting `lv.0` | no (note content) |
| `RESULTS_gating.txt` | Full results of the retired retrieval-structure line (summary in 專題總覽 §11.1) | no (quotes note content) |

The retrieval-structure scripts (`gating_experiment.py`, `calibrate_gate.py`, `run_eval.py`,
the B0 kit, `runs/`) were removed on 2026-09-21; they remain in git history.

`g0/transcripts/` files keep their B0 layout: the corrected text is everything after the
`---` line. Their headings were added for the old structure experiment, so strip Markdown
heading marks before computing character error rate against a plain transcription.

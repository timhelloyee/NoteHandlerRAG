# eval/

Research measurements for the current line: **handwriting recognition and answers, measured
without user correction, against NotebookLM** (see `handoff/實驗架構.md` and `handoff/專題總覽.md` §9, §14).

| Path | What it is | In git |
|---|---|---|
| `r0/手寫筆記稿.md` | Note pages to copy by hand for the R0 NotebookLM baseline | yes |
| `r0/題組與標準答案.md` | R0 questions, gold answers and the scoring sheet | no (answer key) |
| `r0/標準辨識稿.md` | Transcript of the 8 handwritten pages (drafted by Claude, checked word by word by the author; changes logged in its header), with each diagram's nodes and arrows — the gold for recognition | no (answer key) |
| `r0/關鍵事實.md` / `關鍵事實.sha256.txt` | Facts each answer must contain, words it must not, evidence pages; the hash taken before any NoteHandler answer existed, and the one change since | no (answer key) |
| `r0/NotebookLM_提問單.md` | NotebookLM setup, the 23 questions as asked, and its answers | no |
| `r0/run_baseline.py` | Uploads the 8 pages and asks the 23 questions through the app's own functions, into a scratch store | yes |
| `r0/score.py` | Answer verdicts, character error rate, and three deterministic "unsupported content" rules with their error-injection check | yes |
| `r0/compare_runs.py` | Several runs side by side, with every character each one misread | yes |
| `r0/run_vote.py` | Reads each page N times with sampling on and votes per character; scores the vote and the "reads disagree = suspect" flag | yes |
| `r0/run_qwen.py` | Bundles the app's image prompt and the scans for a rented GPU, reads them with Qwen3-VL, and scores it against Gemini | yes |
| `r0/run_claude.py` | Reads the 8 pages with Claude through the API; scores it beside Gemini and Qwen, and the three-model vote | yes |
| `r0/judge_test.py` | Rule-built test answers (gold, look-alike swaps, cut facts) to compare an LLM judge with the mechanical verdicts | yes |
| `r0/confusable_flag.py` | Mines look-alike character pairs from G0 and tests them as a misread flag on R0 | yes |
| `r0/ping_gemini.py` | Pings gemini-3.7/3.8-flash from a UserPromptSubmit hook; wakes Claude to run R0 when one answers | yes |
| `r0/run_tiles.py` | Cuts each page into bands at blank rows, enlarges and reads each band alone, joins the readings | yes |
| `r0/sinica_freq.py` / `r0/classify_errors.py` | Word counts from the Academia Sinica Balanced Corpus online search (cached); rule-based class for every misread | yes |
| `r0/compare_ngram.py` / `r0/compare_wordfreq.py` | The same word pairs in Google Books Ngram (Simplified Chinese) and in wordfreq | yes |
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

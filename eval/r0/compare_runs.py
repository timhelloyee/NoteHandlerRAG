"""Side-by-side numbers for several NoteHandler runs, and the exact characters each misread.

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/compare_runs.py baseline baseline_r2 prompt_f3b

Run names are folders under eval/r0/results/. Uses score.py's metrics unchanged, so a
difference between two runs is a difference in the runs, not in how they were scored.
Prints a Markdown table and, per run, every substitution or deletion against 標準辨識稿.md.
"""
import difflib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_baseline import parse_questions                                   # noqa: E402
from score import (Grounding, edit_counts, gold_text_sections, model_body,  # noqa: E402
                   notehandler_answers, parse_facts, prompt_texts, score_answer, text_only)


def main(runs: list):
    gold = gold_text_sections(HERE / "標準辨識稿.md")
    keys = parse_facts(HERE / "關鍵事實.md")
    qs = parse_questions(HERE / "NotebookLM_提問單.md")
    subj = {q["id"]: q["subject"] for q in qs}
    g = Grounding(HERE / "標準辨識稿.md", qs, extra_glue=prompt_texts())

    rows, misreads = [], []
    for run in runs:
        d = HERE / "results" / run
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        n = s = dl = 0
        errs = []
        for p in sorted(gold):
            ref = text_only(gold[p])
            hyp = text_only(model_body((d / "transcripts" / f"R0_p{p}.md").read_text(encoding="utf-8")))
            a, b, _ = edit_counts(ref, hyp)
            n, s, dl = n + len(ref), s + a, dl + b
            sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
            errs += [f"p{p} {ref[i1:i2]}→{hyp[j1:j2] or '∅'}" for op, i1, i2, j1, j2 in sm.get_opcodes()
                     if op in ("replace", "delete") and i2 - i1 <= 4 and j2 - j1 <= 6]
        ans = notehandler_answers(d)
        ok = sum(score_answer(*ans[q], keys[q], q.startswith("D"))["verdict"] == "正確" for q in keys)
        wrong = [q for q in keys if score_answer(*ans[q], keys[q], q.startswith("D"))["verdict"] != "正確"]
        gr = [g.check(ans[q][0], subj[q]) for q in keys]
        out = sum(r["outside_chars"] for r in gr) / max(sum(r["chars"] for r in gr), 1)
        rel = sum(len(r["relations"]) for r in gr)
        rows.append(f"| {run} | {meta['model']} | `{meta['prompt_sha256'][:8]}` | {s + dl}/{n}（{(s + dl) / n:.1%}） "
                    f"| {ok}/23 | {'、'.join(wrong) or '—'} | {rel} | {out:.1%} |")
        misreads.append(f"- **{run}**：" + "；".join(errs))

    print("| 執行 | 模型 | 提示詞 | 漏/錯字 | 答案正確 | 答錯的題 | 錯誤關係 | 筆記外用詞 |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print()
    print("\n".join(misreads))


if __name__ == "__main__":
    main(sys.argv[1:] or ["baseline"])

"""Flag characters a model tends to write in place of what the notes say (汽 for 氣, 型 for 形 …).

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/confusable_flag.py

Cross-model checking (run_qwen.py) cannot see an error both models make. This rule needs no
second model: the pairs (notes wrote a, the model wrote b) are mined from G0 — 66 pages of
the author's real notes, Gemini's old whole-page reading against the corrected transcript —
and every b in a new transcript is flagged. The rule is then tested on R0, which played no
part in the mining. G0's transcripts were corrected by an AI, not typed by hand, so a pair
it missed is missing here too. Writes results/confusable_flag.md.
"""
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
G0 = HERE.parent / "g0"
sys.path.insert(0, str(HERE))
from run_vote import PAGES, detection, vote                      # noqa: E402
from score import gold_text_sections, model_body, text_only        # noqa: E402

MIN_COUNT = 2                                                      # a one-off is often alignment noise


def g0_pairs() -> Counter:
    base = {(d["file"].removesuffix(".pdf"), p["page"]): p["text"]
            for d in json.loads((G0 / "baseline_gemini_whole_page.json").read_text(encoding="utf-8"))
            for p in d["page_texts"]}
    pairs = Counter()
    for f in sorted((G0 / "transcripts").glob("*.md")):
        stem, page = f.stem.rsplit("__p", 1)
        corrected = f.read_text(encoding="utf-8").split("\n---\n", 1)[1]
        ref, hyp = text_only(corrected), text_only(model_body(base[(stem, int(page))]))
        for op, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
            if op == "replace" and i2 - i1 == j2 - j1 <= 2:
                pairs.update((a, b) for a, b in zip(ref[i1:i2], hyp[j1:j2]) if a != b)
    return Counter({p: n for p, n in pairs.items() if n >= MIN_COUNT})


def main():
    pairs = g0_pairs()
    suspect = {b for _, b in pairs}
    gold = gold_text_sections(HERE / "標準辨識稿.md")
    res = HERE / "results"
    load = lambda f: text_only(model_body(f.read_text(encoding="utf-8")))
    tally = {k: Counter() for k in ("易混字清單", "Gemini 對照 Qwen", "兩者聯集")}
    caught = []
    for p in PAGES:
        ref = text_only(gold[p])
        gem, qwen = load(res / "baseline" / "transcripts" / f"R0_p{p}.md"), load(res / "qwen_out" / "t0" / f"R0_p{p}.md")
        lex = [c in suspect for c in gem]
        cross = vote([gem, qwen])[2]
        for name, flags in (("易混字清單", lex), ("Gemini 對照 Qwen", cross),
                            ("兩者聯集", [a or b for a, b in zip(lex, cross)])):
            tally[name].update(detection(ref, gem, flags))
        for op, i1, i2, j1, j2 in SequenceMatcher(None, ref, gem, autojunk=False).get_opcodes():
            if op == "replace" and i2 - i1 == j2 - j1:
                caught += [f"p{p} {a}→{b}：{'有' if lex[j1 + t] else '無'}標記{'、Qwen 也讀錯' if not cross[j1 + t] else ''}"
                           for t, (a, b) in enumerate(zip(ref[i1:i2], gem[j1:j2]))]
    lines = ["# 易混字清單：從 G0 挖字對，在 R0 測試", "",
             f"G0 挖出 {len(pairs)} 組字對（出現 ≥ {MIN_COUNT} 次）："
             + "、".join(f"{a}→{b}（{n}）" for (a, b), n in pairs.most_common()), "",
             "## 找 Gemini 的讀錯（R0，Gemini 溫度 0）", "",
             "| 方法 | 標記字數 | 抓到 | 精確率 | 召回率 |", "|---|---|---|---|---|"]
    for name, c in tally.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        lines.append(f"| {name} | {tp + fp} | {tp}/{tp + fn} | {tp / max(tp + fp, 1):.0%} | {tp / max(tp + fn, 1):.0%} |")
    lines += ["", "## Gemini 每個可對齊的讀錯字", ""] + [f"- {c}" for c in caught]
    (res / "confusable_flag.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

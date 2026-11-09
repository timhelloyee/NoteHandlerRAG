"""Multi-read voting on the R0 pages: read each page N times with sampling on, vote per character.

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/run_vote.py            # read + score
    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/run_vote.py --score-only

At temperature 0 the model reads a page the same way every time (baseline vs baseline_r2), so
there is nothing to vote on. Here only this process raises the temperature, by wrapping
rag_core._vision_generate: the app's prompt, model and post-processing are untouched, and
nothing in the product changes.

Voting happens on the content characters (score.text_only of the transcript), the same
projection the character error rate is computed on. One read, the one most similar to the
others, is the pivot; every other read is aligned to it and votes at each pivot position.
Positions where the reads do not all agree are flagged as suspect — which needs no gold
transcript, so it is also tested as a way to find misreads automatically.

Writes --out (default eval/r0/results/vote_t07): reads/R0_pN_k.md, meta.json, vote.md.
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from run_baseline import SUBJECT_PAGES, git_state, retry           # noqa: E402
from score import edit_counts, gold_text_sections, model_body, text_only  # noqa: E402

PAGES = [n for ps in SUBJECT_PAGES.values() for n in ps]


def read_pages(out: Path, n: int, temperature: float, model: str):
    os.environ["HF_TOKEN"] = ""                      # same isolation as run_baseline.py
    os.environ["VECTORSTORE_ROOT"] = str(out / "store")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from app import rag_core                         # noqa: E402

    original = rag_core._vision_generate
    rag_core._vision_generate = lambda contents, config, model=None: original(
        contents, {**config, "temperature": temperature}, model=model)
    (out / "reads").mkdir(parents=True, exist_ok=True)
    for p in PAGES:
        for k in range(1, n + 1):
            f = out / "reads" / f"R0_p{p}_{k}.md"
            if f.is_file():                          # resumable after a 503 abort
                continue
            print(f"read p{p} #{k}")
            text = retry(lambda: rag_core.describe_image(str(HERE / f"R0_p{p}.jpg"), model=model), f"p{p}#{k}")
            f.write_text(text, encoding="utf-8")
    return rag_core


def vote(reads: list) -> tuple:
    """(voted text, pivot text, suspect flag per pivot position)."""
    piv = max(range(len(reads)), key=lambda i: sum(
        SequenceMatcher(None, reads[i], r, autojunk=False).ratio() for r in reads))
    P = reads[piv]
    cols = [[c] for c in P]
    for k, r in enumerate(reads):
        if k == piv:
            continue
        for op, i1, i2, j1, j2 in SequenceMatcher(None, P, r, autojunk=False).get_opcodes():
            if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
                for t in range(i2 - i1):
                    cols[i1 + t].append(r[j1 + t])
            elif op == "delete":
                for t in range(i1, i2):
                    cols[t].append(None)
            # an insert, or a replace of a different length, casts no vote here
    out, flags = [], []
    for col in cols:
        cnt = Counter(col)
        top = cnt.most_common(1)[0][1]
        best = col[0] if cnt[col[0]] == top else cnt.most_common(1)[0][0]   # ties keep the pivot
        flags.append(len(cnt) > 1 or len(col) < len(reads))
        if best is not None:
            out.append(best)
    return "".join(out), P, flags


def wrong_positions(ref: str, hyp: str) -> set:
    """Gold positions that hyp substitutes or drops."""
    bad = set()
    for op, i1, i2, _, _ in SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            bad.update(range(i1, i2))
    return bad


def detection(ref: str, pivot: str, flags: list) -> Counter:
    """Suspect flags vs real errors (tp/fp/fn/tn), over pivot characters aligned one-to-one
    with the gold text. An unequal replace is a diagram description lined up against a
    missing gold line, not a run of misread characters, so it is left out."""
    c = Counter()
    for op, i1, i2, j1, j2 in SequenceMatcher(None, ref, pivot, autojunk=False).get_opcodes():
        if op not in ("equal", "replace") or i2 - i1 != j2 - j1:
            continue
        for t in range(j1, j2):
            err, flag = op == "replace", flags[t]
            c["tp" if err and flag else "fn" if err else "fp" if flag else "tn"] += 1
    return c


def score(out: Path, n: int) -> list:
    gold = gold_text_sections(HERE / "標準辨識稿.md")
    lines = []
    tot = Counter()
    single = [Counter() for _ in range(n)]
    for p in PAGES:
        ref = text_only(gold[p])
        base = text_only(model_body((HERE / "results" / "baseline" / "transcripts" / f"R0_p{p}.md").read_text(encoding="utf-8")))
        tot["base"] += sum(edit_counts(ref, base)[:2])
        reads = [text_only(model_body((out / "reads" / f"R0_p{p}_{k}.md").read_text(encoding="utf-8")))
                 for k in range(1, n + 1)]
        voted, pivot, flags = vote(reads)
        s, d, _ = edit_counts(ref, voted)
        tot["N"] += len(ref)
        tot["voted"] += s + d
        for k, r in enumerate(reads):
            a, b, _ = edit_counts(ref, r)
            single[k]["err"] += a + b
        # Error types: at each gold position, how many of the n reads got it wrong.
        per_read = [wrong_positions(ref, r) for r in reads]
        for j in set().union(*per_read):
            m = sum(j in w for w in per_read)
            tot["stable" if m == n else "unstable"] += 1
            if m == n:
                tot.setdefault("stable_chars", [])
                tot["stable_chars"].append(f"p{p}「{ref[j]}」")
        tot.update(detection(ref, pivot, flags))
        lines.append(f"| p{p} | {len(ref)} | {s + d} | {sum(flags)} |")
    N = tot["N"]
    singles = [c["err"] for c in single]
    prec = tot["tp"] / max(tot["tp"] + tot["fp"], 1)
    rec = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
    return [
        "## 辨識層", "",
        "| | 漏/錯字 | 比例 |", "|---|---|---|",
        f"| Gemini 單次讀（溫度 0，baseline） | {tot['base']}/{N} | {tot['base'] / N:.1%} |",
        f"| 單次讀（本次溫度，5 次各自） | {'、'.join(map(str, singles))} / {N} | "
        f"{min(singles) / N:.1%}–{max(singles) / N:.1%} |",
        f"| **投票後** | {tot['voted']}/{N} | **{tot['voted'] / N:.1%}** |", "",
        "## 錯誤類型（每個標準字在 5 次中錯幾次）", "",
        f"- 穩定的錯（5 次都錯）：{tot['stable']} 字 — {'、'.join(tot.get('stable_chars', [])) or '無'}",
        f"- 不穩定的錯（有錯有對）：{tot['unstable']} 字", "",
        "## 「5 次不一致＝可疑」當自動找錯工具", "",
        "只計算能對齊到標準稿的字（圖的文字描述不在標準稿的「文字」段，不列入）。", "",
        f"- 被標記可疑 {tot['tp'] + tot['fp']} 字，其中真的讀錯 {tot['tp']} 字 → 精確率 **{prec:.0%}**",
        f"- 真的讀錯 {tot['tp'] + tot['fn']} 字，其中被標記 {tot['tp']} 字 → 召回率 **{rec:.0%}**", "",
        "## 各頁", "", "| 頁 | 標準字數 | 投票後漏/錯字 | 可疑字數（含圖的描述） |", "|---|---|---|---|", *lines,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--out", default=str(HERE / "results" / "vote_t07"))
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.score_only:
        rc = read_pages(out, args.n, args.temperature, args.model)
        (out / "meta.json").write_text(json.dumps({
            "finished": datetime.now().isoformat(timespec="seconds"), "model": args.model,
            "temperature": args.temperature, "n": args.n, "git": git_state(),
            "note": "temperature overridden in this process only; top_p 0.95 / top_k 20 as in the app",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [f"# 多次辨識投票（{args.n} 次，溫度 {args.temperature}，{args.model}）", ""] + score(out, args.n)
    (out / "vote.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    assert vote(["abc", "abc", "axc"])[0] == "abc"
    assert vote(["abc", "axc", "axc"])[0] == "axc"
    assert vote(["abc", "ac", "ac"])[0] == "ac"
    main()

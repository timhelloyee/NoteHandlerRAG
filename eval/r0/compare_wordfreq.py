"""Do the misreads move toward the words common in Simplified-Chinese text rather than in Taiwan?

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/compare_wordfreq.py

Takes the word pairs classify_errors.py compared with the Academia Sinica corpus (Taiwan,
1981–2007, results/error_classes.md) and looks the same pairs up in wordfreq's Chinese list,
which is mostly Simplified-Chinese text. A pair is compared only when wordfreq treats both
forms as one word; a form it splits into pieces gets a pieced-together number, not a word count.
Writes results/error_classes_wordfreq.md.
"""
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import wordfreq                                                   # noqa: E402

HERE = Path(__file__).resolve().parent
ROW = re.compile(r"^\| p(\d) \| (.+?)→(.+?) \| (\S+) \| (.*?)／(.*?) \| (\d*)／(\d*) \| (.+) \|$")


def direction(g: float, o: float) -> str:
    return "往高頻" if o > g else "往低頻" if o < g else "同頻"


def main():
    rows, tally = [], defaultdict(Counter)
    for line in (HERE / "results" / "error_classes.md").read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if not m or m.group(4) == "刪除":
            continue
        p, g, o, cls, wg, wo, fg, fo, runs = m.groups()
        one_word = all(len(wordfreq.tokenize(w, "zh")) == 1 for w in (wg, wo))
        zg, zo = wordfreq.zipf_frequency(wg, "zh"), wordfreq.zipf_frequency(wo, "zh")
        wf = direction(zg, zo) if one_word else "非單一詞"
        rows.append(f"| p{p} | {g}→{o} | {wg}／{wo} | {fg}／{fo} | {cls} | {zg:.2f}／{zo:.2f} | {wf} | {runs} |")
        for run in runs.split("、"):
            tally[run][(cls, wf)] += 1
    out = ["# 同一批詞對：中研院（台灣）vs wordfreq（簡體為主）", "",
           "wordfreq 的數字是 Zipf 值（每十億詞出現次數的 log10）。", "",
           "| 頁 | 標準稿→讀成 | 比較的詞 | 中研院詞頻 | 中研院分類 | wordfreq Zipf | wordfreq 方向 | 出現在 |",
           "|---|---|---|---|---|---|---|---|", *rows, "", "## 各讀法：中研院分類 × wordfreq 方向（處數）", ""]
    for run, c in tally.items():
        out.append(f"- **{run}**：" + "；".join(f"{a}／{b} {n}" for (a, b), n in sorted(c.items())))
    (HERE / "results" / "error_classes_wordfreq.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

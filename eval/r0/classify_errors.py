"""Classify every misread in the R0 runs by fixed rules, with word frequencies from the
Academia Sinica Balanced Corpus 4.0 (sinica_freq.py) instead of anyone's sense of "the
textbook form".

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/classify_errors.py

For each substitution, windows around it (up to 3 characters of context, total length <= 5)
are looked up in both forms, gold and read; the longest window where either form is a corpus
word is the one compared (fG = gold form's count, fO = read form's count):

  刪除               a gold character is missing
  往高頻替換         fO > fG > 0, or fO > 0 = fG (the read form is the more frequent word)
  往低頻替換         fG > fO > 0
  同頻               fG == fO > 0
  非詞替換           fO == 0 < fG (the read form is not a corpus word)
  語料庫無資料       no window is a corpus word in either form (rare proper names, slips)
  證據不足           the compared words have fewer than MIN_HITS hits in both forms

The search reports at most 20000 hits, so very common words (是、的、之、與) tie at 20000.
MIN_HITS was set after the first run showed comparisons like 0 vs 1.

Writes results/error_classes.md.
"""
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score import gold_text_sections, model_body, text_only    # noqa: E402
from sinica_freq import freq                                   # noqa: E402

RUNS = {"Gemini 3.1 Flash Lite 整頁": "baseline/transcripts", "Gemini 3.1 Flash Lite 切塊": "tiles_gemini31/transcripts",
        "Gemini 3.1 Flash Lite F3b 提示詞": "prompt_f3b/transcripts", "Gemini 3 Flash 整頁": "model_g3flash/transcripts",
        "Qwen3-VL-8B 整頁": "qwen_out/t0", "Claude Opus 5.5 整頁": "claude_opus55/transcripts",
        "Claude Opus 5.5 切塊": "tiles_opus55/transcripts", "Claude Sonnet 5.5 整頁": "claude_sonnet55/transcripts",
        "Claude Sonnet 5.5 切塊": "tiles_sonnet55/transcripts"}
MAX_CTX, MAX_LEN, MIN_HITS = 3, 5, 5


def compare(ref, i1, i2, hyp, j1, j2):
    """(class, gold window, read window, fG, fO)."""
    best = None
    for a in range(MAX_CTX + 1):
        for b in range(MAX_CTX + 1):
            if max(i2 - i1, j2 - j1) + a + b > MAX_LEN or i1 - a < 0 or j1 - a < 0:
                continue
            g, o = ref[i1 - a:i2 + b], hyp[j1 - a:j2 + b]
            fg, fo = freq(g), freq(o)
            if max(fg, fo) and (best is None or len(g) > len(best[0]) or
                                (len(g) == len(best[0]) and max(fg, fo) > max(best[2], best[3]))):
                best = (g, o, fg, fo)
    if best is None:
        return "語料庫無資料", ref[i1:i2], hyp[j1:j2], 0, 0
    g, o, fg, fo = best
    cls = ("證據不足" if max(fg, fo) < MIN_HITS else "非詞替換" if fo == 0 else "往高頻替換" if fo > fg else "往低頻替換" if fo < fg else "同頻")
    return cls, g, o, fg, fo


def main():
    gold = gold_text_sections(HERE / "標準辨識稿.md")
    rows, tally = [], defaultdict(Counter)
    seen = {}
    for name, d in RUNS.items():
        for p in sorted(gold):
            ref = text_only(gold[p])
            hyp = text_only(model_body((HERE / "results" / d / f"R0_p{p}.md").read_text(encoding="utf-8")))
            for op, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
                if op == "delete":
                    tally[name]["刪除"] += i2 - i1
                    seen.setdefault((p, i1, ref[i1:i2], "∅"), ("刪除", "", "", "", "", []))[5].append(name)
                elif op == "replace" and i2 - i1 <= 4 and j2 - j1 <= 6:   # longer = a diagram description, not a misread
                    key = (p, i1, ref[i1:i2], hyp[j1:j2])   # position too: a page can repeat an error
                    if key not in seen:
                        seen[key] = (*compare(ref, i1, i2, hyp, j1, j2), [])
                    seen[key][5].append(name)
                    tally[name][seen[key][0]] += i2 - i1
                    print(name, key, seen[key][:5], flush=True)
    classes = ["往高頻替換", "往低頻替換", "同頻", "非詞替換", "證據不足", "語料庫無資料", "刪除"]
    lines = ["# R0 錯字分類（中研院平衡語料庫 4.0 詞頻）", "",
             "每格是錯的標準字數。分類規則見 classify_errors.py 開頭。", "",
             "| 讀法 | " + " | ".join(classes) + " | 合計 |", "|---|" + "---|" * (len(classes) + 1)]
    for name in RUNS:
        lines.append(f"| {name} | " + " | ".join(str(tally[name][c]) for c in classes) + f" | {sum(tally[name].values())} |")
    lines += ["", "## 每一個錯誤", "", "| 頁 | 標準稿→讀成 | 分類 | 比較的詞（標準稿／讀成） | 詞頻（標準稿／讀成） | 出現在 |",
              "|---|---|---|---|---|---|"]
    for (p, _, g, o), (cls, wg, wo, fg, fo, names) in sorted(seen.items(), key=lambda kv: (kv[1][0], kv[0])):
        lines.append(f"| p{p} | {g}→{o} | {cls} | {wg}／{wo} | {fg}／{fo} | {'、'.join(names)} |")
    (HERE / "results" / "error_classes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

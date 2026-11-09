"""Taiwan vs mainland usage for the misread word pairs.

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/compare_ngram.py

Takes the word pairs classify_errors.py compared in the Academia Sinica corpus (Taiwan,
Traditional, results/error_classes.md), converts both forms to Simplified (OpenCC t2s) and
looks up each whole string in Google Books Ngram, Chinese (Simplified) 2019 corpus (id 34),
mean relative frequency 1980–2019. Google keeps only strings seen 40+ times, so "absent"
means rare, counted as 0. The Ngram JSON endpoint is not a published API. Answers are
cached in results/ngram_freq.json. Writes results/error_classes_ngram.md.
"""
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from opencc import OpenCC

HERE = Path(__file__).resolve().parent
CACHE = HERE / "results" / "ngram_freq.json"
ROW = re.compile(r"^\| p(\d) \| (.+?)→(.+?) \| (\S+) \| (.*?)／(.*?) \| (\d*)／(\d*) \| (.+) \|$")
t2s = OpenCC("t2s").convert


def ngram(words: list) -> dict:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    todo = [w for w in dict.fromkeys(words) if w not in cache]
    for i in range(0, len(todo), 10):
        chunk = todo[i:i + 10]
        r = requests.get("https://books.google.com/ngrams/json", timeout=30, headers={"User-Agent": "Mozilla/5.0"},
                         params={"content": ",".join(chunk), "year_start": 1980, "year_end": 2019,
                                 "corpus": "34", "smoothing": 0})
        r.raise_for_status()
        got = {d["ngram"]: sum(d["timeseries"]) / len(d["timeseries"]) for d in r.json()}
        cache.update({w: got.get(w, 0.0) for w in chunk})
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(2)
    return cache


def main():
    pairs = [m.groups() for line in (HERE / "results" / "error_classes.md").read_text(encoding="utf-8").splitlines()
             if (m := ROW.match(line)) and m.group(4) != "刪除"]
    f = ngram([t2s(w) for p in pairs for w in (p[4], p[5])])
    rows, tally = [], defaultdict(Counter)
    for p, g, o, cls, wg, wo, fg, fo, runs in pairs:
        sg, so = t2s(wg), t2s(wo)
        ng, no = f[sg], f[so]
        nd = "無資料" if ng == no == 0 else "往高頻" if no > ng else "往低頻" if no < ng else "同頻"
        rows.append(f"| p{p} | {g}→{o} | {wg}／{wo} | {fg}／{fo} | {cls} | {sg}／{so} | {ng:.1e}／{no:.1e} | {nd} | {runs} |")
        for run in runs.split("、"):
            tally[run][(cls.removesuffix("替換"), nd)] += 1
    out = ["# 同一批詞對：中研院（台灣繁體）vs Google Books Ngram（簡體中文 2019）", "",
           "Ngram 數字是 1980–2019 年的平均相對頻率；0 表示少於 40 次、Google 不收錄。", "",
           "| 頁 | 標準稿→讀成 | 比較的詞 | 中研院詞頻 | 中研院分類 | 簡體 | Ngram 頻率 | Ngram 方向 | 出現在 |",
           "|---|---|---|---|---|---|---|---|---|", *rows, "", "## 各讀法：中研院方向 × Ngram 方向（處數）", ""]
    out += [f"- **{run}**：" + "；".join(f"台灣{a}／大陸{b} {n}" for (a, b), n in sorted(c.items())) for run, c in tally.items()]
    (HERE / "results" / "error_classes_ngram.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

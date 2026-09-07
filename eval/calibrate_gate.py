"""Re-measure RELEVANCE_CEILING for whatever embedder EMBED_MODEL selects.

The ceiling separates "the notes contain an answer" from "they don't", and it is a raw
distance, so it is a property of the EMBEDDER, not of the app — swapping encoders
invalidates it. 0.38 was measured for multilingual-e5-base, whose unrelated Chinese
pairs still score ~0.77 cosine; a model with wider separation needs a different number
and would otherwise refuse every question.

Ingests the cached B1 corpus, then reports the best (smallest) distance for questions
whose answer IS in the notes versus questions about topics that are not, and suggests a
ceiling midway between the two populations.

    EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B python eval/calibrate_gate.py
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_TOKEN"] = ""
os.environ["WHOLE_CORPUS_CHARS"] = "0"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from app import rag_core as rc  # noqa: E402
import gating_experiment as ge  # noqa: E402

# Absent topics: plausible school questions on subjects the notes genuinely do not
# cover, so a low distance here is a false accept, not a lucky hit.
ABSENT = [
    "光合作用的暗反應在哪裡進行？",
    "莎士比亞的四大悲劇有哪些？",
    "法國大革命的三個階段是什麼？",
    "如何計算矩陣的行列式？",
    "凱因斯學派對財政政策的主張是什麼？",
    "台灣的板塊交界屬於哪一種類型？",
    "Python 的 list comprehension 怎麼寫？",
    "細胞膜的流體鑲嵌模型是誰提出的？",
]


def best_distance(col, question, n=5):
    res = col.query(query_embeddings=[rc.embed_query(question)], n_results=n,
                    include=["distances"])
    return res["distances"][0][0]


def main():
    cases = json.loads((ROOT / "eval/chunk_cases.json").read_text(encoding="utf-8"))["cases"]
    user = f"cal_{int(time.time())}"
    print(f"embedder: {rc.EMBED_MODEL} ({rc.EMBED_DIM}d), current ceiling {rc.RELEVANCE_CEILING}")
    ge.ingest(user, False, ge.usable_pages())
    col = rc.get_user_collection(user)

    present = sorted(best_distance(col, c["question"]) for c in cases)
    absent = sorted(best_distance(col, q) for q in ABSENT)

    def summarise(label, xs):
        print(f"  {label:8} n={len(xs):2}  min {xs[0]:.3f}  median {xs[len(xs)//2]:.3f}"
              f"  max {xs[-1]:.3f}")
    print()
    summarise("present", present)
    summarise("absent", absent)

    # The gate should sit above every real answer and below every absent one. When those
    # two populations overlap there is no such value, and the overlap is the finding.
    hi_present, lo_absent = present[-1], absent[0]
    if hi_present < lo_absent:
        print(f"{chr(10)}separated: no overlap. suggested RELEVANCE_CEILING = "
              f"{(hi_present + lo_absent) / 2:.3f}"
              f"  (gap {lo_absent - hi_present:.3f})")
    else:
        bad = sum(1 for x in absent if x <= hi_present)
        print(f"{chr(10)}OVERLAP: worst present {hi_present:.3f} >= best absent "
              f"{lo_absent:.3f} ({bad}/{len(absent)} absent questions inside the present "
              f"range) — any single ceiling trades refusals against false answers.")
        # Best achievable split: the threshold maximising correct accept/reject.
        cand = sorted(set(present + absent))
        best = max(cand, key=lambda t: sum(x <= t for x in present) + sum(x > t for x in absent))
        print(f"  best single split {best:.3f}: accepts {sum(x <= best for x in present)}"
              f"/{len(present)} present, rejects {sum(x > best for x in absent)}/{len(absent)} absent")

    shutil.rmtree(ROOT / "vectorstores" / rc._safe_name(user), ignore_errors=True)


if __name__ == "__main__":
    main()

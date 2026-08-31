"""Gating experiment (plan §03): is structure-preserving ingestion worth building?

Ingests the same source notes under competing configurations into throwaway collections
and compares RETRIEVAL quality on the labelled cases in cases.json:

    B1  current prose extraction  + _chunk_text(450, 50)          — the baseline
    B4  nested-Markdown extraction + chunk_by_structure(+ancestors) — the cheap rival

If B4 captures most of the available gain, the expensive layout-detection branch has no
justification and the project should pivot (see plan §03). This script answers that.

Metrics are Recall@1/@3 and MRR at NOTE level. Recall@10 is deliberately not the headline:
on a small corpus the top-10 contains nearly every note, so it saturates and discriminates
nothing — rank of the first correct chunk is what moves.

    python eval/gating_experiment.py                 # both configs
    python eval/gating_experiment.py --configs B1    # just one
    python eval/gating_experiment.py --keep          # don't delete the collections

Forces WHOLE_CORPUS_CHARS=0 (retrieval path) and HF_TOKEN='' (never touches the private
Dataset). Ingestion calls Gemini once per image/PDF per config, so it costs real quota.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_TOKEN"] = ""          # never sync the private Dataset
os.environ["WHOLE_CORPUS_CHARS"] = "0"  # force retrieval, or every config scores alike
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from app import rag_core as rc  # noqa: E402

CONFIGS = {
    "B1": {"structured": False, "desc": "prose extraction + 450/50 sliding window"},
    "B4": {"structured": True, "desc": "nested Markdown + structure chunking + ancestors"},
}


def sources():
    """The research corpus: pdf/ only. docs/ and images/ were demo material and are
    deliberately excluded — the study is on real multi-page handwritten notes."""
    return [(p.name, p, "pdf") for p in sorted(ROOT.glob("pdf/*.pdf"))]


def ingest(user, structured):
    """Ingest every source note under one configuration. Returns per-note chunk counts."""
    counts = {}
    for name, path, kind in sources():
        if kind == "text":
            text = path.read_text(encoding="utf-8")
        elif kind == "image":
            text = rc.describe_image(str(path), structured=structured)
        else:
            text = rc.describe_pdf(str(path), structured=structured)
        rc._add_note(user, name, text, kind, source=str(path), structured=structured)
        ids = [i for i in rc.get_user_collection(user).get()["ids"] if rc._base_id(i) == name]
        counts[name] = len(ids)
        print(f"    {name:32} {kind:5} -> {len(ids):2} chunk(s), {len(text):5} chars")
    return counts


def score(user, cases, top_k=10):
    """Recall@1/@3 and MRR at note level, plus the rank of the first correct chunk."""
    col = rc.get_user_collection(user)
    n = len(col.get()["ids"])
    hits1 = hits3 = 0
    rr_total = 0.0
    rows = []
    for c in cases:
        want = set(c["notes"])
        res = col.query(query_embeddings=[rc.embed_query(c["question"])],
                        n_results=min(top_k, n), include=["metadatas", "distances"])
        ranked = [rc._base_id(i) for i in res["ids"][0]]
        rank = next((i + 1 for i, nid in enumerate(ranked) if nid in want), None)
        hits1 += int(rank == 1)
        hits3 += int(rank is not None and rank <= 3)
        rr_total += (1 / rank) if rank else 0.0
        rows.append((c["id"], rank, ranked[0] if ranked else None))
    total = len(cases)
    return {"recall@1": hits1 / total, "recall@3": hits3 / total,
            "mrr": rr_total / total, "n": total, "rows": rows}


def score_chunks(user, cases, top_k=10):
    """Rank of the first retrieved chunk that CONTAINS the answer span.

    Config-independent: chunk boundaries differ between B1 and B4, so gold is the answer
    text, not a chunk id. This is the metric that can actually see chunking quality —
    note-level recall saturates once each subject has only one note.
    """
    col = rc.get_user_collection(user)
    n = len(col.get()["ids"])
    rows, rr, hits1, hits3 = [], 0.0, 0, 0
    for c in cases:
        res = col.query(query_embeddings=[rc.embed_query(c["question"])],
                        n_results=min(top_k, n), include=["documents"])
        docs = res["documents"][0]
        rank = next((i + 1 for i, d in enumerate(docs) if c["span"] in (d or "")), None)
        hits1 += int(rank == 1)
        hits3 += int(rank is not None and rank <= 3)
        rr += (1 / rank) if rank else 0.0
        rows.append((c["id"], c["type"], rank))
    t = len(cases)
    return {"recall@1": hits1 / t, "recall@3": hits3 / t, "mrr": rr / t,
            "n": t, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).parent
    cases = [c for c in json.loads((here / "cases.json")
                                   .read_text(encoding="utf-8"))["cases"] if c.get("notes")]
    chunk_cases = json.loads((here / "chunk_cases.json").read_text(encoding="utf-8"))["cases"]
    print(f"gating experiment — {len(cases)} note-level, {len(chunk_cases)} chunk-level "
          f"cases | WHOLE_CORPUS_CHARS={rc.WHOLE_CORPUS_CHARS}\n")

    stamp = int(time.time())
    results = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        user = f"exp_{name.lower()}_{stamp}"
        print(f"[{name}] {cfg['desc']}")
        counts = ingest(user, cfg["structured"])
        r = score(user, cases)
        rc_ = score_chunks(user, chunk_cases)
        r["chunks"] = sum(counts.values())
        results[name] = (user, r, rc_)
        print(f"    -> {r['chunks']} chunks | note-level MRR {r['mrr']:.3f}"
              f" | chunk-level MRR {rc_['mrr']:.3f}\n")

    print(f"{'config':6} {'chunks':>7} | {'note R@1':>9} {'note MRR':>9}"
          f" | {'chunk R@1':>10} {'chunk R@3':>10} {'chunk MRR':>10}")
    print("-" * 74)
    for name, (_, r, k) in results.items():
        print(f"{name:6} {r['chunks']:7} | {r['recall@1']:9.2f} {r['mrr']:9.3f}"
              f" | {k['recall@1']:10.2f} {k['recall@3']:10.2f} {k['mrr']:10.3f}")

    if len(results) == 2 and "B1" in results and "B4" in results:
        b1, b4 = results["B1"][2], results["B4"][2]
        print(f"\nchunk-level MRR delta (B4 - B1): {b4['mrr'] - b1['mrr']:+.3f}")
        print("rank of first chunk containing the answer (None = not in top-10):")
        for (cid, kind, r1), (_, _, r4) in zip(b1["rows"], b4["rows"]):
            mark = "  " if r1 == r4 else ("+ " if (r4 or 99) < (r1 or 99) else "- ")
            print(f"  {mark}{cid:18} {kind:10} B1={str(r1):>4}  B4={str(r4):>4}")

    if not args.keep:
        for user, _, _ in results.values():
            shutil.rmtree(ROOT / "vectorstores" / rc._safe_name(user), ignore_errors=True)
        print("\n(throwaway collections removed; pass --keep to inspect them)")


if __name__ == "__main__":
    main()

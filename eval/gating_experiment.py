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
import random
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

CORPUS = Path('eval/corpus')

CONFIGS = {
    "B1": {"structured": False, "desc": "prose extraction + 450/50 sliding window"},
    "B4": {"structured": True, "desc": "nested Markdown + structure chunking + ancestors"},
    "B0": {"structured": True, "corpus": "corpus_b0.json",
           "desc": "oracle: human-corrected text + human structure tree"},
}


def load_corpus(structured, corpus_file=None):
    """Page texts produced by build_corpus.py — cached, so a run costs no API calls and
    B1/B4 are compared on exactly the extractions that were measured and inspected."""
    f = CORPUS / (corpus_file or ("corpus_b4.json" if structured else "corpus_b1.json"))
    if not f.exists():
        sys.exit(f"missing {f} — run: python eval/build_corpus.py"
                 f"{' --structured' if structured else ''}")
    return json.loads(f.read_text(encoding="utf-8"))


def usable_pages():
    """Page ids that are non-blank in BOTH corpora.

    The configurations disagree about blank pages — the structured prompt describes an
    empty page with headings, so it escapes the blank filter that catches the prose
    version. Comparing different page sets would confound the result, so both configs
    index exactly the same pages.
    """
    keep = None
    for cfg in CONFIGS.values():
        f = CORPUS / (cfg.get("corpus") or
                      ("corpus_b4.json" if cfg["structured"] else "corpus_b1.json"))
        if not f.exists():
            continue
        ids = {f"{n['file']}#p{pg['page']:02d}"
               for n in json.loads(f.read_text(encoding="utf-8"))
               for pg in n["page_texts"] if not pg["blank"]}
        keep = ids if keep is None else (keep & ids)
    return keep


def ingest(user, structured, pages_allowed, corpus_file=None):
    """One retrievable unit per PAGE, id '<file>#pNN'.

    Pages, not whole documents: with six PDFs the correct document is trivially rank 1
    for every question (measured — both configurations tied at MRR 1.000), so document
    level cannot see chunking at all. Sixty-odd pages give retrieval something to get
    wrong. _base_id keeps '<file>#p07' intact because the suffix is not purely numeric,
    so chunk ids '<file>#p07#2' still resolve back to the page.
    """
    counts = {}
    for note in load_corpus(structured, corpus_file):
        for pg in note["page_texts"]:
            note_id = f"{note['file']}#p{pg['page']:02d}"
            if note_id not in pages_allowed:
                continue
            rc._add_note(user, note_id, pg["text"], "pdf",
                         source=f"pdf/{note['file']}#p{pg['page']}",
                         structured=structured)
            counts[note_id] = len([i for i in rc.get_user_collection(user).get()["ids"]
                                   if rc._base_id(i) == note_id])
    print(f"    {len(counts)} pages -> {sum(counts.values())} chunks")
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


def paired_bootstrap(ranks_a, ranks_b, iters=10000, seed=0):
    """95% CI for the paired MRR difference (B4 - B1).

    Paired, because both configurations answer the same probes: the pairing removes
    per-question difficulty, which is the dominant source of variance with only ~30
    probes. A CI spanning zero means the corpus cannot distinguish the configurations —
    which is a result, not a failure, and must be reported as such.
    """
    rr = lambda rank: (1.0 / rank) if rank else 0.0
    diffs = [rr(b) - rr(a) for a, b in zip(ranks_a, ranks_b)]
    n = len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    point = sum(diffs) / n
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    return point, lo, hi, wins, losses, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).parent
    chunk_cases = json.loads((here / "chunk_cases.json").read_text(encoding="utf-8"))["cases"]
    # Same probes scored two ways: did the right PAGE surface, and did a chunk that
    # actually contains the answer surface. The second is what chunking can move.
    cases = [{"id": c["id"], "question": c["question"], "notes": [c["page"]]}
             for c in chunk_cases]
    print(f"gating experiment: {len(chunk_cases)} probes, scored at page and chunk level"
          f" | WHOLE_CORPUS_CHARS={rc.WHOLE_CORPUS_CHARS}" + chr(10))

    allowed = usable_pages()
    print(f"    {len(allowed)} pages usable in both configurations" + chr(10))
    stamp = int(time.time())
    results = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        user = f"exp_{name.lower()}_{stamp}"
        print(f"[{name}] {cfg['desc']}")
        counts = ingest(user, cfg["structured"], allowed, cfg.get("corpus"))
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

    def ranks(rows, idx):
        return [r[idx] for r in rows]

    names = list(results)
    for level, idx, key in (("page ", 1, 1), ("chunk", 2, 2)):
        print(f"{chr(10)}--- {level}-level pairwise (paired bootstrap, 10k resamples) ---")
        for a in names:
            for b in names:
                if names.index(b) <= names.index(a):
                    continue
                ra = ranks(results[a][key]["rows"], idx)
                rb = ranks(results[b][key]["rows"], idx)
                point, lo, hi, wins, losses, n = paired_bootstrap(ra, rb)
                verdict = ("no detectable difference" if lo <= 0 <= hi else
                           (f"{b} better" if lo > 0 else f"{a} better"))
                print(f"  {b} - {a}: {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                      f"({wins}W/{losses}L/{n - wins - losses}T)  -> {verdict}")

    print(f"{chr(10)}--- per-probe chunk rank (None = not in top-10) ---")
    hdr = "  " + "probe".ljust(26) + "type".ljust(11) + "".join(x.rjust(6) for x in names)
    print(hdr)
    for k in range(len(results[names[0]][2]["rows"])):
        row = results[names[0]][2]["rows"][k]
        cells = "".join(str(results[nm][2]["rows"][k][2]).rjust(6) for nm in names)
        print(f"  {row[0][:25]:26}{row[1]:11}{cells}")

    if not args.keep:
        for user, _, _ in results.values():
            shutil.rmtree(ROOT / "vectorstores" / rc._safe_name(user), ignore_errors=True)
        print("\n(throwaway collections removed; pass --keep to inspect them)")


if __name__ == "__main__":
    main()

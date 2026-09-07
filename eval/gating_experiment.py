"""Gating experiment (plan §03): is structure-preserving ingestion worth building?

SECONDARY LINE, 2026-09-07. Read this before quoting any number below.

This script measures a code path that PRODUCTION DISABLES BY DEFAULT, in a mode
production never uses:

  * rag_query() calls _whole_corpus() first, and with WHOLE_CORPUS_CHARS=60_000 a
    corpus under that size skips _retrieve() entirely — the whole note set goes to
    Gemini. The measured corpus is 52,875 chars, so a real user with these notes never
    retrieves at all. That is why this file forces WHOLE_CORPUS_CHARS=0 below; without
    it every configuration scores identically, because chunking cannot change an answer
    that was built from the entire corpus.
  * The structured configurations pass structured=True, which enables STRUCTURED_RULE
    and chunk_by_structure. NOTHING IN THE APP EVER SETS IT — add_image_note and
    add_pdf_note both call the describers without it.

So the results here describe what would happen once a student's notes outgrow the
whole-corpus cap, not what any user experiences today. The project's primary axis is
whether the output helps a user understand their own notes; this line is kept for the
write-up and is no longer being invested in. The code stays (merge_target defaults to 0,
so production behaviour is unchanged), but do not use these numbers to justify product
work.

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
           "desc": "human-corrected text + human structure tree"},
    # Length-matched arms. Structure chunking splits on every heading and lands a median
    # chunk of ~140 chars against B1's 421, so B4/B0 vs B1 measures chunk LENGTH at least
    # as much as it measures structure and neither can be attributed. These merge
    # consecutive sections into subtree blocks until the length profile matches B1's,
    # which is the only way to read the structure effect on its own.
    "B4L": {"structured": True, "merge": 450,
            "desc": "B4, merged to B1's chunk-length profile"},
    "B0L": {"structured": True, "corpus": "corpus_b0.json", "merge": 450,
            "desc": "B0, merged to B1's chunk-length profile"},
    # Positive control: identical chunks to B0L with the ancestor path removed, so it
    # isolates ancestor injection with length, text and chunk count all held constant.
    # Each section's own heading has already moved into the body by then, so this strips
    # the ANCESTOR chain, not headings in general. If deleting a deliberately large
    # signal is not detectable here, the probe set cannot detect the small effects the
    # study is actually looking for, and no B0/B1/B4 comparison is interpretable.
    "B0L-strip": {"structured": True, "corpus": "corpus_b0.json", "merge": 450,
                  "strip": True, "desc": "B0L minus the ancestor path (positive control)"},
}

_PREFIX_END = " ›\n"          # tail of rag_core._ancestor_prefix


def strip_ancestors(chunks):
    """Drop the ancestor prefix rag_core writes at the head of each structured chunk."""
    out = []
    for c in chunks:
        i = c.find(_PREFIX_END)
        out.append(c[i + len(_PREFIX_END):] if 0 <= i <= 200 else c)
    return out


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


def ingest_cfg(user, cfg, pages_allowed):
    """Ingest one configuration, returning per-page chunk counts and chunk lengths."""
    lengths = []
    counts = {}
    for note in load_corpus(cfg["structured"], cfg.get("corpus")):
        for pg in note["page_texts"]:
            note_id = f"{note['file']}#p{pg['page']:02d}"
            if note_id not in pages_allowed:
                continue
            chunks = None
            if cfg.get("strip"):
                chunks = strip_ancestors(
                    rc.chunk_by_structure(pg["text"], merge_target=cfg.get("merge", 0)))
            rc._add_note(user, note_id, pg["text"], "pdf",
                         source=f"pdf/{note['file']}#p{pg['page']}",
                         structured=cfg["structured"], merge_target=cfg.get("merge", 0),
                         chunks=chunks)
            ids = [i for i in rc.get_user_collection(user).get()["ids"]
                   if rc._base_id(i) == note_id]
            counts[note_id] = len(ids)
    docs = rc.get_user_collection(user).get(include=["documents"])["documents"]
    lengths = sorted(len(d or "") for d in docs)
    med = lengths[len(lengths) // 2]
    q1, q3 = lengths[len(lengths) // 4], lengths[3 * len(lengths) // 4]
    print(f"    {len(counts)} pages -> {sum(counts.values())} chunks"
          f" | median {med} chars, IQR [{q1}, {q3}], {sum(1 for x in lengths if x < 100)} under 100")
    return counts, {"median": med, "q1": q1, "q3": q3,
                    "under100": sum(1 for x in lengths if x < 100)}


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
    # Two-sided bootstrap p: twice the resample mass on the far side of zero. Needed
    # because a CI alone cannot be Holm-corrected — Holm ranks p values.
    p = min(1.0, 2 * min(sum(1 for x in means if x <= 0),
                         sum(1 for x in means if x >= 0)) / iters)
    return point, lo, hi, wins, losses, n, p


def holm(pvals):
    """Holm-Bonferroni thresholds for a family of p values, returned in input order.

    The project's own statistics section requires this and the previous runs did not
    apply it: B0-B1 under Qwen3 (raw p = 0.041) reads as significant alone and does not
    survive a family of three. Step-down: sort ascending, compare the i-th against
    alpha/(m-i), and stop at the first failure — everything after it also fails.
    """
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    out = [None] * m
    still = True
    for rank, i in enumerate(order):
        thr = 0.05 / (m - rank)
        still = still and pvals[i] < thr
        out[i] = (thr, still)
    return out


def mde(ranks_a, ranks_b, power=0.80, alpha=0.05):
    """Smallest paired MRR difference this probe set could detect, and the n that the
    observed difference would need. Reported so a null reads as a bounded effect rather
    than as a failed experiment."""
    import math
    rr = lambda r: (1.0 / r) if r else 0.0
    d = [rr(b) - rr(a) for a, b in zip(ranks_a, ranks_b)]
    n = len(d)
    mean = sum(d) / n
    sd = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
    z = 1.959964 + 0.8416212          # z_{alpha/2} + z_{beta} for 95% / 80%
    detectable = z * sd / math.sqrt(n) if n else float("inf")
    needed = (z * sd / mean) ** 2 if mean else float("inf")
    return sd, detectable, needed


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
    # Probes clustered on a page share text and structure, so the resampling unit must
    # be the page once density rises. Today every probe sits on its own page (design
    # effect 1) and page-level resampling is identical to probe-level; the counts are
    # printed so the day that stops being true is visible rather than silent.
    n_pages = len({c["page"] for c in chunk_cases})
    print(f"gating experiment: {len(chunk_cases)} probes over {n_pages} distinct pages"
          f" ({len(chunk_cases) / n_pages:.2f} probes/page)"
          f" | WHOLE_CORPUS_CHARS={rc.WHOLE_CORPUS_CHARS}"
          f" | embedder {rc.EMBED_MODEL}" + chr(10))

    allowed = usable_pages()
    print(f"    {len(allowed)} pages usable in both configurations" + chr(10))
    stamp = int(time.time())
    results = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        user = f"exp_{name.lower()}_{stamp}"
        print(f"[{name}] {cfg['desc']}")
        counts, lens = ingest_cfg(user, cfg, allowed)
        r = score(user, cases)
        rc_ = score_chunks(user, chunk_cases)
        r["chunks"] = sum(counts.values())
        r["lens"] = lens
        results[name] = (user, r, rc_)
        print(f"    -> {r['chunks']} chunks | note-level MRR {r['mrr']:.3f}"
              f" | chunk-level MRR {rc_['mrr']:.3f}\n")

    # The chunk-length columns are not decoration: without them a difference between
    # configurations cannot be attributed to chunking strategy rather than chunk size.
    print(f"{'config':10} {'chunks':>7} {'median':>7} {'IQR':>13} {'<100c':>6}"
          f" | {'note MRR':>9} | {'chunk R@1':>10} {'chunk R@3':>10} {'chunk MRR':>10}")
    print("-" * 100)
    for name, (_, r, k) in results.items():
        L = r["lens"]
        print(f"{name:10} {r['chunks']:7} {L['median']:7} [{L['q1']:4},{L['q3']:4}]"
              f" {L['under100']:9} | {r['mrr']:9.3f}"
              f" | {k['recall@1']:10.2f} {k['recall@3']:10.2f} {k['mrr']:10.3f}")

    def ranks(rows, idx):
        return [r[idx] for r in rows]

    names = list(results)
    # Comparison family, declared here rather than chosen after seeing the numbers: all
    # pairs at one scoring level within one run are corrected together.
    for level, idx, key in (("page ", 1, 1), ("chunk", 2, 2)):
        print(f"{chr(10)}--- {level}-level pairwise (paired bootstrap 10k, "
              f"Holm-Bonferroni over {len(names) * (len(names) - 1) // 2} pairs) ---")
        rows = []
        for a in names:
            for b in names:
                if names.index(b) <= names.index(a):
                    continue
                ra = ranks(results[a][key]["rows"], idx)
                rb = ranks(results[b][key]["rows"], idx)
                rows.append((a, b, ra, rb, paired_bootstrap(ra, rb)))
        for (a, b, ra, rb, st), (thr, ok) in zip(rows, holm([r[4][6] for r in rows])):
            point, lo, hi, wins, losses, n, p = st
            sd, det, need = mde(ra, rb)
            verdict = ("SIGNIFICANT" if ok else
                       ("not significant (raw p < .05, lost to Holm)" if p < 0.05
                        else "not significant"))
            print(f"  {b} - {a}: {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
                  f"  ({wins}W/{losses}L/{n - wins - losses}T)"
                  f"  raw p={p:.4f} Holm thr={thr:.4f} -> {verdict}")
            print(f"      MDE at n={n}: {det:+.3f} (80% power); this effect would need "
                  f"n={need:.0f}" if need == need and need != float('inf')
                  else f"      MDE at n={n}: {det:+.3f} (80% power)")

    print(f"{chr(10)}--- per-probe chunk rank (None = not in top-10) ---")
    hdr = "  " + "probe".ljust(26) + "type".ljust(11) + "".join(x.rjust(6) for x in names)
    print(hdr)
    for k in range(len(results[names[0]][2]["rows"])):
        row = results[names[0]][2]["rows"][k]
        cells = "".join(str(results[nm][2]["rows"][k][2]).rjust(6) for nm in names)
        print(f"  {row[0][:25]:26}{row[1]:11}{cells}")

    # Per-probe ranks, keyed by embedder, so runs from DIFFERENT embedding models can be
    # compared later without holding two models in memory at once (compare_runs.py).
    runs = ROOT / "eval" / "runs"
    runs.mkdir(exist_ok=True)
    slug = rc.EMBED_MODEL.replace("/", "_")
    out = runs / f"{slug}.json"
    out.write_text(json.dumps({
        "embed_model": rc.EMBED_MODEL, "dim": rc.EMBED_DIM,
        "configs": {name: {"chunks": r["chunks"], "lens": r["lens"],
                           "page_ranks": {row[0]: row[1] for row in r["rows"]},
                           "chunk_ranks": {row[0]: row[2] for row in k["rows"]},
                           "types": {row[0]: row[1] for row in k["rows"]}}
                    for name, (_, r, k) in results.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{chr(10)}per-probe ranks -> {out}")

    if not args.keep:
        for user, _, _ in results.values():
            shutil.rmtree(ROOT / "vectorstores" / rc._safe_name(user), ignore_errors=True)
        print("\n(throwaway collections removed; pass --keep to inspect them)")


if __name__ == "__main__":
    main()

"""Compare two gating_experiment runs that used DIFFERENT embedding models.

gating_experiment.py can only compare configurations inside one run, because the
embedder is loaded once at import. This script reads the per-probe rank dumps it
writes to eval/runs/ and does the same paired bootstrap ACROSS runs — the probes are
identical, so the pairing still holds and per-question difficulty still cancels.

    python eval/compare_runs.py intfloat_multilingual-e5-base Qwen_Qwen3-Embedding-0.6B

Loads no model and makes no API calls.
"""
import json
import random
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def load(slug):
    f = RUNS / f"{slug}.json"
    if not f.exists():
        sys.exit(f"missing {f} — run: EMBED_MODEL=... python eval/gating_experiment.py"
                 f"{chr(10)}available: " + ", ".join(p.stem for p in RUNS.glob('*.json')))
    return json.loads(f.read_text(encoding="utf-8"))


def paired_bootstrap(ranks_a, ranks_b, iters=10000, seed=0):
    """95% CI for the paired MRR difference (b - a). Same procedure as the in-run
    comparison, kept separate so this script needs no torch import."""
    rr = lambda r: (1.0 / r) if r else 0.0
    diffs = [rr(b) - rr(a) for a, b in zip(ranks_a, ranks_b)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    point = sum(diffs) / n
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    p = min(1.0, 2 * min(sum(1 for x in means if x <= 0),
                         sum(1 for x in means if x >= 0)) / iters)
    return point, means[int(0.025 * iters)], means[int(0.975 * iters)], wins, losses, n, p


def holm(pvals):
    """Holm-Bonferroni thresholds, returned in input order. Duplicated from
    gating_experiment rather than imported, because importing it would load torch and
    this script deliberately runs without a model."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    out = [None] * m
    still = True
    for rank, i in enumerate(order):
        thr = 0.05 / (m - rank)
        still = still and pvals[i] < thr
        out[i] = (thr, still)
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a_slug, b_slug = sys.argv[1], sys.argv[2]
    A, B = load(a_slug), load(b_slug)
    print(f"A = {A['embed_model']} ({A['dim']}d){chr(10)}B = {B['embed_model']} ({B['dim']}d)")

    shared = [c for c in A["configs"] if c in B["configs"]]
    mrr = lambda rs: sum((1 / r) if r else 0.0 for r in rs) / len(rs)
    for level in ("chunk_ranks", "page_ranks"):
        # Family = every configuration compared across the two encoders at one level.
        print(f"{chr(10)}--- {level.split('_')[0]}-level, B - A (paired bootstrap 10k, "
              f"Holm-Bonferroni over {len(shared)} comparisons) ---")
        stats = []
        for cfg in shared:
            ra_map, rb_map = A["configs"][cfg][level], B["configs"][cfg][level]
            probes = [p for p in ra_map if p in rb_map]
            ra = [ra_map[p] for p in probes]
            rb = [rb_map[p] for p in probes]
            stats.append((cfg, ra, rb, paired_bootstrap(ra, rb)))
        for (cfg, ra, rb, st), (thr, ok) in zip(stats, holm([s[3][6] for s in stats])):
            point, lo, hi, w, l, n, p = st
            verdict = ("SIGNIFICANT" if ok else
                       ("not significant (raw p < .05, lost to Holm)" if p < 0.05
                        else "not significant"))
            print(f"  {cfg:10}: {mrr(ra):.3f} -> {mrr(rb):.3f}  diff {point:+.3f}"
                  f"  95% CI [{lo:+.3f}, {hi:+.3f}]  ({w}W/{l}L/{n - w - l}T)"
                  f"  raw p={p:.4f} Holm thr={thr:.4f} -> {verdict}")

    # Which probes moved, and in which direction — the per-probe detail that says whether
    # a headline gain came from a broad shift or from one or two probes.
    print(f"{chr(10)}--- per-probe chunk rank (probes that moved in any config) ---")
    hdr = "  " + "probe".ljust(26) + "type".ljust(11) + "".join(
        f"{c}:A".rjust(7) + f"{c}:B".rjust(7) for c in shared)
    print(hdr)
    types = A["configs"][shared[0]]["types"]
    for probe in A["configs"][shared[0]]["chunk_ranks"]:
        cells, moved = "", False
        for c in shared:
            ra = A["configs"][c]["chunk_ranks"].get(probe)
            rb = B["configs"][c]["chunk_ranks"].get(probe)
            moved |= (ra != rb)
            cells += str(ra).rjust(7) + str(rb).rjust(7)
        if moved:
            print(f"  {probe[:25]:26}{types.get(probe, ''):11}{cells}")


if __name__ == "__main__":
    main()

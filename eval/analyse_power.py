"""How much of 'no significant difference' is 'no difference' and how much is 'not enough data'?

Reads the per-probe ranks from RESULTS_gating.txt and reports, for each pair of configs:
  * the paired effect size and its spread
  * the number of probes that would be needed to detect that effect at 80% power
  * the same restricted to the subgroups the hypothesis actually predicts

A wide CI has two very different causes — a real effect drowned in noise, or an effect
that is genuinely near zero. The required-N figure separates them: if detecting the
observed effect would take thousands of probes, the effect is negligible regardless of
how much data is collected, and the probe design is what needs to change.
"""
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
txt = (ROOT / "eval/RESULTS_gating.txt").read_text(encoding="utf-8")
block = txt.split("--- per-probe chunk rank")[1].splitlines()[2:]

rows = []
for line in block:
    m = re.match(r"\s+(\S+)\s+(fact|hierarchy)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
    if m:
        pid, kind, *ranks = m.groups()
        rows.append((pid, kind, [None if r == "None" else int(r) for r in ranks]))
print(f"parsed {len(rows)} probes\n")

NAMES = ["B1", "B4", "B0"]
rr = lambda r: (1.0 / r) if r else 0.0


def report(label, subset, a, b):
    ia, ib = NAMES.index(a), NAMES.index(b)
    d = [rr(r[2][ib]) - rr(r[2][ia]) for r in subset]
    n = len(d)
    if n < 2:
        return
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    half = 1.96 * sd / math.sqrt(n) if sd else 0.0
    wins = sum(1 for x in d if x > 0)
    losses = sum(1 for x in d if x < 0)
    # paired-design N for 80% power at the observed effect
    need = math.ceil(7.85 * (sd / abs(mean)) ** 2) if mean else float("inf")
    need_s = f"{need:,}" if need != float("inf") else "n/a"
    print(f"  {label:34} n={n:3}  {b}-{a} = {mean:+.3f} +/- {half:.3f}  "
          f"sd={sd:.3f}  ({wins}W/{losses}L/{n-wins-losses}T)  N@80%power={need_s}")


for a, b in (("B1", "B4"), ("B1", "B0"), ("B4", "B0")):
    print(f"=== {b} vs {a} (chunk level) ===")
    report("all probes", rows, a, b)
    report("hierarchy only (H2 prediction)", [r for r in rows if r[1] == "hierarchy"], a, b)
    report("single-fact only", [r for r in rows if r[1] == "fact"], a, b)
    headroom = [r for r in rows if r[2][0] != 1]
    report("only where B1 is not already #1", headroom, a, b)
    print()

ties = sum(1 for r in rows if r[2][0] == r[2][1] == r[2][2])
b1_first = sum(1 for r in rows if r[2][0] == 1)
print(f"probes identical across all three configs : {ties}/{len(rows)}")
print(f"probes where B1 already ranks the answer #1: {b1_first}/{len(rows)}"
      f"  -> only {len(rows)-b1_first} probes can move at all")

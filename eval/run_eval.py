"""Regression harness for NoteHandlerRAG.

Answers a fixed set of questions against a running app and scores three things:

  answer accuracy  — every expected string appears in the answer (specific + global cases)
  refusal accuracy — absent topics are declined instead of hallucinated
  retrieval recall — the expected note was actually retrieved  (--local only)

Usage
    python eval/run_eval.py                       # against the deployed Space
    python eval/run_eval.py --base http://localhost:8000
    python eval/run_eval.py --local               # import rag_core directly, adds recall
    python eval/run_eval.py --kind global         # run one category
    python eval/run_eval.py --json out.json       # machine-readable, for before/after diffs

--local imports the app in-process and forces HF_TOKEN="", so the private Dataset is
never touched and a failed run cannot corrupt production data.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "https://timhelloyee-notehandler.hf.space"
DEFAULT_USER = "timhelloyee@gmail.com"
REFUSAL_MARKERS = ("沒有記載", "沒有相關", "查無")


def load_cases(kind=None):
    data = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    return [c for c in cases if not kind or c["kind"] == kind]


class HttpRunner:
    """Exercises the real HTTP API — what the user actually experiences."""

    def __init__(self, base, user, model=None):
        import requests
        self.requests = requests
        self.base, self.user, self.model = base.rstrip("/"), user, model

    def ask(self, question):
        payload = {"user_id": self.user, "question": question, "top_k": 10}
        if self.model:
            payload["model"] = self.model
        r = self.requests.post(f"{self.base}/api/query", json=payload, timeout=300)
        if r.status_code != 200:
            return f"<HTTP {r.status_code}: {r.text[:120]}>", None
        return r.json().get("answer", ""), None


class LocalRunner:
    """Imports rag_core in-process so retrieval can be inspected, not just the answer."""

    def __init__(self, user, model=None):
        os.environ["HF_TOKEN"] = ""  # never sync the private Dataset from an eval run
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        from app import rag_core
        self.rc = rag_core
        self.user, self.model = user, model

    def ask(self, question):
        chunks = self.rc._retrieve(self.user, question, top_k=10)
        answer = self.rc.rag_query(self.user, question, top_k=10, model=self.model)
        return answer, chunks


def grade(case, answer, chunks):
    """Returns (passed, list of failure reasons).

    A refusal is judged by the *shape* of the answer, not by the phrase appearing
    anywhere: a real answer may legitimately close with 「參考資料中沒有記載其他…」
    while still answering the question fully. Real refusals are a single short line.
    """
    reasons = []
    body = answer.strip()
    refused = any(m in body for m in REFUSAL_MARKERS) and len(body) < 120

    if case.get("refusal"):
        if not refused:
            reasons.append("did not refuse an absent topic")
    else:
        # No separate refusal penalty here — a refusal fails `contains` on its own.
        missing = [s for s in case.get("contains", []) if s not in answer]
        if missing:
            reasons.append(f"missing {missing}")

    # Retrieval recall is only checkable when we can see the chunks (--local).
    if chunks is not None and case.get("notes"):
        joined = "\n".join(chunks)
        # A chunk belongs to a note if the note's distinctive text is present. Use the
        # subject tag / filename stem, whichever the stored description carries.
        got = any(n.split(".")[0] in joined or n in joined for n in case["notes"])
        if not got:
            reasons.append("expected note not retrieved")
    return not reasons, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--user", default=DEFAULT_USER)
    ap.add_argument("--model", default=None, help="pin a Gemini model id")
    ap.add_argument("--local", action="store_true", help="run in-process, adds recall")
    ap.add_argument("--kind", choices=["specific", "global", "absent"])
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    cases = load_cases(args.kind)
    runner = LocalRunner(args.user, args.model) if args.local \
        else HttpRunner(args.base, args.user, args.model)
    where = "local rag_core" if args.local else args.base
    print(f"NoteHandlerRAG eval — {len(cases)} cases against {where}")
    print(f"user={args.user} model={args.model or '(server default)'}\n")

    results, by_kind = [], {}
    t_start = time.time()
    for c in cases:
        t0 = time.time()
        try:
            answer, chunks = runner.ask(c["question"])
        except Exception as e:
            answer, chunks = f"<error: {e!r}>", None
        passed, reasons = grade(c, answer, chunks)
        took = time.time() - t0
        results.append({"id": c["id"], "kind": c["kind"], "passed": passed,
                        "reasons": reasons, "seconds": round(took, 1),
                        "answer": answer[:400]})
        agg = by_kind.setdefault(c["kind"], [0, 0])
        agg[1] += 1
        agg[0] += int(passed)
        flag = "PASS" if passed else "FAIL"
        print(f"{flag} [{c['kind']:8}] {c['id']:26} {took:5.1f}s  {c['question'][:38]}")
        if not passed:
            for r in reasons:
                print(f"        └─ {r}")
            print(f"        └─ answer: {answer[:150].replace(chr(10), ' ')}")

    total_pass = sum(r["passed"] for r in results)
    print(f"\n{'':-<70}")
    for kind, (p, n) in sorted(by_kind.items()):
        print(f"  {kind:9} {p}/{n}  ({100*p//max(n,1)}%)")
    print(f"  {'TOTAL':9} {total_pass}/{len(results)}  "
          f"({100*total_pass//max(len(results),1)}%)   in {time.time()-t_start:.0f}s")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"where": where, "passed": total_pass, "total": len(results),
             "by_kind": by_kind, "results": results},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {args.json_out}")

    return 0 if total_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

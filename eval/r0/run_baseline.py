"""R0 baseline: upload the 8 handwritten pages and ask the 23 questions, as the app does.

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/run_baseline.py --dry-run
    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/run_baseline.py

It calls the app's own functions (add_image_note, rag_answer) with the app's own prompts,
so what is measured is the product, not a copy of it. Like scripts/dev_server.py it clears
HF_TOKEN and uses a fresh scratch store before importing the app: nothing here can touch
./vectorstores or the private HF Dataset.

Questions are read from NotebookLM_提問單.md, the sheet NotebookLM was asked from, so both
systems get the same wording. Each question is asked inside its own subject's category,
matching the three notebooks on the NotebookLM side.

Writes to --out (default eval/r0/results/baseline, git-ignored):
    transcripts/R0_pN.md   what the model read from each page (the stored note text)
    answers.json           one record per question, written after every question
    answers.md             the same, for reading
    meta.json              model, date, commit, prompts: what has to stay fixed to compare runs
"""
import argparse
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SHEET = HERE / "NotebookLM_提問單.md"
SUBJECT_PAGES = {"地科": [1, 2, 3], "生物": [4, 5, 6], "歷史": [7, 8]}


def parse_questions(sheet: Path) -> list[dict]:
    """[{id, subject, question}] in sheet order. Subjects come from the '## R0 <科>' headings."""
    out, subject = [], None
    lines = sheet.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^## R0 (\S+?)（", line)
        if m:
            subject = m.group(1)
            continue
        m = re.match(r"^### ([A-D]\d+)\s*$", line)
        if m and subject:
            out.append({"id": m.group(1), "subject": subject, "question": lines[i + 1].strip()})
    return out


def retry(fn, what, tries=6, wait=30):
    """Gemini answers 503 under load, sometimes for minutes; back off (30, 60, 120 s …)
    rather than let a transient failure end a 31-call run."""
    for k in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:                      # noqa: BLE001 - logged and re-raised
            if k == tries:
                raise
            delay = wait * 2 ** (k - 1)
            print(f"  {what}: {type(e).__name__}: {e} — retry {k}/{tries - 1} in {delay}s")
            time.sleep(delay)


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--short", "app").splitlines()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--out", default=str(HERE / "results" / "baseline"))
    ap.add_argument("--dry-run", action="store_true", help="check inputs only; no API calls")
    args = ap.parse_args()

    questions = parse_questions(SHEET)
    pages = {n: HERE / f"R0_p{n}.jpg" for ps in SUBJECT_PAGES.values() for n in ps}
    missing = [str(p) for p in pages.values() if not p.is_file()]
    counts = {s: sum(q["subject"] == s for q in questions) for s in SUBJECT_PAGES}
    assert not missing, f"missing pages: {missing}"
    assert counts == {"地科": 10, "生物": 8, "歷史": 5}, f"unexpected question counts: {counts}"
    assert all(q["question"] for q in questions), "a question line is empty"
    if args.dry_run:
        for q in questions:
            print(q["id"], q["subject"], q["question"])
        print(f"OK: {len(questions)} questions, {len(pages)} pages, model {args.model}")
        return

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty — use a new --out so runs never mix")
    (out / "transcripts").mkdir(parents=True)

    # Same isolation as scripts/dev_server.py, and it must happen before the import.
    os.environ["HF_TOKEN"] = ""
    os.environ["VECTORSTORE_ROOT"] = str(out / "store")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from app import rag_core                        # noqa: E402

    model = rag_core.resolve_model(args.model)
    assert model == args.model, f"{args.model} is not in AVAILABLE_MODELS"
    user = "r0-baseline"
    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "whole_corpus_chars": rag_core.WHOLE_CORPUS_CHARS,
        "git": git_state(),
        "image_prompt_source": inspect.getsource(rag_core.describe_image),
        "answer_prompt": rag_core.PROMPT_TEMPLATE,
    }
    meta["prompt_sha256"] = hashlib.sha256(
        (meta["image_prompt_source"] + meta["answer_prompt"]).encode("utf-8")).hexdigest()[:16]

    # 1. Ingest: one category per subject, each page into its subject's category.
    cats = {}
    for subject, nums in SUBJECT_PAGES.items():
        cats[subject] = rag_core.create_category(user, subject)["id"]
        for n in nums:
            file_id = pages[n].name
            print(f"describe {file_id} -> {subject}")
            retry(lambda: rag_core.add_image_note(user, file_id, str(pages[n]), source=f"r0/{file_id}",
                                                  model=model, category=cats[subject]), file_id)
            text = rag_core.get_note_text(user, file_id) or ""
            (out / "transcripts" / f"R0_p{n}.md").write_text(text, encoding="utf-8")

    # 2. Ask: each question in its own subject's category, saved as it goes.
    records = []
    for q in questions:
        print(f"ask {q['id']}")
        rec = {**q, "category": q["subject"]}
        try:
            r = retry(lambda: rag_core.rag_answer(user, q["question"], model=model,
                                                  category=cats[q["subject"]]), q["id"])
            rec.update(answer=r["answer"], sources=r["sources"], dropped=r["dropped"])
        except Exception as e:                      # noqa: BLE001 - recorded, run continues
            rec.update(answer=None, error=f"{type(e).__name__}: {e}")
        records.append(rec)
        (out / "answers.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# R0 baseline — {model}（{meta['started']}）\n"]
    for r in records:
        cited = "、".join(s["id"] + (f" 第 {s['page']} 頁" if s.get("page") else "") for s in r.get("sources") or [])
        md += [f"## {r['id']}（{r['subject']}）", r["question"], "",
               r["answer"] if r["answer"] is not None else f"**失敗**：{r['error']}", "",
               f"- 引用來源：{cited or '無'}", ""]
    (out / "answers.md").write_text("\n".join(md), encoding="utf-8")
    failed = [r["id"] for r in records if r["answer"] is None]
    print(f"done: {len(records) - len(failed)}/{len(records)} answered" + (f", failed {failed}" if failed else ""))


if __name__ == "__main__":
    main()

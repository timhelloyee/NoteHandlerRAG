"""Build the research corpus from pdf/ — one Gemini call per PAGE, cached to disk.

Why per page rather than per document:
  * Size — lv.0 (14.4 MB) and 地科 (15.9 MB) exceed Gemini's inline request limit when
    base64-encoded, so a whole-document call simply fails.
  * Fidelity — a 23-page scan in one call comes back summarised. Measured earlier on a
    4-page PDF, whole-document extraction dropped a table outright (「（表略）」).
  * Provenance — page numbers give the research a unit to label and cite.

Pages are split with pypdf (no rasterising, so nothing is re-encoded) and each one-page
PDF is sent on its own. Every result is cached under eval/cache/, so a re-run costs
nothing and an interrupted run resumes — plan §R5 asks for exactly this so evaluations
stay reproducible when the hosted model changes underneath us.

    python eval/build_corpus.py                 # prose extraction (config B1)
    python eval/build_corpus.py --structured    # nested Markdown (config B4)
    python eval/build_corpus.py --files lv.0_compressed.pdf
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_TOKEN", "")            # never touch the private Dataset
os.environ.setdefault("WHOLE_CORPUS_CHARS", "0")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from pypdf import PdfReader, PdfWriter   # noqa: E402
from google.genai import types           # noqa: E402
from app import rag_core as rc           # noqa: E402

CACHE = Path("eval/cache")
CORPUS = Path("eval/corpus")

# The subject the note actually belongs to, for checking Gemini's own classification.
EXPECTED_SUBJECT = {
    "lv.0_compressed.pdf": "程式設計",
    "一下公民三段.pdf": "公民",
    "一下地科三段.pdf": "地球科學",
    "一下歷史三段.pdf": "歷史",
    "一下生物三段_compressed.pdf": "生物",
    "哈佛高二下二段段複.pdf": "物理",
}


def page_bytes(reader, index):
    """One page as a standalone PDF, so each call carries only that page."""
    w = PdfWriter()
    w.add_page(reader.pages[index])
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def describe_page(pdf_name, index, data, structured, model=None):
    """Gemini description of a single page, cached on (file, page, mode, model, bytes)."""
    key = hashlib.md5(
        f"{pdf_name}|{index}|{structured}|{model or rc.DEFAULT_MODEL}".encode()
        + hashlib.md5(data).digest()
    ).hexdigest()[:16]
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))["text"], True

    prompt = rc.PDF_PAGE_PROMPT + (rc.STRUCTURED_RULE if structured else "")
    resp = rc._vision_generate(
        contents=[types.Part.from_bytes(data=data, mime_type="application/pdf"),
                  types.Part.from_text(text=prompt)],
        config={"temperature": 0, "top_p": 0.95, "top_k": 20},
        model=model,
    )
    text = rc._clean_description(resp.text or "")
    cached.write_text(json.dumps(
        {"file": pdf_name, "page": index + 1, "structured": structured,
         "model": model or rc.DEFAULT_MODEL, "text": text},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return text, False


def subject_of(text):
    return rc.subject_of(text)


def is_blank(text):
    """A photographed notebook has blank pages; Gemini reports them honestly as
    「無法辨識」+「沒有任何文字」. Embedding those would put 'this page is empty' into the
    index as a retrievable passage, so they are recorded in the manifest but excluded
    from the corpus. Excluded pages are counted, not silently dropped."""
    t = text or ""
    return rc.subject_of(t) == "無法辨識" and len(t) < 160 and (
        "沒有任何文字" in t or "空白" in t or "無內容" in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured", action="store_true")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    mode = "B4-structured" if args.structured else "B1-prose"

    pdfs = sorted(Path("pdf").glob("*.pdf"))
    if args.files:
        want = set(args.files)
        pdfs = [p for p in pdfs if p.name in want]
    print(f"building corpus [{mode}] from {len(pdfs)} PDF(s) in pdf/\n")

    manifest, t0 = [], time.time()
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        n = len(reader.pages)
        pages, hits = [], 0
        print(f"{pdf.name}  ({n} pages, {pdf.stat().st_size/1e6:.1f} MB)")
        for i in range(n):
            text, cached = describe_page(pdf.name, i, page_bytes(reader, i),
                                         args.structured, args.model)
            hits += cached
            blank = is_blank(text)
            pages.append({"page": i + 1, "text": text, "subject": subject_of(text),
                          "chars": len(text), "blank": blank})
            print(f"   p{i+1:<3} {'cache' if cached else 'API  '} "
                  f"{len(text):5} chars  subject={subject_of(text)}"
                  f"{'  [blank - excluded]' if blank else ''}")
        kept = [p for p in pages if not p["blank"]]
        subjects = [p["subject"] for p in kept] or ["?"]
        top = max(set(subjects), key=subjects.count)
        expected = EXPECTED_SUBJECT.get(pdf.name, "?")
        flag = "OK " if top == expected else "MISMATCH"
        print(f"   -> {len(kept)}/{n} pages kept, {sum(p['chars'] for p in kept)} chars, " + f"{hits}/{n} cached, majority subject={top} (expected {expected}) [{flag}]" + chr(10))
        manifest.append({"file": pdf.name, "pages": n, "kept_pages": len(kept),
                         "mode": mode, "expected_subject": expected,
                         "majority_subject": top,
                         "chars": sum(p["chars"] for p in kept), "page_texts": pages})

    out = CORPUS / f"corpus_{'b4' if args.structured else 'b1'}.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tot_pages = sum(m["kept_pages"] for m in manifest)
    tot_chars = sum(m["chars"] for m in manifest)
    print(f"{'file':34} {'pages':>5} {'chars':>7}  subject")
    print("-" * 66)
    for m in manifest:
        mark = " " if m["majority_subject"] == m["expected_subject"] else "!"
        print(f"{mark}{m['file']:33} {m['kept_pages']:5} {m['chars']:7}  "
              f"{m['majority_subject']}")
    print(f"\n{tot_pages} pages, {tot_chars} chars -> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

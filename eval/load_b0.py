"""Turn the corrected eval/b0/pages/ files into corpus_b0.json.

Reads back what a human wrote: everything after the '---' separator is the corrected,
structured text for that page. Reports which pages are still untouched so a partial B0
(one subject at a time) is visible rather than silently mixing corrected and raw pages.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "eval/b0/pages"
B1 = json.loads((ROOT / "eval/corpus/corpus_b1.json").read_text(encoding="utf-8"))


def corrected_body(path):
    text = path.read_text(encoding="utf-8")
    _, _, body = text.partition("\n---\n")
    return body.strip()


def main():
    original = {}
    for note in B1:
        for pg in note["page_texts"]:
            if not pg["blank"]:
                pid = f"{note['file']}#p{pg['page']:02d}"
                original[pid] = re.sub(r"關鍵字[:：].*", "", pg["text"]).strip()

    manifest, edited, untouched = [], [], []
    for note in B1:
        pages = []
        for pg in note["page_texts"]:
            if pg["blank"]:
                pages.append({**pg, "blank": True})
                continue
            pid = f"{note['file']}#p{pg['page']:02d}"
            f = PAGES / (pid.replace("#", "__").replace(".pdf", "") + ".md")
            if not f.exists():
                untouched.append(pid)
                continue
            body = corrected_body(f)
            changed = body != original[pid]
            (edited if changed else untouched).append(pid)
            blank = len(body) < 20 or "（無內容）" in body
            pages.append({"page": pg["page"], "text": body, "chars": len(body),
                          "subject": pg["subject"], "blank": blank})
        manifest.append({"file": note["file"], "pages": note["pages"],
                         "kept_pages": sum(1 for p in pages if not p["blank"]),
                         "mode": "B0-oracle", "expected_subject": note["expected_subject"],
                         "majority_subject": note["majority_subject"],
                         "chars": sum(p["chars"] for p in pages if not p["blank"]),
                         "page_texts": pages})

    out = ROOT / "eval/corpus/corpus_b0.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"corrected pages : {len(edited)}")
    print(f"still untouched : {len(untouched)}")
    if untouched:
        subj = {}
        for pid in untouched:
            subj[pid.split('#')[0]] = subj.get(pid.split('#')[0], 0) + 1
        for k, v in subj.items():
            print(f"    {k:34} {v} page(s) unedited")
        print("\nA配置只在整科完成時才可信：混合校正與未校正的頁面會讓校正過的頁面"
              "單純因為文字較準而佔便宜。")
    print(f"\n-> {out}")
    if not edited:
        sys.exit("nothing corrected yet")


if __name__ == "__main__":
    main()

"""Add notes to a local NoteHandler store from the command line (replaces NoteHandler_Input.ipynb).

The notebook kept its own copy of the ingestion code — whole-document ids, no chunking,
an old image prompt — which drifted from the app until the two could no longer share a
store. This script calls the app's own functions in app/rag_core.py instead, so a note
added here is stored exactly as an upload through the web page would be: same prompts,
same chunking, same categories.

Like scripts/dev_server.py it never touches the private Hugging Face Dataset: HF_TOKEN is cleared
and the store defaults to .devstore/, the folder the local web page uses.

    python scripts/notehandler_input.py add samples/text/a.txt samples/images/Apple.jpg --user alice
    python scripts/notehandler_input.py add pdf/lv.0_compressed.pdf --user alice --category 程式設計
    python scripts/notehandler_input.py delete a.txt --user alice
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # the project root; this file is in scripts/

TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
PDF_EXTS = {".pdf"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", default="alice", help="whose store to use (default: alice)")
    ap.add_argument("--store", default=".devstore", help="store folder (default: .devstore)")
    sub = ap.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="add text, image or PDF notes")
    add.add_argument("files", nargs="+", type=Path)
    add.add_argument("--category", help="category name; created if it does not exist")
    add.add_argument("--model", help="Gemini model for images and PDFs (default: the app's)")

    delete = sub.add_parser("delete", help="delete a note by its id (the file name)")
    delete.add_argument("note_ids", nargs="+")
    return ap.parse_args()


def category_id(rc, user: str, name: str | None) -> str | None:
    """The id of the category with this name, creating it when missing."""
    if not name:
        return None
    for c in rc.list_categories(user):
        if c["name"] == name:
            return c["id"]
    created = rc.create_category(user, name)
    print(f"建立類別「{created['name']}」")
    return created["id"]


def add(rc, user: str, files: list[Path], category: str | None, model: str | None) -> int:
    failed = 0
    for path in files:
        if not path.is_file():
            print(f"找不到檔案：{path}")
            failed += 1
            continue
        file_id, ext = path.name, path.suffix.lower()
        source = f"upload/{file_id}"        # what the web upload records
        if ext in TEXT_EXTS:
            added = rc.add_text_note(user, file_id, path.read_text(encoding="utf-8"), source,
                                     category=category)
        elif ext in IMAGE_EXTS:
            added = bool(rc.add_image_note(user, file_id, str(path), source,
                                           model=model, category=category))
        elif ext in PDF_EXTS:
            added = bool(rc.add_pdf_note(user, file_id, str(path), source,
                                         model=model, category=category))
        else:
            print(f"不支援的檔案類型：{path}")
            failed += 1
            continue
        print(f"{file_id} 已加入" if added else f"{file_id} 已存在，跳過")
    return failed


def delete(rc, user: str, note_ids: list[str]) -> int:
    failed = 0
    for note_id in note_ids:
        if rc.delete_note(user, note_id):
            print(f"{note_id} 已刪除")
        else:
            print(f"找不到筆記：{note_id}")
            failed += 1
    return failed


def main() -> int:
    args = parse_args()
    if args.command == "add":           # resolve against the caller's folder before chdir
        args.files = [p.resolve() for p in args.files]
    # Isolate BEFORE importing the app: rag_core restores from the Dataset at import time.
    os.chdir(ROOT)
    os.environ["HF_TOKEN"] = ""
    os.environ["VECTORSTORE_ROOT"] = args.store
    sys.path.insert(0, str(ROOT))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from app import rag_core as rc

    if args.command == "add":
        category = category_id(rc, args.user, args.category)
        return 1 if add(rc, args.user, args.files, category, args.model) else 0
    return 1 if delete(rc, args.user, args.note_ids) else 0


if __name__ == "__main__":
    sys.exit(main())

"""List and ask about a local NoteHandler store from the command line (replaces NoteHandler_opus_Output.ipynb).

The notebook opened the store directly and rebuilt its own retrieval and prompt, which no
longer matched the app. This script calls app/rag_core.py, so an answer here is the answer
the web page would give — including whole-corpus mode and asking inside one category.

Like dev_server.py it never touches the private Hugging Face Dataset: HF_TOKEN is cleared
and the store defaults to .devstore/, the folder the local web page uses.

    python notehandler_output.py list --user alice
    python notehandler_output.py ask "粒線體的功能是什麼？" --user alice
    python notehandler_output.py ask "When did the French Revolution begin?" --user alice --category B
    python notehandler_output.py ask --user alice          # prompts for the question
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", default="alice", help="whose store to use (default: alice)")
    ap.add_argument("--store", default=".devstore", help="store folder (default: .devstore)")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show every note, grouped by category")

    ask = sub.add_parser("ask", help="ask a question about the notes")
    ask.add_argument("question", nargs="?", help="asked interactively when omitted")
    ask.add_argument("--category", help="only search this category (default: every category)")
    ask.add_argument("--model", help="Gemini model (default: the app's)")
    ask.add_argument("--top-k", type=int, default=10)
    return ap.parse_args()


def list_notes(rc, user: str) -> int:
    categories = {c["id"]: c["name"] for c in rc.list_categories(user)}
    notes = rc.list_notes(user)
    print(f"共有 {len(notes)} 筆筆記、{len(categories)} 個類別")
    for cid, name in list(categories.items()) + [(None, "未分類")]:
        group = [n for n in notes if n["category"] == cid]
        if cid is not None or group:
            print(f"\n【{name}】{len(group)} 筆")
        for n in group:
            preview = " ".join((n["preview"] or "").split())[:100]
            print(f"  - {n['id']}  [{n['type']}]  {preview}")
    return 0


def ask(rc, user: str, question: str | None, category_name: str | None,
        model: str | None, top_k: int) -> int:
    category = None
    if category_name:
        matches = [c["id"] for c in rc.list_categories(user) if c["name"] == category_name]
        if not matches:
            print(f"找不到類別：{category_name}")
            return 1
        category = matches[0]
    question = (question or input("輸入問題: ")).strip()
    if not question:
        print("問題不可為空")
        return 1
    model = rc.resolve_model(model)
    print(f"（範圍：{category_name or '全部類別'}，模型：{model}）\n")
    print(rc.rag_query(user, question, top_k=top_k, model=model, category=category))
    return 0


def main() -> int:
    args = parse_args()
    # Isolate BEFORE importing the app: rag_core restores from the Dataset at import time.
    os.chdir(ROOT)
    os.environ["HF_TOKEN"] = ""
    os.environ["VECTORSTORE_ROOT"] = args.store
    sys.path.insert(0, str(ROOT))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from app import rag_core as rc

    if args.command == "list":
        return list_notes(rc, args.user)
    return ask(rc, args.user, args.question, args.category, args.model, args.top_k)


if __name__ == "__main__":
    sys.exit(main())

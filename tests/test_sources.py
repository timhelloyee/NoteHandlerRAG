"""Checks that answers cite their sources, and that only real sources survive.

    python tests/test_sources.py

Standalone like the other test scripts. Gemini is never called: the answer step is a
stub that records the prompt and replies with whatever citations a test wants, so the
server's checking of those citations can be tested exactly.
"""
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = tempfile.mkdtemp(prefix="nh-test-sources-")
os.environ["HF_TOKEN"] = ""
os.environ["VECTORSTORE_ROOT"] = STORE
os.environ.pop("SPACE_ID", None)
os.environ["NOTEHANDLER_DEV"] = "1"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient            # noqa: E402

import app.main as main                               # noqa: E402
from app import rag_core as rc                        # noqa: E402

client = TestClient(main.app)
RESULTS = []

# A three-page "PDF" note (the page headings are what describe_pdf writes) and a text note.
PDF_ID = "lv.0_compressed.pdf"
PDF_TEXT = ("這是一份【程式設計】科的筆記。\n\n"
            "## 第 3 頁\n字串的基本操作：長度、串接。\n\n"
            "## 第 5 頁\n迴圈與條件判斷。\n\n"
            "## 第 12 頁\n`s.find(\"ABC\")` 找出字串第一次出現的位置。")
TXT_ID = "note_B.txt"
TXT_TEXT = "The French Revolution began in 1789."


def test(fn):
    RESULTS.append(fn)
    return fn


class ScriptedModel:
    """Replies with a fixed answer and keeps the prompts it received."""
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt, model=None):
        self.prompts.append(prompt)
        return self.reply


def seed(user):
    rc.add_text_note(user, PDF_ID, PDF_TEXT, source=f"upload/{PDF_ID}")
    rc.add_text_note(user, TXT_ID, TXT_TEXT, source=f"upload/{TXT_ID}")
    return {"X-Dev-User": user}


def ask(headers, reply, **body):
    model, real = ScriptedModel(reply), rc._llm_invoke
    rc._llm_invoke = model
    try:
        r = client.post("/api/query", headers=headers, json={"question": "q", **body})
    finally:
        rc._llm_invoke = real
    assert r.status_code == 200, r.text
    return r.json(), model.prompts[-1] if model.prompts else ""


# --- tests ------------------------------------------------------------------------------
@test
def test_context_labels_every_note_with_its_file_name():
    h = seed("labels")
    _, prompt = ask(h, "ok")
    assert f"【來源：{PDF_ID}】" in prompt and f"【來源：{TXT_ID}】" in prompt, prompt
    assert "## 第 12 頁" in prompt, "page headings must reach the model"
    assert "〔來源：檔名 第 N 頁〕" in prompt, "the prompt must ask for citations"


@test
def test_valid_citations_become_sources():
    h = seed("valid")
    data, _ = ask(h, f"find 回傳位置〔來源：{PDF_ID} 第 12 頁〕。1789 年〔來源：{TXT_ID}〕。")
    assert [(s["id"], s["page"]) for s in data["sources"]] == [(PDF_ID, 12), (TXT_ID, None)], data
    assert data["dropped"] == 0, data
    for s in data["sources"]:
        assert s["marker"] in data["answer"], (s, data["answer"])


@test
def test_made_up_pages_and_files_are_removed():
    h = seed("invalid")
    reply = (f"甲〔來源：{PDF_ID} 第 99 頁〕乙〔來源：ghost.pdf 第 2 頁〕"
             f"丙〔來源：{PDF_ID} 第 3 頁；ghost.pdf 第 5 頁〕")
    data, _ = ask(h, reply)
    assert "99" not in data["answer"] and "ghost" not in data["answer"], data["answer"]
    assert [(s["id"], s["page"]) for s in data["sources"]] == [(PDF_ID, 3)], data
    assert data["dropped"] == 3, data


@test
def test_citation_variants_are_normalised():
    h = seed("variants")
    reply = (f"一〔來源：{PDF_ID} 第 3、5 頁〕二【來源：{TXT_ID} 第 1 頁】"
             f"三 [來源: {TXT_ID}] 四〔來源：{PDF_ID} 第3-5頁〕")
    data, _ = ask(h, reply)
    assert [(s["id"], s["page"]) for s in data["sources"]] == \
        [(PDF_ID, 3), (PDF_ID, 5), (TXT_ID, None)], data
    assert data["dropped"] == 0, data
    assert "【" not in data["answer"] and "[來源" not in data["answer"], data["answer"]


@test
def test_a_note_outside_the_chosen_category_cannot_be_cited():
    h = seed("scoped")
    cid = client.post("/api/categories", headers=h, json={"name": "程式"}).json()["id"]
    assert client.post(f"/api/notes/{PDF_ID}/move", headers=h, json={"category": cid}).status_code == 200
    data, prompt = ask(h, f"一〔來源：{PDF_ID} 第 3 頁〕二〔來源：{TXT_ID}〕", category=cid)
    assert TXT_ID not in prompt, "a note from another category was sent"
    assert [(s["id"], s["page"]) for s in data["sources"]] == [(PDF_ID, 3)], data
    assert data["dropped"] == 1, data


@test
def test_retrieved_chunks_keep_their_page():
    """A chunk that starts in the middle of a page is labelled with that page."""
    user = "chunks"
    long_page = "## 第 7 頁\n" + "。".join(f"第七頁的第 {i} 個句子內容" for i in range(80))
    rc.add_text_note(user, "long.pdf", "這是一份【物理】科的筆記。\n\n" + long_page,
                     source="upload/long.pdf")
    rc.WHOLE_CORPUS_CHARS = 0
    try:
        data, prompt = ask({"X-Dev-User": user}, "位置〔來源：long.pdf 第 7 頁〕",
                           question="第七頁的第 60 個句子內容是什麼")
    finally:
        rc.WHOLE_CORPUS_CHARS = 60000
    assert "【來源：long.pdf】" in prompt, prompt[:500]
    assert "## 第 7 頁（續）" in prompt, "a mid-page chunk was sent without its page"
    assert [(s["id"], s["page"]) for s in data["sources"]] == [("long.pdf", 7)], data


@test
def test_refusal_has_no_sources():
    h = seed("refusal")
    data, _ = ask(h, rc.REFUSAL)
    assert data["sources"] == [] and data["dropped"] == 0, data


@test
def test_empty_store_refuses_without_calling_the_model():
    data, prompt = ask({"X-Dev-User": "empty"}, "should not be used")
    assert data["answer"] == rc.REFUSAL and prompt == "", data


# --- runner -----------------------------------------------------------------------------
def run():
    main.DEV_MODE = True
    failed = 0
    for fn in RESULTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            print("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    rc.SharedSystemClient.clear_system_cache()
    shutil.rmtree(STORE, ignore_errors=True)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run() else 0)

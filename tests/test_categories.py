"""Checks for flat, independent note categories.

    python tests/test_categories.py

Standalone like test_auth_and_sync.py: prints PASS/FAIL per check and exits non-zero on
failure. Runs against a temporary store in local (sync disabled) mode with the dev user
header, and never calls Gemini — the answer step is replaced by a stub that records the
prompt, so a test can see exactly which notes were sent to the model.
"""
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = tempfile.mkdtemp(prefix="nh-test-categories-")
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
SENTENCES = {
    "A": "The mitochondria is the powerhouse of the cell.",
    "B": "The French Revolution began in 1789.",
    "C": "Water boils at one hundred degrees Celsius at sea level.",
}


def test(fn):
    RESULTS.append(fn)
    return fn


def as_user(name):
    return {"X-Dev-User": name}


def make_category(h, name):
    r = client.post("/api/categories", headers=h, json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def upload(h, name, body, category=""):
    return client.post("/api/upload", headers=h, data={"category": category},
                       files={"file": (name, body.encode("utf-8"), "text/plain")})


def notes_by_id(h):
    return {n["id"]: n for n in client.get("/api/notes", headers=h).json()}


class PromptRecorder:
    """Stands in for the Gemini call and keeps every prompt it was given."""
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, model=None):
        self.prompts.append(prompt)
        return "stub answer"


def setup_abc(user):
    """Three categories A, B, C, each holding one English sentence as a .txt note."""
    h = as_user(user)
    ids = {}
    for key, sentence in SENTENCES.items():
        ids[key] = make_category(h, key)
        r = upload(h, f"{key.lower()}.txt", sentence, ids[key])
        assert r.status_code == 200 and r.json()["added"], r.text
    return h, ids


# --- tests ------------------------------------------------------------------------------
@test
def test_each_category_is_its_own_folder():
    h, ids = setup_abc("folders")
    base = Path(rc._user_dir("folders")) / rc.CATEGORIES_DIR
    assert sorted(p.name for p in base.iterdir()) == sorted(ids.values())
    for cid in ids.values():
        folder = base / cid
        assert (folder / rc.CATEGORY_FILE).is_file()
        # No category lives inside another: nothing under a category folder is itself
        # a category (only Chroma's own index files and originals/).
        assert not (folder / rc.CATEGORIES_DIR).exists(), f"nested categories under {cid}"
    listed = notes_by_id(h)
    for key, cid in ids.items():
        assert listed[f"{key.lower()}.txt"]["category"] == cid, listed


@test
def test_names_are_validated():
    h = as_user("names")
    make_category(h, "生物")
    cases = [("", 400), ("   ", 400), ("生物", 409), ("a/b", 400), ("a\\b", 400),
             ("未分類", 400), ("x" * 31, 400)]
    for name, status in cases:
        r = client.post("/api/categories", headers=h, json={"name": name})
        assert r.status_code == status, f"{name!r} -> {r.status_code} {r.text}"
    cid = make_category(h, "化學")
    assert client.put(f"/api/categories/{cid}", headers=h, json={"name": "生物"}).status_code == 409
    r = client.put(f"/api/categories/{cid}", headers=h, json={"name": "化學 二"})
    assert r.status_code == 200 and r.json()["name"] == "化學 二", r.text
    assert client.put("/api/categories/c123456789abc", headers=h, json={"name": "x"}).status_code == 404
    assert client.put("/api/categories/..", headers=h, json={"name": "x"}).status_code in (404, 405)


@test
def test_question_in_a_category_reads_only_that_category():
    h, ids = setup_abc("scoped")
    recorder, real = PromptRecorder(), rc._llm_invoke
    opened = []
    real_get = rc.get_user_collection

    def spy(user_id, category=None):
        opened.append(category)
        return real_get(user_id, category)

    rc._llm_invoke, rc.get_user_collection = recorder, spy
    try:
        for mode in ("60000", "0"):            # whole-corpus path, then forced retrieval
            rc.WHOLE_CORPUS_CHARS = int(mode)
            for key, cid in ids.items():
                opened.clear()
                r = client.post("/api/query", headers=h,
                                json={"question": "What does the note say?", "category": cid})
                assert r.status_code == 200, r.text
                prompt = recorder.prompts[-1]
                assert SENTENCES[key] in prompt, f"{key}: own note missing ({mode})"
                for other, sentence in SENTENCES.items():
                    if other != key:
                        assert sentence not in prompt, f"{key}: leaked {other} ({mode})"
                assert set(opened) == {cid}, f"{key}: opened {opened} ({mode})"
    finally:
        rc._llm_invoke, rc.get_user_collection = real, real_get
        rc.WHOLE_CORPUS_CHARS = 60000


@test
def test_question_without_a_category_reads_every_category():
    h, ids = setup_abc("everything")
    upload(h, "loose.txt", "An uncategorized sentence about volcanoes.")
    recorder, real = PromptRecorder(), rc._llm_invoke
    rc._llm_invoke = recorder
    try:
        r = client.post("/api/query", headers=h, json={"question": "Summarize my notes."})
        assert r.status_code == 200, r.text
        prompt = recorder.prompts[-1]
        for sentence in list(SENTENCES.values()) + ["volcanoes"]:
            assert sentence in prompt, f"missing {sentence!r}"
    finally:
        rc._llm_invoke = real


@test
def test_moving_a_note_keeps_it_whole_without_re_embedding():
    h, ids = setup_abc("mover")
    real_embed = rc.embed_text

    def no_embedding(texts):
        raise AssertionError("moving a note must not re-embed it")

    rc.embed_text = no_embedding
    try:
        r = client.post("/api/notes/a.txt/move", headers=h, json={"category": ids["B"]})
        assert r.status_code == 200, r.text
    finally:
        rc.embed_text = real_embed
    listed = notes_by_id(h)
    assert listed["a.txt"]["category"] == ids["B"], listed["a.txt"]
    assert client.get("/api/notes/a.txt", headers=h).json()["text"] == SENTENCES["A"]
    # Into uncategorized and back.
    assert client.post("/api/notes/a.txt/move", headers=h, json={"category": None}).status_code == 200
    assert notes_by_id(h)["a.txt"]["category"] is None
    assert client.post("/api/notes/nope.txt/move", headers=h, json={"category": None}).status_code == 404
    assert client.post("/api/notes/b.txt/move", headers=h,
                       json={"category": "c123456789abc"}).status_code == 404


@test
def test_original_moves_with_its_note():
    h = as_user("pictures")
    a, b = make_category(h, "A"), make_category(h, "B")
    rc.add_text_note("pictures", "scan.png", "A scanned page.", source="upload/scan.png", category=a)
    src = Path(tempfile.mkdtemp()) / "scan.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\2" * 16)
    rc.save_original("pictures", "scan.png", str(src), category=a)
    assert client.post("/api/notes/scan.png/move", headers=h, json={"category": b}).status_code == 200
    assert rc.get_original("pictures", "scan.png", a) is None, "original left in the old folder"
    assert rc.get_original("pictures", "scan.png", b), "original not in the new folder"
    assert client.get("/api/notes/scan.png/original", headers=h).status_code == 200


@test
def test_note_ids_are_unique_across_categories():
    h, ids = setup_abc("unique")
    r = upload(h, "a.txt", "A different body.", ids["C"])
    assert r.status_code == 200 and r.json()["added"] is False, r.text
    assert notes_by_id(h)["a.txt"]["category"] == ids["A"]


@test
def test_editing_keeps_the_category():
    h, ids = setup_abc("editor")
    r = client.put("/api/notes/b.txt", headers=h, json={"content": "The Revolution began in 1789."})
    assert r.status_code == 200, r.text
    assert notes_by_id(h)["b.txt"]["category"] == ids["B"]


@test
def test_deleting_a_category_keeps_its_notes():
    h, ids = setup_abc("deleter")
    r = client.delete(f"/api/categories/{ids['C']}", headers=h)
    assert r.status_code == 200 and r.json()["moved"] == 1, r.text
    assert not Path(rc._store_dir("deleter", ids["C"])).exists(), "category folder still on disk"
    listed = notes_by_id(h)
    assert listed["c.txt"]["category"] is None, listed["c.txt"]
    assert client.get("/api/notes/c.txt", headers=h).json()["text"] == SENTENCES["C"]
    cats = [c["id"] for c in client.get("/api/categories", headers=h).json()]
    assert ids["C"] not in cats and ids["A"] in cats, cats
    # The remaining categories still work after the store cache was cleared.
    recorder, real = PromptRecorder(), rc._llm_invoke
    rc._llm_invoke = recorder
    try:
        r = client.post("/api/query", headers=h, json={"question": "q", "category": ids["A"]})
        assert r.status_code == 200 and SENTENCES["A"] in recorder.prompts[-1], r.text
    finally:
        rc._llm_invoke = real


@test
def test_unknown_category_is_refused_everywhere():
    h = as_user("strict")
    bad = "c123456789abc"
    assert upload(h, "x.txt", "text", bad).status_code == 404
    assert client.post("/api/query", headers=h, json={"question": "q", "category": bad}).status_code == 404
    assert client.delete(f"/api/categories/{bad}", headers=h).status_code == 404
    # Nothing was created for the unknown id.
    assert not Path(rc._user_dir("strict"), rc.CATEGORIES_DIR, bad).exists()


@test
def test_users_do_not_share_categories():
    h, ids = setup_abc("owner2")
    other = as_user("intruder2")
    assert client.get("/api/categories", headers=other).json() == []
    assert client.post("/api/query", headers=other,
                       json={"question": "q", "category": ids["A"]}).status_code == 404
    assert client.delete(f"/api/categories/{ids['A']}", headers=other).status_code == 404


@test
def test_category_writes_refused_when_store_unavailable():
    h = as_user("offline")
    cid = make_category(h, "A")
    rc._set_sync("unavailable", "雲端筆記庫拒絕了存取")
    try:
        for r in (client.post("/api/categories", headers=h, json={"name": "B"}),
                  client.put(f"/api/categories/{cid}", headers=h, json={"name": "B"}),
                  client.delete(f"/api/categories/{cid}", headers=h),
                  client.post("/api/notes/x.txt/move", headers=h, json={"category": cid})):
            assert r.status_code == 503, f"{r.request.method} {r.request.url.path} -> {r.status_code}"
    finally:
        rc._set_sync("disabled", "")


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

"""Checks for server-side sign-in and cloud-sync safety.

    python tests/test_auth_and_sync.py

Standalone on purpose (the project does not depend on pytest). Each test prints PASS or
FAIL and the script exits non-zero if any failed. No real Google account or Hugging Face
token is used: Google verification is replaced by a fake where a test needs a *valid*
sign-in, and the Dataset API by a fake that records what would have been uploaded. The
one exception is test_forged_token_rejected, which lets the real verifier fetch Google's
public certificates — it reports SKIP when offline.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = tempfile.mkdtemp(prefix="nh-test-store-")
os.environ["HF_TOKEN"] = ""
os.environ["VECTORSTORE_ROOT"] = STORE
os.environ.pop("SPACE_ID", None)
os.environ.pop("NOTEHANDLER_DEV", None)
os.environ["WHOLE_CORPUS_CHARS"] = "60000"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient            # noqa: E402
from google.auth import exceptions as gexc            # noqa: E402

import app.main as main                               # noqa: E402
from app import rag_core as rc                        # noqa: E402

client = TestClient(main.app)
RESULTS = []


# --- helpers ----------------------------------------------------------------------------
class FakeVerifier:
    """Stands in for id_token.verify_oauth2_token. Maps token strings to outcomes."""
    def __init__(self, table):
        self.table, self.calls = table, 0

    def __call__(self, token, request, audience, clock_skew_in_seconds=0):
        self.calls += 1
        outcome = self.table[token]
        if isinstance(outcome, Exception):
            raise outcome
        assert audience == main.GOOGLE_CLIENT_ID, "wrong audience passed to verifier"
        return outcome


class FakeHfApi:
    """Records uploads instead of talking to the Hub; can be told to fail."""
    def __init__(self, fail_times=0):
        self.fail_times, self.uploads = fail_times, []

    def create_repo(self, *a, **k):
        return None

    def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type, **kwargs):
        originals = os.path.join(folder_path, rc.ORIGINALS_DIR)
        self.uploads.append({"folder": folder_path, "kwargs": kwargs,
                             "originals": sorted(os.listdir(originals))
                             if os.path.isdir(originals) else []})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("503 Service Unavailable")


def claims(email, verified=True, ttl=3600):
    return {"email": email, "email_verified": verified, "exp": time.time() + ttl,
            "aud": main.GOOGLE_CLIENT_ID, "iss": "https://accounts.google.com"}


def use_verifier(table):
    fake = FakeVerifier(table)
    main.id_token.verify_oauth2_token = fake
    main._verified_tokens.clear()
    return fake


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def reset_sync(state="disabled"):
    rc._hf_api = None
    rc._unsynced_users.clear()
    rc._set_sync(state, "")


def upload_text(headers, name="note.txt", body="粒線體是細胞的發電廠"):
    return client.post("/api/upload", headers=headers,
                       files={"file": (name, body.encode("utf-8"), "text/plain")})


def test(fn):
    RESULTS.append(fn)
    return fn


_real_verify = main.id_token.verify_oauth2_token


# --- authentication ---------------------------------------------------------------------
@test
def test_every_user_endpoint_requires_sign_in():
    main.DEV_MODE = False
    reset_sync()
    calls = [
        ("GET", "/api/notes", {}),
        ("GET", "/api/notes/x.txt", {}),
        ("GET", "/api/notes/x.txt/original", {}),
        ("PUT", "/api/notes/x.txt", {"json": {"content": "a"}}),
        ("DELETE", "/api/notes/x.txt", {}),
        ("POST", "/api/query", {"json": {"question": "hi"}}),
        ("POST", "/api/upload", {"files": {"file": ("a.txt", b"x", "text/plain")}}),
        ("POST", "/api/notes/x.txt/move", {"json": {"category": None}}),
        ("GET", "/api/categories", {}),
        ("POST", "/api/categories", {"json": {"name": "A"}}),
        ("PUT", "/api/categories/c000000000000", {"json": {"name": "B"}}),
        ("DELETE", "/api/categories/c000000000000", {}),
    ]
    for method, path, kw in calls:
        r = client.request(method, path, **kw)
        assert r.status_code == 401, f"{method} {path} -> {r.status_code}, expected 401"


@test
def test_email_in_request_no_longer_names_a_user():
    """The old attack: pass someone's address. It must not matter at all now."""
    main.DEV_MODE = False
    r = client.get("/api/notes", params={"user_id": "timhelloyee@gmail.com"})
    assert r.status_code == 401, r.status_code
    r = client.post("/api/query", json={"user_id": "timhelloyee@gmail.com", "question": "q"})
    assert r.status_code == 401, r.status_code


@test
def test_malformed_authorization_rejected():
    main.DEV_MODE = False
    for h in ({"Authorization": "Bearer"}, {"Authorization": "Basic abc"},
              {"Authorization": "bearer "}, {"Authorization": "token"}):
        assert client.get("/api/notes", headers=h).status_code == 401, h


@test
def test_forged_token_rejected():
    """A token that *claims* to be a real address but is not signed by Google."""
    main.DEV_MODE = False
    main.id_token.verify_oauth2_token = _real_verify
    main._verified_tokens.clear()
    b64 = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    forged = ".".join([b64({"alg": "RS256", "kid": "not-a-google-key", "typ": "JWT"}),
                       b64(claims("timhelloyee@gmail.com")),
                       base64.urlsafe_b64encode(b"forged-signature").rstrip(b"=").decode()])
    r = client.get("/api/notes", headers=bearer(forged))
    if r.status_code == 503:
        return "SKIP (offline: could not fetch Google certificates)"
    assert r.status_code == 401, f"forged token -> {r.status_code}"


@test
def test_verified_token_cached_and_rules_applied():
    main.DEV_MODE = False
    reset_sync()
    fake = use_verifier({
        "good": claims("alice@example.com"),
        "unverified": claims("bob@example.com", verified=False),
        "expired": ValueError("Token expired"),
        "offline": gexc.TransportError("no route to host"),
    })
    assert client.get("/api/notes", headers=bearer("good")).status_code == 200
    assert client.get("/api/notes", headers=bearer("good")).status_code == 200
    assert fake.calls == 1, f"verifier called {fake.calls} times for one token"
    assert client.get("/api/notes", headers=bearer("unverified")).status_code == 401
    assert client.get("/api/notes", headers=bearer("expired")).status_code == 401
    r = client.get("/api/notes", headers=bearer("offline"))
    assert r.status_code == 503, f"transport failure -> {r.status_code}, expected 503"


@test
def test_users_cannot_see_or_touch_each_others_notes():
    main.DEV_MODE = False
    reset_sync()
    use_verifier({"a": claims("owner@example.com"), "b": claims("intruder@example.com")})
    r = upload_text(bearer("a"), name="private.txt")
    assert r.status_code == 200 and r.json()["added"], r.text
    ids_b = [n["id"] for n in client.get("/api/notes", headers=bearer("b")).json()]
    assert "private.txt" not in ids_b, "intruder can list the owner's note"
    assert client.get("/api/notes/private.txt", headers=bearer("b")).status_code == 404
    assert client.delete("/api/notes/private.txt", headers=bearer("b")).status_code == 404
    ids_a = [n["id"] for n in client.get("/api/notes", headers=bearer("a")).json()]
    assert "private.txt" in ids_a, "owner's note vanished after the intruder's delete"


@test
def test_dev_header_only_works_in_dev_mode():
    reset_sync()
    main.DEV_MODE = False
    assert client.get("/api/notes", headers={"X-Dev-User": "devtest"}).status_code == 401
    main.DEV_MODE = True
    assert client.get("/api/notes", headers={"X-Dev-User": "devtest"}).status_code == 200
    main.DEV_MODE = False


@test
def test_dev_mode_refused_on_a_space():
    """Evaluated in a fresh process, since DEV_MODE is decided at import."""
    env = dict(os.environ, NOTEHANDLER_DEV="1", SPACE_ID="timhelloyee/NoteHandler",
               HF_TOKEN="", VECTORSTORE_ROOT=tempfile.mkdtemp(prefix="nh-space-"),
               PYTHONIOENCODING="utf-8")
    # Explicit UTF-8: the child logs non-ASCII, and Windows would otherwise decode the
    # pipe with the ANSI code page and crash the reader thread.
    out = subprocess.run(
        [sys.executable, "-c",
         "import app.main as m; print('DEVMODE', m.DEV_MODE); "
         "print('EDITOR', any('ui-editor' in getattr(r,'path','') for r in m.app.routes))"],
        cwd=ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace",
        timeout=600)
    assert "DEVMODE False" in out.stdout, out.stdout + out.stderr[-800:]
    assert "EDITOR False" in out.stdout, out.stdout


@test
def test_cors_preflight_allows_put_with_authorization():
    r = client.options("/api/notes/x.txt", headers={
        "Origin": "https://timhelloyee.github.io",
        "Access-Control-Request-Method": "PUT",
        "Access-Control-Request-Headers": "authorization,content-type",
    })
    assert r.status_code == 200, r.status_code
    assert "PUT" in r.headers.get("access-control-allow-methods", ""), r.headers
    assert "authorization" in r.headers.get("access-control-allow-headers", "").lower(), r.headers


# --- cloud sync -------------------------------------------------------------------------
@test
def test_local_mode_reports_disabled_and_allows_writes():
    main.DEV_MODE = True
    reset_sync("disabled")
    h = client.get("/api/health").json()["sync"]
    assert h["state"] == "disabled" and h["writable"] is True, h
    r = upload_text({"X-Dev-User": "local"}, name="local.txt")
    assert r.status_code == 200 and r.json()["sync"] == "disabled", r.text


@test
def test_unavailable_store_refuses_every_write_and_query():
    main.DEV_MODE = True
    reset_sync("unavailable")
    rc._set_sync("unavailable", "雲端筆記庫拒絕了存取")
    h = {"X-Dev-User": "local"}
    assert client.get("/api/health").json()["sync"]["writable"] is False
    for r in (upload_text(h, name="blocked.txt"),
              client.put("/api/notes/local.txt", headers=h, json={"content": "x"}),
              client.delete("/api/notes/local.txt", headers=h),
              client.post("/api/query", headers=h, json={"question": "q"})):
        assert r.status_code == 503, f"{r.request.method} {r.request.url.path} -> {r.status_code}"
        assert "避免覆蓋" in r.json()["detail"], r.json()
    assert client.get("/api/notes", headers=h).status_code == 200, "listing should still work"
    reset_sync()


@test
def test_failed_restore_marks_store_unavailable():
    rc._hf_api, rc.HF_TOKEN = FakeHfApi(), "hf_fake"
    original = rc.snapshot_download
    def boom(*a, **k):
        raise RuntimeError("401 Client Error ... Invalid username or password.")
    rc.snapshot_download = boom
    try:
        rc._restore_from_dataset()
        s = rc.sync_status()
        assert s["state"] == "unavailable" and not s["writable"], s
        assert "HF_TOKEN" in s["message"], s["message"]
        assert "Invalid username" not in s["message"], "raw error leaked to the page"
    finally:
        rc.snapshot_download, rc.HF_TOKEN = original, ""
        reset_sync()


@test
def test_missing_token_on_a_space_is_unavailable_not_disabled():
    reset_sync()
    old = rc.SYNC_REQUIRED
    rc.SYNC_REQUIRED = True
    try:
        rc._restore_from_dataset()
        assert rc.sync_status()["state"] == "unavailable", rc.sync_status()
    finally:
        rc.SYNC_REQUIRED = old
        reset_sync()


@test
def test_successful_restore_is_ok():
    rc._hf_api, rc.HF_TOKEN = FakeHfApi(), "hf_fake"
    snap = tempfile.mkdtemp(prefix="nh-snap-")
    os.makedirs(os.path.join(snap, "someone"))
    original = rc.snapshot_download
    rc.snapshot_download = lambda *a, **k: snap
    try:
        rc._restore_from_dataset()
        assert rc.sync_status()["state"] == "ok", rc.sync_status()
    finally:
        rc.snapshot_download, rc.HF_TOKEN = original, ""
        reset_sync()


@test
def test_failed_sync_is_reported_then_repaired_by_next_write():
    main.DEV_MODE = True
    reset_sync("ok")
    rc._hf_api = FakeHfApi(fail_times=1)
    h = {"X-Dev-User": "syncer"}
    r = upload_text(h, name="first.txt")
    assert r.status_code == 200 and r.json()["sync"] == "unsaved", r.text
    assert client.get("/api/health").json()["sync"]["writable"] is True, "unsaved must not block"
    r = upload_text(h, name="second.txt")
    assert r.json()["sync"] == "ok", r.text
    assert rc.sync_status()["state"] == "ok"
    reset_sync()


@test
def test_original_is_in_the_folder_when_sync_runs():
    """The upload that reaches the Dataset must already contain the original file."""
    main.DEV_MODE = True
    reset_sync("ok")
    api = FakeHfApi()
    rc._hf_api = api
    original_describe = rc.describe_image
    rc.describe_image = lambda path, model=None: "這是一份【物理】科的筆記。\n\n波的繞射"
    img = Path(tempfile.mkdtemp()) / "wave.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    try:
        rc.add_image_note("imager", "wave.png", str(img), source="upload/wave.png")
        assert api.uploads, "no sync happened"
        assert "wave.png" in api.uploads[-1]["originals"], api.uploads[-1]
        # The Dataset must mirror the folder, or deleted categories come back on restart.
        assert api.uploads[-1]["kwargs"].get("delete_patterns") == "*", api.uploads[-1]
    finally:
        rc.describe_image = original_describe
        reset_sync()


@test
def test_no_orphan_original_when_storing_fails():
    main.DEV_MODE = True
    reset_sync("ok")
    rc._hf_api = FakeHfApi()
    original_describe, original_add = rc.describe_image, rc._add_note
    rc.describe_image = lambda path, model=None: "這是一份【物理】科的筆記。"
    def failing_add(*a, **k):
        raise RuntimeError("embedding failed")
    rc._add_note = failing_add
    img = Path(tempfile.mkdtemp()) / "broken.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        try:
            rc.add_image_note("imager2", "broken.png", str(img), source="upload/broken.png")
        except RuntimeError:
            pass
        assert rc.get_original("imager2", "broken.png") is None, "orphan original left behind"
    finally:
        rc.describe_image, rc._add_note = original_describe, original_add
        reset_sync()


@test
def test_original_response_varies_by_user():
    main.DEV_MODE = True
    reset_sync()
    src = Path(tempfile.mkdtemp()) / "pic.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\1" * 32)
    # An original is served only for a note that exists (it is located by the note's id).
    rc.add_text_note("viewer", "pic.png", "這是一份【物理】科的筆記。", source="upload/pic.png")
    rc.save_original("viewer", "pic.png", str(src))
    r = client.get("/api/notes/pic.png/original", headers={"X-Dev-User": "viewer"})
    assert r.status_code == 200, r.status_code
    assert "authorization" in r.headers.get("vary", "").lower(), r.headers
    other = client.get("/api/notes/pic.png/original", headers={"X-Dev-User": "someone-else"})
    assert other.status_code == 404, "another user could fetch this original"


# --- runner -----------------------------------------------------------------------------
def run():
    failed = 0
    for fn in RESULTS:
        try:
            note = fn()
            print(f"  {'SKIP' if note else 'PASS'}  {fn.__name__}" + (f"  — {note}" if note else ""))
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            print("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    main.id_token.verify_oauth2_token = _real_verify
    shutil.rmtree(STORE, ignore_errors=True)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run() else 0)

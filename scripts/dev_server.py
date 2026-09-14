"""Local dev server, isolated from the user's real notes.

`uvicorn app.main:app` picks up .env, which carries HF_TOKEN — so it would restore the
private Dataset over ./vectorstores on startup and push every local upload back to it.
That is fine in production and wrong on a laptop: test uploads would land in the real
store. This entry point clears HF_TOKEN and points the store at a scratch directory
before importing the app, so nothing here touches the private Dataset.

    python scripts/dev_server.py [--port 8000] [--store .devstore]

For the deployed Space, keep using `uvicorn app.main:app` (see Dockerfile).
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--store", default=".devstore")
args = ap.parse_args()

# Run from the project folder whatever the caller's working directory is, so the scratch
# store and the app import resolve there — a launcher in a parent folder once started the
# app with plain uvicorn instead, which restored the real Dataset.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)                       # `app` lives in the project root, not scripts/

os.environ["HF_TOKEN"] = ""                  # no Dataset restore, no Dataset push
os.environ["VECTORSTORE_ROOT"] = args.store    # never the real ./vectorstores
os.environ["NOTEHANDLER_DEV"] = "1"            # mounts /ui-editor, which writes to disk

import uvicorn  # noqa: E402  (after the env is set)

print(f"[dev] store={args.store}  HF sync disabled  UI editor at /ui-editor")
uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, reload=False)

"""Strip re-reading: cut each R0 page into horizontal bands at blank rows, enlarge each band,
read the bands one by one, and join the readings in order.

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/run_tiles.py --model gemini-3.1-flash-lite --out eval/r0/results/tiles_gemini31
    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/run_tiles.py --model claude-opus-5-5 --out eval/r0/results/tiles_opus55

The hypothesis (Lee et al. 2026, FaithC4): a model rewrites a character toward what the
surrounding text makes likely. A band shows less of that context and more pixels per
character. Everything else is the whole-page condition: the app's own image prompt (Gemini
through rag_core.describe_image, Claude through the same text), the same model settings.
Cuts fall only inside blank bands, so no line of writing is split; a diagram may be.

Writes --out: strips/R0_pN_k.jpg, raw/R0_pN_k.md, transcripts/R0_pN.md (the bands' bodies
joined), meta.json. Score with compare_runs.py.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from run_vote import PAGES                                        # noqa: E402
from score import model_body                                      # noqa: E402

DARK, BLANK_FRAC, MIN_GAP, MIN_H, SCALE = 140, 0.003, 15, 300, 2


def strips(img: Image.Image) -> list:
    """Bands of the page cut at the middle of blank row runs, each at least MIN_H tall;
    bands with no ink are dropped."""
    a = np.asarray(img.convert("L"))
    blank = (a < DARK).mean(axis=1) < BLANK_FRAC
    cuts, start = [0], None
    for y, b in enumerate(list(blank) + [False]):
        if b and start is None:
            start = y
        elif not b and start is not None:
            if y - start >= MIN_GAP:
                cuts.append((start + y) // 2)
            start = None
    cuts.append(len(a))
    bands, top = [], 0
    for c in cuts[1:]:
        if c - top >= MIN_H or c == len(a):
            if (a[top:c] < DARK).mean() > 0.0005:            # has ink
                bands.append(img.crop((0, top, img.width, c)))
            top = c
    return bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="medium", help="Claude only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    for d in ("strips", "raw", "transcripts"):
        (out / d).mkdir(parents=True, exist_ok=True)

    from run_claude import app_prompt
    prompt = app_prompt()
    if args.model.startswith("claude"):
        from run_claude import client, read_image
        c = client()
        read = lambda path: read_image(c, args.model, args.effort, prompt, path.read_bytes())
    else:
        os.environ["HF_TOKEN"] = ""                  # same isolation as run_baseline.py
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        from app import rag_core
        from run_baseline import retry
        read = lambda path: (retry(lambda: rag_core.describe_image(str(path), model=args.model), path.name), {})

    meta_f = out / "meta.json"
    usage = json.loads(meta_f.read_text(encoding="utf-8"))["usage"] if meta_f.is_file() else {}
    for p in PAGES:
        bands = strips(Image.open(HERE / f"R0_p{p}.jpg"))
        bodies = []
        for k, band in enumerate(bands, 1):
            name = f"R0_p{p}_{k}"
            raw = out / "raw" / f"{name}.md"
            if not raw.is_file():                    # resumable
                path = out / "strips" / f"{name}.jpg"
                band.resize((band.width * SCALE, band.height * SCALE), Image.LANCZOS).save(path, quality=92)
                text, u = read(path)
                raw.write_text(text, encoding="utf-8")
                usage[name] = u
                print(f"{name} ({band.height}px): {u}", flush=True)
            bodies.append(model_body(raw.read_text(encoding="utf-8")).strip())
        (out / "transcripts" / f"R0_p{p}.md").write_text("\n\n".join(bodies) + "\n", encoding="utf-8")
        meta_f.write_text(json.dumps({
            "finished": datetime.now().isoformat(timespec="seconds"), "model": args.model,
            "effort": args.effort if args.model.startswith("claude") else None,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "scale": SCALE,
            "usage": usage, "note": "strips cut at blank rows, each enlarged and read alone",
        }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

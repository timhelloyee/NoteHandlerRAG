"""Read the R0 pages with an open model (Qwen3-VL) on a rented GPU, to compare with Gemini.

    1. here:     python eval/r0/run_qwen.py --bundle        -> results/qwen_bundle/ to copy over
    2. GPU box:  pip install -U transformers accelerate pillow
                 python run_qwen.py                         -> qwen_out/ to copy back
    3. here:     python eval/r0/run_qwen.py --score eval/r0/results/qwen_out

The GPU box needs only this file, prompt.txt and the 8 scans: no app code, no API keys.
prompt.txt is the app's own image prompt, cut from describe_image() in rag_core.py, so
both models get the same instruction. Each page is read once at temperature 0 (t0/) and
--n times at 0.7 with Gemini's top_p/top_k (reads/), the same as run_vote.py.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def bundle():
    src = (HERE.parent.parent / "app" / "rag_core.py").read_text(encoding="utf-8")
    body = src[src.index("def describe_image"):]
    prompt = re.search(r'prompt = """(.*?)"""', body, re.S).group(1).replace("\\\\", "\\")
    out = HERE / "results" / "qwen_bundle"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    shutil.copy(__file__, out / "run_qwen.py")
    for p in range(1, 9):
        shutil.copy(HERE / f"R0_p{p}.jpg", out)
    print(f"{out}  (prompt sha256 {hashlib.sha256(prompt.encode()).hexdigest()[:8]})")


def read_all(args):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    prompt = (HERE / "prompt.txt").read_text(encoding="utf-8")
    out = Path(args.out)
    for d in ("t0", "reads"):
        (out / d).mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype="auto", device_map="auto")
    proc = AutoProcessor.from_pretrained(args.model)

    def read(page, temperature, seed):
        img = Image.open(HERE / f"R0_p{page}.jpg").convert("RGB")
        if args.max_side:                            # local smoke test only
            img.thumbnail((args.max_side, args.max_side))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
        sampling = (dict(do_sample=False) if temperature == 0 else
                    dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=20))
        torch.manual_seed(seed)
        ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, **sampling)
        return proc.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    jobs = [(p, 0, out / "t0" / f"R0_p{p}.md") for p in args.pages]
    jobs += [(p, k, out / "reads" / f"R0_p{p}_{k}.md") for p in args.pages for k in range(1, args.n + 1)]
    for p, k, f in jobs:
        if f.is_file():                              # resumable if the session drops
            continue
        print(f"p{p} {'t0' if k == 0 else f'#{k}'}", flush=True)
        f.write_text(read(p, 0 if k == 0 else args.temperature, k), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "model": args.model, "temperature": args.temperature, "n": args.n, "pages": args.pages,
        "max_side": args.max_side, "transformers": transformers.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }, indent=2), encoding="utf-8")


def score(run: Path):
    sys.path.insert(0, str(HERE))
    from difflib import SequenceMatcher
    from run_vote import PAGES, detection, vote, wrong_positions
    from run_vote import score as vote_score
    from score import edit_counts, gold_text_sections, model_body, text_only

    gold = gold_text_sections(HERE / "標準辨識稿.md")
    load = lambda f: text_only(model_body(f.read_text(encoding="utf-8")))
    gem_dir = HERE / "results" / "baseline" / "transcripts"
    n = err_q = err_g = both = 0
    det, misreads = {}, []
    for p in PAGES:
        ref, qwen, gem = text_only(gold[p]), load(run / "t0" / f"R0_p{p}.md"), load(gem_dir / f"R0_p{p}.md")
        n += len(ref)
        err_q += sum(edit_counts(ref, qwen)[:2])
        err_g += sum(edit_counts(ref, gem)[:2])
        both += len(wrong_positions(ref, qwen) & wrong_positions(ref, gem))
        # Cross-model check: flag every Gemini character Qwen reads differently.
        _, _, flags = vote([gem, qwen])
        for key, v in detection(ref, gem, flags).items():
            det[key] = det.get(key, 0) + v
        misreads += [f"p{p} {ref[i1:i2]}→{qwen[j1:j2] or '∅'}" for op, i1, i2, j1, j2 in
                     SequenceMatcher(None, ref, qwen, autojunk=False).get_opcodes()
                     if op in ("replace", "delete") and i2 - i1 <= 4 and j2 - j1 <= 6]
    tp, fp, fn = det.get("tp", 0), det.get("fp", 0), det.get("fn", 0)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    report = [
        f"# Qwen 與 Gemini 比較（{meta['model']}，{meta['gpu']}）", "",
        "## 單次讀，溫度 0", "",
        "| 模型 | 漏/錯字 | 比例 |", "|---|---|---|",
        f"| Gemini（baseline） | {err_g}/{n} | {err_g / n:.1%} |",
        f"| Qwen | {err_q}/{n} | {err_q / n:.1%} |", "",
        f"- 兩個模型錯在同一個標準字：{both} 字（模型共同的習慣，跨模型比對抓不到）",
        f"- Qwen 讀錯的字：{'；'.join(misreads) or '無'}", "",
        "## 「Gemini 和 Qwen 不一致＝可疑」當自動找錯工具（找 Gemini 的錯）", "",
        f"- 被標記可疑 {tp + fp} 字，其中 Gemini 真的讀錯 {tp} 字 → 精確率 **{tp / max(tp + fp, 1):.0%}**",
        f"- Gemini 真的讀錯 {tp + fn} 字，其中被標記 {tp} 字 → 召回率 **{tp / max(tp + fn, 1):.0%}**",
        "- 對照：Gemini 自己讀 5 次投票是精確率 18%、召回率 36%（vote_t07）", "",
    ] + vote_score(run, meta["n"])
    (run / "score.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="store_true")
    ap.add_argument("--score", metavar="RUN")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--pages", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-side", type=int, default=0, help="shrink the scan (smoke test only)")
    ap.add_argument("--out", default="qwen_out")
    args = ap.parse_args()
    if args.bundle:
        bundle()
    elif args.score:
        score(Path(args.score))
    else:
        read_all(args)


if __name__ == "__main__":
    main()

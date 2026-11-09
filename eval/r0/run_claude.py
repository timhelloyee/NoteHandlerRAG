"""Read the R0 pages with Claude through the API, and score it beside Gemini and Qwen.

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/run_claude.py            # read + score
    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/run_claude.py --score-only

Same instruction as the other two models: the app's image prompt, cut from rag_core.py
(run_qwen.bundle). Claude takes no temperature, so each page is read once at the model's
own sampling; thinking cannot be turned off on Opus 5.5, so effort is fixed and recorded.
No refusal fallback: a fallback would answer with another model and mix the conditions,
so a refusal is recorded as it is. Needs ANTHROPIC_API_KEY in .env.

Writes --out (default results/claude_opus55): transcripts/R0_pN.md, meta.json, score.md.
"""
import argparse
import base64
import hashlib
import json
import re
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_vote import PAGES, detection, vote, wrong_positions  # noqa: E402
from score import edit_counts, gold_text_sections, model_body, text_only  # noqa: E402


def read_all(out: Path, model: str, effort: str, prompt: str):
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(HERE.parent.parent / ".env")
    client = anthropic.Anthropic()
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    usage = {}
    for p in PAGES:
        f = out / "transcripts" / f"R0_p{p}.md"
        if f.is_file():                              # resumable
            continue
        img = base64.standard_b64encode((HERE / f"R0_p{p}.jpg").read_bytes()).decode()
        r = client.messages.create(
            model=model, max_tokens=16000, output_config={"effort": effort},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}},
                {"type": "text", "text": prompt}]}])
        text = "".join(b.text for b in r.content if b.type == "text")
        usage[p] = {"input": r.usage.input_tokens, "output": r.usage.output_tokens, "stop": r.stop_reason}
        print(f"p{p}: {r.stop_reason}, in {r.usage.input_tokens}, out {r.usage.output_tokens}", flush=True)
        f.write_text(text, encoding="utf-8")
    return usage


def score(out: Path) -> list:
    gold = gold_text_sections(HERE / "標準辨識稿.md")
    load = lambda f: text_only(model_body(f.read_text(encoding="utf-8")))
    res = HERE / "results"
    others = {"Gemini": res / "baseline" / "transcripts", "Qwen": res / "qwen_out" / "t0"}
    n, err, shared, misreads = 0, {k: 0 for k in ("Claude", "Jury", *others)}, {k: 0 for k in others}, []
    det = {}
    for p in PAGES:
        ref = text_only(gold[p])
        reads = {"Claude": load(out / "transcripts" / f"R0_p{p}.md"),
                 **{k: load(d / f"R0_p{p}.md") for k, d in others.items()}}
        n += len(ref)
        for k, hyp in reads.items():
            err[k] += sum(edit_counts(ref, hyp)[:2])
        mine = wrong_positions(ref, reads["Claude"])
        for k in others:
            shared[k] += len(mine & wrong_positions(ref, reads[k]))
        jury, pivot, flags = vote([reads["Gemini"], reads["Qwen"], reads["Claude"]])
        err["Jury"] += sum(edit_counts(ref, jury)[:2])
        for key, v in detection(ref, pivot, flags).items():
            det[key] = det.get(key, 0) + v
        misreads += [f"p{p} {ref[i1:i2]}→{reads['Claude'][j1:j2] or '∅'}" for op, i1, i2, j1, j2 in
                     SequenceMatcher(None, ref, reads["Claude"], autojunk=False).get_opcodes()
                     if op in ("replace", "delete") and i2 - i1 <= 4 and j2 - j1 <= 6]
    tp, fp, fn = det.get("tp", 0), det.get("fp", 0), det.get("fn", 0)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    row = lambda k, label: f"| {label} | {err[k]}/{n} | {err[k] / n:.1%} |"
    return [f"# Claude 與 Gemini、Qwen 比較（{meta['model']}，effort {meta['effort']}）", "",
            "| 讀法 | 漏/錯字 | 比例 |", "|---|---|---|",
            row("Gemini", "Gemini 3.1 Flash Lite（baseline，溫度 0）"), row("Qwen", "Qwen3-VL-8B（溫度 0）"),
            row("Claude", f"Claude（{meta['model']}）"), row("Jury", "三模型逐字多數決"), "",
            f"- Claude 和 Gemini 錯在同一個標準字：{shared['Gemini']} 字；和 Qwen：{shared['Qwen']} 字",
            f"- Claude 讀錯的字：{'；'.join(misreads) or '無'}", "",
            "## 「三個模型不一致＝可疑」當找錯工具（找多數決基準稿的錯）", "",
            f"- 標記 {tp + fp} 字，真的錯 {tp} 字 → 精確率 **{tp / max(tp + fp, 1):.0%}**",
            f"- 真的錯 {tp + fn} 字，被標記 {tp} 字 → 召回率 **{tp / max(tp + fn, 1):.0%}**", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--out", default=str(HERE / "results" / "claude_opus55"))
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.score_only:
        src = (HERE.parent.parent / "app" / "rag_core.py").read_text(encoding="utf-8")   # as run_qwen.bundle
        prompt = re.search(r'prompt = """(.*?)"""', src[src.index("def describe_image"):], re.S).group(1).replace("\\\\", "\\")
        old = out / "meta.json"
        usage = json.loads(old.read_text(encoding="utf-8"))["usage"] if old.is_file() else {}
        usage.update(read_all(out, args.model, args.effort, prompt))
        (out / "meta.json").write_text(json.dumps({
            "finished": datetime.now().isoformat(timespec="seconds"), "model": args.model,
            "effort": args.effort, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "usage": usage, "note": "no temperature (API removed it); thinking always on; no fallback",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    report = score(out)
    (out / "score.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

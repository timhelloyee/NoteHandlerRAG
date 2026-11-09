"""Does an LLM judge agree with the mechanical key-fact verdicts, and does it accept look-alike words?

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/judge_test.py
    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/judge_test.py --score-only

Test answers, all built by rule, none written by a model:
  gold     the standard answer itself                          -> should be 正確
  similar  one key-fact character swapped for a look-alike that a model really misread
           on these pages (德/地, 線/腺, 形/型 …)               -> should not be 正確
  missing  one key fact cut out of the standard answer         -> should be 部分
  real     the 69 answers NotebookLM, NoteHandler and F3b gave -> no built-in truth;
           compared with the mechanical verdict

Accuracy here means "consistent with the user's own notes", so a look-alike word is wrong
even when it is the textbook spelling. Two judge prompts: a usual "same meaning" one and
a strict one that says so. Writes results/judge_test/{verdicts.json, report.md}.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from run_baseline import parse_questions, retry                     # noqa: E402
from score import (gold_text_sections, model_body, notebooklm_answers,  # noqa: E402
                   notehandler_answers, parse_facts, score_answer, standard_answers, text_only)

OUT = HERE / "results" / "judge_test"
MODEL = "gemini-3.1-flash-lite"
VERDICTS = ("正確", "部分", "錯誤")
ABSTAIN_GOLD = "筆記中沒有提到這個內容。"

BASE = """你是評分老師。根據標準答案，判斷學生回答的對錯。

題目：{q}
標準答案：{gold}
學生回答：{a}

"""
PROMPTS = {
    "一般": BASE + "只要意思和標準答案相符就算正確；缺少部分重點算部分；意思不符算錯誤。\n只輸出一個詞：正確、部分或錯誤。",
    "嚴格": BASE + ("這裡的「正確」是指和使用者自己的筆記一致。專有名詞、譯名、數字與用字必須和標準答案相同，"
                    "寫法不同就算錯，即使是同義詞、常見寫法或課本上的寫法；只有語序與連接詞可以不同。\n"
                    "全部重點都對算正確；有對有錯或缺少重點算部分；都不對算錯誤。\n只輸出一個詞：正確、部分或錯誤。"),
}


FUNCTION = set("的之是為乃時")                     # swapping these changes no fact


def confusions() -> Counter:
    """(gold char, read char) -> how often a saved transcript misread one as the other."""
    gold = gold_text_sections(HERE / "標準辨識稿.md")
    res = HERE / "results"
    files = [f for d in ("baseline/transcripts", "prompt_f3b/transcripts", "qwen_out/t0")
             for f in sorted((res / d).glob("R0_p*.md"))]
    files += sorted((res / "vote_t07" / "reads").glob("*.md")) + sorted((res / "qwen_out" / "reads").glob("*.md"))
    pairs = Counter()
    for f in files:
        p = int(re.match(r"R0_p(\d+)", f.name).group(1))
        ref, hyp = text_only(gold[p]), text_only(model_body(f.read_text(encoding="utf-8")))
        for op, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
            if op == "replace" and i2 - i1 == j2 - j1 <= 2:
                pairs.update(zip(ref[i1:i2], hyp[j1:j2]))
    return Counter({p: n for p, n in pairs.items() if n >= 2})   # a one-off is often alignment noise


def look_alike(text: str, s: int, e: int, pairs: Counter):
    """(position, replacement) for the most often misread character in text[s:e]. A pair seen
    only the other way round (the model wrote a, the notes had b) is the fallback."""
    for direction in (0, 1):
        cands = [(n, i, pair[1 - direction]) for pair, n in pairs.items() for i in range(s, e)
                 if pair[direction] == text[i] and text[i] not in FUNCTION]
        if cands:
            _, i, b = max(cands)
            return i, b
    return None


def raw_facts() -> dict:
    """關鍵事實 as written (not normalised), so spans can be found in the standard answer."""
    out = {}
    for line in (HERE / "關鍵事實.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and re.fullmatch(r"[ABC]\d+", cells[0]):
            out[cells[0]] = [f.split("／") for f in cells[1].split("；")]
    return out


def span(text: str, variants: list):
    for v in variants:
        m = re.search(r"\s*".join(map(re.escape, v)), text, re.I)
        if m:
            return m
    return None


def build_items() -> list:
    qs = parse_questions(HERE / "NotebookLM_提問單.md")
    question = {q["id"]: q["question"] for q in qs}
    gold = {k: v[1] for k, v in standard_answers(HERE / "題組與標準答案.md", qs).items()}
    gold.update({q: ABSTAIN_GOLD for q in question if q.startswith("D")})
    pairs, facts = confusions(), raw_facts()
    items = []

    def add(kind, qid, answer, truth, note=""):
        items.append({"id": f"{kind}:{qid}:{len(items)}", "kind": kind, "qid": qid, "q": question[qid],
                      "gold": gold[qid], "answer": answer, "truth": truth, "note": note})

    for qid in question:
        add("gold", qid, gold[qid], "正確")
        for k, variants in enumerate(facts.get(qid, [])):
            m = span(gold[qid], variants)
            if not m:
                continue
            s, e = m.span()
            swap = look_alike(gold[qid], s, e, pairs)
            if swap:
                j, b = swap
                swapped = gold[qid][:j] + b + gold[qid][j + 1:]
                add("similar", qid, swapped, "非正確", f"{gold[qid][s:e]}→{swapped[s:e]}")
            if len(facts[qid]) > 1:                  # every occurrence, or a repeat keeps the fact
                cut = re.sub(rf"(?<![A-Za-z]){re.escape(m.group(0))}(?![A-Za-z])", "", gold[qid])  # ADP, not NADP
                add("missing", qid, cut.strip("、，；→ "), "部分",
                    f"刪掉「{m.group(0)}」")
    systems = {"NotebookLM": notebooklm_answers(HERE / "NotebookLM_提問單.md"),
               "NoteHandler": notehandler_answers(HERE / "results" / "baseline"),
               "F3b": notehandler_answers(HERE / "results" / "prompt_f3b")}
    for name, answers in systems.items():
        for qid in question:
            add("real", qid, answers[qid][0], "", name)
    keys = parse_facts(HERE / "關鍵事實.md")
    for it in items:
        it["mechanical"] = score_answer(it["answer"], set(), keys[it["qid"]], it["qid"].startswith("D"))["verdict"]
    return items


def judge(items: list) -> dict:
    os.environ["HF_TOKEN"] = ""                      # no store is touched; keep it that way
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from app import rag_core                         # noqa: E402

    path = OUT / "verdicts.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    todo = [(it, name) for it in items for name in PROMPTS if f"{it['id']}|{name}" not in done]

    def one(job):
        it, name = job
        text = retry(lambda: rag_core._llm_invoke(
            PROMPTS[name].format(q=it["q"], gold=it["gold"], a=it["answer"]), model=MODEL), it["id"])
        m = re.search("|".join(VERDICTS), text)
        return f"{it['id']}|{name}", m.group(0) if m else "?"

    with ThreadPoolExecutor(4) as pool:
        for n, (key, v) in enumerate(pool.map(one, todo), 1):
            done[key] = v
            if n % 20 == 0:
                path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{n}/{len(todo)}", flush=True)
    path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    return done


def hit(truth: str, verdict: str) -> bool:
    return verdict != "正確" if truth == "非正確" else verdict == truth


def report(items: list, done: dict) -> list:
    judges = ["機械判定"] + [f"LLM（{n}）" for n in PROMPTS]
    verdict = lambda it, j: it["mechanical"] if j == "機械判定" else done[f"{it['id']}|{j[4:-1]}"]
    lines = [f"# LLM 評審 vs 機械判定（評審模型 {MODEL}，溫度 0）", "",
             "## 一、規則造出的答案（對錯已知）", "",
             "| 答案類型 | 應判 | 題數 | " + " | ".join(judges) + " |",
             "|---|---|---|" + "---|" * len(judges)]
    for kind, truth in (("gold", "正確"), ("similar", "不是正確"), ("missing", "部分")):
        group = [it for it in items if it["kind"] == kind]
        cells = [f"{sum(hit(it['truth'], verdict(it, j)) for it in group)}/{len(group)}" for j in judges]
        lines.append(f"| {kind} | {truth} | {len(group)} | " + " | ".join(cells) + " |")
    lines += ["", "### 相似字被判成「正確」的題（相似詞誤判）", "",
              "| 題 | 換掉的字 | " + " | ".join(judges) + " |", "|---|---|" + "---|" * len(judges)]
    for it in (it for it in items if it["kind"] == "similar"):
        lines.append(f"| {it['qid']} | {it['note']} | " + " | ".join(verdict(it, j) for j in judges) + " |")
    lines += ["", "## 二、三個系統的真實回答（69 題）：LLM 與機械判定是否一致", ""]
    real = [it for it in items if it["kind"] == "real"]
    for j in judges[1:]:
        c = Counter((it["mechanical"], verdict(it, j)) for it in real)
        agree = sum(v for (a, b), v in c.items() if a == b)
        lines += [f"**{j}**：一致 {agree}/{len(real)}", "",
                  "| 機械＼LLM | " + " | ".join(VERDICTS) + " |", "|---|---|---|---|"]
        lines += [f"| {a} | " + " | ".join(str(c[(a, b)]) for b in VERDICTS) + " |" for a in VERDICTS]
        diff = [f"{it['note']} {it['qid']}（機械 {it['mechanical']}，LLM {verdict(it, j)}）"
                for it in real if it["mechanical"] != verdict(it, j)]
        lines += ["", "不一致：" + ("；".join(diff) or "無"), ""]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = build_items()
    (OUT / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    done = (json.loads((OUT / "verdicts.json").read_text(encoding="utf-8")) if args.score_only
            else judge(items))
    rep = report(items, done)
    (OUT / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    assert hit("非正確", "部分") and not hit("非正確", "正確") and hit("部分", "部分")
    main()

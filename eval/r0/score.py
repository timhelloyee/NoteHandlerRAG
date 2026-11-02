"""Mechanical first-pass scoring for R0. Humans confirm the 部分 rows and every flag.

    C:/Users/user/anaconda3/envs/ragtraining/python.exe eval/r0/score.py [--run eval/r0/results/baseline]

Answer layer, for both systems, against 關鍵事實.md:
    verdict   正確 / 部分 / 錯誤 from which key facts appear (and in order, where required)
    flags     不可出現 words found: a human decides (they may sit inside a negation)
    citation  cited pages vs the evidence pages; an abstention must cite nothing
Recognition layer, NoteHandler only (NotebookLM never shows what it read), against 標準辨識稿.md:
    CER on the 文字 section of each page, punctuation ignored. The model's transcript also
    describes the diagrams, which the 文字 section does not contain, so insertions are
    reported apart: 漏/錯字率 = (substitutions + deletions) / gold length is the figure that
    the extra diagram text cannot inflate.

Writes <run>/score.md. Nothing here calls a model.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁺", "0123456789+")


def norm(s: str) -> str:
    """The normalisation 關鍵事實.md specifies: drop markup, unify widths, case and subscripts."""
    s = unicodedata.normalize("NFKC", s or "").translate(SUB).lower()
    s = re.sub(r"\\x?(?:right)?arrow|\\to\b", "→", s)
    s = re.sub(r"\\[a-z]+", "", s)                  # any other LaTeX command: markup, not content
    return re.sub(r"[\s*$\\{}^_]", "", s)


def parse_facts(path: Path) -> dict:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and re.fullmatch(r"[A-D]\d+", cells[0]):
            qid, facts, forbid, order, pages = cells
            rows[qid] = {
                "facts": [[norm(v) for v in f.split("／")] for f in facts.split("；")],
                "forbid": [] if forbid == "—" else [norm(v) for v in forbid.split("／")],
                "ordered": order == "是",
                "pages": {int(p) for p in re.findall(r"p(\d+)", pages)},
            }
    return rows


def first_hit(text: str, variants: list) -> int:
    hits = [text.find(v) for v in variants if v and v in text]
    return min(hits) if hits else -1


def score_answer(answer: str, cited: set, key: dict, abstain: bool) -> dict:
    t = norm(answer)
    pos = [first_hit(t, f) for f in key["facts"]]
    found = sum(p >= 0 for p in pos)
    if found == len(pos):
        verdict = "正確"
        if key["ordered"] and pos != sorted(pos):
            verdict = "部分"
    else:
        verdict = "部分" if found else "錯誤"
    flags = [v for v in key["forbid"] if v in t]
    if abstain:
        citation = "正確" if not cited else "不該有出處"
    elif not cited:
        citation = "沒有出處"
    elif cited <= key["pages"]:
        citation = "正確"
    else:
        citation = "有錯頁" if cited & key["pages"] else "全錯"
    return {"verdict": verdict, "found": f"{found}/{len(pos)}", "flags": flags, "citation": citation,
            "cited": sorted(cited)}


# --- the two answer sources ----------------------------------------------------------
def notebooklm_answers(sheet: Path) -> dict:
    """{qid: (answer, cited pages)}. The closing suggestion paragraph (💡/🌋/🌱/🫀 …) is the
    product offering a next question, not part of the answer, so it is dropped."""
    out = {}
    text = sheet.read_text(encoding="utf-8")
    for m in re.finditer(r"^### ([A-D]\d+)\n.*?\n\n- 回答：(.*?)\n- 引用來源：(.*?)$", text, re.S | re.M):
        qid, body, cites = m.groups()
        paras = [p for p in body.strip().split("\n\n")]
        if paras and unicodedata.category(paras[-1].strip()[:1] or " ") == "So":
            paras = paras[:-1]
        out[qid] = ("\n\n".join(paras), {int(n) for n in re.findall(r"R0_p(\d+)", cites)})
    return out


def notehandler_answers(run: Path) -> dict:
    out = {}
    for r in json.loads((run / "answers.json").read_text(encoding="utf-8")):
        pages = {int(n) for s in r.get("sources") or [] for n in re.findall(r"R0_p(\d+)", s["id"])}
        out[r["id"]] = (r.get("answer") or "", pages)
    return out


# --- recognition ------------------------------------------------------------------------
def text_only(s: str) -> str:
    """Characters that carry content: CJK, letters, digits. Punctuation and markup are not
    scored — 「，」 against 「、」 is not a recognition error a reader would notice."""
    s = norm(s)
    return "".join(ch for ch in s if ch.isalnum())


def edit_counts(ref: str, hyp: str) -> tuple:
    """Levenshtein with the S/D/I split. Pages are a few hundred characters: O(n·m) is fine."""
    n, m = len(ref), len(hyp)
    prev = [(j, 0, 0, j) for j in range(m + 1)]            # (cost, S, D, I)
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            c, s, d, ins = prev[j - 1]
            best = (c, s, d, ins) if ref[i - 1] == hyp[j - 1] else (c + 1, s + 1, d, ins)
            c, s, d, ins = prev[j]
            best = min(best, (c + 1, s, d + 1, ins))
            c, s, d, ins = cur[j - 1]
            best = min(best, (c + 1, s, d, ins + 1))
            cur.append(best)
        prev = cur
    return prev[m][1:]


def gold_text_sections(path: Path) -> dict:
    """{page: text of its '### 文字' section} from 標準辨識稿.md."""
    out, page, grab, buf = {}, None, False, []
    for line in path.read_text(encoding="utf-8").splitlines() + ["## end"]:
        m = re.match(r"^## p(\d+)|^## end", line)
        if m:
            if page:
                out[page] = "\n".join(buf)
            page, grab, buf = (int(m.group(1)) if m.group(1) else None), False, []
        elif line.startswith("### "):
            grab = line.strip() == "### 文字"
        elif grab:
            buf.append(line)
    return out


def model_body(text: str) -> str:
    """The transcript minus the furniture the prompt asks for (subject line, keyword line)."""
    text = re.sub(r"這是一份[【\[]?[^】\]\s]+?[】\]]?科的筆記。?", "", text, count=1)
    return re.sub(r"關鍵字[:：].*$", "", text, flags=re.S)


# --- grounding: content the notes do not support, by three fixed rules ---------------------
# Every rule compares an answer with the HAND-TYPED transcript, never with a system's own
# reading: otherwise a misread word would count as "in the notes". The rules are fixed
# before scoring and run the same way on both systems; validate() measures how often
# each one catches a known-bad sentence and how often it flags a known-good one.
SUBJECT_PAGES = {"地科": [1, 2, 3], "生物": [4, 5, 6], "歷史": [7, 8]}
CITE_MARK = re.compile(r"〔來源：[^〕]*〕")
LIST_NUM = re.compile(r"(?m)^\s*\d+[.、)]\s*")
NAME_PUNCT = re.compile(r"[、，,（）()]")
CLAUSE_END = re.compile(r"[。！？；!?;\n]")
# A relation is claimed when two diagram nodes are joined by one of these within a clause.
CONNECT = re.compile(r"→|->|⟶|⇒|導致|引發|引起|造成|帶動|衍生|促成|催生|產生|形成|變成|轉變成|轉變為|"
                     r"轉為|成為|進入|流向|流入|流經|流回|送到|送往|送回|回到|經由|經過|再到")
NEGATION = re.compile(r"不|沒有|並非|而非|無")
# Discourse words any answer may use. Fixed here, before scoring; content words are not.
GLUE = """根據 筆記 手寫 內容 參考資料 資料 來源 相關 資訊 訊息 如下 以下 整理 整合 主要 包括 包含 以及 其中 分別
依序 過程 結果 原因 說明 詳細 重點 經過 哪些 什麼 為何 因此 所以 而且 並且 同時 另外 此外 最終 最後 首先 接著
然後 屬於 代表 例如 細節 部分 圖中 圖示 圖表 標示 標註 記載 提到 提及 沒有 未 需要 可以 進行 透過 藉由 其 此
該 這 這些 一個 兩個 兩者 之間 關係 影響 作用 功能 定義 類型 特徵 例子 階段 路徑 路線 構造 物質 位置 方式 程度
回答 問題 依據 根據圖 如圖 由此 可知 即 也就是 也 就 會 能 是 為 的 在 與 和 及 或 而 但 並 被 讓 使 將 把 從 由
向 往 至 到 後 前 中 上 下 內 外 時 再 又 都 皆 均 各 每 等 等等 如 若 當 則 因 故 不 非 無 有 了 著 過 嗎 呢 吧"""


def gold_pages(path: Path) -> dict:
    """{page: {"cells": [text], "edges": [(src, dst, label, color)], "notes": [text]}}.
    Cells hold only handwriting; blockquote lines are the transcriber's notes, kept apart."""
    out, page, in_fig = {}, None, False
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## p(\d+)", line)
        if m:
            page, in_fig = int(m.group(1)), False
            out[page] = {"cells": [], "edges": [], "notes": []}
            continue
        if page is None or not line.strip():
            continue
        if line.startswith("### "):
            in_fig = line.startswith("### 圖")
            continue
        if line.startswith(">"):
            out[page]["notes"].append(line)
            continue
        if line.startswith("節點："):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: ") or cells[:2] == ["起點", "終點"]:
                continue
            if in_fig and len(cells) == 4:
                out[page]["edges"].append(tuple(cells))
                out[page]["cells"] += cells[:3]       # the colour cell is an annotation
            else:
                out[page]["cells"] += cells
            continue
        out[page]["cells"].append(line)
    return out


def bigrams(s: str) -> set:
    s = text_only(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


class Grounding:
    def __init__(self, gold_path: Path, questions: list, extra_glue: str = ""):
        self.pages = gold_pages(gold_path)
        words = (GLUE + " " + extra_glue).split()
        glue = set().union(*(bigrams(w) for w in words))
        self.glue_chars = {w for w in GLUE.split() if len(w) == 1}
        self.vocab = {}
        for subj, nums in SUBJECT_PAGES.items():
            v = set(glue)
            for n in nums:
                for c in self.pages[n]["cells"]:
                    v |= bigrams(c)
            for q in questions:
                if q["subject"] == subj:
                    v |= bigrams(q["question"])
            self.vocab[subj] = v
        # Diagram nodes, each reachable by its full name and by the name without (…).
        self.alias, self.adj, self.node_page = {}, {}, {}
        for n, pg in self.pages.items():
            for src, dst, *_ in pg["edges"]:
                self.adj.setdefault(src, set()).add(dst)
                for name in (src, dst):
                    self.node_page[name] = n
                    for form in {name, re.sub(r"（[^）]*）", "", name)}:
                        for key in {norm(form), NAME_PUNCT.sub("", norm(form))}:
                            if len(key) >= 2:
                                self.alias.setdefault(key, name)
        self.alias_by_len = sorted(self.alias, key=len, reverse=True)
        # Rule 3 attributes, read from the transcript: legend colours and node-text colour.
        legend = {}
        for c in self.pages[6]["cells"]:
            legend.update(dict(re.findall(r"(紅|藍)：(\S+?血)", c)))
        self.blood = {}
        for name in {x for e in self.pages[6]["edges"] for x in e[:2]}:
            colours = {e[3] for e in self.pages[6]["edges"] if name in e[:2]}
            if len(colours) == 1 and next(iter(colours)) in legend:
                self.blood[name] = legend[next(iter(colours))]
        m = re.search(r"節點文字都是(.)色", "".join(self.pages[6]["notes"]))
        self.text_colour = {name: m.group(1) for name in self.blood} if m else {}

    def mentions(self, t: str) -> list:
        out, i = [], 0
        while i < len(t):
            for a in self.alias_by_len:
                if t.startswith(a, i):
                    out.append((i, i + len(a), self.alias[a]))
                    i += len(a)
                    break
            else:
                i += 1
        return out

    def reachable(self, x: str, y: str) -> bool:
        seen, stack = set(), [x]
        while stack:
            n = stack.pop()
            for m in self.adj.get(n, ()):
                if m == y:
                    return True
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return False

    def check(self, answer: str, subject: str) -> dict:
        t = norm(CITE_MARK.sub("", answer))
        ms = self.mentions(t)
        # Rule 1: a relation between two nodes must be an arrow, or a path of arrows.
        bad_rel = []
        for (s1, e1, x), (s2, e2, y) in zip(ms, ms[1:]):
            gap = t[e1:s2]
            if (x == y or self.node_page[x] != self.node_page[y] or len(gap) > 15
                    or CLAUSE_END.search(gap) or NEGATION.search(gap) or not CONNECT.search(gap)):
                continue
            if y not in self.adj.get(x, ()) and not self.reachable(x, y):
                bad_rel.append(f"{x}→{y}")
        # Rule 3: a stated attribute must match the transcript.
        bad_attr = []
        for m in re.finditer(r"充氧血|缺氧血", t):
            before = [mm for mm in ms if mm[1] <= m.start()
                      and not re.search(r"[。！？；!?;\n，,：:]", t[mm[1]:m.start()])]
            if before and before[-1][2] in self.blood and self.blood[before[-1][2]] != m.group():
                bad_attr.append(f"{before[-1][2]}＝{m.group()}")
        for s, e, name in ms:
            m = re.match(r"[（(]?(紅|藍)字", t[e:e + 4])
            if m and name in self.text_colour and self.text_colour[name] != m.group(1):
                bad_attr.append(f"{name}＝{m.group(1)}字")
        # Rule 2: characters no bigram of the notes, the question or the glue list covers.
        c = text_only(LIST_NUM.sub("", CITE_MARK.sub("", answer)))
        v = self.vocab[subject]
        covered = [c[i] in self.glue_chars or c[max(i - 1, 0):i + 1] in v or c[i:i + 2] in v
                   for i in range(len(c))]
        spans = re.findall(r"#+", "".join("." if k else "#" for k in covered))
        span_text, pos = [], 0
        for i, k in enumerate(covered + [True]):
            if not k and (i == 0 or covered[i - 1]):
                pos = i
            if k and i > 0 and not covered[i - 1]:
                span_text.append(c[pos:i])
        return {"relations": bad_rel, "attributes": bad_attr,
                "outside_chars": sum(not k for k in covered), "chars": len(c), "spans": span_text}


def validate(g: Grounding, answer_key: dict) -> list:
    """Known-good and known-bad sentences for each rule, generated by fixed procedures.
    answer_key: {qid: (subject, standard answer)} — correct meaning, textbook wording."""
    rows = []
    # Rule 1: every arrow stated forwards is good; every pair with no path is bad.
    good = bad = good_hit = bad_hit = 0
    subj_of = {n: s for s, ps in SUBJECT_PAGES.items() for n in ps}
    for n, pg in g.pages.items():
        nodes = sorted({x for e in pg["edges"] for x in e[:2]})
        for x in nodes:
            for y in nodes:
                if x == y:
                    continue
                r = g.check(f"{x}導致{y}。", subj_of[n])["relations"]
                if y in g.adj.get(x, ()):
                    good, good_hit = good + 1, good_hit + bool(r)
                elif not g.reachable(x, y):
                    bad, bad_hit = bad + 1, bad_hit + bool(r)
    rows.append(("規則 1 圖的關係", f"{bad_hit}/{bad}", f"{good_hit}/{good}",
                 "每條箭頭寫成「X導致Y」為好句；兩節點間沒有任何路徑為壞句"))
    # Rule 3: each node with a known blood type, stated rightly and wrongly.
    good = bad = good_hit = bad_hit = 0
    flip = {"充氧血": "缺氧血", "缺氧血": "充氧血"}
    for name, b in g.blood.items():
        good_hit += bool(g.check(f"{name}裡是{b}。", "生物")["attributes"])
        bad_hit += bool(g.check(f"{name}裡是{flip[b]}。", "生物")["attributes"])
        good, bad = good + 1, bad + 1
    rows.append(("規則 3 屬性", f"{bad_hit}/{bad}", f"{good_hit}/{good}",
                 "血管節點的血液種類，照圖寫為好句、寫反為壞句"))
    # Rule 2: facts from outside the notes, and textbook names for things the notes spell
    # differently, are bad; the question set's standard answers (right meaning, own
    # wording) are good — so the false-alarm figure is the cost of paraphrase.
    outside = [("地科", "海嘯是海底地震造成海水大規模擾動形成的。"),
               ("地科", "花崗岩主要由石英、長石和雲母組成。"),
               ("地科", "錯動型邊界的例子是美國聖安地列斯斷層。"),
               ("地科", "常見地形是轉型斷層。"),
               ("地科", "地震規模每增加1，能量約增加31.6倍。"),
               ("生物", "葉綠素主要吸收紅光和藍紫光。"),
               ("生物", "上、下腔靜脈把血液送回右心房。"),
               ("生物", "光反應在葉綠餅中進行，屬於光依賴反應。"),
               ("歷史", "法國大革命發生在1789年。"),
               ("歷史", "動力由人力、畜力轉為蒸汽。"),
               ("歷史", "1825年史蒂芬生的蒸汽火車行駛於英國第一條公共鐵路。"),
               ("歷史", "工業革命始於十八世紀中葉的英格蘭。")]
    bad_hit = sum(g.check(s, subj)["outside_chars"] > 0 for subj, s in outside)
    good_hit = sum(g.check(a, subj)["outside_chars"] > 0 for subj, a in answer_key.values())
    rows.append(("規則 2 筆記外用詞", f"{bad_hit}/{len(outside)}", f"{good_hit}/{len(answer_key)}",
                 "壞句：筆記以外的事實與課本譯名；好句：題組標準答案（意思對、措辭不同，被標記代表改寫）"))
    return rows


def standard_answers(path: Path, questions: list) -> dict:
    subj = {q["id"]: q["subject"] for q in questions}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and re.fullmatch(r"[ABC]\d+", cells[0]):
            out[cells[0]] = (subj[cells[0]], re.sub(r"（[^）]*）", "", cells[2]))
    return out


def prompt_texts() -> str:
    """Both systems' answer instructions: words a system was told to use are not note content."""
    src = (HERE.parent.parent / "app" / "rag_core.py").read_text(encoding="utf-8")
    nh = re.search(r'PROMPT_TEMPLATE = """(.*?)"""', src, re.S).group(1)
    lm = re.search(r"```text\n(.*?)```", (HERE / "NotebookLM_提問單.md").read_text(encoding="utf-8"), re.S).group(1)
    return nh + " " + lm


def grounding_section(systems: list, g: Grounding, questions: list, val: list) -> list:
    subj = {q["id"]: q["subject"] for q in questions}
    out = ["", "## 沒有依據的內容（確定性規則）", "",
           "三條規則都拿回答對照**手打的標準稿**，兩個系統套用同一套規則，重跑結果相同。", "",
           "| 系統 | 錯誤關係（規則 1） | 屬性矛盾（規則 3） | 筆記外字數（規則 2） | 筆記外字比例 |", "|---|---|---|---|---|"]
    details = []
    for name, answers in systems:
        res = {qid: g.check(answers.get(qid, ("", set()))[0], subj[qid]) for qid in subj}
        rel = sum(len(r["relations"]) for r in res.values())
        att = sum(len(r["attributes"]) for r in res.values())
        oc, ch = sum(r["outside_chars"] for r in res.values()), sum(r["chars"] for r in res.values())
        out.append(f"| {name} | {rel} | {att} | {oc} | {oc / max(ch, 1):.1%} |")
        details += ["", f"### {name}", "", "| 題 | 錯誤關係 | 屬性矛盾 | 筆記外字數 | 筆記外片段 |", "|---|---|---|---|---|"]
        for qid, r in res.items():
            details.append(f"| {qid} | {'、'.join(r['relations'])} | {'、'.join(r['attributes'])} | "
                           f"{r['outside_chars']} | {'、'.join(r['spans'][:10])} |")
    out += details
    out += ["", "### 規則本身的可靠度（錯誤注入）", "",
            "| 規則 | 壞句抓到 | 好句誤報 | 怎麼產生 |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in val]
    out += ["", "限制：規則 2 只看用詞，抓不到用筆記裡的詞拼成的錯誤句子；規則 1 只認得標準稿裡的節點名稱，"
            "回答換了說法的節點不會被檢查（會漏抓，不會誤報）。"]
    return out


# --- report -----------------------------------------------------------------------------
def answer_table(name: str, answers: dict, keys: dict) -> tuple:
    lines = [f"### {name}", "", "| 題 | 判定 | 事實 | 出處 | 引用頁 | 人工複核 |", "|---|---|---|---|---|---|"]
    tally = {}
    for qid in keys:
        ans, cited = answers.get(qid, ("", set()))
        r = score_answer(ans, cited, keys[qid], abstain=qid.startswith("D"))
        tally.setdefault(qid[0], []).append(r)
        lines.append(f"| {qid} | {r['verdict']} | {r['found']} | {r['citation']} | "
                     f"{'、'.join(f'p{p}' for p in r['cited']) or '—'} | {'、'.join(r['flags']) or ''} |")
    return lines, tally


def summary_row(name: str, tally: dict) -> str:
    cells = []
    for t in "ABCD":
        rs = tally.get(t, [])
        cells.append(f"{sum(r['verdict'] == '正確' for r in rs)}/{len(rs)}")
    allr = [r for rs in tally.values() for r in rs]
    ok = sum(r["verdict"] == "正確" for r in allr)
    cite = sum(r["citation"] == "正確" for r in allr)
    flags = sum(bool(r["flags"]) for r in allr)
    return f"| {name} | " + " | ".join(cells) + f" | {ok}/{len(allr)} | {cite}/{len(allr)} | {flags} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE / "results" / "baseline"))
    args = ap.parse_args()
    run = Path(args.run)
    keys = parse_facts(HERE / "關鍵事實.md")
    assert len(keys) == 23, f"關鍵事實.md: expected 23 rows, got {len(keys)}"

    systems = [("NotebookLM", notebooklm_answers(HERE / "NotebookLM_提問單.md"))]
    if (run / "answers.json").is_file():
        systems.append((f"NoteHandler（{run.name}）", notehandler_answers(run)))

    out = ["# R0 機械初評", "", "程式初判；「部分」與「人工複核」欄有內容的題目要人工確認。", "",
           "## 答案層", "", "| 系統 | A 正確 | B 正確 | C 正確 | D 拒答正確 | 合計正確 | 出處正確 | 需複核 |",
           "|---|---|---|---|---|---|---|---|"]
    details = []
    for name, answers in systems:
        lines, tally = answer_table(name, answers, keys)
        out.append(summary_row(name, tally))
        details += [""] + lines
    out += details

    tdir = run / "transcripts"
    if tdir.is_dir():
        gold = gold_text_sections(HERE / "標準辨識稿.md")
        out += ["", "## 辨識層（只有 NoteHandler）", "",
                "| 頁 | 標準字數 | 錯字 S | 漏字 D | 多字 I | CER | 漏/錯字率 |", "|---|---|---|---|---|---|---|"]
        tot = [0, 0, 0, 0]
        for p in sorted(gold):
            f = tdir / f"R0_p{p}.md"
            if not f.is_file():
                continue
            ref, hyp = text_only(gold[p]), text_only(model_body(f.read_text(encoding="utf-8")))
            s, d, i = edit_counts(ref, hyp)
            n = len(ref)
            tot = [tot[0] + n, tot[1] + s, tot[2] + d, tot[3] + i]
            out.append(f"| p{p} | {n} | {s} | {d} | {i} | {(s + d + i) / n:.1%} | {(s + d) / n:.1%} |")
        n, s, d, i = tot
        out.append(f"| 合計 | {n} | {s} | {d} | {i} | {(s + d + i) / n:.1%} | {(s + d) / n:.1%} |")
        out += ["", "多字 I 包含模型對圖的文字描述（標準稿的「文字」段不含圖），所以 CER 偏高是預期的；"
                "比較辨識品質時看「漏/錯字率」。"]

    from run_baseline import parse_questions
    questions = parse_questions(HERE / "NotebookLM_提問單.md")
    g = Grounding(HERE / "標準辨識稿.md", questions, extra_glue=prompt_texts())
    val = validate(g, standard_answers(HERE / "題組與標準答案.md", questions))
    out += grounding_section(systems, g, questions, val)

    run.mkdir(parents=True, exist_ok=True)
    (run / "score.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:12]))
    print(f"-> {run / 'score.md'}")


if __name__ == "__main__":
    # Self-check of the pieces the verdicts rest on.
    assert norm("NADP^{+}") == norm("NADP⁺") == "nadp+" and "nadp+" not in norm("NADPH")
    assert norm("$H_2O$") == norm("H₂O") == "h2o"
    assert norm(r"\xrightarrow{\text{光}}") == "→光" and norm(r"\(O_2\)") == "(o2)"
    assert edit_counts("abc", "axc") == (1, 0, 0) and edit_counts("abc", "ac") == (0, 1, 0)
    k = {"facts": [["a"], ["b"]], "forbid": [], "ordered": True, "pages": {1}}
    assert score_answer("b then a", {1}, k, False)["verdict"] == "部分"
    assert score_answer("a then b", {1, 2}, k, False)["citation"] == "有錯頁"
    main()

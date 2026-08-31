"""Export the B0 (oracle) working kit: the problem set plus one editable file per page.

B0 is the ceiling condition in the plan's experiment matrix: human-corrected text with a
human-marked structure tree. It is the only configuration that can separate "structure
helps" from "the extraction just got more accurate", which is the question a judge will
ask first. Everything else in the matrix is comparable only against it.

Structure is expressed as nested Markdown headings, because chunk_by_structure() already
consumes exactly that — so the corrected files feed the experiment with no extra tooling.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "b0"
PAGES = OUT / "pages"

corpus = json.loads((ROOT / "eval/corpus/corpus_b1.json").read_text(encoding="utf-8"))
probes = json.loads((ROOT / "eval/chunk_cases.json").read_text(encoding="utf-8"))["cases"]

by_page = {}
for p in probes:
    by_page.setdefault(p["page"], []).append(p)

PAGES.mkdir(parents=True, exist_ok=True)
written = 0
for note in corpus:
    for pg in note["page_texts"]:
        if pg["blank"]:
            continue
        pid = f"{note['file']}#p{pg['page']:02d}"
        safe = pid.replace("#", "__").replace(".pdf", "")
        mine = by_page.get(pid, [])
        body = re.sub(r"關鍵字[:：].*", "", pg["text"]).strip()
        spans = "".join(f"\n- `{p['span']}`  （{p['id']}：{p['question']}）" for p in mine)
        header = [
            f"<!-- {pid} | source: pdf/{note['file']} page {pg['page']} -->",
            f"# {pid}",
            "",
            "## 這一頁的正解關鍵詞" if mine else "## 這一頁沒有對應的題目",
            ("下列詞是這一頁所回答的題目的正解。修正時請保留這些詞的原樣，"
             "否則該題無法計分：" + spans) if mine else
            "這一頁不是任何題目的正解，但仍會進入索引作為干擾項，所以一樣需要校正。",
            "",
            "## 修正後的內容（請直接改下面這一段）",
            "",
            "規則：",
            "1. 對照原始 PDF 這一頁，把辨識錯的字、公式、數字改正確。",
            "2. 用巢狀 Markdown 標題標出這一頁「本來就有」的層級：",
            "   `#` 整頁主題、`##` 大標、`###` 次標，條列用 `-` 並以縮排表示層級。",
            "   標題請用筆記上實際寫的字，不要自己發明分類。",
            "3. 只寫這一頁看得到的東西，不要補充其他頁或課本內容。",
            "4. 看不懂的地方寫「（無法辨識）」，不要猜。",
            "",
            "---",
            "",
        ]
        (PAGES / f"{safe}.md").write_text(
            "\n".join(header) + body + "\n", encoding="utf-8")
        written += 1

# --- the problem set, human readable -----------------------------------------
lines = ["# NoteHandler 檢索題組（B0 用）", "",
         f"共 {len(probes)} 題，涵蓋 {len({p['page'] for p in probes})} 個正解頁面。", "",
         "每一題的「正解頁」是唯一含有該關鍵詞的頁面（已用程式驗證）。",
         "評分方式：問題向量化後檢索，看含有正解關鍵詞的片段排在第幾名。", "",
         "| # | 題目 | 型別 | 正解頁 | 正解關鍵詞 |", "|---|---|---|---|---|"]
for i, p in enumerate(probes, 1):
    lines.append(f"| {i} | {p['question']} | {p['type']} | `{p['page']}` | `{p['span']}` |")
(OUT / "probes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

readme = f"""# B0（oracle）校正說明

## 這是什麼

B0 是實驗矩陣裡的**天花板**組：人工校正過的文字 ＋ 人工標註的結構樹。
它是唯一能分辨「結構有用」和「只是辨識變準了」的組別——而這正是評審一定會問的問題。
沒有 B0，B1 與 B4 之間看不出差異這件事就無法解讀：可能是結構不值錢，
也可能是 LLM 自己推的結構不準。兩者的處置完全相反。

## 你要做的事

`pages/` 底下有 {written} 個檔案，一頁一個。每個檔案裡是目前 Gemini 的辨識結果。
請對照原始 PDF：

1. **改錯字**：手寫辨識常錯，公式與數字尤其要看。
2. **標結構**：用 `#` / `##` / `###` 與縮排 `-` 標出這一頁本來就有的層級。
   標題用筆記上實際寫的字。
3. **保留正解關鍵詞**：檔案最上面列出的詞請原樣保留，否則那一題無法計分。
4. **不要補充**：只寫這一頁看得到的內容。

## 重要：請以「整個科目」為單位完成

檢索是在所有頁面之間排名的。如果只校正了 29 個正解頁、其他 37 頁維持原樣，
那些被校正過的頁面會單純因為「文字比較準」而佔便宜，量到的就不是結構的效果。

建議順序（工作量由小到大）：

| 科目 | 頁數 | 題數 |
|---|---|---|
| 一下公民三段 | 3 | 2 |
| 哈佛高二下二段段複 | 4 | 3 |
| 一下地科三段 | 10 | 4 |
| 一下歷史三段 | 10 | 6 |
| 一下生物三段 | 18 | 8 |
| lv.0 | 21 | 6 |

先做完公民＋物理（7 頁、5 題）就能跑第一次 B0 對照；之後每完成一科，
就能把該科加進比較。

## 注意

- 地科 p07–p10 只有一個手寫姓名（`124 14 林祐霆`），沒有內容。
  這幾頁可以留白或直接寫「（無內容）」，我會排除。
- 改完把 `pages/` 交回即可，我會建出 corpus_b0.json 並跑 B0 vs B1 vs B4。
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print(f"wrote {written} page files + probes.md + README.md -> {OUT}")

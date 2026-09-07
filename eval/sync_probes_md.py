"""Regenerate eval/b0/probes.md from eval/chunk_cases.json.

probes.md is the human-readable copy the annotator works from; chunk_cases.json is what
the harness scores. They drifted once already — the 2026-08-31 fix to probes 17 and 18
landed in the JSON only, so the markdown kept handing the annotator gold keywords that
Gemini had invented. Generating one from the other removes the failure mode.

    python eval/sync_probes_md.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cases = json.loads((ROOT / "eval/chunk_cases.json").read_text(encoding="utf-8"))["cases"]
pages = len({c["page"] for c in cases})

head = f"""# NoteHandler 檢索題組（B0 用）

共 {len(cases)} 題，涵蓋 {pages} 個正解頁面。

**這份檔案由 `eval/chunk_cases.json` 產生，不要手動編輯。** 產生方式見
`eval/sync_probes_md.py`。先前兩份曾經不同步：第 17、18 題的關鍵詞在 2026-08-31
已於 JSON 修正，這裡卻還留著舊值。

每一題的「正解頁」是唯一含有該關鍵詞的頁面，已用程式對 B0／B1／B4 三份語料驗證，
且關鍵詞不會出現在自己的題目裡（否則量到的是字面比對，不是檢索）。
評分方式：問題向量化後檢索，看含有正解關鍵詞的片段排在第幾名。

| # | 題目 | 型別 | 正解頁 | 正解關鍵詞 |
|---|---|---|---|---|
"""
rows = "".join(f"| {i} | {c['question']} | {c['type']} | `{c['page']}` | `{c['span']}` |\n"
               for i, c in enumerate(cases, 1))
(ROOT / "eval/b0/probes.md").write_text(head + rows, encoding="utf-8")
print(f"regenerated eval/b0/probes.md: {len(cases)} rows, {pages} pages")

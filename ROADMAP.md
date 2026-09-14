# NoteHandler 改良藍圖

> **給另一個 AI 模型閱讀的交接文件。** 內容是 2026-09-14 的專案現況、已完成的工作、研究線結論，以及依優先順序排列、尚未完成的改良方向。
>
> 更細的技術紀錄在 `PROJECT_OVERVIEW.md`；研究數據在 `eval/RESULTS_gating.txt`。

### 狀態標記

| 標記 | 意思 |
|---|---|
| ✅ | 已完成，**已 commit 並部署**（線上版本已有） |
| 🟡 | 已完成並通過驗證，但**只在本機**：尚未 commit、尚未部署 |
| ◐ | 做了一部分 |
| ⬜ | 尚未開始 |

**一句話現況：** 產品主線的第一步（讓使用者能對照原稿檢查辨識結果）已上線；兩個信任問題（身分驗證、同步失敗）已在本機修好，**卡在 commit 與部署**；專題目標中「以科目別整理」與「答案可追溯」尚未開始。

---

## 0. 讀這份文件前要先知道的

### 專題目標（排序判準）

**讓使用者用 NoteHandler 做出以科目別的筆記整理與查詢；判準是模型的輸出能不能讓使用者看懂自己的筆記內容。**

檢索排名等數值只是手段，不是目標。使用者在 2026-09 明確要求：不要過度強調數值高低，而忽略輸出是否讓使用者明白自己的筆記。B0（人工校正版面的 oracle 實驗）目前忽略。

### 系統概要

- 使用者上傳手寫繁體中文筆記（圖片、PDF、文字檔）。Gemini 逐頁辨識成 Markdown，以 `intfloat/multilingual-e5-base` 嵌入後存進 Chroma。
- 問答：語料總字數在 `WHOLE_CORPUS_CHARS`（預設 60,000）以內時，**整份筆記直接送進 Gemini，完全不走檢索**；超過上限才用檢索。
- 部署：前端在 GitHub Pages（`docs/index.html`），後端在 HF Space（Docker，`https://timhelloyee-notehandler.hf.space`）。筆記存在私有 HF Dataset `timhelloyee/notehandler-store`，不進 git。
- 主要檔案：`app/rag_core.py`（約 1,100 行）、`app/main.py`（約 380 行）、`app/static/index.html`（約 1,000 行）、`app/static/ui_editor.html`（本機樣式編輯器）。

### 必須遵守的規則

| 規則 | 原因 |
|---|---|
| 用 `C:/Users/user/anaconda3/envs/ragtraining/python.exe`，不要用裸 `python` | 裸 `python` 沒有 torch |
| 在 PowerShell 跑測試前設 `$env:PYTHONIOENCODING="utf-8"` | 預設 cp950 會在輸出中文時崩潰 |
| 改完 `app/static/index.html` 一定要複製到 `docs/index.html` | Pages 用的是手動副本，兩份必須 byte-identical（目前一致） |
| 部署時 GitHub（`origin`）和 HF Space（`hf`）都要推 | 兩個是獨立的 remote |
| **後端和 `docs/index.html` 必須一起部署** | 新後端會拒絕舊前端的請求（401），新前端也無法搭配舊後端 |
| 本機開發用 `dev_server.py`，不要直接跑 `uvicorn app.main:app` | `.env` 含 `HF_TOKEN`，直接跑會還原並寫回真實的私有 Dataset |
| 不要放寬 `unavailable` 狀態下的寫入封鎖 | 筆記庫沒載入時寫入，會用近乎空白的資料覆蓋雲端上的真實筆記 |
| 兩個 HF token 分開：Space secret 要能寫 **Dataset**；筆電 push token 只給 **Space** | 筆電 token 外洩時，不應連帶讓人能改寫筆記 |
| 絕不要求、顯示 token 值，也不要把 token 放進指令或 remote URL | 會留在 shell 歷史與 git 設定中 |
| 量測用 Qwen3 時必須 `EMBED_DTYPE=float32` | fp16 下重跑結果不一致（已證實） |
| 絕不把筆記內容、`vectorstores/`、`pdf/`、`.devstore/`、`eval/b0/pages*` 放進 git | 筆記是使用者的私人資料 |
| force-push 需要使用者明確同意 | 曾為清除個資而改寫過歷史 |

---

## 1. 三件決定方向的事

1. **真實使用者不會走到檢索。** `pdf/` 內 62 頁的辨識文字共 52,875 字元，低於上限 60,000，問答時整份筆記直接送進 Gemini。過去一個月在切塊與嵌入模型上的比較，量的是一條目前沒人會走的路。→ 研究線已降為次要（✅，見第 5 節）。
2. **專題目標只做了一半。** 使用者已經能對照原稿檢查辨識結果（✅），但「以科目別整理」目前只是辨識提示詞裡的一條規則，資料庫沒有科目欄位，清單也不分組；答案沒有出處，無法從答案回頭對照原稿。→ 方向 A、B（⬜）。
3. **兩個信任問題**（🟡 本機已修，待部署，見第 4 節）：
   - 只要知道 email 就能讀取、刪除任何人的筆記。
   - 同步 token 失效時，app 顯示「尚無筆記」而不是「無法載入」。

---

## 2. 現況

### 部署與版本

| 項目 | 狀態 |
|---|---|
| 本機分支 | `main`，HEAD = `c6b14e2`，與 `origin/main`、`hf/main` 相同 |
| 線上（GitHub Pages + HF Space） | 執行 `c6b14e2`，Space 為 cpu-basic |
| 線上筆記 | 1 筆：`lv.0_compressed.pdf`，`has_original=False`（沒有原稿，無法並排比對） |
| Space `HF_TOKEN` | 9/14 曾失效（log：`401 Invalid username or password`，清單變空）；使用者更換後已成功還原上述 1 筆 |
| 本機未提交 | 8 個修改檔（+879／−205 行）＋ 3 個新檔：`ROADMAP.md`、`app/static/ui_editor.html`、`tests/` |

> ⚠️ **線上版本仍然有身分驗證漏洞**，要等第 4 節的改動部署後才會修好。這是目前最優先的一件事。

### 功能狀態

| 功能 | 狀態 | 說明 |
|---|---|---|
| 上傳文字、圖片、PDF | ✅ | PDF 逐頁平行辨識；空白頁略過，但保留實體頁碼 `## 第 N 頁` |
| 問答 | ✅ | 語料在上限內整份送出；超過上限才改用檢索 |
| 模型選擇 | ✅ | 7 個 Gemini 模型；額度用盡時自動改用第二組金鑰 |
| 原稿與辨識文字並排修正 | ✅ | 見第 3 節 |
| 完整內容與匯出 | ✅ | 「展開更多／顯示較少」、「儲存 .md」 |
| Google 登入由後端驗證 | 🟡 | 見 4.1；線上仍是舊版（前端自行解碼 email） |
| 雲端同步狀態提示 | 🟡 | 見 4.2；線上仍是舊版 |
| UI 樣式編輯器 | 🟡 | 見 4.4；只在本機開發模式存在，部署版本沒有這些路由 |
| 科目分類 | ⬜ | 不是資料欄位；純文字筆記不分類；清單不分組（方向 A） |
| 答案出處 | ⬜ | API 回應只有 `answer` 與 `model`（方向 B） |

---

## 3. ✅ 已完成並部署（`c6b14e2` 以前）

以下已在 GitHub 與 HF Space 上線。

| 項目 | 內容 | 位置 |
|---|---|---|
| 原稿／辨識文字並排 | 編輯器改兩欄：左邊原稿（圖片或 PDF），右邊可編輯文字；純文字筆記退回單欄並說明沒有原檔；手機寬度堆疊 | `index.html` 的 `openEditor`、`originalPane` |
| 原稿可內嵌顯示 | `/api/notes/{id}/original` 改為 `inline`；`list_notes` 回傳 `has_original`，不必先吃一次 404 | `main.py`、`rag_core.list_notes` |
| PDF 頁碼跳轉 | 解析 `## 第 N 頁` 產生頁碼按鈕，同時跳 PDF 與捲動文字。**白頁閃爍已根治**：每次跳頁建立新的 iframe、以世代計數器丟棄過期載入、250 ms 後才換到前景（原因是移除 `src` 會觸發一次 `about:blank` 的 load 事件，吃掉了一次性監聽器） | `pageJumper`、`framePdf` |
| 不需懂 LaTeX 也能改 | 右欄下方即時排版預覽（400 ms 防抖，可隱藏） | `renderRich` |
| 完整內容 | 清單預覽下方「展開更多／顯示較少」 | 清單列 |
| 匯出 | 每筆筆記「儲存 .md」 | `downloadMarkdown`、`fetchNoteText` |
| 原稿快取 | `ETag`／`If-None-Match` 回 304（兩個標頭都存在且相符才回，修掉曾讓每次 GET 都變 304 的 bug） | `note_original` |
| 研究線降級 | `eval/gating_experiment.py` 檔頭與 `PROJECT_OVERVIEW.md` 註明：量的是生產環境預設不會走的路徑；程式碼保留、不刪 | 第 5 節 |
| 研究 harness 修正 | 切塊器碎片 bug（`_chunk_text` 回退點須 ≥ `size//2`）、chunk 長度對照組、6 題洩題題目改寫 | `rag_core._chunk_text`、`eval/` |
| 本機隔離開發伺服器 | `dev_server.py`：清空 `HF_TOKEN`、改用 `.devstore/` | `dev_server.py` |

---

## 4. 🟡 已完成、只在本機（未 commit、未部署）

### 4.1 由後端驗證使用者身分

**原本的問題：** 前端用 `decodeJwt(response.credential)` 在瀏覽器解出 email，再以 `user_id` 參數送出，後端直接相信。9/14 在沒有登入的情況下，只帶 email 就列出了線上筆記。

**已實作：**

- 前端把 Google ID token 放進 `Authorization: Bearer` 送出。`app/main.py` 的 `current_user` 用 `google-auth` 的 `id_token.verify_oauth2_token` 驗證簽章、`aud`（本 app 的 client id）、到期時間與 `email_verified`。**所有端點都不再接受 `user_id`。**
- 驗證過的 token 快取到到期為止，避免每個請求都下載 Google 憑證。驗證失敗回 401；連不到 Google 回 503。
- 原稿圖片與 PDF 改由前端帶 token 抓取，再以 `blob:` URL 顯示（`<img>`/`<iframe>` 無法帶標頭；把 token 或 email 放進 URL 會進到伺服器 log）。附帶好處：PDF 每開一次編輯器只下載一次，跳頁不再重抓。
- `/original` 回應加上 `Vary: Authorization, X-Dev-User`（URL 對所有使用者相同，又快取一小時，沒有 Vary 會讓同一瀏覽器的下一位使用者拿到上一位的快取）。
- CORS 補上 `PUT`（原本從 GitHub Pages 編輯筆記一直會在 preflight 失敗）。
- ID token 約一小時到期：前端只在 token 仍有效時還原登入，並用 `auto_select` + One Tap 讓回訪使用者自動重新登入；收到 401 時明確登出；登出時呼叫 `disableAutoSelect`。
- 舊的 `localStorage.nh_user`（只存 email）會被刪除，不再讓任何人登入。
- **本機開發：** `dev_server.py` 接受 `X-Dev-User` 標頭代替 Google（前端在 localhost 讀 `localStorage.nh_dev_user`）。`DEV_MODE` 在偵測到 `SPACE_ID` 時拒絕啟用，Space 上無法藉此冒充使用者。
- `requirements.txt` 新增 `google-auth>=2.20` 與 `requests`。

### 4.2 同步失敗時讓使用者知道

**原本的問題：** 9/14 的 log 顯示 `vectorstore restore skipped (… 401 Invalid username or password)`，app 照常啟動並顯示「尚無筆記」。這段期間上傳的筆記只存在容器暫存，重啟就消失。

**已實作：** `rag_core.sync_status()`，由 `/api/health` 回傳（不需登入）。

| 狀態 | 條件 | 行為 |
|---|---|---|
| `ok` | 已從 Dataset 還原 | 無橫幅 |
| `disabled` | 沒有 token，且不在 Space 上（本機開發） | 灰色提示「本機模式」 |
| `unavailable` | 還原失敗，或在 Space 上（`SPACE_ID` 存在）卻沒有 token | 紅色橫幅；**上傳、修改、刪除與問答一律回 503**（`require_loaded()`） |
| `unsaved` | 還原成功，但之後某次同步失敗 | 黃色橫幅；仍可寫入，下一次寫入時整個資料夾重新上傳即修復 |

- 使用者看到的訊息是可行動的說明（例如「HF_TOKEN 無效、已撤銷，或沒有這個 Dataset 的權限」），原始錯誤只寫進 log。
- 筆記庫無法載入時，清單顯示「雲端筆記庫目前無法載入」，不再顯示「尚無筆記」；編輯與刪除按鈕停用。
- 上傳回應帶 `sync` 欄位；同步失敗時上傳結果會註明「尚未同步到雲端」。
- `add_pdf_note` 在呼叫 Gemini **之前**就檢查 `require_loaded()`，不會白花辨識額度。

### 4.3 順帶修正：原稿晚一步才同步

上傳時原稿是在 `_add_note` **同步之後**才存檔，所以原稿要等使用者下一次有變更才會上傳到 Dataset。現在由 `_add_described_note` 先存原稿再同步，儲存失敗時會刪掉原稿，不留孤兒檔。線上那筆 `has_original=False` 很可能就是這個 bug 造成的（修好後仍需方向 F 替它補上原檔）。

### 4.4 UI 樣式編輯器（僅本機開發模式）

- `/ui-editor`：17 個設計變數（頁寬、背景、字級、行高等）與 3 個字型位置（內文、標題、程式碼），字型為下拉選單並標明套用位置，說明文字用一般英文而非變數名。
- 同源 iframe 即時預覽；儲存時只改寫 `index.html` 的 `:root` 區塊、留下 `.html.bak`，並同步到 `docs/index.html`；可「還原上一版」。
- 值以白名單正則驗證，不能注入其他 CSS。路由只在 `DEV_MODE` 存在，Space 上回 404。

### 4.5 驗證（2026-09-14 重跑確認）

- `tests/test_auth_and_sync.py`：**18/18 通過**。獨立腳本，不需 pytest，也不需要真實帳號（Google 驗證與 HF API 用假物件取代）。其中一項讓真正的 Google 驗證器處理一個宣稱是 `timhelloyee@gmail.com`、但簽章是偽造的 token，確認會被拒絕。
- 瀏覽器實測：舊 email 金鑰無法登入、過期 token 會被清除並提示、原稿以 `blob:` 載入（圖片 3000×2409 正常顯示）、PDF 跳頁後恰好一個前景 frame、存檔往返正常、所有請求 URL 都不含 `user_id` 或 email。
- 模擬壞掉的 Space（`SPACE_ID` 存在、無 token）：紅色橫幅、上傳按鈕停用、`X-Dev-User` 被忽略（401）、`/ui-editor` 回 404。
- 研究 harness 迴歸：B1 chunk MRR 仍為 0.799，與紀錄一致。
- `app/static/index.html` 與 `docs/index.html` 雜湊相同。
- **尚未驗證：** 真實 Google 帳號在 GitHub Pages 上登入新後端（只能部署後測）。

```bash
C:/Users/user/anaconda3/envs/ragtraining/python.exe tests/test_auth_and_sync.py
```

---

## 5. 研究線學到了什麼（✅ 已降為次要，不再投入）

數字以修正題組後為準（29 題、62 頁、float32）。程式碼與數據保留，可作為報告的一節。

| 發現 | 證據 |
|---|---|
| **換嵌入模型是最大的槓桿** | e5 → Qwen3-0.6B，基準組 chunk MRR 0.799 → 0.871（差 +0.071，95% CI −0.017～+0.158，9 勝 2 負）。方向一致，但 n=29 下未達顯著 |
| **保留結構測不出效果** | LLM 結構切塊相對基準：e5 +0.001、Qwen3 +0.027。Holm 校正後 chunk 層級沒有任何比較成立 |
| **正對照沒有通過** | 刻意刪除整條祖先路徑：e5 反而變好 +0.052，Qwen3 變差 −0.023。兩個模型方向不一致，題組量不到結構效果 |
| **三個缺陷改變了結論** | 切塊器產生重複碎片；chunk 長度與策略混淆；6 題答案寫在題目裡（修正後主要結論正負號翻轉） |
| **MDE 太大** | n=29 下可偵測的最小效果約 +0.07～+0.13，是實際追逐效果量的 2–4 倍 |

生產環境的 `STRUCTURED_RULE`、`chunk_by_structure`、`merge_target` 是死碼：`add_image_note`／`add_pdf_note` 從不傳 `structured=True`，`merge_target` 預設 0。

---

## 6. 接下來的改良方向（全部 ⬜）

每一項都列出證據、做法、完成的樣子與粗估時間（一個人）。

### 6.0 最優先：部署第 4 節（約 1 小時）

不是新功能，但線上漏洞要靠它修。步驟：

1. commit 本機改動（一個或依主題拆成數個 commit）。
2. 同時推 `origin main` 與 `hf main`（後端與 `docs/index.html` 必須同版）。
3. Space 重建約 2–3 分鐘後檢查 `/api/health`：`sync.state` 應為 `ok`。若是 `unavailable`，修 Space secret，**不要改程式**。
4. 在 GitHub Pages（不是 Space 的 iframe 內）用真實帳號登入，確認清單、原稿、跳頁、編輯存檔都正常。
5. 確認 Google OAuth 的已授權 JavaScript 來源包含實際使用的網址。

### 6.1 接著：把專題目標做完整

#### A. 讓科目成為真正的欄位（1–2 天）

- **證據：** 科目只存在於辨識提示詞第 1 條規則與 `_SUBJECT_TAG` 正則（`rag_core.subject_of()`，找不到時回傳 `?`）。Chroma metadata 沒有 `subject`；`.txt`/`.md` 不經 Gemini，完全不分類；清單只顯示 `[type] 檔名`。
- **做法：**
  - 上傳時把 `subject_of()` 的結果寫入 metadata，並用同一函式回填既有筆記。
  - PDF 是逐頁辨識，**每一頁都有自己的科目標籤**：筆記層級取多數決，頁層級保留在該頁 chunk 的 metadata。
  - 純文字筆記：上傳時讓使用者選科目，或以一次輕量呼叫判斷。
  - `list_notes` 回傳 `subject`；清單依科目分組並顯示筆數。
  - 編輯器加科目下拉選單，用 `collection.update` 只改 metadata、不重新嵌入。
  - 11 個科目：國文、英文、數學、物理、化學、地球科學、生物、地理、公民、歷史、程式設計。
- **完成的樣子：** 打開 app 先看到各科目與筆數，點「生物」只顯示生物筆記；科目標錯時改下拉選單即可。
- **注意：** 寫入路徑要沿用 `require_loaded()` 並回傳 `sync`；`update_note` 目前會刪除再重建 chunk，回填與編輯時要保留既有 metadata；新端點要用 `Depends(current_user)`，並補進 `tests/test_auth_and_sync.py` 的「每個端點都要登入」檢查。

#### B. 答案附上出處（1–2 天）

- **證據：** `QueryResponse` 只有 `{answer, model}`；送進模型的參考資料以 `"\n---\n".join(docs)` 串接，沒有檔名與頁碼，模型無從引用。
- **做法：**
  - 每段參考資料前加來源標頭，並保留既有 `## 第 N 頁` 標記。整份模式與檢索模式都要加。
  - 提示詞（`rag_core.PROMPT_TEMPLATE`）要求每個重點後標註來源。
  - 後端解析引用、剔除不存在的檔名或頁碼，回傳 `sources: [{id, page}]`。
  - 前端顯示為可點的來源標籤，點下去開啟**已上線**的並排編輯器（`openEditor`），並用 `originalPane` 回傳的 `goto(page)` 跳到該頁。

```
送進模型的參考資料
【來源：lv.0_compressed.pdf】
## 第 12 頁
- `s.find("ABC");`：找出字串的第一次出現位置

模型的回答
`find` 會回傳子字串第一次出現的位置〔來源：lv.0_compressed.pdf 第 12 頁〕
```

- **使用流程：** 提問 → 答案附來源 → 點來源標籤 → 原稿並排、跳到第 12 頁 → 修正辨識錯誤 → 下次回答使用更正後的內容。
- **完成的樣子：** 每個答案都能一鍵回到原稿；引用到不存在頁碼的內容不會顯示。
- **依賴：** 點來源跳回原稿需要原檔存在，線上那筆要先做 F。

#### C. 用科目延後進入檢索（半天–1 天，需先完成 A）

- **證據：** 整份模式才是使用者實際得到的體驗，而檢索路徑驗證最不足（正對照沒有通過）。62 頁已有 52,875 字元，離上限 60,000 不遠。
- **做法：** 在 `rag_query` 中，超過上限時先把範圍縮到相關科目（使用者選的篩選，或從問題判斷），整科送出；只有單一科目本身也超過上限時才改用檢索。
- **完成的樣子：** 筆記累積到數百頁後，問生物的問題仍然依據整份生物筆記回答，而不是 10 個片段。

### 6.2 之後：幫使用者看懂自己的筆記

#### D. 標出模型沒把握的地方（約 1 天）

- **證據：** 人工校正時發現 Gemini 在程式設計筆記中寫出了原稿上不存在的「記憶體位址」與「結構化綁定」。兩份辨識提示詞（`describe_image` 內的提示詞與 `PDF_PAGE_PROMPT`）的第 4 條**已經**要求把看不清的地方標成「（無法辨識）」，但介面完全沒有凸顯，而「推測」的內容目前沒有任何標記。
- **做法：**
  - 沿用既有的「（無法辨識）」，另加一個「推測」標記（例如 `〔?…〕`），寫進提示詞第 4 條。
  - 編輯器以底色標出這兩種標記並提供「下一處」跳轉；清單列顯示「3 處待確認」之類的數量。
  - 使用者對照原稿確認後，刪掉標記即視為已確認。
- **完成的樣子：** 使用者知道該對照原稿檢查哪幾處，不必整份重讀。既有筆記要重新辨識才會出現推測標記；「（無法辨識）」在現有筆記上馬上就能標出來。

#### E. 「筆記裡沒有」成為明確狀態（約半天）

- **證據：** 拒答字串「參考資料中沒有記載相關訊息」與一般短答案在畫面上長得一樣。
- **做法：** API 回傳 `status`（`answered` / `not_in_notes`）；介面用不同樣式，並建議換個問法、改選科目或確認筆記是否已上傳。

#### F. 替舊筆記補上原檔（約半天）

- **證據：** 線上唯一的 `lv.0_compressed.pdf` 是 `has_original=False`；重新上傳會重跑辨識並蓋掉手動修正。造成它的 bug 已在 4.3 修好，但已缺原檔的筆記不會自己恢復。
- **做法：** 新增「補上原檔」端點：只呼叫 `save_original` 並同步，不重新辨識、不改內文。檢查上傳檔的類型與原筆記相同（PDF 對 PDF），頁數不符時警告。前端在「沒有原始檔」的說明旁放上傳按鈕。

#### G. 評量改成量「看不看得懂」（1–2 天，最好在 B 之後）

- **證據：** 過去主要指標是檢索排名，但使用者碰不到檢索。`eval/run_eval.py` 的本地資料缺少物理筆記，28 題中 7 題在 `--local` 下必定失敗。
- **做法：** 在 `pdf/` 語料上逐題檢查：答案是否包含正確事實、引用是否指向正確頁、是否出現筆記裡沒有的內容、該拒答時有沒有拒答。
- **完成的樣子：** 每次改提示詞或換模型，都能用數字回答「答案是否可信、能不能追溯到原稿」。

#### H. 科目統整與複習（視範圍，需先完成 A、B）

- **做法：** 每個科目提供「產生統整」與「產生小測驗」，內容同樣附來源。問答輸入框的範例文字本來就是「輸出每一個科目統整後的筆記內容」。統整結果可沿用已上線的「儲存 .md」匯出。

---

## 7. 建議順序

| 順序 | 項目 | 狀態 | 粗估 |
|---|---|---|---|
| 1 | 原稿並排、PDF 跳頁、預覽、展開更多、儲存 .md、研究線降級 | ✅ | — |
| 2 | 後端驗證身分、同步失敗提示、原稿先存再同步、UI 樣式編輯器 | 🟡 本機完成 | — |
| 3 | **commit 並部署第 2 項**（6.0） | ⬜ | 約 1 小時 |
| 4 | 科目成為真正的欄位（A） | ⬜ | 1–2 天 |
| 5 | 替舊筆記補上原檔（F，B 的前提） | ⬜ | 約半天 |
| 6 | 答案附上出處（B） | ⬜ | 1–2 天 |
| 7 | 用科目延後進入檢索（C） | ⬜ | 半天–1 天 |
| 8 | 沒把握的標記、拒答狀態（D、E） | ⬜ | 約 1.5 天 |
| 9 | 評量改成量答案是否可信、可追溯（G） | ⬜ | 1–2 天 |
| 10 | 科目統整與複習（H） | ⬜ | 視範圍 |

與上一版的差異：F 從第 5 順位提前到 B 之前，因為「點來源跳回原稿」在線上那筆沒有原檔的筆記上無法運作。

---

## 8. 暫停不做

| 項目 | 理由 |
|---|---|
| 版面偵測與結構保留實驗（B0、B4、M1–M3） | 優化的是使用者走不到的路徑，而且正對照沒有通過 |
| ColPali（B7）與 contextual retrieval（B5） | 同上；目前 4.3 GB 顯示卡也跑不動 ColPali |
| 現在就把線上嵌入模型換成 Qwen3 | 只影響檢索路徑；需要重新嵌入所有筆記、重新校準 `_GATE_BY_MODEL` 門檻，Space 的 CPU 上也會變慢。等「用科目延後進入檢索」也撐不住時再評估 |
| 用 PDF.js 取代瀏覽器內建 PDF 檢視器 | 曾試過，canvas 重繪同樣會閃白且難以驗證；目前的新 iframe 方案已解決白頁 |

---

## 9. 零碎待辦

- [ ] **commit 並部署第 4 節的改動**（見 6.0）。
- [ ] 刪除已併入 main 的分支 `claude/note-review-ui-and-eval-fixes`（`git branch --merged main` 已確認合併）。
- [ ] 讓 `docs/index.html` 自動同步或加上 CI 檢查（目前靠手動複製；UI 樣式編輯器存檔時會自動同步，但手改 `index.html` 不會）。
- [ ] 編輯筆記時保留關鍵字：刪掉結尾「關鍵字：」那一行會讓該筆記失去關鍵字加成（`_keywords_of`）。可在儲存時若缺少該行就沿用舊的 metadata。
- [ ] 空白頁判斷不要依賴 Gemini 違反自己的規則：`is_blank_page` 要求 `subject_of(t) == "無法辨識"`，但提示詞第 1 條只允許 11 個科目，模型遵守規則時空白頁反而偵測不到。
- [ ] 把其他手動驗證過的行為寫成測試。◐ 目前已涵蓋：身分驗證、使用者隔離、同步狀態、`Vary`。尚缺：切塊不產生碎片、304 快取、樣式編輯器拒絕注入、`docs/` 與 `app/static/` 一致。
- [x] 修好 Space secret `HF_TOKEN`（9/14 更換後已成功還原筆記）。
- [ ] 重新建立筆電 push token（fine-grained，只給 Space 寫入權，不含筆記 Dataset），並以 `hf auth login --add-to-git-credential` 存進 Git Credential Manager。
- [ ] （使用者端）輪替曾貼在對話中的 HF／Gemini／LangChain 金鑰。

---

## 10. 相關檔案

| 檔案 | 內容 |
|---|---|
| `PROJECT_OVERVIEW.md` | 完整技術交接，含每次修正的來龍去脈 |
| `eval/RESULTS_gating.txt` | 研究線所有實驗的完整數據 |
| `eval/markdown_review/修改提議2_實測回覆_v3.md` | 研究流程審查的最終版回覆 |
| `tests/test_auth_and_sync.py` | 身分驗證與同步安全的 18 項檢查 |
| `dev_server.py` | 隔離的本機開發伺服器（不碰私有 Dataset） |
| `app/static/ui_editor.html` | 本機樣式編輯器 |
| `.claude/launch.json` | Browser pane 啟動設定（使用 `dev_server.py`） |

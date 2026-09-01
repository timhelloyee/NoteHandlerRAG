<!-- lv.0_compressed.pdf#p13 | source: pdf/lv.0_compressed.pdf page 13 -->
# lv.0_compressed.pdf#p13

## 這一頁沒有對應的題目
這一頁不是任何題目的正解，但仍會進入索引作為干擾項，所以一樣需要校正。

## 修正後的內容（請直接改下面這一段）

規則：
1. 對照原始 PDF 這一頁，把辨識錯的字、公式、數字改正確。
2. 用巢狀 Markdown 標題標出這一頁「本來就有」的層級：
   `#` 整頁主題、`##` 大標、`###` 次標，條列用 `-` 並以縮排表示層級。
   標題請用筆記上實際寫的字，不要自己發明分類。
3. 只寫這一頁看得到的東西，不要補充其他頁或課本內容。
4. 看不懂的地方寫「（無法辨識）」，不要猜。

---

- `count`，適用於 vector、string，說某一元素的出現次數
  - `count(s.begin(), s.end(), 'element');`
- `string a, b, c;`
  - 若 `c = a + b` ⇒ 慢，`a += b` ⇒ 快 ⇒ 之後一律用 `+=`
- 字串限定 `reverse(始位, 停位)`
- `string s = "12698745"`
  - `cout << stoll(s) + 1 << '\n';`
  - ⇒ 会輸出 12698746（以 long long 格式）
- `stoll` ⇒ long long
  `stold` ⇒ long double
- `to_string(long long / long double)` 可將其轉換為字串
- 前面記得補 0
- `cin.get();` 可以讀入空白，但是 ∈ per 字元
- `getline(cin, s);`（string s）可以讀入空白，且是 ∈ 一整行
- 可以用運算式就盡量用，判斷式太慢
- `stringstream ss;` ⇒ ∈ 一種資料形態
  - 用法 ex. `string s = 'A  B'`
  - ⇒ `ss << s;` `ss = 'A';`
  - 欲輸出完 stringstream 的值，用 `while (ss >> a) { 欲對 a 做的事 }`

## ＃ 寫二維陣列時，最內層較舉的變數要放在陣列的最右邊

- reason：Catch（快取）
  - 自己 ⟶ time min，具有的記憶體也 min
  - 附近
  - 有點遠
  - 超遠！！！ ⟶ time max，具有的記憶體也 max
- ⇒ 而 long long 佔 8 bytes，int 佔 4 bytes，
  故 int 可放較多於快速運行区，∴ 時間：int = 2×(long long)

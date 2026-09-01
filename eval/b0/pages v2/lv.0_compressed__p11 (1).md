<!-- lv.0_compressed.pdf#p11 | source: pdf/lv.0_compressed.pdf page 11 -->
# lv.0_compressed.pdf#p11

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

- 比較 vector 大小
  - 〈法 1〉：`Va.size()` □ `Vb.size()`
  - 〈法 2〉：`Va` □ `Vb`
- vector 也可以直接運用比較運算子 ✓
  - ex. `Va = {1, 2, 5}`、`Vb = {1, 2, 5}`
  - `if (Va == Vb) ⇒ cout "Yes";` ⇒ 成立！
- `vector<pair<int, int>>` ✓ ⇒ vector.pair 的 `push_back`
  - first：`v.push_back({值, 值});`
- `vector<pair<int, int>> x;`
  `vector<pair<int, int>> y;`
  - cmp 中，可直接比較 `x > y;`
- `v.back = v[v.size()-1];`

- 語法書網址：https://hackmd.io/@aacp/library

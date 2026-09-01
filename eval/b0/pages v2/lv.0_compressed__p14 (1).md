<!-- lv.0_compressed.pdf#p14 | source: pdf/lv.0_compressed.pdf page 14 -->
# lv.0_compressed.pdf#p14

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

# 第七堂

- 不可以 list 做陣列名
- auto 用法：
  - `auto x = 1;` ⇒ x ∈ INT
  - `x = 1LL;` ⇒ x ∈ long long
  - ⋮
  - `x = true & true;` ⇒ x ∈ INT
  - `x = b.begin();` ⇒ iterator
- 不可 `auto = b.size() - 1;`
- `size()` 危險！！！ ⇒ 在使用 `size()` 之前，先用 `int k = v.size();`
- `int cur = arr[pl[idx[i]]];`
  - `cur++;` ⇒ 改变 cur 的值
- `int &cur = arr[pl[idx[i]]];`
  - `cur++;` ⇒ 改变 `arr[pl[idx[i]]]` 的值
- `for (值：資料型態)`
  - ex. `vector<int> d = {6, 9, 8};`
  - `for (int x : d)`
  - = `for (int i = 0; i < d.size(); i++) { x = d[i], 動作… }`
- vector 的 size 可以直接 `v.size(6);` ⇒ 会 v 的 size = 6

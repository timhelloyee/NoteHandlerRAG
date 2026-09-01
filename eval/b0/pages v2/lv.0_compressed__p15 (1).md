<!-- lv.0_compressed.pdf#p15 | source: pdf/lv.0_compressed.pdf page 15 -->
# lv.0_compressed.pdf#p15

## 這一頁的正解關鍵詞
下列詞是這一頁所回答的題目的正解。修正時請保留這些詞的原樣，否則該題無法計分：
- `結構化綁定`  （cs-structured-binding：C++ 的結構化綁定怎麼使用？）

## 修正後的內容（請直接改下面這一段）

規則：
1. 對照原始 PDF 這一頁，把辨識錯的字、公式、數字改正確。
2. 用巢狀 Markdown 標題標出這一頁「本來就有」的層級：
   `#` 整頁主題、`##` 大標、`###` 次標，條列用 `-` 並以縮排表示層級。
   標題請用筆記上實際寫的字，不要自己發明分類。
3. 只寫這一頁看得到的東西，不要補充其他頁或課本內容。
4. 看不懂的地方寫「（無法辨識）」，不要猜。

---
- `int d[3] = {2, 3};`
  `auto &[aa, bb, cc] = d;`
  - ⇒ `aa = d[0]`、`bb = d[1]`、`cc = d[2]`
- most 普及用於 pair
  - `tuple<int, char, double, float> d {1, n, 6.4, 1.989};`
  - `cout << get<2>(d);` ⇒ cout 出 6.4
  - `auto [a, b, c] = make_tuple(d, e, f);`
  - ⇒ `a = d`、`b = e`、`c = f`

- 少用 define！！！
- `A ? B : C`
  - A ⇒ 問；`B : C` ⇒ 同形態
  - `if A == true` ⇒ = B
  - `else` ⇒ = C
  - ex. `int res = x < y ? x : y;`（x = 5, y = 11）⇒ `cout << 5;`
- 少用逗號
  - `vector<pair<int, int>> pa(N);`
  - `for (auto &[p, a] : pa)`
  - `(a, b, c)` ⇒ 必為 c，易出現在 bug 中
  - C 語言靠右對齊，靠左對齊 `±%n`【資形(資料型態)】【變數】;
    - (n ∈ ℕ)（補空格直到共 n）
    - (0n ∈ ℕ)（補 0 直到共 n）
- 用 `signed main`

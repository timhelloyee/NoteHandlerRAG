<!-- lv.0_compressed.pdf#p19 | source: pdf/lv.0_compressed.pdf page 19 -->
# lv.0_compressed.pdf#p19

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

# 第九堂

- 指定一結構体：`struct A { string name; int score; }`
  - ⇒ 在 main 裡 ⇒ `A n;`
  - ⇒ `cin >> n.name >> n.score;`
- struct v.s. class
  - ⇒ 前者為可供外在使用，後者為只可供內部使用
  - ⇒ 而 `struct { private: … }`：此變數只可內
  - ⇒ `class { public: … }`：此變數可供外部
- 亦可利用成員函式改变其功能
  - ex. `public: void read() { cin >> name >> score; }`
  - ⇒ 代表可在外部用 `.read` 來輸入此二變數（一致性）
- struct 的裡面加上
  - `bool operator < (const Rest &b) const`
  - `{ 裡面是 cmp 內的功能 }`
  - ⇒ 会支援 sort，且 sort 裡面無須再宣告 cmp
- 如果 ∈ 二項的 structure，也可用 `make_pair(_, _)` 來比較
- 直接在 struct 內初始化：
  - ① `struct A { int n  Data() { n = 0; } }`
  - ② `struct A { int n  Data(int d) { n = d; } }`（此項被劃掉）
    `main { Data d(5); }`
  - ③ `struct A { int n }`
    `main { Data d = { 1 }; }`
  - ④ `struct A { int n  Data(int d) : n(2*d) { } }`
- overload 運算子：
  - 判斷：bool
  - 運算：函式名

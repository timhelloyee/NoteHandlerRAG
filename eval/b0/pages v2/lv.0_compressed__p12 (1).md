<!-- lv.0_compressed.pdf#p12 | source: pdf/lv.0_compressed.pdf page 12 -->
# lv.0_compressed.pdf#p12

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

# 第六堂

- string 的二種神奇宣告
  - `(string a = "AA")` ＝ `(char c[] = "AA", string a = c)`
  - ＝ `(string a = string(2, 'AA'))`
- 記得指定 string 的值 & 大小（if needed）
- vector 有的功能 string 都有！！！只是大小是 `s.length()`
- 多出的功能有：
  - ① `.substr(位置, 長度);`
  - ② `s.find("ABC");` 找出字串的第一次出現位
  - ③ `s.find_last_of('Te');` 会輸出含有裡面任 1 元素的最末位
  - ④ `s.find_first_of('Te');` 会輸出含有裡面任 1 元素的最初位
- C++ 中，vector 可以有 `vector<vector<vector<…<int>>…>>`，因此也存在 `vector<vector<string>>`
  - ex. "Hello"
- 注意！！！題目讀值不一定要讀完，有些讀完甚至会 TLE！！！
- 注意！！！比較函式左右至少要有一个是變數 string，如果皆 ∈【置變含式】，則無法比較

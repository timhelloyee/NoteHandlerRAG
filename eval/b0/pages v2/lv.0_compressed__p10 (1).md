<!-- lv.0_compressed.pdf#p10 | source: pdf/lv.0_compressed.pdf page 10 -->
# lv.0_compressed.pdf#p10

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

# 第五堂

- 盡量不用 vector 的功能，時間 complex 度較大
- `v.resize(大小)` 可以改动 vector 大小
- vector 可以直接 `=`，ex. `vector<int> a;` `vector<int> b = a;`
- 注意 `v.erase()`、`v.insert()` 不可亂用！！！
- `| 1 | 2 | 3 | 4 |`
  - 左端 ↑ `.begin()`
  - 右端 ↑ `.end()`
- `v.insert(v.begin() + 位置, 值);`
- `vector<int>::iterator iter;`
  - 一个指標的概念
  - ex. `v.begin();`

```cpp
vector<int> d{1,2,3,4};
vector<int>::iterator iter = d.begin();
cout << *iter << endl; // 輸出 1
iter += 2;
cout << *iter << endl; // 輸出 3
iter--;
cout << *iter << endl; // 輸出 2
cout << d.end() - iter << endl; // 輸出 3
```

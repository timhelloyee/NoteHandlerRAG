"""Embed every file in docs/ and images/ into a user's ChromaDB vectorstore.

  - text files  -> multilingual-e5-base text embedding of the file content
  - image files -> Gemini Flash description, then text embedding of the description

Retrieval embedder is intfloat/multilingual-e5-base (繁中+英文皆強，跨語言檢索佳)，
取代原本對中文較弱的 CLIP 文字編碼器。E5 需要前綴：文件用 "passage: "，查詢用
"query: "（查詢端在 test_query.py / notebook 中處理）。

Usage:
    python embed_folder.py alice
"""
import os
import re
import sys
import glob
import time

import torch
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

load_dotenv()

for _key in ("GEMINI_API_KEY_PRIMARY", "GEMINI_API_KEY_SECONDARY"):
    if not os.environ.get(_key):
        raise RuntimeError(f"缺少環境變數 {_key}，請在 .env 檔中設定")

# 兩把金鑰：圖片描述優先用備用金鑰（與 Input.ipynb 一致），失敗時改用主要金鑰
gemini_clients = [
    genai.Client(api_key=os.environ["GEMINI_API_KEY_SECONDARY"]),
    genai.Client(api_key=os.environ["GEMINI_API_KEY_PRIMARY"]),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置: {device}")

EMBED_MODEL = "intfloat/multilingual-e5-base"
embedder = SentenceTransformer(EMBED_MODEL, device=device)


def embed_text(texts):
    """Embed note passages. E5 expects a 'passage: ' prefix for documents."""
    inputs = [f"passage: {t}" for t in texts]
    vecs = embedder.encode(inputs, normalize_embeddings=True)
    return vecs.tolist()


def get_user_collection(user_id):
    chroma_client = chromadb.PersistentClient(path=f"./vectorstores/{user_id}")
    return chroma_client.get_or_create_collection(user_id)


def describe_image(image_path, max_retries=5):
    with open(image_path, "rb") as f:
        image_data = f.read()
    ext = image_path.split(".")[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    parts = [
        types.Part.from_bytes(data=image_data, mime_type=mime),
        types.Part.from_text(text="""請詳細描述這張圖片中有什麼內容。注意: 1.不要使用粗體 2.圖片應該以左右反轉的方式描述 3. 在開頭輸出
            你判斷這是什麼科目(國文、英文、數學、物理、化學、地球科學、生物、地理、公民、歷史擇一)的筆記，格式為: 這是一份【科目】科的筆記。
            """),
    ]
    last_err = None
    for attempt in range(max_retries):
        client = gemini_clients[attempt % len(gemini_clients)]  # 交替使用兩把金鑰
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=parts,
                config={"temperature": 0, "top_p": 0.95, "top_k": 20},
            )
            return response.text
        except Exception as e:  # 503/429 等暫時性錯誤 -> 退避重試
            last_err = e
            wait = 2 ** attempt
            print(f"  Gemini 失敗 (attempt {attempt + 1}/{max_retries}): {e}; {wait}s 後重試")
            time.sleep(wait)
    raise RuntimeError(f"describe_image 失敗 {max_retries} 次: {last_err}")


# 切塊大小（字元）。理由有二：
#   1. E5 對短塊能完整嵌入 -> 檢索更準（舊 CLIP 編碼器只看前 77 個 token，整篇長
#      筆記只有開頭會被嵌入）。
#   2. NoteMind 的 context window 有限，短塊讓多個檢索結果能一起塞進模型。
CHUNK_CHARS = 140
CHUNK_OVERLAP = 20

# 句末標點與換行：切塊時優先在這些位置斷開，避免把句子切一半。
# 不含 ASCII 句點，以免拆開小數（例如 3.14）。
_SENTENCE_END = re.compile(r"[。！？；!?\n]")


def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Split text into <=size-char chunks. Each chunk ends at the last sentence
    boundary inside the window (a run-on with no boundary is hard-cut at `size`),
    and consecutive chunks share `overlap` chars so context isn't lost at the seam.
    Returns non-empty chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    boundaries = [m.end() for m in _SENTENCE_END.finditer(text)]  # 理想切點：句末之後
    n = len(text)
    chunks, start = [], 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            # 把切點往回挪到視窗內最後一個句末；沒有句末就硬切在 size 上限
            end = max((b for b in boundaries if start < b <= end), default=end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # 重疊；start+1 保證前進，避免死迴圈
    return chunks


def _add_chunks(collection, file_id, full_text, source, kind):
    """Store one vector per chunk, id = '<file>#<i>' (single chunk keeps the bare id)."""
    chunks = chunk_text(full_text)
    ids = [file_id if len(chunks) == 1 else f"{file_id}#{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=[embed_text([c])[0] for c in chunks],
        documents=chunks,
        metadatas=[{"type": kind, "source": source, "chunk": i, "n_chunks": len(chunks)}
                   for i in range(len(chunks))],
    )
    print(f"{file_id} 已加入 ({kind}, {len(chunks)} 塊)")


def add_txt(collection, path):
    file_id = os.path.basename(path)
    if collection.get(ids=[file_id])["ids"] or collection.get(ids=[f"{file_id}#0"])["ids"]:
        print(f"{file_id} 已存在，跳過")
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    _add_chunks(collection, file_id, content, path, "text")


def add_image(collection, path):
    file_id = os.path.basename(path)
    if collection.get(ids=[file_id])["ids"] or collection.get(ids=[f"{file_id}#0"])["ids"]:
        print(f"{file_id} 已存在，跳過")
        return
    description = describe_image(path)
    print(f"圖片描述 [{file_id}]：{description[:120].strip()}...")
    _add_chunks(collection, file_id, description, path, "image")


def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else "alice"
    collection = get_user_collection(user_id)

    txt_files = sorted(glob.glob("docs/*.txt"))
    img_files = sorted(
        p for ext in ("jpg", "jpeg", "png") for p in glob.glob(f"images/*.{ext}")
    )

    print(f"\n== docs ({len(txt_files)}) ==")
    for p in txt_files:
        add_txt(collection, p)

    print(f"\n== images ({len(img_files)}) ==")
    for p in img_files:
        add_image(collection, p)

    total = len(collection.get()["ids"])
    print(f"\n完成。{user_id} 的 vectorstore 現有 {total} 筆資料。")


if __name__ == "__main__":
    main()

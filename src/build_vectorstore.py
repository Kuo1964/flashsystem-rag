"""
build_vectorstore.py
====================
將所有 Document 段落 embedding 後寫入 Chroma 向量資料庫。
使用 HuggingFace sentence-transformers（本地推理，無需 API key）。

Collection 命名策略：
  - "flashsystem_pdf"   ← PDF 文件
  - "flashsystem_web"   ← IBM Docs 網頁
  - "flashsystem_json"  ← 結構化 JSON 文件
  - "flashsystem_all"   ← 全部合併（預設查詢用）

去重策略：
  每個 Document 以 SHA-1(source + page + content[:200]) 產生確定性 ID，
  寫入前查詢 collection 已有的 ID，只寫入新增段落。
"""

import hashlib
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# 每批次寫入的 Document 數量（避免記憶體溢出）
_BATCH_SIZE = 500

# 預設 Embedding 模型（多語言，適合技術文件）
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


_embeddings_cache: dict = {}


def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> HuggingFaceEmbeddings:  # type: ignore[return]
    """
    取得 HuggingFace Embedding 模型實例（同一模型只載入一次，後續重用）。
    首次執行時會自動下載模型（約 1GB）。
    """
    if model_name not in _embeddings_cache:
        _embeddings_cache[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
    return _embeddings_cache[model_name]


def build_vectorstore(
    documents: List[Document],
    persist_dir: str,
    collection_name: str = "flashsystem_all",
    embed_model: str = DEFAULT_EMBED_MODEL,
    reset: bool = False,
) -> Chroma:
    """
    將 documents 寫入 Chroma 向量資料庫。

    Args:
        documents:       待寫入的 Document 列表。
        persist_dir:     Chroma 持久化目錄。
        collection_name: Chroma collection 名稱。
        embed_model:     HuggingFace 模型名稱。
        reset:           True 則清除舊資料後重建。

    Returns:
        Chroma: 可供查詢的向量資料庫實例。
    """
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(embed_model)

    if reset:
        # 刪除指定 collection
        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            client.delete_collection(collection_name)
            print(f"[INFO] 已清除舊 collection：{collection_name}")
        except Exception:
            pass

    if not documents:
        print("[WARN] 沒有文件可寫入，回傳空的 vectorstore。")
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    print(f"[INFO] 開始處理 {len(documents)} 個段落到 collection='{collection_name}' ...")

    # ── 建立 / 取得 vectorstore 實例 ──
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # ── 查詢已存在的 ID ──
    existing_ids: set = set()
    try:
        existing = vectorstore._collection.get(include=[])  # 只取 IDs，不取向量
        existing_ids = set(existing["ids"])
    except Exception:
        pass

    # ── 為每個 Document 產生確定性 ID，過濾已存在的與同批內重複的 ──
    new_docs: List[Document] = []
    new_ids:  List[str]     = []
    seen_ids: set           = set()
    for doc in documents:
        src  = doc.metadata.get("source", "")
        page = str(doc.metadata.get("page", ""))
        raw  = f"{src}::{page}::{doc.page_content[:200]}"
        doc_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        if doc_id not in existing_ids and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            new_docs.append(doc)
            new_ids.append(doc_id)

    skipped = len(documents) - len(new_docs)
    if skipped:
        print(f"[INFO] 跳過 {skipped} 個重複段落，新增 {len(new_docs)} 個段落。")
    else:
        print(f"[INFO] 無重複，寫入全部 {len(new_docs)} 個段落。")

    if not new_docs:
        print("[INFO] 無新段落，向量資料庫維持不變。")
        return vectorstore

    # ── 分批寫入 ──
    for i in tqdm(range(0, len(new_docs), _BATCH_SIZE), desc="寫入向量資料庫"):
        batch_docs = new_docs[i : i + _BATCH_SIZE]
        batch_ids  = new_ids[i : i + _BATCH_SIZE]
        vectorstore.add_documents(batch_docs, ids=batch_ids)

    print(f"[INFO] 向量資料庫建立完成，路徑：{persist_dir}")
    return vectorstore


def load_vectorstore(
    persist_dir: str,
    collection_name: str = "flashsystem_all",
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> Chroma:
    """
    載入既有的 Chroma 向量資料庫（不重新 embedding）。

    Args:
        persist_dir:     Chroma 持久化目錄。
        collection_name: 要載入的 collection 名稱。
        embed_model:     需與建立時相同的模型。

    Returns:
        Chroma: 可供查詢的向量資料庫實例。
    """
    embeddings = get_embeddings(embed_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    count = vectorstore._collection.count()
    print(f"[INFO] 載入 collection='{collection_name}'，共 {count} 個向量")
    return vectorstore


if __name__ == "__main__":
    # 快速驗證：建立一個包含少量測試資料的 vectorstore
    test_docs = [
        Document(page_content="IBM FlashSystem 9500 is a high-performance NVMe storage system.", metadata={"source": "test", "source_type": "test"}),
        Document(page_content="The mkvolume command creates a new volume in the FlashSystem.", metadata={"source": "test", "source_type": "test"}),
        Document(page_content="FlashSystem supports iSCSI, FC, and NVMe-oF host connections.", metadata={"source": "test", "source_type": "test"}),
    ]
    vs = build_vectorstore(test_docs, persist_dir="/tmp/rag_test", collection_name="test", reset=True)
    results = vs.similarity_search("how to create a volume", k=2)
    for r in results:
        print(f"  → {r.page_content[:80]}")

"""
migrate_embeddings.py
=====================
把既有 Chroma collection 的文字，用新的 embedding 模型重新向量化後寫入「新的 collection」。

為什麼需要：
  不同 embedding 模型的向量維度不同（mpnet=768、bge-m3=1024），無法在同一個 collection 內混用，
  所以換模型必須重新向量化。文字與 metadata 已存在於舊 collection，不需要重新爬取。

特性：
  - 只讀舊 collection，不修改、不刪除。
  - 沿用舊 ID（SHA-1），之後 build_vectorstore 的去重邏輯仍相容。
  - 可續傳／可增量：跳過新 collection 已有的 ID，所以中斷後重跑、或爬蟲又新增資料後重跑都只處理差額。
  - 分頁讀取（避免 too many SQL variables）。

用法（需使用 .venv 的 Python）：
  .venv/bin/python3 scripts/migrate_embeddings.py --dst-dir /Users/yhkuo/RAG/vectorstore_bge --device mps
  .venv/bin/python3 scripts/migrate_embeddings.py --dst-dir ... --limit 2000   # 只轉 2000 筆（測速、試跑）
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from dotenv import load_dotenv

load_dotenv()

_PAGE = 1000          # 每次從舊 collection 讀取的筆數
_MAX_CHARS = 8000     # 單筆文字上限，保護性設定


def main() -> int:
    ap = argparse.ArgumentParser(description="以新 embedding 模型重建 collection（不需重爬）")
    ap.add_argument("--vector-dir", default=os.environ.get("RAG_VECTOR_DIR", "/Users/yhkuo/RAG/vectorstore"))
    ap.add_argument("--src", default="flashsystem_all", help="來源 collection（唯讀）")
    ap.add_argument("--dst-dir", default="", help="目標向量庫目錄（預設與來源相同）。"
                    "來源有爬蟲正在寫入時，務必指定不同目錄，避免兩個行程同時寫同一個 SQLite")
    ap.add_argument("--dst", default="flashsystem_all", help="目標 collection")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--batch", type=int, default=32, help="encode batch size")
    ap.add_argument("--max-seq", type=int, default=512, help="模型最大輸入 token 數（1000 字元約 250~500 token）")
    ap.add_argument("--limit", type=int, default=0, help="最多處理幾筆新資料（0 = 全部）")
    args = ap.parse_args()

    dst_dir = args.dst_dir or args.vector_dir
    if Path(dst_dir).resolve() == Path(args.vector_dir).resolve() and args.src == args.dst:
        print("[ERROR] 來源與目標不可相同", file=sys.stderr)
        return 2

    from sentence_transformers import SentenceTransformer

    src = chromadb.PersistentClient(path=args.vector_dir).get_collection(args.src)
    dst_client = chromadb.PersistentClient(path=dst_dir)
    dst = dst_client.get_or_create_collection(args.dst)

    total_src = src.count()
    done_ids = set(dst.get(include=[])["ids"])
    print(f"[INFO] 來源 {args.vector_dir}/{args.src}：{total_src} 筆；"
          f"目標 {dst_dir}/{args.dst} 已有 {len(done_ids)} 筆", flush=True)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = min(args.max_seq, model.max_seq_length)
    print(f"[INFO] 模型 {args.model}｜device={args.device}｜max_seq_length={model.max_seq_length}", flush=True)

    written = 0
    t0 = time.time()
    offset = 0
    while offset < total_src:
        r = src.get(include=["documents", "metadatas"], limit=_PAGE, offset=offset)
        offset += _PAGE
        ids, docs, metas = [], [], []
        for i, d, m in zip(r["ids"], r["documents"], r["metadatas"]):
            if i in done_ids or not d:
                continue
            ids.append(i)
            docs.append(d[:_MAX_CHARS])
            metas.append(m)
            if args.limit and written + len(ids) >= args.limit:
                break
        if ids:
            vecs = model.encode(
                docs, batch_size=args.batch, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True,
            )
            dst.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs.tolist())
            done_ids.update(ids)
            written += len(ids)
            rate = written / max(time.time() - t0, 1e-6)
            print(f"[進度] 讀到 {min(offset, total_src)}/{total_src}｜本次新增 {written}｜"
                  f"{rate:.1f} 筆/秒｜目標共 {len(done_ids)} 筆", flush=True)
        if args.limit and written >= args.limit:
            break

    print(f"[INFO] 完成。本次新增 {written} 筆，目標 collection 共 {dst.count()} 筆（來源 {src.count()} 筆）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

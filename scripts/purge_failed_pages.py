#!/usr/bin/env python3
"""清除 refetch_failed.txt 內網址的所有 chunk（這些頁面重爬後仍無主文，只剩導覽殘留）。

預設 dry-run，只顯示會刪多少；加 --apply 才真的刪除。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import chromadb  # noqa: E402

_BATCH = 200


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urls-file", default=None, help="網址清單（預設為向量庫目錄下 refetch_failed.txt）")
    ap.add_argument("--apply", action="store_true", help="真的刪除（預設只 dry-run）")
    args = ap.parse_args()

    vdir = os.path.expanduser(os.environ.get("RAG_VECTOR_DIR", "~/RAG/vectorstore_bge"))
    coll_name = os.environ.get("RAG_COLLECTION", "flashsystem_all")
    path = args.urls_file or os.path.join(vdir, "refetch_failed.txt")
    with open(path, encoding="utf-8") as f:
        urls = sorted({l.strip() for l in f if l.strip()})

    col = chromadb.PersistentClient(path=vdir).get_collection(coll_name)
    print(f"[INFO] 向量庫 {vdir}｜collection {coll_name}｜目前 {col.count()} 筆｜清單 {len(urls)} 個網址")

    total = 0
    for i in range(0, len(urls), _BATCH):
        batch = urls[i : i + _BATCH]
        ids = col.get(where={"source": {"$in": batch}}, include=[])["ids"]
        total += len(ids)
        if args.apply and ids:
            col.delete(ids=ids)
    verb = "已刪除" if args.apply else "將刪除（dry-run）"
    print(f"[完成] {verb} {total} 筆；collection 現有 {col.count()} 筆")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

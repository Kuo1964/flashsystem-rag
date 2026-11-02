"""
refetch_bad_pages.py
====================
重爬「抓到導覽骨架而不是主文」的網頁，並更新向量庫。

背景：IBM Docs 是單頁應用，爬取當下偶爾拿到尚未渲染完成的頁面，主文選擇器沒命中時，
舊版爬蟲會退回整個 body，把 "Documentation / My IBM / Log in / ..." 的導覽殼切成 chunk 寫進庫。

流程（每一頁獨立處理，中斷後重跑只會處理剩下的）：
  1. 在向量庫找出含骨架 chunk 的頁面（chunk 文字以 "Documentation\nMy IBM\nLog in" 開頭）。
  2. 用 ingest_web._fetch_text 重抓：主文沒出現會自動多等、重載，仍失敗才放棄。
  3. 抓到正常主文 → 切 chunk、用目前設定的 embedding 模型向量化 → 先寫入新 chunk，
     再刪除該頁「所有」舊 chunk。順序確保任何時間點該頁都不會是空的。
  4. 抓不到主文 → 記入 failed 檔。預設保留原樣；加 --purge-unfetchable 則刪除該頁的骨架 chunk
     （版本首頁／landing page 本來就沒有主文，骨架內容只是雜訊）。

用法（需使用 .venv 的 Python；爬蟲和遷移不可同時對同一個向量庫寫入）：
  .venv/bin/python3 scripts/refetch_bad_pages.py --list-only             # 只統計，不改任何東西
  .venv/bin/python3 scripts/refetch_bad_pages.py --limit 20              # 先試跑 20 頁
  .venv/bin/python3 scripts/refetch_bad_pages.py --workers 3             # 全量
"""

import argparse
import hashlib
import os
import sqlite3
import sys
import time
import queue
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import List

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

# 重爬只需要頁面主文，不需要圖片／字型／樣式／追蹤腳本；封鎖它們可讓每頁從約 12 秒降到約 4 秒。
# 實測 20 頁（messages／reference／commands／其他）主文文字與不封鎖時完全相同。
# 注意：封鎖後 TOC 目錄的子連結會少一大半，所以這個設定只用於「已知網址」的修復，
# 不可套用在需要探索子連結的一般爬蟲（src/ingest_web.py）。
_BLOCKED_URLS = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.woff", "*.woff2", "*.ttf", "*.css",
    "*google-analytics*", "*googletagmanager*", "*doubleclick*", "*demdex*", "*adobedtm*",
    "*tiqcdn*", "*qualtrics*", "*newrelic*", "*nr-data*", "*medallia*", "*bam.nr*",
    "*cookielaw*", "*onetrust*", "*segment*", "*hotjar*", "*chat*",
]
_SKELETON_LIKE = "Documentation\nMy IBM\nLog in%"
_SKELETON_PREFIX = "Documentation\nMy IBM\nLog in"


def find_bad_pages(vector_dir: str) -> List[str]:
    """用唯讀 SQLite 找出含骨架 chunk 的網頁網址（排序固定，便於續跑）。"""
    db = sqlite3.connect(f"file:{vector_dir}/chroma.sqlite3?mode=ro", uri=True, timeout=30)
    try:
        rows = db.execute(
            "SELECT DISTINCT m.string_value FROM embedding_metadata m "
            "JOIN embedding_metadata d ON d.id = m.id "
            "WHERE m.key = 'source' AND m.string_value LIKE 'http%' "
            "AND d.key = 'chroma:document' AND d.string_value LIKE ?",
            (_SKELETON_LIKE,),
        ).fetchall()
    finally:
        db.close()
    return sorted(r[0] for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="重爬骨架頁並更新向量庫")
    ap.add_argument("--vector-dir", default=os.environ.get("RAG_VECTOR_DIR", "/Users/yhkuo/RAG/vectorstore_bge"))
    ap.add_argument("--collection", default=os.environ.get("RAG_COLLECTION", "flashsystem_all"))
    ap.add_argument("--workers", type=int, default=3, help="並行瀏覽器數（1~4）")
    ap.add_argument("--limit", type=int, default=0, help="最多處理幾頁（0 = 全部）")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=150)
    ap.add_argument("--failed-file", default="", help="失敗網址記錄檔（預設放在向量庫目錄）")
    ap.add_argument("--urls-file", default="", help="只處理此檔列出的網址（每行一個；仍須是骨架頁才會處理）")
    ap.add_argument("--list-only", action="store_true", help="只統計壞頁數，不爬也不改")
    ap.add_argument("--purge-unfetchable", action="store_true",
                    help="重抓後仍沒有主文的頁面：刪除它們的骨架 chunk（只刪文字以導覽骨架開頭的 chunk）")
    args = ap.parse_args()
    args.workers = max(1, min(4, args.workers))

    bad = find_bad_pages(args.vector_dir)
    print(f"[INFO] 向量庫 {args.vector_dir}：找到 {len(bad)} 個骨架頁", flush=True)
    if args.list_only or not bad:
        return 0
    if args.urls_file:
        wanted = set(Path(args.urls_file).read_text(encoding="utf-8").split())
        bad = [u for u in bad if u in wanted]
        print(f"[INFO] 依 --urls-file 篩選後剩 {len(bad)} 頁", flush=True)
    if args.limit:
        bad = bad[: args.limit]

    import chromadb
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    from src.build_vectorstore import DEFAULT_EMBED_MODEL, get_embeddings
    from src.ingest_web import _fetch_text, _make_driver, _parse_meta

    col = chromadb.PersistentClient(path=args.vector_dir).get_collection(args.collection)
    print(f"[INFO] embedding 模型：{DEFAULT_EMBED_MODEL}｜collection 現有 {col.count()} 筆", flush=True)
    emb = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    failed_path = Path(args.failed_file or Path(args.vector_dir) / "refetch_failed.txt")

    parallel = args.workers > 1
    handles = []
    try:
        for _ in range(args.workers):
            handles.append(_make_driver(isolated=parallel))
    except Exception:
        for h in handles:
            try:
                h.quit()
            except Exception:
                pass
        raise

    for h in handles:
        raw = h.driver if parallel else h
        raw.execute_cdp_cmd("Network.enable", {})
        raw.execute_cdp_cmd("Network.setBlockedURLs", {"urls": _BLOCKED_URLS})

    free = queue.Queue()
    for h in handles:
        free.put(h)

    def fetch(url):
        """取一個閒置瀏覽器抓一頁，抓完歸還。任何瀏覽器都能立刻接下一頁，不必等同組其他頁。"""
        h = free.get()
        try:
            raw = h.driver if parallel else h
            return url, _fetch_text(url, raw)[0]
        finally:
            free.put(h)

    fixed = failed = dropped = added = purged = 0
    t0 = time.time()

    def handle(url, text):
        nonlocal fixed, failed, dropped, added, purged
        if not text or len(text.strip()) <= 100:
            failed += 1
            with open(failed_path, "a", encoding="utf-8") as f:
                f.write(url + "\n")
            if args.purge_unfetchable:
                r = col.get(where={"source": {"$eq": url}}, include=["documents"])
                skel = [i for i, d in zip(r["ids"], r["documents"])
                        if d.startswith(_SKELETON_PREFIX)]
                if skel:
                    col.delete(ids=skel)
                    purged += len(skel)
            return

        meta = _parse_meta(url)
        base_meta = {"source": url, "source_type": "web",
                     "product": meta["product"], "version": meta["version"]}
        chunks = [c for c in splitter.split_text(text) if c.strip()]
        ids, docs, seen = [], [], set()
        for c in chunks:
            cid = hashlib.sha1(f"{url}::::{c[:200]}".encode("utf-8")).hexdigest()
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            docs.append(c)
        if not ids:
            failed += 1
            return

        old = col.get(where={"source": {"$eq": url}}, include=[])["ids"]
        vecs = emb.embed_documents(docs)
        # 先寫新，再刪舊（只刪不在新 ID 集合內的）：任何時間點該頁都不會是空的
        col.upsert(ids=ids, documents=docs, metadatas=[base_meta] * len(ids), embeddings=vecs)
        stale = [i for i in old if i not in seen]
        if stale:
            col.delete(ids=stale)
        fixed += 1
        added += len(ids)
        dropped += len(stale)

    try:
        # 滾動式並行：最多 workers*2 頁同時在途，誰先抓完就先處理（寫入向量庫在主執行緒逐頁進行）
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            todo = iter(bad)
            pending = set()
            done = 0
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < args.workers * 2:
                    try:
                        pending.add(ex.submit(fetch, next(todo)))
                    except StopIteration:
                        exhausted = True
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    url, text = fut.result()
                    handle(url, text)
                    done += 1
                    if done % 15 == 0 or done >= len(bad):
                        rate = done / max(time.time() - t0, 1e-6) * 60
                        print(f"[進度] {done}/{len(bad)}｜已修復 {fixed}｜失敗 {failed}｜"
                              f"新增 {added} 段｜換掉骨架 {dropped} 段｜清除無主文 {purged} 段｜{rate:.1f} 頁/分", flush=True)
    finally:
        for h in handles:
            try:
                h.quit()
            except Exception:
                pass

    print(f"[完成] 處理 {len(bad)} 頁：修復 {fixed}、失敗 {failed}；"
          f"新增 {added} 段、換掉舊骨架 {dropped} 段、清除無主文頁骨架 {purged} 段；collection 現有 {col.count()} 筆", flush=True)
    if failed:
        print(f"[INFO] 失敗網址已記入 {failed_path}（重跑腳本會自動重新嘗試仍是骨架的頁面）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

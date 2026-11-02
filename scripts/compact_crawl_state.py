"""
compact_crawl_state.py
======================
把舊格式的續爬記錄檔（每頁內嵌完整連結）轉成字典編碼的新格式。

安全做法：
  - 只「讀」來源檔，輸出到另一個新檔（預設 <來源>.compact），絕不改寫來源。
  - 轉完會重新載入新舊兩檔，確認「已完成頁面」與「每頁子連結」完全一致，才算成功。
  - 要替換原檔請自行 mv，並先保留舊檔備份。

用法：
  .venv/bin/python3 scripts/compact_crawl_state.py <來源.jsonl> [輸出.jsonl]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.crawl_state import CrawlState  # noqa: E402


def compact(src: Path, dst: Path) -> None:
    if dst.exists():
        sys.exit(f"[錯誤] 輸出檔已存在，請先移除或換名稱：{dst}")

    t0 = time.time()
    print(f"[INFO] 載入來源（相容舊格式）：{src}")
    state = CrawlState(str(src))
    print(f"[INFO] 載入完成：{len(state)} 頁，{len(state._link_by_id)} 個不同連結，"
          f"耗時 {time.time() - t0:.1f} 秒")

    with dst.open("w", encoding="utf-8") as f:
        for link in state._link_by_id:
            f.write(json.dumps({"d": link}, ensure_ascii=False) + "\n")
        for url, ids in state._done.items():
            f.write(json.dumps({"url": url, "ids": list(ids)}, ensure_ascii=False) + "\n")
        f.flush()

    print(f"[INFO] 已寫出：{dst}")
    print(f"[INFO] 大小 {src.stat().st_size / 1e6:.1f} MB → {dst.stat().st_size / 1e6:.1f} MB")

    print("[INFO] 驗證：重新載入新檔並逐頁比對 ...")
    new = CrawlState(str(dst))
    if len(new) != len(state):
        sys.exit(f"[錯誤] 頁數不一致：舊 {len(state)}，新 {len(new)}")
    for url in state._done:
        if not new.is_done(url) or new.links_of(url) != state.links_of(url):
            sys.exit(f"[錯誤] 頁面內容不一致：{url}")
    print(f"[OK] 驗證通過：{len(new)} 頁，所有頁面與子連結（含順序）完全一致")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    s = Path(sys.argv[1])
    d = Path(sys.argv[2]) if len(sys.argv) > 2 else s.with_name(s.name + ".compact")
    compact(s, d)

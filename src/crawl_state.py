"""
crawl_state.py
==============
記錄「已爬取並寫入向量庫」的網頁，程序重啟後可跳過已完成的頁面。

檔案格式：JSON Lines（每行一筆，只追加不改寫，中途當機最多損失最後一行）。

  {"url": "...", "links": ["...", ...]}

  url   已完成（段落已寫入向量庫）的頁面。
  links 該頁面發現的子連結。續爬時直接用這些連結延伸佇列，
        不必重新載入已完成的頁面，速度差距非常大。

重要：只有在段落「確實寫入向量庫之後」才可呼叫 commit()，
      否則當機時會有頁面被標記完成、資料卻沒進資料庫。
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class CrawlState:
    def __init__(self, state_file: str):
        self.path = Path(state_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # url -> 該頁面發現的子連結
        self._done: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        bad = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._done[rec["url"]] = list(rec.get("links", []))
                except (json.JSONDecodeError, KeyError, TypeError):
                    bad += 1  # 當機時可能留下半行，略過即可
        if bad:
            print(f"[WARN] 爬取記錄檔有 {bad} 行無法解析，已略過：{self.path}")

    def __len__(self) -> int:
        return len(self._done)

    def is_done(self, url: str) -> bool:
        return url in self._done

    def links_of(self, url: str) -> List[str]:
        return self._done.get(url, [])

    def commit(self, records: Iterable[tuple]) -> None:
        """
        將 (url, links) 記錄為已完成。呼叫前必須確保這些頁面的段落已寫入向量庫。
        """
        records = list(records)
        if not records:
            return
        with self.path.open("a", encoding="utf-8") as f:
            for url, links in records:
                if url in self._done:
                    continue
                self._done[url] = list(links)
                f.write(json.dumps({"url": url, "links": list(links)}, ensure_ascii=False) + "\n")
            f.flush()


def open_state(state_file: Optional[str]) -> Optional[CrawlState]:
    """state_file 為 None 或空字串時回傳 None（不使用續爬記錄）。"""
    if not state_file:
        return None
    return CrawlState(state_file)


def default_state_file(vector_dir: str, collection: str) -> str:
    """
    續爬記錄檔的預設路徑：與向量庫放在一起，依 collection 名稱區分。
    與向量庫綁在一起的好處：刪除或重建向量庫時，記錄不會殘留成「假完成」。
    """
    return str(Path(vector_dir) / f"crawl_state_{collection}.jsonl")


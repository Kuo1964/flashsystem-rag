"""
crawl_state.py
==============
記錄「已爬取並寫入向量庫」的網頁，程序重啟後可跳過已完成的頁面。

檔案格式：JSON Lines（每行一筆，只追加不改寫，中途當機最多損失最後一行）。

為了避免檔案暴增（IBM Docs 每頁都帶整份目錄的數千個連結，
實測重複率 99.9%），連結採「字典編碼」：每個不同的連結只存一次，
之後以整數編號引用。共有三種記錄行，可混在同一個檔案：

  {"d": "<連結>"}                   定義一個新連結；編號 = 目前已定義的連結數（從 0 起算）
  {"url": "...", "ids": [3, 7, 9]}  已完成頁面；ids 為該頁子連結的編號
  {"url": "...", "links": ["..."]}  舊格式（連結直接內嵌），仍可讀取

  url   已完成（段落已寫入向量庫）的頁面。
  ids   該頁面發現的子連結。續爬時直接用這些連結延伸佇列，
        不必重新載入已完成的頁面，速度差距非常大。

重要：只有在段落「確實寫入向量庫之後」才可呼叫 commit()，
      否則當機時會有頁面被標記完成、資料卻沒進資料庫。
      字典行一定寫在引用它的頁面行「之前」，所以當機時
      不會出現引用了未定義編號的頁面行。
"""

import json
from array import array
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class CrawlState:
    def __init__(self, state_file: str):
        self.path = Path(state_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 連結字典：編號 <-> 連結
        self._link_by_id: List[str] = []
        self._id_by_link: Dict[str, int] = {}
        # url -> 該頁面發現的子連結編號（用 array('I') 每個編號只占 4 bytes）
        self._done: Dict[str, array] = {}
        self._load()
        self._ensure_trailing_newline()

    def _ensure_trailing_newline(self) -> None:
        """
        若上次當機留下沒換行的半行，補一個換行。
        否則下次追加的內容會黏在半行後面，連同新資料一起被當成壞行丟掉，
        字典編號也會因此錯位。
        """
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return
        with self.path.open("rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")

    def _intern(self, link: str) -> int:
        """取得連結編號；沒有就新增（只加進記憶體，寫檔由 commit 負責）。"""
        idx = self._id_by_link.get(link)
        if idx is None:
            idx = len(self._link_by_id)
            self._link_by_id.append(link)
            self._id_by_link[link] = idx
        return idx

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
                    if "d" in rec:                       # 字典行
                        self._intern(rec["d"])
                    elif "ids" in rec:                   # 新格式頁面行
                        ids = rec["ids"]
                        n = len(self._link_by_id)
                        if any((not isinstance(i, int)) or i < 0 or i >= n for i in ids):
                            raise KeyError("ids 引用了未定義的連結編號")
                        self._done[rec["url"]] = array("I", ids)
                    else:                                # 舊格式頁面行
                        self._done[rec["url"]] = array(
                            "I", (self._intern(l) for l in rec.get("links", []))
                        )
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                    bad += 1  # 當機時可能留下半行，略過即可
        if bad:
            print(f"[WARN] 爬取記錄檔有 {bad} 行無法解析，已略過：{self.path}")

    def __len__(self) -> int:
        return len(self._done)

    def is_done(self, url: str) -> bool:
        return url in self._done

    def links_of(self, url: str) -> List[str]:
        return [self._link_by_id[i] for i in self._done.get(url, [])]

    def commit(self, records: Iterable[tuple]) -> None:
        """
        將 (url, links) 記錄為已完成。呼叫前必須確保這些頁面的段落已寫入向量庫。
        """
        records = [(u, l) for u, l in records if u not in self._done]
        if not records:
            return

        lines: List[str] = []
        known = len(self._link_by_id)
        staged: Dict[str, array] = {}
        for url, links in records:
            staged[url] = array("I", (self._intern(l) for l in links))
        # 本批新增的連結（編號 >= known）先寫字典行，再寫頁面行
        for link in self._link_by_id[known:]:
            lines.append(json.dumps({"d": link}, ensure_ascii=False))
        for url, ids in staged.items():
            lines.append(json.dumps({"url": url, "ids": list(ids)}, ensure_ascii=False))

        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
                f.flush()
        except Exception:
            # 寫檔失敗（例如 SSD 暫時 I/O 錯誤）：檔案可能已寫入一部分，
            # 直接從檔案重新載入，讓記憶體與檔案永遠一致，再把例外丟出去。
            self._link_by_id, self._id_by_link, self._done = [], {}, {}
            try:
                self._load()
                self._ensure_trailing_newline()
            except Exception:
                pass  # 磁碟仍不可用；呼叫端會收到下面的原始例外
            raise
        self._done.update(staged)


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

"""
ingest_web.py
=============
讀取網址清單，用 Selenium headless Chrome 渲染 IBM Docs SPA，
遞迴抓取子頁面（僅限同根網址），切割成文字段落並附上 metadata。

IBM Docs 使用 JavaScript SPA，純 requests 只能拿到空殼 HTML；
必須讓瀏覽器執行 JS 後才能取得真實內容。

公開 API：
  load_urls(url_file, ...)           ← 從 txt 檔讀取 URL 清單（CLI 用）
  load_urls_from_list(urls, ...)     ← 直接傳入 URL 字串列表（GUI 用）
"""

import os
import re
import shutil
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.crawl_state import open_state

# IBM Docs 主文內容的 CSS 選擇器（依序優先嘗試）
_CONTENT_SELECTORS = [
    "main[role='main']",
    "article.ibm-content",
    "div#content",
    "div.content",
    "main",
    "article",
]

# 從 IBM Docs URL 解析版本與產品
_DOCS_RE = re.compile(r"ibm\.com/docs/en/([^/]+)/([^/\s?#]+)")

# 等待 JS 渲染完成的秒數（IBM Docs 通常 3~5 秒）
_JS_WAIT = 4.0

# 頁面骨架（導覽殼）的開頭特徵：IBM Docs 是單頁應用，內容尚未渲染完成時，
# 主文選擇器抓不到任何東西，退回整個 body 就會得到這段導覽文字而不是主文。
_SKELETON_PREFIX = "Documentation\nMy IBM\nLog in"
# 偵測到「主文尚未出現」時的補救：先多等一會兒重讀，再重新載入一次，仍失敗才放棄
_EMPTY_CONTENT_EXTRA_WAIT = 8.0

# 沒有主文的頁面（版本首頁／landing page）只有目錄連結。範圍內連結達此數量才視為
# landing page：不寫入 chunk，但記錄為已完成並用它的連結展開佇列。
# 連結很少的多半是尚未渲染完成的殼，維持「不記錄、下次再試」。
_LANDING_MIN_LINKS = 5

# 載入失敗的重試設定
_MAX_RETRIES = 3          # 最多重試次數（不含首次嘗試）
_RETRY_BASE_DELAY = 5.0   # 首次重試等待秒數，之後指數倍增（5 → 10 → 20）
_NETWORK_DOWN_DELAY = 30.0  # 偵測到網路中斷時的額外等待秒數


def _parse_meta(url: str) -> dict:
    m = _DOCS_RE.search(url)
    if m:
        return {"product": m.group(1), "version": m.group(2)}
    return {"product": "unknown", "version": "unknown"}


# 舊式網址：IBM 部分版本（例如 8.6.0）的目錄連結是 …/docs/en/STSLR9_8.6.x/<doc>/<file>.html，
# 載入後會被導向正式網址 …/flashsystem-9x00/8.6.0?topic=…。
# 這種連結不含根網址，過去被範圍過濾擋掉，導致 8.6.0 只爬到根頁附近的少數頁面。
_LEGACY_LINK_RE = re.compile(r"ibm\.com/docs/en/[A-Za-z0-9]+_(\d+\.\d+)\.x/")
# 舊式連結目錄裡還有 PDF、影片等檔案連結，載入後是 404 頁（"This isn't the page you were looking for"），
# 只放行網頁（.htm／.html）。
_LEGACY_PAGE_EXT = (".htm", ".html")


def _in_scope(link: str, root_clean: str) -> bool:
    """連結是否屬於這個根網址：網址在根網址底下，或是同一個 主.次 版本的舊式「網頁」網址。"""
    if root_clean in link:
        return True
    m = _LEGACY_LINK_RE.search(link)
    if not m or not link.split("?")[0].lower().endswith(_LEGACY_PAGE_EXT):
        return False
    version = _parse_meta(root_clean)["version"]
    return version != "unknown" and ".".join(version.split(".")[:2]) == m.group(1)


def _final_url(url: str, driver) -> str:
    """瀏覽器目前實際停留的網址（舊式網址會被導向正式網址）；取不到就沿用原網址。"""
    try:
        cur = (driver.current_url or "").split("#")[0].rstrip("/")
    except Exception:
        return url
    return cur or url


class _DriverHandle:
    """
    一個瀏覽器實例 + 它專屬的 chromedriver 副本與 Chrome profile 目錄。
    quit() 會一併清掉暫存檔，避免並行爬取後在 /tmp 留下殘骸。
    """

    def __init__(self, driver, tmp_dir: Optional[str] = None):
        self.driver = driver
        self._tmp_dir = tmp_dir

    def quit(self) -> None:
        try:
            self.driver.quit()
        finally:
            if self._tmp_dir:
                shutil.rmtree(self._tmp_dir, ignore_errors=True)


def _make_driver(isolated: bool = False):
    """
    建立 Selenium headless Chrome WebDriver。
    使用 webdriver-manager 自動下載對應版本的 chromedriver。

    isolated=True（並行爬取用）：每個實例使用自己複製出來的 chromedriver
    與獨立的 Chrome profile，多個瀏覽器之間才不會互相干擾。
    回傳值：isolated=False 回傳 driver 本身（相容舊用法）；
            isolated=True 回傳 _DriverHandle（需呼叫 quit() 清理暫存）。
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    tmp_dir = tempfile.mkdtemp(prefix="rag_crawl_") if isolated else None

    options = Options()
    if isolated:
        options.add_argument(f"--user-data-dir={tmp_dir}/profile")
    options.add_argument("--headless=new")          # headless 模式
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,800")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    driver_path = ChromeDriverManager().install()
    if isolated:
        # 每個實例用自己的 chromedriver 副本。實測同一個路徑的 chromedriver
        # 同時啟動第二個實例會被系統終止（SIGKILL），複製到不同路徑則可並行。
        own_path = os.path.join(tmp_dir, "chromedriver")
        shutil.copy2(driver_path, own_path)
        os.chmod(own_path, 0o755)
        driver_path = own_path

    try:
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        # 隱藏 webdriver 特徵
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
    except Exception:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return _DriverHandle(driver, tmp_dir) if isolated else driver


def _load_with_retry(url: str, driver) -> bool:
    """
    載入頁面，失敗時以指數退避重試。

    Returns:
        True 表示載入成功；False 表示重試用盡仍失敗。
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            driver.get(url)
            time.sleep(_JS_WAIT)  # 等待 JS 渲染完成
            return True
        except Exception as e:
            # 只取錯誤訊息第一行，避免印出整段 Stacktrace
            first_line = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            if attempt >= _MAX_RETRIES:
                print(f"[ERROR] 放棄載入 {url}（已重試 {_MAX_RETRIES} 次）：{first_line}")
                return False

            wait = _RETRY_BASE_DELAY * (2 ** attempt)
            if "ERR_INTERNET_DISCONNECTED" in first_line or "ERR_NAME_NOT_RESOLVED" in first_line:
                wait = max(wait, _NETWORK_DOWN_DELAY)  # 網路中斷，多等一會兒
            print(
                f"[WARN] 載入失敗，{wait:.0f} 秒後重試 "
                f"({attempt + 1}/{_MAX_RETRIES})：{url}\n       {first_line}"
            )
            time.sleep(wait)
    return False


def _extract_content(url: str, driver) -> tuple[Optional[str], list[str]]:
    """
    解析目前瀏覽器中的頁面，回傳 (主文, 子頁面連結)。
    主文區塊沒出現（尚未渲染完成）或抓到的是導覽骨架時，主文回傳 None。
    """
    soup = BeautifulSoup(driver.page_source, "lxml")

    # 從整頁收集同網域子連結（在移除 nav 之前）
    # IBM Docs TOC 在 nav 裡，必須先收集連結再移除
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    url_clean = url.split("#")[0].rstrip("/")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        href_clean = href.split("#")[0].rstrip("/")
        if href_clean.startswith(base) and href_clean != url_clean:
            links.append(href_clean)

    # 移除 nav / footer / script / style（取連結後再移除，不影響主文擷取）
    for tag in soup.find_all(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    # 找主文區塊。全部選擇器都沒命中 = 內容還沒渲染出來，
    # 不可退回整個 body（會把導覽骨架當成主文寫進向量庫）。
    content_el = None
    for sel in _CONTENT_SELECTORS:
        content_el = soup.select_one(sel)
        if content_el:
            break
    if content_el is None:
        return None, links

    text = content_el.get_text(separator="\n", strip=True)
    if text.startswith(_SKELETON_PREFIX):
        return None, links
    return text, links


def _fetch_text(url: str, driver) -> tuple[str, list[str], str]:
    """
    用 Selenium 渲染頁面，回傳 (主文字串, 子頁面連結列表, 實際網址)。
    實際網址是載入後瀏覽器停留的網址：舊式網址會被導向正式網址，一般頁面則與 url 相同。
    載入失敗會自動重試；重試用盡仍失敗則回傳空內容（呼叫端不會記錄為已完成，下次續爬會再試）。

    主文沒出現時（IBM Docs 偶爾回傳尚未渲染完成的頁面）：
      1) 多等一會兒後重新讀取目前頁面；
      2) 仍沒有就重新載入一次；
      3) 還是沒有就放棄，絕不把導覽骨架當主文回傳。
    """
    if not _load_with_retry(url, driver):
        return "", [], url

    text, links = _extract_content(url, driver)
    if text is None:
        time.sleep(_EMPTY_CONTENT_EXTRA_WAIT)
        text, links = _extract_content(url, driver)
    if text is None and _load_with_retry(url, driver):
        time.sleep(_EMPTY_CONTENT_EXTRA_WAIT)
        text, links = _extract_content(url, driver)
    if text is None:
        print(f"[WARN] 頁面主文未出現，略過（不記錄，下次續爬會再試）：{url}")
        return "", links, url

    return text, links, _final_url(url, driver)


def _crawl_urls(
    root_urls: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    max_pages_per_root: int = 200,
    delay: float = 0.5,
    on_flush: Optional[Callable[[List[Document]], None]] = None,
    flush_every: int = 100,
    state_file: Optional[str] = None,
    workers: int = 1,
) -> List[Document]:
    """
    核心爬取邏輯：對 root_urls 逐一做廣度優先爬取。

    Args:
        root_urls:           已清洗的根網址列表（無空行、無 # 註解）。
        chunk_size:          段落最大字元數。
        chunk_overlap:       段落重疊字元數。
        max_pages_per_root:  每個根網址最多爬取的子頁面數。
        delay:               每次請求間的額外等待秒數（JS 等待之外）。
        on_flush:            每累積 flush_every 個段落（以及結束 / 發生例外時的剩餘段落）
                             就以該批 Document 呼叫一次，用於邊爬邊寫入向量庫。
        flush_every:         觸發 on_flush 的段落數（預設 100）。
        state_file:          續爬記錄檔路徑。已記錄的頁面重啟後不再重爬，
                             改用記錄中的子連結延伸佇列。None 表示不使用。
                             注意：需搭配 on_flush，否則無法確保資料已寫入。
        workers:             同時使用的瀏覽器數量（預設 1 = 原本的單瀏覽器行為）。
                             只有「載入網頁」會並行；佇列、去重、分塊、寫入向量庫與
                             續爬記錄都由主執行緒單獨處理，因此「寫入成功才記錄」不變。

    Returns:
        List[Document]: 帶有 metadata 的文字段落列表（包含已 flush 的段落）。
    """
    workers = max(1, int(workers))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # 瀏覽器池。workers=1 維持原本行為（單一、非隔離的 driver，同步抓取）。
    # workers>1 時每個瀏覽器各自一份 chromedriver 副本與 profile。
    parallel = workers > 1
    print(f"[INFO] 啟動 {workers} 個 headless Chrome...")
    handles: list = []
    try:
        for _ in range(workers):
            handles.append(_make_driver(isolated=parallel))
    except Exception:
        for h in handles:
            try:
                h.quit()
            except Exception:
                pass
        raise
    free_drivers: deque = deque(handles)   # 目前閒置的瀏覽器
    executor = ThreadPoolExecutor(max_workers=workers) if parallel else None

    def _submit(url: str):
        """
        把載入工作交給一個閒置瀏覽器，回傳 (future, handle)。
        呼叫端必須在取得結果後把 handle 放回 free_drivers。
        """
        h = free_drivers.popleft()
        raw = h.driver if parallel else h
        return executor.submit(_fetch_text, url, raw), h

    state = open_state(state_file) if on_flush else None
    if state_file and not on_flush:
        print("[WARN] 未提供 on_flush，無法確認資料已寫入，已停用續爬記錄。")
    if state is not None and len(state):
        print(f"[INFO] 續爬記錄：已有 {len(state)} 個頁面完成，將直接跳過（{state.path}）")

    all_docs: List[Document] = []
    pending: List[Document] = []  # 尚未 flush 的段落
    # 已處理但尚未寫入向量庫的頁面 (url, links)；flush 成功後才寫入續爬記錄
    pending_pages: List[tuple] = []

    # 進度統計（只計算 flush 成功、確實已寫入的部分）
    written_chunks = 0                       # 本次執行已寫入的段落數
    written_pages = 0                        # 本次執行已完成的頁面數
    resumed_pages = len(state) if state is not None else 0  # 啟動前已完成的頁面數

    def _flush() -> None:
        nonlocal pending, pending_pages, written_chunks, written_pages
        if on_flush and (pending or pending_pages):
            batch, pending = pending, []
            pages, pending_pages = pending_pages, []
            if batch:
                on_flush(batch)          # 例外會往外拋，此時不記錄這些頁面
            if state is not None:
                state.commit(pages)      # 確定寫入成功後才標記完成
            written_chunks += len(batch)
            written_pages += len(pages)
            # tqdm.write 會自動避開進度條，不會把畫面弄亂
            tqdm.write(
                f"[進度] 已完成 {resumed_pages + written_pages} 頁"
                f"（本次 +{written_pages}，先前 {resumed_pages}）"
                f"｜本次已寫入 {written_chunks} 個段落（這批 {len(batch)}）"
            )

    # global_visited 只用來去除真正跨根網址的重複頁面（完全相同 URL）
    # 不同版本的相同 topic 應各自爬取，故以 root_clean 為 key 分開管理
    global_visited: Set[str] = set()

    def _handle_page(url, text, child_links, final_url, meta, root_clean,
                     visited, queue, queued) -> None:
        """處理一頁的抓取結果：分塊、累積、必要時 flush、延伸佇列。只在主執行緒呼叫。"""
        if not text:
            # 載入失敗（重試用盡）或主文未出現：通常不記錄，下次重啟會再試一次。
            # 例外：沒有主文、但範圍內連結夠多的 landing page，只用來展開佇列。
            scoped = [l for l in child_links if _in_scope(l, root_clean)]
            if len(scoped) < _LANDING_MIN_LINKS:
                return
            pending_pages.append((url, child_links))
            if len(pending_pages) >= flush_every:
                _flush()
            for link in scoped:
                if link not in visited and link not in queued:
                    queued.add(link)
                    queue.append(link)
            return

        # 只有「舊式網址」才改用導向後的正式網址當 source；一般網址維持原行為。
        # 導向到根網址以外的頁面、或正式網址已經抓過（或正在抓）的頁面，不重複收錄。
        redirected = bool(_LEGACY_LINK_RE.search(url)) and final_url != url
        source = final_url if redirected else url
        store = len(text.strip()) > 100  # 過濾空頁/403頁
        if redirected:
            if root_clean not in final_url or final_url in visited:
                store = False
            else:
                visited.add(final_url)

        if store:
            for chunk in splitter.split_text(text):
                if not chunk.strip():
                    continue
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "source_type": "web",
                        "product": meta["product"],
                        "version": meta["version"],
                    },
                )
                all_docs.append(doc)
                pending.append(doc)

        # 頁面處理完畢，等 flush 成功後才記錄為已完成（舊式與正式網址都記，續爬時都不再重爬）
        pending_pages.append((url, child_links))
        if redirected:
            pending_pages.append((final_url, child_links))

        # 累積達 flush_every 個段落就先寫入
        # （頁面數過多但段落很少時也要定期 flush，避免記錄長期落後）
        if len(pending) >= flush_every or len(pending_pages) >= flush_every:
            _flush()

        # 只加入同根網址底下的子頁面（含同版本的舊式網址）
        for link in child_links:
            if _in_scope(link, root_clean) and link not in visited and link not in queued:
                queued.add(link)
                queue.append(link)

    try:
        for root_url in tqdm(root_urls, desc="爬取根網址"):
            meta = _parse_meta(root_url)
            root_clean = root_url.split("#")[0].rstrip("/")
            # 用 deque（popleft 是 O(1)），並以 queued 記錄「已排入佇列」的網址。
            # IBM Docs 每頁都帶整份目錄的數千個連結，若不在入列前去重，
            # 續爬時佇列會膨脹到數百萬筆，list.pop(0) 又是 O(n)，整個程式卡在這裡。
            queue: deque = deque([root_clean])
            queued: Set[str] = {root_clean}
            # 每個根網址用獨立的 visited，不與其他根網址共享
            visited: Set[str] = set()
            page_count = 0   # 已「送出抓取」的頁數（含尚在進行中的），用來限制 max_pages
            # 進行中的抓取，依送出順序排列：(url, future, handle)。
            # 結果一律依送出順序處理，使 BFS 順序與單瀏覽器完全一致。
            inflight: deque = deque()

            def _drain_one() -> None:
                """等最早送出的那一頁完成，處理它的結果並歸還瀏覽器。"""
                url, fut, h = inflight.popleft()
                try:
                    text, child_links, final_url = fut.result()
                finally:
                    free_drivers.append(h)
                _handle_page(url, text, child_links, final_url, meta, root_clean,
                             visited, queue, queued)
                if delay > 0:
                    time.sleep(delay)

            while True:
                # 1) 還有閒置瀏覽器、佇列有網址、且 max_pages 沒滿 → 再送出一頁
                if queue and page_count < max_pages_per_root and free_drivers:
                    url = queue.popleft()
                    if url in visited:
                        continue
                    visited.add(url)

                    # 已完成的頁面：不重爬，直接取用記錄的子連結延伸佇列
                    if state is not None and state.is_done(url):
                        for link in state.links_of(url):
                            if _in_scope(link, root_clean) and link not in visited and link not in queued:
                                queued.add(link)
                                queue.append(link)
                        continue  # 不佔用 max_pages 額度，續爬時可往更深處前進

                    page_count += 1
                    if parallel:
                        fut, h = _submit(url)
                        inflight.append((url, fut, h))
                        continue
                    # workers=1：同步抓取，行為與改寫前完全相同
                    text, child_links, final_url = _fetch_text(url, free_drivers[0])
                    _handle_page(url, text, child_links, final_url, meta, root_clean,
                                 visited, queue, queued)
                    if delay > 0:
                        time.sleep(delay)
                    continue

                # 2) 沒有可送出的了：若還有進行中的，等最早的一頁處理完，
                #    它可能帶來新的子連結讓佇列繼續；否則這個根網址結束
                if inflight:
                    _drain_one()
                    continue
                break

            # 爬完一個根網址後，把已訪問的 URL 加入 global_visited
            # 確保完全相同的 URL 在不同根網址間不重複爬取
            global_visited.update(visited)

    finally:
        try:
            _flush()  # 寫入剩餘不足 flush_every 的段落（含發生例外時）
        finally:
            # 先停止接新工作並等進行中的抓取結束，再關瀏覽器；
            # 若在抓取途中直接殺掉 Chrome，工作執行緒會卡在 driver.get。
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            for h in handles:
                try:
                    h.quit()
                except Exception:
                    pass
            print("[INFO] headless Chrome 已關閉")

    print(f"[INFO] 網頁載入完成：{len(root_urls)} 個根網址 → {len(all_docs)} 個段落")
    if on_flush:
        print(
            f"[完成] 本次寫入 {written_chunks} 個段落、完成 {written_pages} 頁；"
            f"累計已完成 {resumed_pages + written_pages} 頁"
        )
    return all_docs


def load_urls(
    url_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    max_pages_per_root: int = 200,
    delay: float = 0.5,
    on_flush: Optional[Callable[[List[Document]], None]] = None,
    flush_every: int = 100,
    state_file: Optional[str] = None,
    workers: int = 1,
) -> List[Document]:
    """
    從 txt 檔讀取根網址清單，爬取每個根網址及其子頁面。（CLI 用）

    Args:
        url_file:            包含 URL 的文字檔（# 開頭為註解）。
        chunk_size:          段落最大字元數。
        chunk_overlap:       段落重疊字元數。
        max_pages_per_root:  每個根網址最多抓取的子頁面數。
        delay:               每次 HTTP 請求之間的等待秒數。
        on_flush:            每累積 flush_every 個段落就呼叫一次（邊爬邊寫入）。
        flush_every:         觸發 on_flush 的段落數（預設 100）。
        state_file:          續爬記錄檔路徑；已完成的頁面重啟後不再重爬。
        workers:             同時使用的瀏覽器數量（預設 1；建議 2~3）。

    Returns:
        List[Document]: 帶有 metadata 的文字段落列表。
    """
    url_file = Path(url_file)
    if not url_file.is_file():
        raise FileNotFoundError(f"URL 檔案不存在：{url_file}")

    root_urls = [
        line.strip()
        for line in url_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not root_urls:
        print("[WARN] URL 檔案內沒有有效網址。")
        return []

    return _crawl_urls(
        root_urls, chunk_size, chunk_overlap, max_pages_per_root, delay,
        on_flush=on_flush, flush_every=flush_every, state_file=state_file,
        workers=workers,
    )


def load_urls_from_list(
    urls: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    max_pages_per_root: int = 200,
    delay: float = 0.5,
    on_flush: Optional[Callable[[List[Document]], None]] = None,
    flush_every: int = 100,
    state_file: Optional[str] = None,
    workers: int = 1,
) -> List[Document]:
    """
    直接接受 URL 字串列表，爬取並回傳段落。（GUI 用）

    空行與 # 開頭的項目會自動略過，與 load_urls() 行為一致。

    Args:
        urls:                URL 字串列表（可含空字串或 # 註解行）。
        chunk_size:          段落最大字元數。
        chunk_overlap:       段落重疊字元數。
        max_pages_per_root:  每個根網址最多抓取的子頁面數。
        delay:               每次 HTTP 請求之間的等待秒數。
        on_flush:            每累積 flush_every 個段落就呼叫一次（邊爬邊寫入）。
        flush_every:         觸發 on_flush 的段落數（預設 100）。
        state_file:          續爬記錄檔路徑；已完成的頁面重啟後不再重爬。
        workers:             同時使用的瀏覽器數量（預設 1；建議 2~3）。

    Returns:
        List[Document]: 帶有 metadata 的文字段落列表。
    """
    root_urls = [
        u.strip()
        for u in urls
        if u.strip() and not u.strip().startswith("#")
    ]

    if not root_urls:
        print("[WARN] 沒有有效的 URL。")
        return []

    return _crawl_urls(
        root_urls, chunk_size, chunk_overlap, max_pages_per_root, delay,
        on_flush=on_flush, flush_every=flush_every, state_file=state_file,
        workers=workers,
    )


if __name__ == "__main__":
    import sys

    url_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/yhkuo/RAG/raw_data/web_urls.txt"
    docs = load_urls(url_file, max_pages_per_root=3)
    if docs:
        print(f"範例段落：\n{docs[0].page_content[:300]}\nMetadata：{docs[0].metadata}")
    else:
        print("沒有取得任何段落。")

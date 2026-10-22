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

import re
import time
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

# 載入失敗的重試設定
_MAX_RETRIES = 3          # 最多重試次數（不含首次嘗試）
_RETRY_BASE_DELAY = 5.0   # 首次重試等待秒數，之後指數倍增（5 → 10 → 20）
_NETWORK_DOWN_DELAY = 30.0  # 偵測到網路中斷時的額外等待秒數


def _parse_meta(url: str) -> dict:
    m = _DOCS_RE.search(url)
    if m:
        return {"product": m.group(1), "version": m.group(2)}
    return {"product": "unknown", "version": "unknown"}


def _make_driver():
    """
    建立 Selenium headless Chrome WebDriver。
    使用 webdriver-manager 自動下載對應版本的 chromedriver。
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
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

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    # 隱藏 webdriver 特徵
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


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


def _fetch_text(url: str, driver) -> tuple[str, list[str]]:
    """
    用 Selenium 渲染頁面，回傳 (主文字串, 子頁面連結列表)。
    載入失敗會自動重試；重試用盡仍失敗則回傳空內容。
    """
    if not _load_with_retry(url, driver):
        return "", []

    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")

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

    # 找主文區塊
    content_el = None
    for sel in _CONTENT_SELECTORS:
        content_el = soup.select_one(sel)
        if content_el:
            break
    if content_el is None:
        content_el = soup.body or soup

    text = content_el.get_text(separator="\n", strip=True)

    return text, links


def _crawl_urls(
    root_urls: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    max_pages_per_root: int = 200,
    delay: float = 0.5,
    on_flush: Optional[Callable[[List[Document]], None]] = None,
    flush_every: int = 100,
    state_file: Optional[str] = None,
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

    Returns:
        List[Document]: 帶有 metadata 的文字段落列表（包含已 flush 的段落）。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    print("[INFO] 啟動 headless Chrome...")
    driver = _make_driver()

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

    try:
        for root_url in tqdm(root_urls, desc="爬取根網址"):
            meta = _parse_meta(root_url)
            root_clean = root_url.split("#")[0].rstrip("/")
            queue: List[str] = [root_clean]
            # 每個根網址用獨立的 visited，不與其他根網址共享
            visited: Set[str] = set()
            page_count = 0

            while queue and page_count < max_pages_per_root:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)

                # 已完成的頁面：不重爬，直接取用記錄的子連結延伸佇列
                if state is not None and state.is_done(url):
                    for link in state.links_of(url):
                        if root_clean in link and link not in visited:
                            queue.append(link)
                    continue  # 不佔用 max_pages 額度，續爬時可往更深處前進

                text, child_links = _fetch_text(url, driver)
                page_count += 1

                if not text:
                    # 載入失敗（重試用盡）：不記錄，下次重啟會再試一次
                    if delay > 0:
                        time.sleep(delay)
                    continue

                if len(text.strip()) > 100:  # 過濾空頁/403頁
                    chunks = splitter.split_text(text)
                    for chunk in chunks:
                        if not chunk.strip():
                            continue
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": url,
                                "source_type": "web",
                                "product": meta["product"],
                                "version": meta["version"],
                            },
                        )
                        all_docs.append(doc)
                        pending.append(doc)

                # 頁面處理完畢，等 flush 成功後才記錄為已完成
                pending_pages.append((url, child_links))

                # 累積達 flush_every 個段落就先寫入
                # （頁面數過多但段落很少時也要定期 flush，避免記錄長期落後）
                if len(pending) >= flush_every or len(pending_pages) >= flush_every:
                    _flush()

                # 只加入同根網址底下的子頁面
                for link in child_links:
                    if root_clean in link and link not in visited:
                        queue.append(link)

                if delay > 0:
                    time.sleep(delay)

            # 爬完一個根網址後，把已訪問的 URL 加入 global_visited
            # 確保完全相同的 URL 在不同根網址間不重複爬取
            global_visited.update(visited)

    finally:
        try:
            _flush()  # 寫入剩餘不足 flush_every 的段落（含發生例外時）
        finally:
            driver.quit()
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
    )


if __name__ == "__main__":
    import sys

    url_file = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/SSD_4TB/RAG/raw_data/web_urls.txt"
    docs = load_urls(url_file, max_pages_per_root=3)
    if docs:
        print(f"範例段落：\n{docs[0].page_content[:300]}\nMetadata：{docs[0].metadata}")
    else:
        print("沒有取得任何段落。")

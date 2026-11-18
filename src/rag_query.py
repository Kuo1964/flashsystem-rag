"""
rag_query.py
============
兩階段 FlashSystem 專家系統。

第一層（本機 RAG 過濾）：
  - 向量相似度搜尋，從 Chroma 取得候選段落
  - 本機評分：去重、相關性篩選、依 source_type 優先排序
  - 輸出：精煉後的 Context 與引用清單

第二層（LLM 深度推理）：
  - 將第一層的 Context 連同原始問題送給 LLM
  - 使用結構化 Prompt 要求輸出：診斷 / 操作步驟 / 風險提示 / 引用來源
  - 輸出：ExpertAnswer（結構化 dataclass）
  - 支援 LLM 供應商：watsonx（預設）、gemini
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ibm import ChatWatsonx

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """第一層過濾的輸出。"""
    question: str
    chunks: List[Document]          # 篩選後的相關段落（已去重排序）
    context_text: str               # 組合成字串的 context，供第二層使用
    source_refs: List[str]          # 引用來源列表
    model_mismatch_warning: str = ""  # 若查詢型號與文件型號不符時的警告訊息


@dataclass
class ExpertAnswer:
    """第二層推理的輸出。"""
    question: str
    diagnosis: str                  # 問題診斷／背景說明
    steps: List[str]                # 操作步驟或解答要點
    risks: List[str]                # 風險提示或注意事項
    summary: str                    # 總結
    sources: List[str]              # 引用來源
    raw_llm_output: str             # LLM 原始輸出（供 debug）
    filter_result: FilterResult     # 保留第一層結果供查閱


# ─────────────────────────────────────────────────────────────────────────────
# 第一層：本機 RAG 過濾器
# ─────────────────────────────────────────────────────────────────────────────

# source_type 優先權（分數越高越優先放在 context 前面）
_SOURCE_PRIORITY: Dict[str, int] = {
    "json_cli":      10,   # CLI 指令白名單：最精確
    "json_feature":  9,    # 功能生命週期：版本判斷關鍵
    "json_parts":    8,    # 零件清單：硬體相關
    "pdf":           5,    # PDF 手冊：詳細但較長
    "web":           4,    # IBM Docs 網頁：廣泛但可能重複
}

# context 最大字元數（防止超過 LLM token 上限）
# 命中段落會展開成「同頁前後文」，單筆變長，因此上限由 6000 提高到 9000
_MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "9000"))
_MODEL_BOOST_K     = 5      # 型號強制補入時，額外用 metadata filter 搜尋的 chunk 數

# 命中段落向前後各展開幾個同頁相鄰段落（0 = 不展開）
_NEIGHBOR_WINDOW = int(os.environ.get("RAG_NEIGHBOR_WINDOW", "1"))
# 只展開這些來源（JSON 本身已是完整條目，不需展開）
_EXPAND_SOURCE_TYPES = {"web", "pdf"}
# 相鄰段落重疊的偵測範圍（字元）；切割時 overlap=150
_OVERLAP_MIN, _OVERLAP_MAX = 20, 300


def _score_chunk(doc: Document) -> float:
    """給每個 chunk 計算優先分數，分數越高越優先。"""
    src_type = doc.metadata.get("source_type", "unknown")
    return float(_SOURCE_PRIORITY.get(src_type, 1))


def _deduplicate(docs: List[Document], threshold: int = 100) -> List[Document]:
    """
    移除內容高度重複的段落（頭100字相同即視為重複）。
    保留分數較高的那一個。
    """
    seen: Dict[str, Document] = {}
    for doc in docs:
        key = doc.page_content[:threshold].strip()
        if key not in seen:
            seen[key] = doc
    return list(seen.values())


def _join_chunks(a: str, b: str) -> str:
    """接合兩個相鄰段落：若 a 的結尾與 b 的開頭重疊（切割時的 overlap）則去掉重疊部分。"""
    for k in range(min(len(a), len(b), _OVERLAP_MAX), _OVERLAP_MIN - 1, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + "\n" + b


def _page_where(doc: Document) -> Optional[Dict[str, Any]]:
    """取得「同一頁」的 metadata 過濾條件；web 以網址為一頁，pdf 以 檔名+頁碼 為一頁。"""
    src = doc.metadata.get("source")
    if not src:
        return None
    if doc.metadata.get("source_type") == "pdf":
        page = doc.metadata.get("page")
        if page is None:
            return None
        return {"$and": [{"source": {"$eq": src}}, {"page": {"$eq": page}}]}
    return {"source": {"$eq": src}}


def _expand_with_neighbors(
    docs: List[Document],
    vectorstore: Chroma,
    window: int = _NEIGHBOR_WINDOW,
) -> List[Document]:
    """
    將命中的 chunk 展開成「同頁前後文」。

    - 同一頁有多個 chunk 命中時合併成一筆，排在第一個命中者的位置。
    - 每個命中 chunk 向前後各取 window 個同頁相鄰 chunk；不連續的區段以 "..." 分隔。
    - 相鄰 chunk 的 overlap 會去重。
    - 找不到同頁資料或查詢失敗時，原樣保留該 chunk。
    """
    if window <= 0:
        return docs

    groups: Dict[str, Dict[str, Any]] = {}   # key → {"first": Document, "texts": set}
    order: List[Any] = []                    # 保持原排序；元素為 key 或原 Document
    for doc in docs:
        where = _page_where(doc) if doc.metadata.get("source_type") in _EXPAND_SOURCE_TYPES else None
        if where is None:
            order.append(doc)
            continue
        key = repr(where)
        if key not in groups:
            groups[key] = {"first": doc, "where": where, "hits": []}
            order.append(key)
        groups[key]["hits"].append(doc.page_content)

    result: List[Document] = []
    for item in order:
        if isinstance(item, Document):
            result.append(item)
            continue
        g = groups[item]
        first: Document = g["first"]
        try:
            page_chunks: List[str] = vectorstore._collection.get(
                where=g["where"], include=["documents"]
            )["documents"]
        except Exception:
            result.extend(Document(page_content=t, metadata=first.metadata) for t in g["hits"])
            continue

        wanted: set = set()
        for hit in g["hits"]:
            if hit in page_chunks:
                i = page_chunks.index(hit)
                wanted.update(range(max(0, i - window), min(len(page_chunks), i + window + 1)))
        if not wanted:
            result.extend(Document(page_content=t, metadata=first.metadata) for t in g["hits"])
            continue

        text = ""
        prev = None
        for i in sorted(wanted):
            if prev is None:
                text = page_chunks[i]
            elif i == prev + 1:
                text = _join_chunks(text, page_chunks[i])
            else:
                text = text + "\n...\n" + page_chunks[i]
            prev = i
        result.append(Document(page_content=text, metadata=first.metadata))
    return result


# ── 識別碼精確比對（錯誤碼／CLI 指令名）──────────────────────────────────────
# 向量檢索分不出 CMMVC5770E 與 CMMVC5767E（語意幾乎相同），所以問題含這類識別碼時，
# 直接用「網址最後一段 topic 名稱」找出對應頁面（例如 ...?topic=messages-cmmvc5770e），
# 補入候選。每個版本／產品各有一頁，取最新版本優先的前幾頁。
_ERROR_CODE_RE = re.compile(r"\b(CMM[A-Z]{2}\d{4}[A-Z]?)\b", re.IGNORECASE)
_EXACT_PAGES_PER_ID = int(os.environ.get("RAG_EXACT_PAGES_PER_ID", "4"))
_EXACT_MAX_IDS = 3
# 這些 CLI 指令名同時是一般英文單字（如 "host connection"），單獨出現時不視為指令，避免誤判
_AMBIGUOUS_CMD_WORDS = frozenset({
    "backup", "clear", "help", "host", "metadata", "ping", "restore", "snap", "traceroute",
    "test", "exit", "include", "lock", "unlock", "start", "stop", "view", "quit", "system",
    "volume", "node", "port", "user", "license",
})

_topic_index_lock = threading.Lock()
_topic_index_cache: Dict[str, Dict[str, List[str]]] = {}


def _persist_dir(vectorstore: Chroma) -> Optional[str]:
    try:
        return vectorstore._client.get_settings().persist_directory
    except Exception:
        return None


def _load_topic_index(vectorstore: Chroma) -> Dict[str, List[str]]:
    """
    建立 {topic 名稱(小寫) → [網址, ...]} 索引，只在第一次查詢時載入並快取。
    以唯讀方式直接讀 Chroma 的 SQLite（約 5 萬個網址，耗時不到 0.1 秒）。
    """
    pdir = _persist_dir(vectorstore)
    if not pdir:
        return {}
    with _topic_index_lock:
        if pdir in _topic_index_cache:
            return _topic_index_cache[pdir]
        index: Dict[str, List[str]] = {}
        try:
            db = sqlite3.connect(f"file:{pdir}/chroma.sqlite3?mode=ro", uri=True, timeout=5)
            try:
                rows = db.execute(
                    "SELECT DISTINCT string_value FROM embedding_metadata "
                    "WHERE key='source' AND string_value LIKE 'http%'"
                )
                for (url,) in rows:
                    if "topic=" in url:
                        index.setdefault(url.rsplit("topic=", 1)[-1].lower(), []).append(url)
            finally:
                db.close()
        except Exception:
            index = {}
        _topic_index_cache[pdir] = index
        return index


def _extract_identifiers(question: str, index: Dict[str, List[str]]) -> List[str]:
    """
    從問題抽出要精確比對的 topic 名稱。
      - 錯誤碼：CMMVC5770E → messages-cmmvc5770e
      - CLI 指令：問題中的英數單字若剛好是 commands-<單字> 頁面（如 lssystem、mkvolume），
        但排除同時是一般英文單字的指令名（host、backup 等）
    只接受索引裡真的存在的 topic，所以不會因誤判而回傳不存在的頁面。
    """
    found: List[str] = []
    for code in _ERROR_CODE_RE.findall(question):
        key = f"messages-{code.lower()}"
        if key in index and key not in found:
            found.append(key)
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", question):
        w = word.lower()
        if w in _AMBIGUOUS_CMD_WORDS:
            continue
        key = f"commands-{w}"
        if key in index and key not in found:
            found.append(key)
    return found[:_EXACT_MAX_IDS]


def _url_version_key(url: str) -> Tuple[int, ...]:
    """網址形如 .../flashsystem-5x00/9.1.3?topic=... → (9, 1, 3)，用於新版本優先排序。"""
    m = re.search(r"/(\d+(?:\.\d+)*)\?topic=", url)
    return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)


def _exact_identifier_docs(
    question: str,
    vectorstore: Chroma,
    where_filter: Optional[Dict[str, Any]] = None,
    pages_per_id: int = _EXACT_PAGES_PER_ID,
) -> List[Document]:
    """
    取回問題中錯誤碼／指令名對應的網頁 chunk（依寫入順序）。

    - 同一識別碼在不同產品／版本各有一頁；使用者有指定 version 時只取該版本，
      否則取最新版本優先的前 pages_per_id 頁，避免塞入太多幾乎相同的內容。
    - 查詢失敗或索引不可用時回傳空列表，不影響原本的向量搜尋流程。
    """
    index = _load_topic_index(vectorstore)
    if not index:
        return []
    wanted_version = None
    if where_filter:
        v = where_filter.get("version")
        if isinstance(v, dict):
            wanted_version = v.get("$eq")

    docs: List[Document] = []
    for key in _extract_identifiers(question, index):
        urls = index[key]
        if wanted_version:
            urls = [u for u in urls if f"/{wanted_version}?" in u]
        urls = sorted(urls, key=_url_version_key, reverse=True)[:pages_per_id]
        if not urls:
            continue
        try:
            r = vectorstore._collection.get(
                where={"source": {"$in": urls}}, include=["documents", "metadatas"]
            )
        except Exception:
            continue
        for d, m in zip(r["documents"], r["metadatas"]):
            docs.append(Document(page_content=d, metadata=m))
    return docs



# 從問題中抽取型號編號（4 位數），對應到可能的文件 source 檔名前綴
# 支援以下格式：FS7600、fs7600、FlashSystem 7600、FS 7600、7600（緊接中文時無 \b）
_BOOST_MODEL_RE = re.compile(
    r"(?:(?:flash\s*system\s*|fs\s*)(\d{4})(?:\b|(?=\s|$|[\u4e00-\u9fff]))"
    r"|(?<!\d)(\d{4})(?=\s*(?:節點|node|每個|電源|PSU|power)))",
    re.IGNORECASE,
)


def _boost_model_docs(
    question: str,
    vectorstore: Chroma,
    existing_docs: List[Document],
    boost_k: int = 5,
    where_filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    若問題中出現型號關鍵字（如 fs7600），以 metadata source filter 直接
    搜尋該型號文件，將結果補入 existing_docs 前端，確保相關文件一定被納入。

    這解決了 embedding 語意偏移問題：
    - 向量搜尋基於語意相似度，當型號間硬體描述用詞相近時（power supply、
      cabling 等），高版本多版本的 FS9200 文件會佔據前排，壓低 FS7600 排名。
    - 強制補入確保「問到哪個型號，就一定有那個型號的文件」。
    """
    # findall 回傳 list of tuples（因為有多個 group），取每個 tuple 中第一個非空的值
    raw_matches = _BOOST_MODEL_RE.findall(question)
    model_nums = {next((g for g in groups if g), None) for groups in raw_matches}
    model_nums.discard(None)
    if not model_nums:
        return existing_docs

    boosted: List[Document] = list(existing_docs)
    existing_keys = {doc.page_content[:100].strip() for doc in existing_docs}

    for model_num in model_nums:
        # 對應的 source 檔名模式，例如 7600 → "fs7600"
        source_pattern = f"fs{model_num}"

        # 用 metadata where filter 直接指定 source 檔名
        boost_filter: Dict[str, Any] = {"source": {"$eq": f"{source_pattern}_pdfguide.pdf"}}
        # 若使用者有額外的 where 條件（如 version），合併進去
        if where_filter:
            boost_filter = {"$and": [boost_filter, where_filter]}

        try:
            boost_docs = vectorstore.similarity_search(
                question,
                k=boost_k,
                filter=boost_filter,
            )
            for doc in boost_docs:
                key = doc.page_content[:100].strip()
                if key not in existing_keys:
                    existing_keys.add(key)
                    boosted.insert(0, doc)   # 插入最前面，確保不被 context 截斷
        except Exception:
            # filter 不符合（例如該型號無文件）時靜默忽略
            pass

    return boosted


def _decompose_expert_queries(question: str) -> List[str]:
    """
    根據問題中的領域關鍵字與意圖，將單一問題拆解/擴展為多個互補維度的檢索查詢矩陣。
    涵蓋：核心架構概念、容錯/高可用/維護、具體操作指令與配置、最佳實踐/遷移策略。
    """
    queries = [question]
    q_lower = question.lower()

    # 1. PBHA / 雙活 / 複製高可用
    if any(k in q_lower for k in ["pbha", "policy-based", "雙活", "双活", "high availability", "active-active", "replication policy", "ha policy"]):
        queries.append("Policy-Based High Availability replication volume group maintenance failover")
        queries.append("chvolumegroupreplication replicationpolicy high availability consistent_synchronized")

    # 2. DRAID / 硬碟更換 / 擴容限制 / 空間新增
    if any(k in q_lower for k in ["draid", "raid", "硬盤", "硬盘", "drive", "expand", "capacity", "擴充", "扩容", "更換", "更换", "新增", "空間", "空间"]):
        queries.append("DRAID distributed raid replace drives with larger capacity array expansion limitations")
        queries.append("charray expandarray drive upgrade distributed raid member drive class")
        queries.append("rmarray rmmdisk force migrate used extents online storage pool")
        queries.append("migratevdisk data migration between storage pools online addvdiskcopy")

    # 3. MDisk / Storage Pool / Extent 在線搬移 / 陣列移除
    if any(k in q_lower for k in ["mdisk", "pool", "extent", "遷移", "迁移", "migrate", "rmmdisk", "rmarray"]):
        queries.append("rmarray rmmdisk force migrate used extents online storage pool")
        queries.append("migratevdisk data migration between storage pools online addvdiskcopy")

    # 4. Safeguarded Copy / 快照 / 安全副本
    if any(k in q_lower for k in ["safeguarded", "snapshot", "快照", "備份", "备份", "ransomware"]):
        queries.append("Safeguarded Copy volume group snapshot policy backup recovery")

    # 5. SVC / 虛擬化控制器擴容
    if any(k in q_lower for k in ["svc", "san volume controller", "virtualize"]):
        queries.append("SVC detectmdisk addmdisk storage pool expansion backend storage")

    # 6. 錯誤碼或特定故障排除
    for err in _ERROR_CODE_RE.findall(question):
        queries.append(f"{err} error code description cause action resolution")

    # 7. CLI 指令名（英數單詞若可能為指令）
    for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_]{3,}\b", question):
        w = word.lower()
        if w in _AMBIGUOUS_CMD_WORDS:
            continue
        if any(w.startswith(prefix) for prefix in ("ls", "ch", "mk", "rm", "expand", "migrate", "detect", "add")):
            queries.append(f"{w} CLI command syntax parameters examples")

    return list(dict.fromkeys(queries))


def _reciprocal_rank_fusion(doc_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) 演算法：
    將多路檢索得到的候選文件清單依據 RRF 分數進行融合加權與去重。
    RRF(d) = sum(1.0 / (k + rank_i(d)))
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            key = doc.page_content[:120].strip()
            if key not in doc_map:
                doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank + 1))

    # 依 RRF 分數降序排列
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[x] for x in sorted_keys]


def _filter_noisy_chunks(docs: List[Document], question: str) -> List[Document]:
    """
    過濾與特定儲存架構或指令操作主題無關的通用物理/環境警語雜訊。
    """
    noise_patterns = [
        re.compile(r"open expansion enclosure lid reduces airflow", re.IGNORECASE),
        re.compile(r"do not leave the lid open for more than 15 minutes", re.IGNORECASE),
        re.compile(r"terms and conditions for information centers", re.IGNORECASE),
        re.compile(r"notices and trademarks", re.IGNORECASE),
    ]

    filtered: List[Document] = []
    for doc in docs:
        content = doc.page_content
        if any(pat.search(content) for pat in noise_patterns):
            continue
        filtered.append(doc)
    return filtered


class LocalRAGFilter:
    """
    第一層：多維度本機向量搜尋 + RRF 融合 + 品質過濾。
    完全在本機執行，不呼叫任何外部 API。
    """

    def __init__(self, vectorstore: Chroma, top_k: int = 30):
        """
        Args:
            vectorstore: 已載入的 Chroma 向量資料庫。
            top_k:       初始搜尋取得的候選段落數（過濾前）。
        """
        self.vectorstore = vectorstore
        self.top_k = top_k

    def filter(
        self,
        question: str,
        version: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> FilterResult:
        """
        執行本機多維度 RAG 檢索與融合過濾。

        Args:
            question:    使用者問題。
            version:     版本篩選（例如 "9.1.3"）。
            source_type: 來源類型篩選（"pdf" / "web" / "json_cli" 等）。

        Returns:
            FilterResult：包含篩選後段落、context 文字、引用來源。
        """
        where: Dict[str, Any] = {}
        if version:
            where["version"] = {"$eq": version}
        if source_type:
            where["source_type"] = {"$eq": source_type}

        search_kwargs: Dict[str, Any] = {
            "k": self.top_k,
            "fetch_k": self.top_k * 8,
        }
        if where:
            search_kwargs["filter"] = where

        # ── 1. 多維度查詢矩陣分解 ──
        sub_queries = _decompose_expert_queries(question)

        # ── 2. 多路並行檢索 ──
        multi_results: List[List[Document]] = []
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )
        for sq in sub_queries:
            try:
                docs = retriever.invoke(sq)
                multi_results.append(docs)
            except Exception:
                pass

        if not multi_results:
            raw_docs = []
        elif len(multi_results) == 1:
            raw_docs = multi_results[0]
        else:
            raw_docs = _reciprocal_rank_fusion(multi_results, k=60)

        # ── 3. 型號強制補入 ──
        raw_docs = _boost_model_docs(
            question=question,
            vectorstore=self.vectorstore,
            existing_docs=raw_docs,
            boost_k=_MODEL_BOOST_K,
            where_filter=where if where else None,
        )

        # ── 4. 錯誤碼／指令名精確比對 ──
        exact_docs = _exact_identifier_docs(question, self.vectorstore, where if where else None)
        exact_keys = {d.page_content[:100].strip() for d in exact_docs}

        # ── 5. 去重與雜訊過濾 ──
        deduped = _deduplicate(exact_docs + raw_docs)
        clean_docs = _filter_noisy_chunks(deduped, question)

        # ── 6. 依 source_type 優先分數排序；精確命中的頁面固定排在最前 ──
        ranked = sorted(
            clean_docs,
            key=lambda d: (d.page_content[:100].strip() in exact_keys, _score_chunk(d)),
            reverse=True,
        )

        # ── 7. 命中的 web / pdf chunk 展開成同頁前後文（同頁合併）──
        ranked = _expand_with_neighbors(ranked, self.vectorstore)

        # ── 5. 截斷到 context 上限 ──
        selected: List[Document] = []
        total_chars = 0
        for doc in ranked:
            length = len(doc.page_content)
            if total_chars + length > _MAX_CONTEXT_CHARS:
                continue   # 展開後單筆可能較長：跳過它，讓後面較短的仍有機會入選
            selected.append(doc)
            total_chars += length

        # ── 6. 組合 context 文字 ──
        context_parts = []
        for i, doc in enumerate(selected, 1):
            src  = doc.metadata.get("source", "unknown")
            ver  = doc.metadata.get("version", "")
            pg   = doc.metadata.get("page", "")
            label = f"[{i}] {src}"
            if ver:
                label += f" ({ver})"
            if pg:
                label += f" p.{pg}"
            context_parts.append(f"{label}\n{doc.page_content}")

        context_text = "\n\n---\n\n".join(context_parts)

        # ── 7. 引用來源列表 ──
        source_refs = []
        seen_src: set = set()
        for i, doc in enumerate(selected, 1):
            src = doc.metadata.get("source", "unknown")
            ver = doc.metadata.get("version", "")
            pg  = doc.metadata.get("page", "")
            key = (src, pg)
            if key not in seen_src:
                seen_src.add(key)
                ref = f"[{i}] {src}"
                if ver:
                    ref += f"  version={ver}"
                if pg:
                    ref += f"  p.{pg}"
                source_refs.append(ref)

        # ── 8. 型號不符警告偵測 ──
        # 從問題中抽取型號關鍵字，並與實際引用的文件來源比對
        mismatch_warning = _detect_model_mismatch(question, selected)

        return FilterResult(
            question=question,
            chunks=selected,
            context_text=context_text,
            source_refs=source_refs,
            model_mismatch_warning=mismatch_warning,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 型號不符偵測
# ─────────────────────────────────────────────────────────────────────────────

# FlashSystem 型號關鍵字對應（型號縮寫 → 文件關鍵字）
_MODEL_KEYWORDS: Dict[str, List[str]] = {
    "5000": ["fs5000", "flashsystem 5000", "5200", "5035", "5015"],
    "7000": ["fs7000", "flashsystem 7000", "7200", "7600"],
    "9000": ["fs9000", "flashsystem 9000", "9200", "9500"],
}

def _detect_model_mismatch(question: str, docs: List[Document]) -> str:
    """
    若問題中提及特定型號，但檢索回來的文件來源都是其他型號，
    則回傳警告字串；否則回傳空字串。
    """
    if not docs:
        return ""

    # 從問題中找出詢問的型號（重用 _BOOST_MODEL_RE，支援中文邊界）
    raw_matches = _BOOST_MODEL_RE.findall(question)
    queried_models = {next((g for g in groups if g), None) for groups in raw_matches}
    queried_models.discard(None)
    if not queried_models:
        return ""

    # 收集文件來源字串（source 檔名 + page_content 前 100 字）
    doc_texts = []
    for doc in docs:
        src = doc.metadata.get("source", "").lower()
        doc_texts.append(src)
        doc_texts.append(doc.page_content[:100].lower())
    combined = " ".join(doc_texts)

    # 判斷查詢型號系列是否有出現在文件來源中
    missing: List[str] = []
    found_families: set = set()
    for model_num in queried_models:
        # 找到這個型號屬於哪個系列
        for family, keywords in _MODEL_KEYWORDS.items():
            if any(model_num in kw for kw in keywords):
                # 檢查文件來源是否包含同系列任一關鍵字
                if not any(kw in combined for kw in keywords):
                    if family not in found_families:
                        missing.append(f"FS{model_num}")
                        found_families.add(family)
                break

    if missing:
        missing_str = "、".join(missing)
        return (
            f"⚠ 知識庫中找不到 {missing_str} 的專屬文件，"
            f"以下回答是根據相近型號的文件推斷，可能存在差異。"
            f"建議將 {missing_str} 的 Hardware Installation Guide 加入知識庫後重建索引。"
        )
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 第二層：watsonx 深度推理
# ─────────────────────────────────────────────────────────────────────────────

_EXPERT_PROMPT_TEMPLATE = """\
你是一位具備原廠實戰經驗的 IBM Storage FlashSystem / IBM Storage Virtualize 資深儲存架構師。
請只依據下方【參考資料】回答使用者問題，並以【繁體中文】輸出可直接執行的專家級方案。

== 回答原則 ==
1. 先檢驗問題的前提：使用者提出的做法或假設，在參考資料所述的產品架構與限制下是否成立？
   若不成立，要明確寫出「不可行」，說明原因與資料依據，再提出可行的替代方案。
   若有多個可行方案，逐一說明可行或被排除的理由，並指出推薦方案。
2. 涉及容量、時間、數量的規劃，只使用問題中給定的數字做推算，並列出推算假設；問題沒給的數字不要自行編造。
3. 所有 CLI 指令的名稱、參數與語法，必須逐字出自【參考資料】。參考資料沒有出現的指令或參數一律不得輸出；
   資料不足時寫「參考資料未涵蓋，需查閱 CLI Guide」，不得憑記憶補寫。不同版本語法可能不同，以參考資料標示的版本為準。
4. 每個階段都要有「驗收標準」：執行後用哪個指令檢查、看到什麼結果才算通過，通過後才能進入下一階段。
5. 只能引用【參考資料】中出現的文件與頁碼（沿用其 [n] 編號、檔名、頁碼或網址）；不得編造文件名稱、章節或頁碼。
6. 參考資料的型號或版本與問題不一致時，要明確說明。

== 參考資料 ==
{context}

== 使用者問題 ==
{question}

== 輸出格式（必須使用以下五個標籤，標籤名稱不可更動；下列為格式說明，不是內容範例） ==

[DIAGNOSIS]
用「-」條列：前提檢驗結論（成立或不成立與原因）、方案比較（各方案可行或排除的理由與推薦方案）、
規劃推算（若問題提供了數字）。

[STEPS]
依執行順序分成多個階段。最上層以「1.」「2.」「3.」編號，每個編號代表一個階段（階段標題寫明目的）；
階段內的細項一律用「-」，不要再用數字編號。每個階段都要包含：
- 動作說明
- 指令區塊（```bash 程式碼區塊，每個指令前以 # 註解說明用途；指令必須出自參考資料）
- 驗收標準（檢查指令與預期結果）

[RISKS]
用「-」條列，每條格式為「異常情境：現象或原因 → 處置與回滾方式」。

[SUMMARY]
用 2～3 句話總結結論與最關鍵的前提條件。

[SOURCES]
每條一行，格式為「[n] 檔名或網址 p.頁碼 — 此來源支持的論點」；沒有頁碼的來源省略頁碼。
"""

_EXPERT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_EXPERT_PROMPT_TEMPLATE,
)

# 逐 section 切割：找到每個標籤的起始位置，取兩標籤之間的文字
_TAG_RE = re.compile(r"\[(DIAGNOSIS|STEPS|RISKS|SUMMARY|SOURCES)\]", re.IGNORECASE)


def _extract_sections(raw: str) -> dict:
    """將 LLM 輸出按 [TAG] 切割成 {tag: content} dict，容許前言文字與標籤大小寫變化。"""
    sections: dict = {}
    matches = list(_TAG_RE.finditer(raw))
    for i, m in enumerate(matches):
        tag = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        sections[tag] = raw[start:end].strip()
    return sections


def _validate_and_sanitize_cli(text: str) -> str:
    """
    CLI 防幻覺校驗器：
    自動校正 LLM 輸出中常見的 Storage Virtualize 指令幻覺與參數錯誤。
    """
    if not text:
        return text

    # 1. mkpool 指令校正 -> mkmdiskgrp
    text = re.sub(r"\bmkpool\s+-name\s+([A-Za-z0-9_-]+)", r"mkmdiskgrp -ext 1024 -name \1", text)
    text = re.sub(r"\bmkpool\b", "mkmdiskgrp", text)

    # 2. lsvolumegroup -replicationpolicy -> lsvolumegroup
    text = re.sub(r"\blsvolumegroup\s+-replicationpolicy\b", "lsvolumegroup", text)

    # 3. mkarray 參數修正：移除錯誤的 -pool 參數
    text = re.sub(r"\bmkarray\b(.*?)(\s+-name\s+[A-Za-z0-9_-]+)?\s+-pool\s+([A-Za-z0-9_-]+)", r"mkarray\1 \3", text)

    return text


def _parse_expert_output(raw: str, question: str, filter_result: FilterResult) -> ExpertAnswer:
    """將 LLM 的結構化文字輸出解析成 ExpertAnswer 並進行 CLI 防幻覺校驗。"""
    raw_sanitized = _validate_and_sanitize_cli(raw)
    sections = _extract_sections(raw_sanitized)

    # 至少要有 DIAGNOSIS 或 SUMMARY 才算成功解析
    if not sections or ("DIAGNOSIS" not in sections and "SUMMARY" not in sections):
        return ExpertAnswer(
            question=question,
            diagnosis=raw_sanitized.strip() if raw_sanitized.strip() else "（LLM 未回傳任何內容）",
            steps=[],
            risks=[],
            summary="（輸出格式解析失敗，請參閱上方診斷欄位的原始 LLM 輸出）",
            sources=filter_result.source_refs,
            raw_llm_output=raw,
            filter_result=filter_result,
        )

    diagnosis   = _validate_and_sanitize_cli(sections.get("DIAGNOSIS", ""))
    steps_raw   = _validate_and_sanitize_cli(sections.get("STEPS", ""))
    risks_raw   = sections.get("RISKS", "")
    summary     = _validate_and_sanitize_cli(sections.get("SUMMARY", ""))
    sources_raw = sections.get("SOURCES", "")

    # 解析 steps：智慧處理可能跨多行的 Mermaid 或程式碼區塊
    steps: List[str] = []
    current_step: List[str] = []
    in_code_block = False

    for line in steps_raw.splitlines():
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            current_step.append(line)
            continue

        if not in_code_block and re.match(r"^[\d１-９]+[.．、]", line.strip()):
            if current_step:
                steps.append("\n".join(current_step).strip())
                current_step = []
            clean_line = re.sub(r"^[\d１-９]+[.．、\s]+", "", line).strip()
            current_step.append(clean_line)
        else:
            if line.strip():
                current_step.append(line)

    if current_step:
        steps.append("\n".join(current_step).strip())

    if not steps and steps_raw:
        steps = [line.strip() for line in steps_raw.splitlines() if line.strip()]

    # 解析 risks：取出 bullet 列表
    risks = [
        line.lstrip("-•*－ ").strip()
        for line in risks_raw.splitlines()
        if line.strip() and line.strip()[0] in "-•*－"
    ]
    if not risks and risks_raw and risks_raw not in ("無", "None", "N/A"):
        risks = [line.strip() for line in risks_raw.splitlines() if line.strip()]

    # sources：優先解析 LLM 生成的具體手冊與章節引用；若為空則 fallback
    sources: List[str] = []
    for line in sources_raw.splitlines():
        s = line.strip()
        if s and (s.startswith("[") or s.startswith("-") or s.startswith("•")):
            sources.append(s)

    if not sources:
        sources = filter_result.source_refs

    return ExpertAnswer(
        question=question,
        diagnosis=diagnosis,
        steps=steps,
        risks=risks,
        summary=summary,
        sources=sources,
        raw_llm_output=raw,
        filter_result=filter_result,
    )


def _build_watsonx_llm(
    model_id: Optional[str] = None,
    project_id: Optional[str] = None,
    api_key: Optional[str] = None,
    url: Optional[str] = None,
) -> ChatWatsonx:
    resolved_model   = model_id   or os.environ.get("WATSONX_MODEL_ID",   "ibm/granite-13b-chat-v2")
    resolved_project = project_id or os.environ.get("WATSONX_PROJECT_ID", "")
    resolved_api_key = api_key    or os.environ.get("WATSONX_APIKEY",     "")
    resolved_url     = url        or os.environ.get("WATSONX_URL",        "https://us-south.ml.cloud.ibm.com")

    if not resolved_api_key:
        raise ValueError("缺少 WATSONX_APIKEY。請在 .env 檔案中設定。")
    if not resolved_project:
        raise ValueError("缺少 WATSONX_PROJECT_ID。請在 .env 檔案中設定。")

    max_tokens = int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "8192"))

    return ChatWatsonx(
        model_id=resolved_model,
        url=resolved_url,
        project_id=resolved_project,
        apikey=resolved_api_key,
        params={
            "max_tokens": max_tokens,
            "temperature": 0.05,
        },
    )


def _build_gemini_llm(
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChatGoogleGenerativeAI:
    resolved_model   = model_id or os.environ.get("GEMINI_MODEL_ID", "gemini-2.0-flash")
    resolved_api_key = api_key  or os.environ.get("GEMINI_API_KEY",  "")
    max_tokens = int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "8192"))

    if not resolved_api_key:
        raise ValueError("缺少 GEMINI_API_KEY。請在 .env 檔案中設定。")

    return ChatGoogleGenerativeAI(
        model=resolved_model,
        google_api_key=resolved_api_key,
        temperature=0.05,
        max_output_tokens=max_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主介面：FlashSystemExpert（兩階段）
# ─────────────────────────────────────────────────────────────────────────────

class FlashSystemExpert:
    """
    IBM FlashSystem 兩階段專家系統。

    使用方式：
        expert = FlashSystemExpert(vectorstore)
        answer = expert.ask("How do I configure iSCSI on FlashSystem 9500?")
        print(expert.format_answer(answer))
    """

    def __init__(
        self,
        vectorstore: Chroma,
        top_k: int = 8,
        llm_provider: str = "watsonx",
        model_id: Optional[str] = None,
        project_id: Optional[str] = None,
        api_key: Optional[str] = None,
        watsonx_url: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Args:
            vectorstore:  已載入的 Chroma 向量資料庫。
            top_k:        第一層搜尋的候選段落數（過濾前）。
            llm_provider: LLM 供應商："watsonx"（預設）或 "gemini"。
            model_id:     模型 ID（watsonx 或 Gemini，可覆寫 .env）。
            project_id:   watsonx Project ID（僅 watsonx 使用）。
            api_key:      API Key（watsonx 使用 IBM Cloud Key；gemini 使用 Google AI Key）。
            watsonx_url:  watsonx endpoint URL（僅 watsonx 使用）。
            verbose:      True 時印出第一層的中間結果。
        """
        self.local_filter = LocalRAGFilter(vectorstore, top_k=top_k)
        if llm_provider == "gemini":
            self.llm = _build_gemini_llm(model_id, api_key)
        else:
            self.llm = _build_watsonx_llm(model_id, project_id, api_key, watsonx_url)
        self.verbose = verbose

    def ask(
        self,
        question: str,
        version: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> ExpertAnswer:
        """
        執行兩階段查詢。

        Args:
            question:    使用者問題。
            version:     版本篩選（例如 "9.1.3"）。
            source_type: 來源類型篩選。

        Returns:
            ExpertAnswer：結構化的完整專家回答。
        """
        # ── 第一層：本機 RAG 過濾 ──
        print("  [第一層] 本機向量搜尋與過濾中...", flush=True)
        filter_result = self.local_filter.filter(question, version, source_type)

        if self.verbose:
            print(f"  [第一層] 找到 {len(filter_result.chunks)} 個相關段落：")
            for ref in filter_result.source_refs:
                print(f"           {ref}")

        if not filter_result.chunks:
            return ExpertAnswer(
                question=question,
                diagnosis="本機知識庫中找不到相關資訊。",
                steps=[],
                risks=[],
                summary="請確認問題是否與 IBM FlashSystem 相關，或嘗試重建向量資料庫。",
                sources=[],
                raw_llm_output="",
                filter_result=filter_result,
            )

        # ── 第二層：watsonx 深度推理 ──
        print("  [第二層] 送往 IBM watsonx 進行深度推理...", flush=True)
        prompt = _EXPERT_PROMPT.format(
            context=filter_result.context_text,
            question=question,
        )
        raw_output = self.llm.invoke(prompt)
        # Chat models (Gemini, etc.) return AIMessage; extract plain text
        if hasattr(raw_output, "content"):
            raw_output = raw_output.content

        return _parse_expert_output(raw_output, question, filter_result)

    def format_answer(self, answer: ExpertAnswer, verbose: bool = False) -> str:
        """將 ExpertAnswer 格式化成易讀的終端輸出。"""
        sep = "─" * 60
        lines = [
            "",
            sep,
            f"  問題：{answer.question}",
            sep,
        ]

        # 型號不符警告（優先顯示在最上方）
        if answer.filter_result.model_mismatch_warning:
            lines += ["", answer.filter_result.model_mismatch_warning]

        lines += [
            "",
            "【診斷】",
            answer.diagnosis,
        ]

        if answer.steps:
            lines += ["", "【操作步驟】"]
            for i, step in enumerate(answer.steps, 1):
                lines.append(f"  {i}. {step}")
        else:
            lines += ["", "【操作步驟】", "  N/A"]

        if answer.risks and answer.risks != ["None"]:
            lines += ["", "【風險與注意事項】"]
            for risk in answer.risks:
                if risk:
                    lines.append(f"  ⚠  {risk}")
        else:
            lines += ["", "【風險與注意事項】", "  無"]

        lines += [
            "",
            "【總結】",
            answer.summary,
        ]

        if answer.sources:
            lines += ["", "【引用來源】"]
            for src in answer.sources:
                lines.append(f"  {src}")

        if verbose and answer.raw_llm_output:
            lines += [
                "",
                "── LLM 原始輸出（debug）──",
                answer.raw_llm_output,
            ]

        lines.append(sep)
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from src.build_vectorstore import load_vectorstore

    persist_dir     = sys.argv[1] if len(sys.argv) > 1 else "/Users/yhkuo/RAG/vectorstore"
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "flashsystem_all"

    vs     = load_vectorstore(persist_dir, collection_name)
    expert = FlashSystemExpert(vs, verbose=True)

    print("IBM FlashSystem 專家系統（兩階段）\n輸入 'exit' 退出。\n")
    while True:
        try:
            q = input("問題：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break
        if q.lower() in ("exit", "quit", "q", ""):
            continue
        ans = expert.ask(q)
        print(expert.format_answer(ans))

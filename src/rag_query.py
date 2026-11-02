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
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ibm import WatsonxLLM

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


class LocalRAGFilter:
    """
    第一層：本機向量搜尋 + 品質過濾。
    完全在本機執行，不呼叫任何外部 API。
    """

    def __init__(self, vectorstore: Chroma, top_k: int = 30):
        """
        Args:
            vectorstore: 已載入的 Chroma 向量資料庫。
            top_k:       初始搜尋取得的候選段落數（過濾前）。
                         過濾後實際使用的段落數會少於此值。
                         預設從 8 提高至 30：fs7600 的 PSU 相關 chunk
                         在向量相似度排名約第 98 位，需要更大的搜尋範圍。
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
        執行本機 RAG 過濾。

        Args:
            question:    使用者問題。
            version:     版本篩選（例如 "9.1.3"）。
            source_type: 來源類型篩選（"pdf" / "web" / "json_cli" 等）。

        Returns:
            FilterResult：包含篩選後段落、context 文字、引用來源。
        """
        # ── 1. 向量搜尋（MMR 保證多樣性）──
        where: Dict[str, Any] = {}
        if version:
            where["version"] = {"$eq": version}
        if source_type:
            where["source_type"] = {"$eq": source_type}

        search_kwargs: Dict[str, Any] = {
            "k": self.top_k,
            "fetch_k": self.top_k * 8,   # 原本 4x 不足，提高至 8x 增加多樣性
        }
        if where:
            search_kwargs["filter"] = where

        raw_docs = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        ).invoke(question)

        # ── 2. 型號強制補入（問題含型號關鍵字時，確保該型號文件一定進入候選）──
        raw_docs = _boost_model_docs(
            question=question,
            vectorstore=self.vectorstore,
            existing_docs=raw_docs,
            boost_k=_MODEL_BOOST_K,
            where_filter=where if where else None,
        )

        # ── 2b. 錯誤碼／指令名精確比對（向量檢索分不出相近的識別碼）──
        exact_docs = _exact_identifier_docs(question, self.vectorstore, where if where else None)
        exact_keys = {d.page_content[:100].strip() for d in exact_docs}

        # ── 3. 去重（精確命中放最前面，同內容時保留精確命中那份）──
        deduped = _deduplicate(exact_docs + raw_docs)

        # ── 4. 依 source_type 優先分數排序；精確命中的頁面固定排在最前 ──
        ranked = sorted(
            deduped,
            key=lambda d: (d.page_content[:100].strip() in exact_keys, _score_chunk(d)),
            reverse=True,
        )

        # ── 4b. 命中的 web / pdf chunk 展開成同頁前後文（同頁合併）──
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
你是一位精通 IBM FlashSystem 儲存系統的專家工程師。

以下是從 IBM FlashSystem 官方文件（PDF 手冊、IBM Docs 網頁、技術白名單）中檢索到的參考段落，以及使用者的問題。

請根據參考資料，以**繁體中文**撰寫詳盡的結構化專家回答。

== 參考資料 ==
{context}

== 使用者問題 ==
{question}

== 回答格式要求 ==
請嚴格依照以下格式輸出，不可新增或刪除任何段落標籤。所有內容必須使用繁體中文。

[DIAGNOSIS]
<說明使用者的問題核心，以及涉及哪些 FlashSystem 概念或元件。>

[STEPS]
<具體的操作步驟、指令或解答，以編號列表呈現。若無步驟請填「不適用」。>
1.
2.
3.

[RISKS]
<使用者必須注意的風險、警告或前提條件，以項目符號列表呈現。若無風險請填「無」。>
-

[SUMMARY]
<用一段話簡潔地總結完整回答。>

[SOURCES]
<列出本回答引用的參考資料編號，例如 [1]、[3]。>
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


def _parse_expert_output(raw: str, question: str, filter_result: FilterResult) -> ExpertAnswer:
    """將 LLM 的結構化文字輸出解析成 ExpertAnswer。"""
    sections = _extract_sections(raw)

    # 至少要有 DIAGNOSIS 或 SUMMARY 才算成功解析
    if not sections or ("DIAGNOSIS" not in sections and "SUMMARY" not in sections):
        # 嘗試把截斷的原始輸出直接當 diagnosis 顯示，而非空白
        return ExpertAnswer(
            question=question,
            diagnosis=raw.strip() if raw.strip() else "（LLM 未回傳任何內容）",
            steps=[],
            risks=[],
            summary="（輸出格式解析失敗，請參閱上方診斷欄位的原始 LLM 輸出）",
            sources=filter_result.source_refs,
            raw_llm_output=raw,
            filter_result=filter_result,
        )

    diagnosis   = sections.get("DIAGNOSIS", "")
    steps_raw   = sections.get("STEPS", "")
    risks_raw   = sections.get("RISKS", "")
    summary     = sections.get("SUMMARY", "")

    # 解析 steps：取出編號列表（相容中英文數字、全形句號）
    steps = [
        re.sub(r"^[\d１-９]+[.．、\s]+", "", line).strip()
        for line in steps_raw.splitlines()
        if re.match(r"^[\d１-９]+[.．、]", line.strip())
    ]
    # 若編號解析失敗但有內容，直接按行保留
    if not steps and steps_raw:
        steps = [line.strip() for line in steps_raw.splitlines() if line.strip()]

    # 解析 risks：取出 bullet 列表
    risks = [
        line.lstrip("-•*－ ").strip()
        for line in risks_raw.splitlines()
        if line.strip() and line.strip()[0] in "-•*－"
    ]
    # 若 bullet 解析失敗但有內容，直接按行保留
    if not risks and risks_raw and risks_raw not in ("無", "None", "N/A"):
        risks = [line.strip() for line in risks_raw.splitlines() if line.strip()]

    # sources：使用第一層檢索到的 source_refs
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
) -> WatsonxLLM:
    resolved_model   = model_id   or os.environ.get("WATSONX_MODEL_ID",   "ibm/granite-13b-chat-v2")
    resolved_project = project_id or os.environ.get("WATSONX_PROJECT_ID", "")
    resolved_api_key = api_key    or os.environ.get("WATSONX_APIKEY",     "")
    resolved_url     = url        or os.environ.get("WATSONX_URL",        "https://us-south.ml.cloud.ibm.com")

    if not resolved_api_key:
        raise ValueError("缺少 WATSONX_APIKEY。請在 .env 檔案中設定。")
    if not resolved_project:
        raise ValueError("缺少 WATSONX_PROJECT_ID。請在 .env 檔案中設定。")

    return WatsonxLLM(
        model_id=resolved_model,
        url=resolved_url,
        project_id=resolved_project,
        apikey=resolved_api_key,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 2048,
            "min_new_tokens": 1,
            "temperature": 0.05,
            "top_k": 50,
            "top_p": 1,
            "repetition_penalty": 1.05,
        },
    )


def _build_gemini_llm(
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChatGoogleGenerativeAI:
    resolved_model   = model_id or os.environ.get("GEMINI_MODEL_ID", "gemini-2.0-flash")
    resolved_api_key = api_key  or os.environ.get("GEMINI_API_KEY",  "")

    if not resolved_api_key:
        raise ValueError("缺少 GEMINI_API_KEY。請在 .env 檔案中設定。")

    return ChatGoogleGenerativeAI(
        model=resolved_model,
        google_api_key=resolved_api_key,
        temperature=0.05,
        max_output_tokens=8192,   # 原本 2048 不足以容納繁體中文完整回答，提高至 8192
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

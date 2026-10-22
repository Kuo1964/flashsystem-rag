"""
ingest_json.py
==============
載入 manual_docs/ 下的結構化 JSON 文件（人工整理的知識庫），
轉換成 LangChain Document 段落，供後續寫入向量資料庫。

支援三種已知的 JSON 格式：
1. feature_lifecycle_matrix.json  - 功能生命週期清單（List[dict]）
2. official_cli_commands_whitelist.json - CLI 指令白名單（dict of dict）
3. official_grounded_parts_from_pdf.json - 零件清單（List[dict]）
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain.schema import Document


# ── 個別格式的轉換函數 ─────────────────────────────────────────────────────


def _feature_to_text(item: Dict[str, Any]) -> str:
    """將 feature_lifecycle_matrix 的一筆記錄轉為可讀文字。"""
    lines = [
        f"Feature: {item.get('feature_name', '')} (ID: {item.get('feature_id', '')})",
        f"Keywords: {', '.join(item.get('keywords', []))}",
        f"Introduced: {item.get('introduced_version', 'N/A')}",
        f"Deprecated: {item.get('deprecated_version', 'N/A')}",
        f"Status in 9.1: {item.get('status_in_9_1', 'N/A')}",
    ]
    if item.get("replacement_feature"):
        lines.append(f"Replacement: {item['replacement_feature']} — {item.get('replacement_description', '')}")
    if item.get("legacy_commands"):
        lines.append(f"Legacy commands: {', '.join(item['legacy_commands'])}")
    if item.get("modern_commands"):
        lines.append(f"Modern commands: {', '.join(item['modern_commands'])}")
    if item.get("guidance_summary"):
        lines.append(f"Guidance: {item['guidance_summary']}")
    return "\n".join(lines)


def _cli_to_text(cmd: str, info: Dict[str, Any]) -> str:
    """將 CLI 指令白名單的一筆記錄轉為可讀文字。"""
    parts = [f"CLI Command: {cmd}"]
    if isinstance(info, dict):
        for k, v in info.items():
            if v:
                parts.append(f"  {k}: {v}")
    elif isinstance(info, str):
        parts.append(f"  Description: {info}")
    return "\n".join(parts)


def _part_to_text(item: Dict[str, Any]) -> str:
    """將零件記錄轉為可讀文字。"""
    return (
        f"Part Number: {item.get('part_number', 'N/A')}\n"
        f"Description: {item.get('description', 'N/A')}\n"
        f"Type: {item.get('type', 'N/A')}\n"
        f"Source PDF: {item.get('source_pdf', 'N/A')}  Page: {item.get('page_number', 'N/A')}"
    )


# ── 主要載入函數 ───────────────────────────────────────────────────────────


def _load_feature_lifecycle(path: Path) -> List[Document]:
    data: List[Dict] = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for item in data:
        text = _feature_to_text(item)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_type": "json_feature",
                    "feature_id": item.get("feature_id", ""),
                    "version": item.get("introduced_version", "unknown"),
                },
            )
        )
    return docs


def _load_cli_whitelist(path: Path) -> List[Document]:
    data: Dict = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for cmd, info in data.items():
        text = _cli_to_text(cmd, info)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_type": "json_cli",
                    "command": cmd,
                },
            )
        )
    return docs


def _load_grounded_parts(path: Path) -> List[Document]:
    data: List[Dict] = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for item in data:
        text = _part_to_text(item)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_type": "json_parts",
                    "part_number": item.get("part_number", ""),
                    "source_pdf": item.get("source_pdf", ""),
                    "page": item.get("page_number", 0),
                },
            )
        )
    return docs


# 檔名 → 對應載入函數的映射
_LOADERS = {
    "feature_lifecycle_matrix.json": _load_feature_lifecycle,
    "official_cli_commands_whitelist.json": _load_cli_whitelist,
    "official_grounded_parts_from_pdf.json": _load_grounded_parts,
}


def load_json_docs(json_dir: str) -> List[Document]:
    """
    載入 json_dir 下所有已知的 JSON 文件。

    Args:
        json_dir: JSON 文件所在目錄。

    Returns:
        List[Document]: 轉換後的 Document 列表。
    """
    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise FileNotFoundError(f"JSON 目錄不存在：{json_dir}")

    all_docs: List[Document] = []
    for filename, loader_fn in _LOADERS.items():
        path = json_dir / filename
        if not path.exists():
            print(f"[WARN] 找不到 {filename}，略過。")
            continue
        try:
            docs = loader_fn(path)
            all_docs.extend(docs)
            print(f"[INFO] {filename} → {len(docs)} 個段落")
        except Exception as e:
            print(f"[ERROR] 載入 {filename} 失敗：{e}")

    print(f"[INFO] JSON 載入完成，共 {len(all_docs)} 個段落")
    return all_docs


if __name__ == "__main__":
    import sys

    json_dir = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/SSD_4TB/RAG/raw_data/manual_docs"
    docs = load_json_docs(json_dir)
    if docs:
        print(f"\n範例段落：\n{docs[0].page_content}\nMetadata：{docs[0].metadata}")

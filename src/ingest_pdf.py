"""
ingest_pdf.py
=============
從指定目錄載入所有 PDF 檔案，切割成適合 embedding 的文字段落，
並加上 metadata（來源檔名、頁碼、版本）。
"""

import os
import re
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm import tqdm


# 從檔名推斷 FlashSystem 版本，例如 "9.1.0_concept_pdfguide.pdf" → "9.1.0"
_VERSION_RE = re.compile(r"^(\d+\.\d+[\.\d]*)_")


def _extract_version(filename: str) -> str:
    m = _VERSION_RE.match(filename)
    return m.group(1) if m else "unknown"


def load_pdfs(pdf_dir: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """
    讀取 pdf_dir 下所有 .pdf 檔案，回傳切割後的 Document 列表。

    Args:
        pdf_dir:       PDF 所在目錄路徑。
        chunk_size:    每個段落的最大 token 數（字元近似）。
        chunk_overlap: 相鄰段落的重疊字元數，保留上下文連貫性。

    Returns:
        List[Document]: 帶有 metadata 的文字段落列表。
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF 目錄不存在：{pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] {pdf_dir} 內找不到任何 PDF 檔案。")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_docs: List[Document] = []

    for pdf_path in tqdm(pdf_files, desc="載入 PDF"):
        filename = pdf_path.name
        version = _extract_version(filename)

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as e:
            print(f"[ERROR] 無法讀取 {filename}：{e}")
            continue

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            # 切割並附加 metadata
            chunks = splitter.split_text(text)
            for chunk in chunks:
                if not chunk.strip():
                    continue
                all_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "source_type": "pdf",
                            "page": page_num,
                            "version": version,
                        },
                    )
                )

    print(f"[INFO] PDF 載入完成：{len(pdf_files)} 個檔案 → {len(all_docs)} 個段落")
    return all_docs


if __name__ == "__main__":
    import sys

    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/yhkuo/RAG/raw_data/pdfs"
    docs = load_pdfs(pdf_dir)
    print(f"範例段落：\n{docs[0].page_content[:300]}\nMetadata：{docs[0].metadata}")

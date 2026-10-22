"""
main.py
=======
IBM FlashSystem 兩階段專家系統主程式入口。

指令：
  build   - 建立向量資料庫（從 PDF、網頁、JSON）
  query   - 互動式查詢介面
  search  - 單次查詢（非互動）

使用範例：
  # 建立向量資料庫（全部來源）
  python main.py build

  # 只從 PDF 建立
  python main.py build --source pdf

  # 互動式查詢（預設顯示兩階段進度）
  python main.py query

  # 互動式查詢 + 顯示第一層過濾中間結果
  python main.py query --verbose

  # 單次查詢
  python main.py search "How to configure iSCSI on FlashSystem 9500?"

  # 帶版本篩選的查詢
  python main.py search "What CLI commands are available?" --version 9.1.3

  # 顯示 LLM 原始輸出（debug）
  python main.py search "mkvolume syntax" --verbose
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── 預設路徑設定（可透過 .env 或命令列參數覆寫）──────────────────────────

DEFAULT_PDF_DIR    = os.environ.get("RAG_PDF_DIR",    "/Volumes/SSD_4TB/RAG/raw_data/pdfs")
DEFAULT_URL_FILE   = os.environ.get("RAG_URL_FILE",   "/Volumes/SSD_4TB/RAG/raw_data/web_urls.txt")
DEFAULT_JSON_DIR   = os.environ.get("RAG_JSON_DIR",   "/Volumes/SSD_4TB/RAG/raw_data/manual_docs")
DEFAULT_VECTOR_DIR = os.environ.get("RAG_VECTOR_DIR", "/Volumes/SSD_4TB/RAG/vectorstore")
DEFAULT_COLLECTION = os.environ.get("RAG_COLLECTION",  "flashsystem_all")


# ─────────────────────────────────────────────────────────────────────────────
# build：建立向量資料庫
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build(args: argparse.Namespace) -> None:
    from src.build_vectorstore import build_vectorstore
    from src.crawl_state import default_state_file
    from src.ingest_json import load_json_docs
    from src.ingest_pdf import load_pdfs
    from src.ingest_web import load_urls

    all_docs = []
    source = args.source  # "all" | "pdf" | "web" | "json"

    # --reset 只在整個流程的第一次寫入時執行，之後的批次寫入一律用 reset=False
    reset_pending = args.reset

    def _write(docs) -> None:
        nonlocal reset_pending
        build_vectorstore(
            documents=docs,
            persist_dir=args.vector_dir,
            collection_name=args.collection,
            reset=reset_pending,
        )
        reset_pending = False

    state_file = default_state_file(args.vector_dir, args.collection)

    # 先把 reset 做掉（避免網頁爬取中途才清空前面已寫入的資料）
    if reset_pending:
        _write([])
        # 向量庫已清空，續爬記錄也必須一併清除，否則會誤判頁面「已完成」而不重爬
        Path(state_file).unlink(missing_ok=True)

    if source in ("all", "pdf"):
        print("\n=== [1/3] 載入 PDF 文件 ===")
        pdf_docs = load_pdfs(
            args.pdf_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        all_docs.extend(pdf_docs)
        if pdf_docs:
            print(f"寫入 {len(pdf_docs)} 個 PDF 段落 ...")
            _write(pdf_docs)

    if source in ("all", "web"):
        print("\n=== [2/3] 爬取網頁文件（每 100 個段落先寫入一次）===")
        web_docs = load_urls(
            args.url_file,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_pages_per_root=args.max_web_pages,
            on_flush=_write,
            flush_every=100,
            state_file=state_file,
        )
        all_docs.extend(web_docs)

    if source in ("all", "json"):
        print("\n=== [3/3] 載入 JSON 文件 ===")
        json_docs = load_json_docs(args.json_dir)
        all_docs.extend(json_docs)
        if json_docs:
            _write(json_docs)

    if not all_docs:
        print("[ERROR] 沒有載入任何文件，請確認資料來源路徑是否正確。")
        sys.exit(1)

    print(f"\n✅ 向量資料庫建立完成！共處理 {len(all_docs)} 個段落，路徑：{args.vector_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# query：互動式兩階段查詢
# ─────────────────────────────────────────────────────────────────────────────

def cmd_query(args: argparse.Namespace) -> None:
    from src.build_vectorstore import load_vectorstore
    from src.rag_query import FlashSystemExpert

    print(f"載入向量資料庫：{args.vector_dir} / {args.collection}")
    vs = load_vectorstore(args.vector_dir, args.collection)

    expert = FlashSystemExpert(
        vs,
        top_k=args.top_k,
        llm_provider=args.llm,
        verbose=args.verbose,
    )

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║      IBM FlashSystem 兩階段專家系統                 ║")
    print("║  第一層：本機 RAG 過濾  →  第二層：watsonx 推理     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("輸入問題後按 Enter，輸入 'exit' 退出。\n")

    version     = args.version if hasattr(args, "version") else None
    source_type = args.source_type if hasattr(args, "source_type") else None

    while True:
        try:
            question = input("❓ 問題：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break

        if question.lower() in ("exit", "quit", "q", ""):
            if question.lower() in ("exit", "quit", "q"):
                break
            continue

        print()
        answer = expert.ask(question, version=version, source_type=source_type)
        print(expert.format_answer(answer, verbose=args.verbose))
        print()


# ─────────────────────────────────────────────────────────────────────────────
# search：單次兩階段查詢
# ─────────────────────────────────────────────────────────────────────────────

def cmd_search(args: argparse.Namespace) -> None:
    from src.build_vectorstore import load_vectorstore
    from src.rag_query import FlashSystemExpert

    vs = load_vectorstore(args.vector_dir, args.collection)

    expert = FlashSystemExpert(
        vs,
        top_k=args.top_k,
        llm_provider=args.llm,
        verbose=args.verbose,
    )

    print()
    answer = expert.ask(
        args.question,
        version=args.version,
        source_type=args.source_type,
    )
    print(expert.format_answer(answer, verbose=args.verbose))


# ─────────────────────────────────────────────────────────────────────────────
# CLI 定義
# ─────────────────────────────────────────────────────────────────────────────

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vector-dir",  default=DEFAULT_VECTOR_DIR, help="Chroma 持久化目錄")
    parser.add_argument("--collection",  default=DEFAULT_COLLECTION,  help="Chroma collection 名稱")
    parser.add_argument("--llm", default="watsonx", choices=["watsonx", "gemini"],
                        help="LLM 供應商：watsonx（預設）或 gemini")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="顯示第一層過濾中間結果與 LLM 原始輸出")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="IBM FlashSystem 兩階段專家系統",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── build ──
    p_build = subparsers.add_parser("build", help="建立向量資料庫")
    p_build.add_argument("--vector-dir",    default=DEFAULT_VECTOR_DIR, help="Chroma 持久化目錄")
    p_build.add_argument("--collection",    default=DEFAULT_COLLECTION,  help="Chroma collection 名稱")
    p_build.add_argument("--source",        default="all",  choices=["all", "pdf", "web", "json"], help="資料來源")
    p_build.add_argument("--pdf-dir",       default=DEFAULT_PDF_DIR,  help="PDF 目錄路徑")
    p_build.add_argument("--url-file",      default=DEFAULT_URL_FILE, help="網址清單檔案路徑")
    p_build.add_argument("--json-dir",      default=DEFAULT_JSON_DIR, help="JSON 文件目錄路徑")
    p_build.add_argument("--chunk-size",    default=1000, type=int,   help="段落最大字元數")
    p_build.add_argument("--chunk-overlap", default=150,  type=int,   help="段落重疊字元數")
    p_build.add_argument("--max-web-pages", default=200,  type=int,   help="每個根網址最多爬取子頁面數")
    p_build.add_argument("--reset",         action="store_true",      help="清除舊資料後重建")
    p_build.set_defaults(func=cmd_build)

    # ── query ──
    p_query = subparsers.add_parser("query", help="互動式兩階段查詢")
    _add_common_args(p_query)
    p_query.add_argument("--top-k",      default=30,   type=int, help="第一層搜尋的候選段落數")
    p_query.add_argument("--version",    default=None,           help="篩選特定版本，例如 9.1.3")
    p_query.add_argument("--source-type",default=None,           help="篩選來源類型：pdf / web / json_cli")
    p_query.set_defaults(func=cmd_query)

    # ── search ──
    p_search = subparsers.add_parser("search", help="單次兩階段查詢")
    _add_common_args(p_search)
    p_search.add_argument("question",                          help="查詢問題")
    p_search.add_argument("--top-k",      default=30,   type=int, help="第一層搜尋的候選段落數")
    p_search.add_argument("--version",    default=None,           help="篩選特定版本，例如 9.1.3")
    p_search.add_argument("--source-type",default=None,           help="篩選來源類型：pdf / web / json_cli 等")
    p_search.set_defaults(func=cmd_search)

    # ── ui ──
    p_ui = subparsers.add_parser("ui", help="啟動 Gradio Web UI")
    p_ui.add_argument("--port",  type=int, default=7860, help="監聽 port（預設 7860）")
    p_ui.add_argument("--share", action="store_true",    help="建立 Gradio 公開連結（謹慎使用）")
    p_ui.set_defaults(func=cmd_ui)

    return parser


def cmd_ui(args: argparse.Namespace) -> None:
    """啟動 Gradio Web UI。"""
    from app import launch
    print(f"啟動 IBM FlashSystem RAG Web UI → http://127.0.0.1:{args.port}")
    launch(port=args.port, share=args.share)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

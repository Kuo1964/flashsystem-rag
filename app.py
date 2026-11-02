"""
app.py
======
IBM FlashSystem RAG 系統 — Gradio Web UI

啟動方式：
  python app.py
  python app.py --port 8080

三個 Tab：
  Tab 1 查詢（Chat）     — 兩階段專家查詢 + 側欄篩選器
  Tab 2 資料管理（Data）  — 網頁爬取 / PDF / JSON 重載 / 全部重建
  Tab 3 設定（Settings） — 環境變數檢視 + Collection 名稱
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ── 預設路徑（從 .env 讀取）─────────────────────────────────────────────────
_PDF_DIR    = os.environ.get("RAG_PDF_DIR",    "/Users/yhkuo/RAG/raw_data/pdfs")
_URL_FILE   = os.environ.get("RAG_URL_FILE",   "/Users/yhkuo/RAG/raw_data/web_urls.txt")
_JSON_DIR   = os.environ.get("RAG_JSON_DIR",   "/Users/yhkuo/RAG/raw_data/manual_docs")
_VECTOR_DIR = os.environ.get("RAG_VECTOR_DIR", "/Users/yhkuo/RAG/vectorstore")
_COLLECTION = os.environ.get("RAG_COLLECTION",  "flashsystem_all")

# ── 全域狀態（跨回呼共用）───────────────────────────────────────────────────
_vectorstore = None      # 已載入的 Chroma 實例
_expert      = None      # 已初始化的 FlashSystemExpert
_expert_cfg  = {}        # 目前 expert 的設定（用於 lazy re-init 判斷）
_build_lock  = threading.Lock()   # 防止同時觸發多次 build


def _mask_key(val: str) -> str:
    """遮蔽 API Key，只顯示前4碼。"""
    if not val or val.startswith("your_"):
        return "（未設定）"
    return val[:4] + "****"


# ─────────────────────────────────────────────────────────────────────────────
# 共用：載入 / 取得 vectorstore
# ─────────────────────────────────────────────────────────────────────────────

def _get_vectorstore(vector_dir: str, collection: str):
    """載入（或重用）Chroma vectorstore。"""
    global _vectorstore
    from src.build_vectorstore import load_vectorstore
    _vectorstore = load_vectorstore(vector_dir, collection)
    return _vectorstore


def _get_expert(
    llm_provider: str,
    top_k: int,
    vector_dir: str,
    collection: str,
):
    """Lazy init FlashSystemExpert；設定相同時重用，否則重新初始化。"""
    global _expert, _expert_cfg, _vectorstore
    cfg = {"llm": llm_provider, "top_k": top_k, "dir": vector_dir, "col": collection}
    if _expert is None or cfg != _expert_cfg:
        from src.rag_query import FlashSystemExpert
        vs = _get_vectorstore(vector_dir, collection)
        _expert = FlashSystemExpert(vs, top_k=top_k, llm_provider=llm_provider)
        _expert_cfg = cfg
    return _expert


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1：查詢（Chat）
# ─────────────────────────────────────────────────────────────────────────────

def chat_fn(
    message: str,
    history: list,
    llm_provider: str,
    version_filter: str,
    top_k: int,
    source_type_filter: str,
    vector_dir: str,
    collection: str,
):
    """處理一次查詢，回傳更新後的 history。"""
    if not message.strip():
        return history

    # 初始化 expert
    try:
        expert = _get_expert(llm_provider, int(top_k), vector_dir, collection)
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ 初始化失敗：{e}"})
        return history

    # 執行兩階段查詢
    version = version_filter.strip() or None
    src_type = source_type_filter if source_type_filter != "全部" else None

    try:
        answer = expert.ask(message, version=version, source_type=src_type)
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ 查詢失敗：{e}"})
        return history

    # 格式化回答為 Markdown
    parts = []

    # 型號不符警告（優先顯示）
    if answer.filter_result.model_mismatch_warning:
        parts.append(f"**{answer.filter_result.model_mismatch_warning}**")

    if answer.diagnosis:
        parts.append(f"**📋 診斷**\n{answer.diagnosis}")

    if answer.steps:
        steps_md = "\n".join(f"{i}. {s}" for i, s in enumerate(answer.steps, 1))
        parts.append(f"**🔧 操作步驟**\n{steps_md}")

    if answer.risks and answer.risks != ["None"]:
        risks_md = "\n".join(f"- ⚠ {r}" for r in answer.risks if r)
        if risks_md:
            parts.append(f"**⚠ 風險與注意事項**\n{risks_md}")

    if answer.summary:
        parts.append(f"**✅ 總結**\n{answer.summary}")

    if answer.sources:
        srcs_md = "\n".join(f"- {s}" for s in answer.sources)
        parts.append(f"**📚 引用來源**\n{srcs_md}")

    response = "\n\n---\n\n".join(parts) if parts else "（無回答）"
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history


def clear_chat():
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2：資料管理（Data）
# ─────────────────────────────────────────────────────────────────────────────

def _invalidate_expert():
    """清除 expert 快取，下次查詢時重新初始化。"""
    global _expert, _expert_cfg, _vectorstore
    _expert = None
    _expert_cfg = {}
    _vectorstore = None


def _clear_crawl_state(vector_dir: str, collection: str) -> None:
    """向量庫被重置時，一併清除網頁續爬記錄，避免頁面被誤判為「已完成」而不再重爬。"""
    from src.crawl_state import default_state_file
    Path(default_state_file(vector_dir, collection)).unlink(missing_ok=True)


def ingest_web_urls(
    url_text: str,
    uploaded_file,
    max_pages: int,
    vector_dir: str,
    collection: str,
    workers: int = 1,
) -> str:
    """爬取網頁並加入向量資料庫。"""
    if not _build_lock.acquire(blocking=False):
        return "⏳ 已有建庫任務進行中，請稍後再試。"
    try:
        from src.build_vectorstore import build_vectorstore
        from src.crawl_state import default_state_file
        from src.ingest_web import load_urls, load_urls_from_list

        urls: list[str] = []

        # 從上傳的 txt 檔解析
        if uploaded_file is not None:
            try:
                content = Path(uploaded_file.name).read_text(encoding="utf-8")
                urls += content.splitlines()
            except Exception as e:
                return f"❌ 無法讀取上傳檔案：{e}"

        # 從 Textarea 解析
        if url_text.strip():
            urls += url_text.strip().splitlines()

        if not urls:
            return "❌ 請輸入至少一個 URL 或上傳 web_urls.txt。"

        # 每爬滿 100 個段落就先寫入向量庫（build_vectorstore 內建去重）
        def _write_batch(batch):
            build_vectorstore(batch, persist_dir=vector_dir, collection_name=collection, reset=False)

        docs = load_urls_from_list(
            urls,
            max_pages_per_root=int(max_pages),
            on_flush=_write_batch,
            flush_every=100,
            workers=int(workers),
            # 記錄已寫入的頁面；重啟後重新爬取會自動跳過這些頁面
            state_file=default_state_file(vector_dir, collection),
        )
        if not docs:
            return "⚠ 爬取完成，但沒有取得任何段落。請確認 URL 是否可存取。"

        _invalidate_expert()
        return f"✅ 完成！爬取 {len(set(u.metadata['source'] for u in docs))} 個頁面，共處理 {len(docs)} 個段落（已邊爬邊寫入，重複者自動略過）。"
    except Exception as e:
        return f"❌ 錯誤：{e}"
    finally:
        _build_lock.release()


def reload_pdf(vector_dir: str, collection: str, reset: bool) -> str:
    """重新載入 PDF 並更新向量資料庫。"""
    if not _build_lock.acquire(blocking=False):
        return "⏳ 已有建庫任務進行中，請稍後再試。"
    try:
        from src.build_vectorstore import build_vectorstore
        from src.ingest_pdf import load_pdfs

        docs = load_pdfs(_PDF_DIR)
        if not docs:
            return f"⚠ 在 {_PDF_DIR} 找不到 PDF 檔案。"
        build_vectorstore(docs, persist_dir=vector_dir, collection_name=collection, reset=reset)
        if reset:
            _clear_crawl_state(vector_dir, collection)
        _invalidate_expert()
        return f"✅ PDF 載入完成！共 {len(docs)} 個段落。{'（已重置）' if reset else ''}"
    except Exception as e:
        return f"❌ 錯誤：{e}"
    finally:
        _build_lock.release()


def reload_json(vector_dir: str, collection: str) -> str:
    """重新載入 JSON 並更新向量資料庫。"""
    if not _build_lock.acquire(blocking=False):
        return "⏳ 已有建庫任務進行中，請稍後再試。"
    try:
        from src.build_vectorstore import build_vectorstore
        from src.ingest_json import load_json_docs

        docs = load_json_docs(_JSON_DIR)
        if not docs:
            return f"⚠ 在 {_JSON_DIR} 找不到 JSON 文件。"
        build_vectorstore(docs, persist_dir=vector_dir, collection_name=collection, reset=False)
        _invalidate_expert()
        return f"✅ JSON 載入完成！共 {len(docs)} 個段落。"
    except Exception as e:
        return f"❌ 錯誤：{e}"
    finally:
        _build_lock.release()


def rebuild_all(vector_dir: str, collection: str, reset: bool) -> str:
    """重建全部資料庫（PDF + JSON；網頁需另行觸發）。"""
    if not _build_lock.acquire(blocking=False):
        return "⏳ 已有建庫任務進行中，請稍後再試。"
    try:
        from src.build_vectorstore import build_vectorstore
        from src.ingest_json import load_json_docs
        from src.ingest_pdf import load_pdfs

        all_docs = []
        pdf_docs = load_pdfs(_PDF_DIR)
        all_docs.extend(pdf_docs)
        json_docs = load_json_docs(_JSON_DIR)
        all_docs.extend(json_docs)

        if not all_docs:
            return "⚠ 找不到任何文件，請確認路徑設定。"

        build_vectorstore(all_docs, persist_dir=vector_dir, collection_name=collection, reset=reset)
        if reset:
            _clear_crawl_state(vector_dir, collection)
        _invalidate_expert()
        return (
            f"✅ 重建完成！PDF {len(pdf_docs)} 個段落 + JSON {len(json_docs)} 個段落，"
            f"共 {len(all_docs)} 個段落。{'（已重置）' if reset else ''}"
        )
    except Exception as e:
        return f"❌ 錯誤：{e}"
    finally:
        _build_lock.release()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3：設定（Settings）
# ─────────────────────────────────────────────────────────────────────────────

def get_settings_display() -> str:
    """取得目前環境變數的遮蔽後顯示文字。"""
    lines = [
        "### 目前環境變數",
        "",
        "**IBM watsonx**",
        f"- `WATSONX_APIKEY`：{_mask_key(os.environ.get('WATSONX_APIKEY', ''))}",
        f"- `WATSONX_PROJECT_ID`：{_mask_key(os.environ.get('WATSONX_PROJECT_ID', ''))}",
        f"- `WATSONX_URL`：{os.environ.get('WATSONX_URL', '（未設定）')}",
        f"- `WATSONX_MODEL_ID`：{os.environ.get('WATSONX_MODEL_ID', 'ibm/granite-13b-chat-v2（預設）')}",
        "",
        "**Google Gemini**",
        f"- `GEMINI_API_KEY`：{_mask_key(os.environ.get('GEMINI_API_KEY', ''))}",
        f"- `GEMINI_MODEL_ID`：{os.environ.get('GEMINI_MODEL_ID', 'gemini-2.0-flash（預設）')}",
        "",
        "**資料路徑**",
        f"- PDF 目錄：`{_PDF_DIR}`",
        f"- URL 清單：`{_URL_FILE}`",
        f"- JSON 目錄：`{_JSON_DIR}`",
        f"- 向量資料庫：`{_VECTOR_DIR}`",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI 建構
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="IBM FlashSystem RAG 專家系統") as demo:

        # ── 頁面標題 ──
        gr.Markdown(
            """
# 🗄️ IBM FlashSystem RAG 專家系統
**兩階段推理：本機 RAG 過濾 → LLM 深度分析**
"""
        )

        # ── 共用狀態（vector_dir / collection） ──
        state_vector_dir = gr.State(_VECTOR_DIR)
        state_collection = gr.State(_COLLECTION)

        with gr.Tabs():

            # ──────────────────────────────────────────────────────────────
            # Tab 1：查詢
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("💬 查詢"):
                with gr.Row():
                    # 左側：Chatbot
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="對話記錄",
                            height=520,
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="輸入你的 FlashSystem 問題…",
                                show_label=False,
                                scale=5,
                            )
                            send_btn  = gr.Button("送出", variant="primary", scale=1)
                            clear_btn = gr.Button("清除", scale=1)

                    # 右側：篩選器
                    with gr.Column(scale=1, min_width=220):
                        gr.Markdown("### ⚙️ 查詢參數")
                        llm_dd = gr.Dropdown(
                            choices=["watsonx", "gemini"],
                            value="watsonx",
                            label="LLM 供應商",
                        )
                        topk_slider = gr.Slider(
                            minimum=4, maximum=20, step=1, value=8,
                            label="Top-K（第一層取回段落數）",
                        )
                        version_box = gr.Textbox(
                            label="版本篩選",
                            placeholder="例如 9.1.3（空白＝不篩選）",
                        )
                        source_dd = gr.Dropdown(
                            choices=["全部", "pdf", "web", "json_cli", "json_feature", "json_parts"],
                            value="全部",
                            label="來源類型篩選",
                        )

                # 事件綁定
                send_btn.click(
                    fn=chat_fn,
                    inputs=[msg_input, chatbot, llm_dd, version_box, topk_slider,
                            source_dd, state_vector_dir, state_collection],
                    outputs=chatbot,
                ).then(lambda: "", outputs=msg_input)

                msg_input.submit(
                    fn=chat_fn,
                    inputs=[msg_input, chatbot, llm_dd, version_box, topk_slider,
                            source_dd, state_vector_dir, state_collection],
                    outputs=chatbot,
                ).then(lambda: "", outputs=msg_input)

                clear_btn.click(fn=clear_chat, outputs=chatbot)

            # ──────────────────────────────────────────────────────────────
            # Tab 2：資料管理
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("📂 資料管理"):

                with gr.Row():
                    t2_vector_dir = gr.Textbox(
                        label="向量資料庫路徑", value=_VECTOR_DIR, scale=3
                    )
                    t2_collection = gr.Textbox(
                        label="Collection 名稱", value=_COLLECTION, scale=1
                    )

                gr.Markdown("---")

                # ── 網頁爬取 ──
                gr.Markdown("### 🌐 網頁爬取")
                with gr.Row():
                    with gr.Column(scale=2):
                        url_textarea = gr.Textbox(
                            label="直接輸入 URL（每行一個）",
                            placeholder="https://www.ibm.com/docs/en/flashsystem-9x00/9.1.3\nhttps://example.com/docs",
                            lines=6,
                        )
                    with gr.Column(scale=1):
                        url_file_upload = gr.File(
                            label="或上傳 web_urls.txt",
                            file_types=[".txt"],
                        )
                        max_pages_slider = gr.Slider(
                            minimum=1, maximum=7000, step=50, value=200,
                            label="每個根網址最多爬取頁數",
                        )
                        workers_slider = gr.Slider(
                            minimum=1, maximum=4, step=1, value=1,
                            label="同時使用的瀏覽器數量（建議 2~3）",
                        )
                web_btn    = gr.Button("🕷️ 爬取並加入 RAG", variant="primary")
                web_status = gr.Textbox(label="爬取狀態", interactive=False)

                web_btn.click(
                    fn=ingest_web_urls,
                    inputs=[url_textarea, url_file_upload, max_pages_slider,
                            t2_vector_dir, t2_collection, workers_slider],
                    outputs=web_status,
                )

                gr.Markdown("---")

                # ── PDF 重載 ──
                gr.Markdown(f"### 📄 PDF 重載　（來源：`{_PDF_DIR}`）")
                with gr.Row():
                    pdf_reset_chk = gr.Checkbox(label="重置（清除舊資料後重建）", value=False)
                    pdf_btn       = gr.Button("🔄 重新載入 PDF", variant="secondary")
                pdf_status = gr.Textbox(label="PDF 狀態", interactive=False)

                pdf_btn.click(
                    fn=reload_pdf,
                    inputs=[t2_vector_dir, t2_collection, pdf_reset_chk],
                    outputs=pdf_status,
                )

                gr.Markdown("---")

                # ── JSON 重載 ──
                gr.Markdown(f"### 🗂️ JSON 重載　（來源：`{_JSON_DIR}`）")
                json_btn    = gr.Button("🔄 重新載入 JSON", variant="secondary")
                json_status = gr.Textbox(label="JSON 狀態", interactive=False)

                json_btn.click(
                    fn=reload_json,
                    inputs=[t2_vector_dir, t2_collection],
                    outputs=json_status,
                )

                gr.Markdown("---")

                # ── 全部重建 ──
                gr.Markdown("### ♻️ 全部重建（PDF + JSON）")
                with gr.Row():
                    rebuild_reset_chk = gr.Checkbox(label="重置（清除舊資料後重建）", value=False)
                    rebuild_btn       = gr.Button("🔨 重建全部資料庫", variant="stop")
                rebuild_status = gr.Textbox(label="重建狀態", interactive=False)

                rebuild_btn.click(
                    fn=rebuild_all,
                    inputs=[t2_vector_dir, t2_collection, rebuild_reset_chk],
                    outputs=rebuild_status,
                )

                # 同步 state（vector_dir / collection 改變時）
                t2_vector_dir.change(lambda v: v, inputs=t2_vector_dir, outputs=state_vector_dir)
                t2_collection.change(lambda v: v, inputs=t2_collection, outputs=state_collection)

            # ──────────────────────────────────────────────────────────────
            # Tab 3：設定
            # ──────────────────────────────────────────────────────────────
            with gr.Tab("⚙️ 設定"):
                gr.Markdown("環境變數唯讀顯示（API Key 已遮蔽）。如需修改，請編輯 `.env` 檔案後重啟應用程式。")
                settings_md = gr.Markdown(get_settings_display())
                refresh_btn = gr.Button("🔁 重新整理")
                refresh_btn.click(fn=get_settings_display, outputs=settings_md)

                gr.Markdown("---")
                gr.Markdown("### Collection 名稱")
                settings_collection = gr.Textbox(
                    label="目前使用的 Collection",
                    value=_COLLECTION,
                )
                apply_btn    = gr.Button("套用", variant="primary")
                apply_status = gr.Textbox(label="狀態", interactive=False)

                def apply_collection(col_name: str):
                    _invalidate_expert()
                    return f"✅ Collection 已切換為：{col_name}（下次查詢生效）"

                apply_btn.click(
                    fn=apply_collection,
                    inputs=settings_collection,
                    outputs=apply_status,
                ).then(
                    fn=lambda v: v,
                    inputs=settings_collection,
                    outputs=state_collection,
                )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def launch(port: int = 7860, share: bool = False) -> None:
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",   # 只綁定本機（安全規範）
        server_port=port,
        share=share,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IBM FlashSystem RAG Web UI")
    parser.add_argument("--port",  type=int, default=7860, help="監聽 port（預設 7860）")
    parser.add_argument("--share", action="store_true",    help="建立 Gradio 公開連結（謹慎使用）")
    args = parser.parse_args()

    print(f"啟動 IBM FlashSystem RAG Web UI → http://127.0.0.1:{args.port}")
    launch(port=args.port, share=args.share)

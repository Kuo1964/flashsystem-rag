# FlashSystem RAG — Gradio GUI 計畫

## 概述

在現有的兩階段專家系統（CLI）上，加上 Gradio Web UI。
使用者可在瀏覽器中完成查詢、網頁爬取、PDF 匯入與參數設定，
不再需要使用命令列。

**範圍：**
- 新增 `app.py`（Gradio 主程式，三個 Tab）
- 修改 `src/ingest_web.py`（支援直接傳入 URL 字串列表）
- 更新 `requirements.txt`（加 gradio）
- 更新 `main.py`（加 `ui` 子指令）

**不在範圍內：**
- 改動向量資料庫邏輯
- 改動兩階段查詢邏輯
- 驗證功能或使用者帳號系統

---

## 子任務

---

### 子任務 1：requirements.txt 加入 gradio

**Intent**
安裝 Gradio 套件，後續子任務才能 import。

**Expected Outcomes**
- `requirements.txt` 含 `gradio>=5.0.0,<6.0.0`

**Todo List**
- [ ] 在 requirements.txt 的 Utilities 區塊加入 `gradio>=5.0.0,<6.0.0`

**Relevant Context**
- 檔案：`flashsystem-rag/requirements.txt`

**Status:** [ ] pending

---

### 子任務 2：修改 ingest_web.py，支援直接傳入 URL 列表

**Intent**
目前 `load_urls()` 只能讀取 txt 檔案。GUI 需要支援：
1. 使用者在 Textarea 直接輸入多個 URL（字串列表）
2. 使用者匯入 web_urls.txt（原本功能保持不變）

最小改動：新增 `load_urls_from_list()` 函數，接受 `List[str]`，
內部共用原本 `load_urls()` 的爬取邏輯（抽出一個 `_crawl_urls()` 私有函數）。

**Expected Outcomes**
- `load_urls_from_list(urls: List[str], ...) -> List[Document]` 函數可用
- 原本 `load_urls(url_file: str, ...)` 行為不變
- 兩者共用同一份爬取邏輯，沒有重複程式碼

**Todo List**
- [ ] 把 `load_urls()` 的核心爬取邏輯抽出成 `_crawl_urls(root_urls, chunk_size, chunk_overlap, max_pages_per_root, delay)`
- [ ] `load_urls()` 改為：讀取 txt → 呼叫 `_crawl_urls()`
- [ ] 新增 `load_urls_from_list(urls, ...)` → 過濾空行和 # 註解 → 呼叫 `_crawl_urls()`

**Relevant Context**
- 檔案：`flashsystem-rag/src/ingest_web.py`
- 現有函數：`load_urls(url_file, chunk_size, chunk_overlap, max_pages_per_root, delay)`
- 爬取核心邏輯位於 `load_urls()` 的 124–172 行（session、queue、BFS、text fetch）

**Status:** [ ] pending

---

### 子任務 3：新增 app.py（Gradio 三 Tab 介面）

**Intent**
建立 `flashsystem-rag/app.py`，使用 Gradio Blocks API 製作三個 Tab：

#### Tab 1：查詢（Chat）
- Chatbot 元件顯示對話歷史（問題 + 結構化回答）
- 文字輸入框 + 送出按鈕
- 側欄篩選器：
  - LLM 供應商（Dropdown：watsonx / gemini）
  - 版本篩選（Textbox，空白 = 不篩選）
  - Top-K（Slider，4–20，預設 8）
  - 來源類型篩選（Dropdown：全部 / pdf / web / json_cli）
- 每次回答後顯示結構化區塊（診斷、步驟、風險、總結、來源）
- FlashSystemExpert 在第一次查詢時才初始化（lazy init），
  切換 LLM 或 top_k 時重新初始化

#### Tab 2：資料管理（Data）
- **網頁來源區塊：**
  - Textarea：直接輸入 URL（每行一個）
  - File Upload：上傳 web_urls.txt
  - 「爬取並加入 RAG」按鈕 → 靜默模式（轉圈）→ 完成後顯示統計
- **PDF 來源區塊：**
  - 顯示目前 PDF 目錄路徑（從 .env 讀取）
  - 「重新載入 PDF」按鈕 → 靜默模式 → 顯示統計
- **JSON 來源區塊：**
  - 「重新載入 JSON」按鈕 → 顯示統計
- **全部重建按鈕：** 勾選 Reset 後按「重建全部資料庫」
- 所有爬取/建庫操作在背景執行，完成後更新狀態文字

#### Tab 3：設定（Settings）
- 顯示目前 .env 設定（唯讀，遮蔽 API Key）
- Chroma collection 名稱（可修改）
- 向量資料庫路徑（顯示，不可修改）
- 儲存按鈕：將修改寫回 session state（不寫入 .env，避免安全問題）

**Expected Outcomes**
- `python app.py` 啟動後，瀏覽器開啟 `http://127.0.0.1:7860`
- 三個 Tab 功能可獨立操作
- 查詢功能正確呼叫 FlashSystemExpert 兩階段流程
- 資料管理功能可觸發爬取並更新 Chroma DB
- 語法驗證通過

**Todo List**
- [ ] 建立 `flashsystem-rag/app.py`
- [ ] 實作 Tab 1（查詢）：Chatbot + 側欄篩選器 + lazy init FlashSystemExpert
- [ ] 實作 Tab 2（資料管理）：網頁爬取（URL 輸入 + txt 匯入）+ PDF + JSON 重載
- [ ] 實作 Tab 3（設定）：唯讀環境變數 + collection 名稱
- [ ] 實作頂部標題列與說明文字

**Relevant Context**
- 匯入：`from src.rag_query import FlashSystemExpert`
- 匯入：`from src.build_vectorstore import load_vectorstore, build_vectorstore`
- 匯入：`from src.ingest_web import load_urls, load_urls_from_list`
- 匯入：`from src.ingest_pdf import load_pdfs`
- 匯入：`from src.ingest_json import load_json_docs`
- FlashSystemExpert 簽名：`(vectorstore, top_k=8, llm_provider="watsonx", model_id=None, ...)`
- 預設路徑從環境變數讀取（RAG_VECTOR_DIR、RAG_PDF_DIR 等）
- Gradio 須綁定 `server_name="127.0.0.1"` 符合安全規範（不 bind 0.0.0.0）

**Status:** [ ] pending

---

### 子任務 4：main.py 加 ui 子指令

**Intent**
讓使用者可以用 `python main.py ui` 啟動 Gradio 介面，
保持 CLI 功能（build / query / search）不受影響。

**Expected Outcomes**
- `python main.py ui` 啟動 Gradio，預設 port 7860
- `python main.py ui --port 8080` 可指定 port
- 原本 build / query / search 指令行為不變
- 語法驗證通過

**Todo List**
- [ ] 在 `main.py` 加 `cmd_ui()` 函數（import app 並呼叫 launch）
- [ ] 在 `build_parser()` 加 `ui` 子指令（--port 參數，預設 7860）

**Relevant Context**
- 檔案：`flashsystem-rag/main.py`
- 現有子指令：build / query / search（不得修改）
- `_add_common_args()` 供 query / search 共用，build 與 ui 不使用

**Status:** [ ] pending

---

## 執行順序

1. 子任務 1（requirements.txt）
2. 子任務 2（ingest_web.py 重構）
3. 子任務 3（app.py 新增）
4. 子任務 4（main.py 加 ui 指令）

---

## 安全注意事項

- Gradio 須綁定 `127.0.0.1`，不可 `0.0.0.0`
- API Key 在設定頁面顯示時必須遮蔽（只顯示前 4 碼 + ****）
- 上傳的 txt 檔案只解析 URL，不執行任何內容

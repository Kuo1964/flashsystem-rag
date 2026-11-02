# FlashSystem RAG 專案交接摘要

> 產生時間：2026-11-01 11:13（CST）。給新對話框接續工作用。
> 請一律用**繁體中文**回覆。安全規範見下方「不可違反的規則」。

---

## 1. 專案目標

建立並維護一個 **IBM FlashSystem 兩階段 RAG 專家系統**：

1. 本機 Chroma 向量資料庫（PDF、網頁、JSON 三種來源）。
2. 兩階段推理：第一層本機 RAG 過濾，第二層 LLM 深度分析（IBM watsonx 或 Google Gemini 可切換）。
3. Gradio Web UI（三分頁：查詢／資料管理／設定）加 CLI（`main.py build/query/search/ui`）。

程式碼位置：`/Users/yhkuo/.bob/playground/flashsystem-rag/`

---

## 2. 不可違反的規則

- Gradio 只綁 `127.0.0.1`，不可綁 `0.0.0.0`。
- API Key 不可 hardcode。`.env` 含 API Key，只能改路徑行，不要顯示其他內容。
- **必須用 `.venv/bin/python3`**（系統 `python3` 沒有 chromadb）。執行腳本要 `PYTHONPATH=.`，或在程式裡 `sys.path.insert(0,'.')`。
- 離線使用 embedding 模型要加 `HF_HUB_OFFLINE=1`。
- 資料變動前先備份。修改後做語法驗證（`py_compile`）並盡量實測。
- 用 `update_todo_list` 時，既有已完成項目不可移除。
- 使用者習慣：我提出方案 → 使用者回覆編號或 yes/no。使用者說「檢查進度」時，要實際查日誌與程序狀態再回報。
- **我無法自己在稍後回來檢查**，只在使用者傳訊息時才動作，要如實告知。

---

## 3. 環境與注意事項

- macOS，Apple M1 Pro，16 GB RAM，MPS 可用。記憶體長期吃緊（swap 常用 1.5～10 GB），可能讓程序進入 `UN` 狀態、速度大降且不理 SIGINT。
- **重爬期間不要同時啟動 `app.py` 或其他寫入向量庫的程序。**
- `SIGINT` 常常無法讓爬蟲結束，需要 `pkill -TERM`，之後 `pkill -f "rag_crawl_"` 並刪 `/tmp/rag_crawl_*`。
- 查日誌（`LC_ALL=C` 避免 `tr: Illegal byte sequence`）：
  ```bash
  LC_ALL=C tr '\r' '\n' < ~/RAG_logs/crawl_860b.log | grep -a 已完成 | tail -2
  ```
- Chroma 查詢技巧：
  - `col.get()` 必須分頁（`limit=5000, offset=…`）。
  - `where_document={"$contains":…}` 在二十萬筆以上很慢，不可用於查詢路徑。
  - 直接用唯讀 SQLite（`file:…/chroma.sqlite3?mode=ro`，表 `embedding_metadata`，`key='source'` 或 `key='chroma:document'`）很快。
- 注意：我先前用 `stat` 或 `date` 時看到的時間曾落後現實，判斷「是否卡住」前要用 `date` 與 `ps etime` 交叉確認。
- `source_type` 實際值：`pdf`、`web`、`json_cli`、`json_feature`、`json_parts`。

---

## 4. 目前狀態

### 4.1 向量庫

- 路徑：`/Users/yhkuo/RAG/vectorstore_bge`，collection `flashsystem_all`，embedding 模型 `BAAI/bge-m3`（1024 維）。
- 筆數：225,091（11:13 時，還會增加）。`quick_check` 在上次重啟前為 ok。
- 舊 mpnet 庫 `/Users/yhkuo/RAG/vectorstore`（約 5 GB）仍保留，內含舊爬蟲記錄；建議再放幾天才決定是否刪。
- `.env` 已指向新庫與 bge-m3；備份 `.env.bak_mpnet`（含 API Key，權限 600，已列入 `.gitignore`）。

### 4.2 進行中的背景工作：8.6.0 完整重爬

- 指令：
  ```bash
  HF_HUB_OFFLINE=1 nohup caffeinate -dims .venv/bin/python3 main.py build --source web \
    --url-file ~/RAG/raw_data/web_urls_860.txt --vector-dir ~/RAG/vectorstore_bge \
    --collection flashsystem_all --max-web-pages 20000 --workers 3 \
    > ~/RAG_logs/crawl_860b.log 2>&1 &
  ```
- PID 38553（11:13 時已跑 49 分鐘）。日誌 `/Users/yhkuo/RAG_logs/crawl_860b.log`（前一輪日誌 `crawl_860.log`，已被停止）。
- 根網址：`~/RAG/raw_data/web_urls_860.txt`，共 3 個：`flashsystem-5x00/8.6.0`、`7x00/8.6.0`、`9x00/8.6.0`。
- 進度：已完成 7,794 頁（續爬基準 7,594 + 本次 200）。目前還在第一個根網址 `5x00`，庫內 8.6.0 5x00 約 4,762 頁，7x00 與 9x00 仍是舊的 5、4 頁。
- 續爬記錄：`/Users/yhkuo/RAG/vectorstore_bge/crawl_state_flashsystem_all.jsonl`（約 31 MB）。重啟會自動接續，不重爬已完成的頁面。
- 速度：整體約 9 頁/分（粗估）。預估剩餘 15～20 小時（**僅為估計**，8.6.0 總頁數未知；參考其他版本每個產品線約 4,700～5,400 頁）。
- 重啟後要先走過約 7,600 個已完成頁重建佇列，這段約 40 分鐘沒有輸出，不是卡住。

停止重爬的方式：
```bash
pkill -TERM -f "Python main.py build"; pkill -f caffeinate
pkill -f "rag_crawl_"; rm -rf /tmp/rag_crawl_*
```

---

## 5. 已完成的工作

| # | 項目 | 說明 |
|---|---|---|
| 1 | 爬取 30 個根網址 | 累計 51,489 頁；向量庫曾達 522,505 筆 |
| 2 | 同頁前後文展開 | `src/rag_query.py`：`_expand_with_neighbors`、`_join_chunks`、`_page_where`；`_NEIGHBOR_WINDOW=1`；`_MAX_CONTEXT_CHARS=9000` |
| 3 | 換 embedding 為 bge-m3 | 舊模型 `max_seq_length=128`，chunk 約 92% 被截斷；後段查詢 recall@5 由 0.29 升到 0.77。遷移腳本 `scripts/migrate_embeddings.py` |
| 4 | 錯誤碼／指令名精確比對 | `CMMVC5770E` 這類向量檢索分不出，改用網址最後一段 `topic=messages-cmmvc5770e` 當索引；`_AMBIGUOUS_CMD_WORDS` 排除 host/backup/help/snap 等一般單字 |
| 5 | 修正「抓到導覽骨架」 | 約 27.6% 網頁（14,217 頁）的 chunk 是導覽殼。`src/ingest_web.py` 的 `_extract_content` 與 `_fetch_text` 不再退回 body。`scripts/refetch_bad_pages.py` 重爬修復：13,029 頁修復，813 頁失敗 |
| 6 | 清除無主文頁 | 失敗的 834 個網址（硬體安裝、安全聲明、版本首頁等）重爬後仍只有目錄與頁尾，已備份後刪除 6,641 筆。腳本 `scripts/purge_failed_pages.py`（預設 dry-run，`--apply` 才刪） |
| 7 | 檢索重測 | 7 項測試全過，且無導覽殘留（`CMMVC5770E`、`CMMVC5767E`、`lssystem`、`mkvdisk`、9.1.3 機架安裝、HyperSwap 等） |
| 8 | `app.py` 重啟 | 曾於收尾驗證後重啟；**目前已關閉**（為了重爬） |
| 9 | 查明版本資料偏少原因 | 見第 6 節 |
| 10 | 修正 8.6.0 爬蟲漏抓 | 見第 6 節。假瀏覽器測試 21 項全過；真瀏覽器試跑 110 頁成功 |

README 已有章節：調整 Embedding 模型、同頁前後文展開、修復抓到導覽骨架的網頁、錯誤碼／指令名精確比對。`.env.example` 已補 `RAG_EMBED_MODEL`、`RAG_NEIGHBOR_WINDOW`、`RAG_MAX_CONTEXT_CHARS`。

---

## 6. 版本資料偏少的調查結論

| 版本 | 原因 | 狀態 |
|---|---|---|
| 8.7.1、8.7.2、8.7.3 | IBM 已下架網頁文件，開啟只有一頁「End of support」公告，連原本存在的網址也被導向 `topic=eos-871-7k` | 無法補。8.7.1、8.7.3 靠 PDF（各約 7,700 筆）；**8.7.2 沒有 PDF，只有 8 筆公告 chunk，問 8.7.2 找不到內容** |
| 8.6.0 | 爬蟲漏抓：8.6.0 目錄連結很少（每頁 35～50 個，8.7.0 約 2,600 個），一部分是舊式網址 `…/docs/en/STSLR9_8.6.x/<doc>/<file>.html`，載入後會導向正式網址；舊爬蟲只收含根網址的連結 | 修正中，重爬進行中 |
| 9.1.2 | 爬蟲只記錄 3 頁；庫內網頁 431 筆、PDF 8,111 筆。開 `9.1.2?topic=messages-cmmvc5770e` 時主文為 0 字 | **未深查**，可能是渲染問題或本來沒有完整網頁 |

### 本次對 `src/ingest_web.py` 的修改

- `_LEGACY_LINK_RE`、`_LEGACY_PAGE_EXT`、`_in_scope(link, root_clean)`：同「主.次」版本的舊式網址視為範圍內，**但只放行 `.htm`／`.html`**（PDF、mp4 載入後是 404 頁）。
- `_final_url(url, driver)`：取瀏覽器實際停留網址。
- `_fetch_text` 現在回傳 `(text, links, final_url)` 三個值。主文未出現時也回傳 `links`。
- `_handle_page`：
  - 只有舊式網址才改用正式網址當 `source`；一般網址行為不變。
  - 導向後的網址若在根網址以外或已抓過，不重複收錄。
  - 續爬記錄同時記舊式與正式網址。
  - 沒有主文、但範圍內連結 ≥ `_LANDING_MIN_LINKS`（5）的 landing page：不寫 chunk，只用連結展開佇列，並記為已完成。
- 呼叫端 `scripts/refetch_bad_pages.py` 的 `_fetch_text(url, raw)[0]` 取第一個值，**仍相容**。
- 測試腳本在 `/tmp/test_legacy.py`（暫存，可能已不存在；需要可重寫）。

---

## 7. 已知問題與待清理

1. **庫內有 14 筆 404 chunk**：內容以 `Go back / This isn't the page you were looking for` 開頭，來源是 `…STHGUJ_8.6.x/…pdf`、`…mp4`。
   - 起因：第一輪爬蟲用的是過寬的舊式連結規則。
   - 新版程式已不再產生，但舊的還在庫裡，爬完後要刪。
2. **62 個 `…8.6.0?topic=STHGUJ_8.6.x/…htm(l)` 來源**：5x00 的舊式網址載入後實際網址仍是這種長格式，不是 `topic=commands-xxx`。
   - 我開 `svc_stopsystem.htm` 驗證，主文有 3,472 字，內容正常。**是有效內容，保留。**
   - 只驗證一頁，沒有全面比對是否與其他頁面重複。
3. 新爬的 chunk 可能夾帶少量骨架頁；爬完後要用 `--list-only` 檢查。
4. 8.6.0 爬蟲 WARN（主文未出現、載入失敗重試）共約 37 次，原因未逐一確認。
5. 向量庫總筆數從最初 522,505 降到約 22 萬，是因為骨架 chunk 被正文取代，**不是資料遺失**。

---

## 8. 下一步（使用者說「重爬完成」之後）

1. **備份** 目前庫：
   ```bash
   sqlite3 "file:$HOME/RAG/vectorstore_bge/chroma.sqlite3?mode=ro" ".backup $HOME/RAG_backup/bge_chroma_after_860.sqlite3"
   ```
   現有備份在 `~/RAG_backup/`：`bge_chroma_before_refetch.sqlite3`（重爬骨架頁前）、`bge_chroma_after_refetch.sqlite3`（骨架頁重爬後、清除前）、`bge_chroma_before_860.sqlite3`（8.6.0 重爬前）。
2. **刪除 404 chunk**：先 dry-run 看筆數，再刪（可參考 `scripts/purge_failed_pages.py` 的寫法，條件改為文件開頭是 "Go back" 且含 "This isn't the page you were looking for"）。
3. **統計**：
   - `HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/refetch_bad_pages.py --list-only`：殘留骨架頁數（理想為 0）。
   - 8.6.0 各產品線頁數、`source_type` 構成、SQLite `quick_check`。
4. **重測檢索**：用 8.6.0 的問題（例如指定 `version="8.6.0"`）、`CMMVC5770E`、`lssystem command parameters` 等，確認各版本都能取回主文且無導覽殘留。
5. **重啟 `app.py`**（綁 `127.0.0.1:7860`），用 Gemini／watsonx 跑完整兩階段問答測試。
6. 更新 README：`scripts/purge_failed_pages.py` 說明、`--workers` 與 chromedriver 副本說明、8.6.0 舊式連結處理、8.7.1～8.7.3 靠 PDF 的說明。

### 可選

- 深查 9.1.2：開 `9.1.2` 的幾個 topic 頁，確認是渲染太慢還是本來沒有網頁；必要時用 `refetch_bad_pages.py` 類似流程補抓。
- 決定是否刪除舊 mpnet 庫 `/Users/yhkuo/RAG/vectorstore`（約 5 GB，建議再保留幾天）。
- 提醒使用者：SSD 不穩，以及 CrowdStrike 與 chromedriver 被殺可能有關，建議問資安。

---

## 9. 檔案與目錄一覽

```
/Users/yhkuo/.bob/playground/flashsystem-rag/
├── .env                      ← 已切到 vectorstore_bge + RAG_EMBED_MODEL=BAAI/bge-m3（含 API Key，勿顯示）
├── .env.bak_mpnet            ← 切換前備份（含 API Key，chmod 600）
├── .env.example / .gitignore / README.md
├── app.py / main.py          ← app.py 目前關閉；main.py build 支援 --workers、--max-web-pages
├── scripts/
│   ├── compact_crawl_state.py
│   ├── migrate_embeddings.py     ← 舊 collection → 新 embedding 模型（可續傳）
│   ├── refetch_bad_pages.py      ← 重爬骨架頁（--list-only、--limit、--urls-file、--purge-unfetchable、--workers）
│   └── purge_failed_pages.py     ← 清除 refetch_failed.txt 內網址的 chunk（預設 dry-run，--apply 才刪）
└── src/
    ├── rag_query.py          ← 同頁展開、錯誤碼／指令名精確比對、context 上限 9000
    ├── build_vectorstore.py  ← DEFAULT_EMBED_MODEL 由 RAG_EMBED_MODEL 控制（預設 bge-m3）
    ├── ingest_web.py         ← 骨架偵測、舊式連結範圍判斷、landing page 處理
    └── crawl_state.py / ingest_pdf.py / ingest_json.py

/Users/yhkuo/RAG/
├── raw_data/{pdfs, web_urls.txt, web_urls_860.txt, manual_docs/}
├── vectorstore/        ← 舊 mpnet 庫（唯讀保留）
└── vectorstore_bge/    ← 目前使用的 bge-m3 庫；crawl_state_flashsystem_all.jsonl、refetch_failed.txt

/Users/yhkuo/RAG_logs/   ← crawl_860.log（已停）、crawl_860b.log（進行中）、refetch_bad2.log（已完成）
/Users/yhkuo/RAG_backup/ ← 三份庫快照（見第 8 節）與舊快照
```

套件版本：chromadb 0.6.3、gradio 6.30、selenium 4.31.0、sentence-transformers 6.1.0、torch 2.14.1。

---

## 10. 建議的新對話開場

> 請先讀 `flashsystem-rag/HANDOVER.md`。8.6.0 重爬正在背景執行（PID 38553），請先「檢查進度」：看 `~/RAG_logs/crawl_860b.log` 與程序狀態，再回報。

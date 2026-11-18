# FlashSystem RAG 專案工作交接與狀態總結

> **建立時間**：2026-11-08（最後更新：2026-11-15）
> **適用情境**：新對話框接續工作、系統上線運維與日常維護標準作業。

---

## 1. 當前狀態速覽（Executive Summary）

1. **8.6.0 全量重爬**：**已 100% 圓滿完成**
   - 涵蓋 5000 / 7000 / 9000 三大產品線（共 86,599 筆 chunk，404/骨架頁 0 筆殘留）。
   - 階段快照備份：`/Users/yhkuo/RAG_backup/bge_chroma_after_860.sqlite3`（約 3.9 GB）。
2. **14 根網址補充爬取（SVC 全版本 + 缺漏系列）**：**已 100% 圓滿完成！**
   - 涵蓋 SVC 9.1.3 ~ 8.4.0 全版本、FS5000 8.4.0 / 8.5.0 / 9.1.0、FS7000 9.1.1 / 8.4.0、FS9000 9.1.0（共 14 根網址）。
   - 累計完成頁面數達 **203,426 頁**；Chroma 向量庫（`flashsystem_all`）總向量數達 **477,839 筆 Chunks**。
   - 404 / 無效頁面清理：0 筆殘留；骨架頁檢查：**0 筆殘留**。
   - 安全快照備份已完成：`/Users/yhkuo/RAG_backup/bge_chroma_after_patch.sqlite3`（3.9 GB）。
3. **Web UI（`app.py`）狀態**：**已上線運行中（PID: 28664）**
   - 監聽位址：`http://127.0.0.1:7860`（嚴格綁定 localhost，無外部曝險）。

---

## 2. 背景任務資訊（Background Process）

- **補充爬取狀態**：**已於 2026-11-15 100% 爬取完畢**
- **累計處理**：14 個根網址、203,426 頁、477,839 個向量段落。
- **補爬清單（`/Users/yhkuo/RAG/raw_data/web_urls_patch.txt`，共 14 個根網址，全數爬取入庫）**：
  1. `https://www.ibm.com/docs/en/sanvolumecontroller/9.1.3`
  2. `https://www.ibm.com/docs/en/sanvolumecontroller/9.1.2`
  3. `https://www.ibm.com/docs/en/sanvolumecontroller/9.1.1`
  4. `https://www.ibm.com/docs/en/sanvolumecontroller/9.1.0`
  5. `https://www.ibm.com/docs/en/sanvolumecontroller/8.7.0`
  6. `https://www.ibm.com/docs/en/sanvolumecontroller/8.6.0`
  7. `https://www.ibm.com/docs/en/sanvolumecontroller/8.5.0`
  8. `https://www.ibm.com/docs/en/sanvolumecontroller/8.4.0`
  9. `https://www.ibm.com/docs/en/flashsystem-5x00/8.4.0`
  10. `https://www.ibm.com/docs/en/flashsystem-5x00/8.5.0`
  11. `https://www.ibm.com/docs/en/flashsystem-7x00/9.1.1`
  12. `https://www.ibm.com/docs/en/flashsystem-5x00/9.1.0`
  13. `https://www.ibm.com/docs/en/flashsystem-9x00/9.1.0`
  14. `https://www.ibm.com/docs/en/flashsystem-7x00/8.4.0`

---

## 3. 安全規範與重要約束（Rules & Guardrails）

1. **Python 環境**：一律使用專案虛擬環境 `flashsystem-rag/.venv/bin/python3`，並加上 `PYTHONPATH=.`。
2. **Embedding 離線調用**：使用 `BAAI/bge-m3` 模型時，環境變數需加 `HF_HUB_OFFLINE=1`。
3. **避免 SQLite 資料庫鎖定**：**爬蟲執行期間絕不可啟動 `app.py` 或任何寫入向量庫的程式**。
4. **網路與主機綁定**：Gradio 服務僅能綁定 `127.0.0.1`，嚴禁綁定 `0.0.0.0`。
5. **金鑰安全**：`.env` 內含敏感 API Key，禁止直接讀取顯示其內容，僅能透過程式讀取或修改路徑。
6. **查證在先（Evidence First）**：所有進度與統計必須以 `ps`、log 檔與 SQLite 實際查詢為準，不可臆測。

---

## 4. 重要檔案與路徑目錄

| 類別 | 路徑 | 說明 |
| :--- | :--- | :--- |
| **專案工作目錄** | `/Users/yhkuo/.bob/playground/flashsystem-rag` | 專案程式碼與 CLI 入口 |
| **專案總綱文件** | [`flashsystem-rag/HANDOVER.md`](HANDOVER.md) | 系統交接、架構與維運指南 |
| **Chroma 向量資料庫** | `/Users/yhkuo/RAG/vectorstore_bge/` | 當前主力向量庫（內含 `chroma.sqlite3`） |
| **爬蟲狀態記錄檔** | `/Users/yhkuo/RAG/vectorstore_bge/crawl_state_flashsystem_all.jsonl` | 續爬依據（已記錄 7.7 萬+ 頁面） |
| **8.6.0 安全快照** | `/Users/yhkuo/RAG_backup/bge_chroma_after_860.sqlite3` | 8.6.0 重爬完成後之完整備份 |
| **全量補爬安全快照** | `/Users/yhkuo/RAG_backup/bge_chroma_after_patch.sqlite3` | 14 根網址補爬完成後之完整備份（3.9 GB） |
| **Web UI 執行日誌** | `/Users/yhkuo/RAG_logs/gradio_app.log` | Gradio Web UI 背景運行日誌 |
| **補爬網址清單** | `/Users/yhkuo/RAG/raw_data/web_urls_patch.txt` | 14 個補爬根網址清單 |

---

## 5. 新對話框接續操作流程（Next Steps SOP）

在新對話框中，請依照以下標準步驟進行：

### Step 1: 檢查背景爬蟲進度
```bash
# 檢查 PID 12251 是否仍在運行
ps -p 12251 -o pid,stat,time,command

# 查看最新 30 行爬取日誌
tail -n 30 /Users/yhkuo/RAG_logs/crawl_patch.log
```
* 若 PID 仍在運行，向使用者回報當前爬取頁數與寫入進度。
* 若 PID 已結束，檢查日誌末端確認是否顯示全部 14 個根網址完成，並進入 Step 2。

---

### Step 2: 向量資料庫快照備份（爬蟲完成後執行）
```bash
cp /Users/yhkuo/RAG/vectorstore_bge/chroma.sqlite3 /Users/yhkuo/RAG_backup/bge_chroma_after_patch.sqlite3
ls -lh /Users/yhkuo/RAG_backup/bge_chroma_after_patch.sqlite3
```

---

### Step 3: 清理無效頁面與檢查骨架頁
```bash
cd /Users/yhkuo/.bob/playground/flashsystem-rag

# 1. 檢查並清理 404 / 無效頁面（若有）
PYTHONPATH=. .venv/bin/python3 scripts/purge_failed_pages.py

# 2. 檢查骨架頁殘留（目標：0 筆）
PYTHONPATH=. .venv/bin/python3 scripts/refetch_bad_pages.py --list-only
```

---

### Step 4: 執行檢索回歸測試（7 組案例）
```bash
cd /Users/yhkuo/.bob/playground/flashsystem-rag

# 測試 1: 錯誤碼精確檢索
HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python3 main.py search "CMMVC5770E" --n-results 3

# 測試 2: 指令名精確檢索
HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python3 main.py search "lssystem command" --n-results 3

# 測試 3: 8.6.0 功能檢索
HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python3 main.py search "8.6.0 Safeguarded Copy" --n-results 3

# 測試 4: SVC 專屬功能檢索
HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python3 main.py search "SAN Volume Controller 2145-SV3 node canister" --n-results 3

# 測試 5: SVC MDisk 擴容架構問題
HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python3 main.py search "SVC expand mdisk addmdisk storage pool" --n-results 3
```

---

### Step 5: 啟動 Gradio Web UI
```bash
cd /Users/yhkuo/.bob/playground/flashsystem-rag
nohup .venv/bin/python3 app.py > /Users/yhkuo/RAG_logs/gradio_app.log 2>&1 &

# 驗證 Web UI 綁定於 127.0.0.1:7860
lsof -iTCP:7860 -sTCP:LISTEN
```
* 當前 Web UI 背景運行 PID: `42895`，監聽於 `http://127.0.0.1:7860`。
* 已修復 `app.py` 中 `import re` 缺失問題，確保 Gemini 供應商在輸出格式化時不再發生 `Error`。
* 已內建 CLI 實體語法原型前置注入、原廠指令白名單約束、後處理 CLI 防幻覺校驗器（`_validate_and_sanitize_cli`）與步驟編號連續性排版修復。
* 最大輸出長度已調至頂規 `8192` tokens（支援環境變數 `RAG_MAX_OUTPUT_TOKENS` 動態調整）。
* 已內建多維度查詢矩陣分解、RRF 排名融合、雜訊過濾與第一性原理 Mermaid / CLI 專家輸出契約。

---

## 6. 重要知識庫與架構備忘

- **SVC MDisk 擴容架構規範**：
  - IBM Spectrum Virtualize **沒有 `expandmdisksize` 指令**。
  - SVC 不支援直接對既有受管 MDisk 進行原地大小擴充（In-place expansion）。
  - 標準擴充作業程序：於後端儲存陣列（如 FS5030）建立新 LUN $\rightarrow$ 在 SVC 執行 `detectmdisk` $\rightarrow$ 使用 `addmdisk -mdisk <mdisk_name> <mdiskgroup_name>` 將新 MDisk 加入既有 Storage Pool。

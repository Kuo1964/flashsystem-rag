# IBM FlashSystem RAG 知識庫

IBM FlashSystem 多版本文件的 RAG（Retrieval-Augmented Generation）系統，
整合 Chroma 向量資料庫與 IBM watsonx.ai，支援 PDF、網頁、結構化 JSON 三種資料來源。

---

## 系統架構

```
raw_data/
├── pdfs/                          ← 103 個 FlashSystem PDF 手冊
├── web_urls.txt                   ← IBM Docs 網址清單（8.5.x ~ 9.1.x）
└── manual_docs/
    ├── feature_lifecycle_matrix.json
    ├── official_cli_commands_whitelist.json
    └── official_grounded_parts_from_pdf.json

vectorstore/                       ← Chroma 持久化向量資料庫

src/
├── ingest_pdf.py                  ← PDF 載入與切割
├── ingest_web.py                  ← 網頁爬取與切割
├── ingest_json.py                 ← JSON 文件載入
├── build_vectorstore.py           ← Chroma 向量資料庫建立
└── rag_query.py                   ← IBM watsonx RAG 查詢介面

main.py                            ← CLI 主程式入口
```

---

## 快速開始

### 1. 建立 Python 虛擬環境並安裝套件

```bash
cd flashsystem-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 設定環境變數

複製 `.env.example` 為 `.env` 並填入你的 IBM watsonx 認證資訊：

```bash
cp .env.example .env
# 編輯 .env，填入：
#   WATSONX_APIKEY      ← IBM Cloud API Key
#   WATSONX_PROJECT_ID  ← watsonx.ai Project ID
```

取得認證資訊：
- API Key：[IBM Cloud IAM](https://cloud.ibm.com/iam/apikeys)
- Project ID：[watsonx.ai Studio](https://dataplatform.cloud.ibm.com/) → 專案設定

### 3. 建立向量資料庫

```bash
# 全部來源（PDF + 網頁 + JSON）
python main.py build

# 只處理 PDF（速度較快，適合初次測試）
python main.py build --source pdf

# 重建（清除舊資料）
python main.py build --reset
```

> ⚠️ 首次執行會自動下載 sentence-transformers 模型（約 1GB）。
> 103 個 PDF 全部處理約需 15–30 分鐘（視機器效能而定）。

### 4. 查詢

```bash
# 互動式查詢
python main.py query

# 單次查詢
python main.py search "How do I create a volume on FlashSystem 9500?"

# 篩選特定版本
python main.py search "What CLI commands are available for replication?" --version 9.1.3

# 只查 PDF 來源
python main.py search "What is the maximum drive count?" --source-type pdf
```

---

## 進階設定

### 變更 LLM 模型

在 `.env` 中設定 `WATSONX_MODEL_ID`，可用模型：

| Model ID | 說明 |
|----------|------|
| `ibm/granite-13b-chat-v2` | 預設，平衡速度與品質 |
| `ibm/granite-3-8b-instruct` | 輕量快速 |
| `meta-llama/llama-3-70b-instruct` | 高品質，速度較慢 |

### 調整 Embedding 模型

在 [`src/build_vectorstore.py`](src/build_vectorstore.py) 中修改 `DEFAULT_EMBED_MODEL`：
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`（預設，多語言）
- `sentence-transformers/all-MiniLM-L6-v2`（英文，速度較快）

### 調整段落切割參數

```bash
python main.py build --chunk-size 800 --chunk-overlap 100
```

---

## 模組說明

| 模組 | 功能 |
|------|------|
| [`src/ingest_pdf.py`](src/ingest_pdf.py) | 讀取 PDF，依頁面切割，附版本 metadata |
| [`src/ingest_web.py`](src/ingest_web.py) | 爬取 IBM Docs，遞迴抓子頁面，附產品/版本 metadata |
| [`src/ingest_json.py`](src/ingest_json.py) | 轉換三種 JSON 格式為可讀文字段落 |
| [`src/build_vectorstore.py`](src/build_vectorstore.py) | HuggingFace embedding + Chroma 批次寫入/載入 |
| [`src/rag_query.py`](src/rag_query.py) | watsonx LLM + MMR 搜尋 + 帶 metadata 篩選的查詢 |
| [`main.py`](main.py) | CLI 入口：`build` / `query` / `search` |

---

## 注意事項

- 網頁爬取預設每個根網址最多 200 個子頁面（`--max-web-pages` 可調整）
- 爬取間隔 0.3 秒（`ingest_web.py` 中 `delay` 參數），避免對 IBM Docs 造成負擔
- 向量資料庫儲存於 `/Volumes/SSD_4TB/RAG/vectorstore`，可在 `.env` 中修改 `RAG_VECTOR_DIR`
- 不要將 `.env` 提交至版本控制（`.gitignore` 已排除）

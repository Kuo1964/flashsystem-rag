"""
單元測試：FlashSystem RAG 多維度專家推理與檢索工作流 (TDD - unittest)
"""
import unittest
from langchain_core.documents import Document

try:
    from src.rag_query import (
        _decompose_expert_queries,
        _filter_noisy_chunks,
        _reciprocal_rank_fusion,
        _EXPERT_PROMPT_TEMPLATE,
        _parse_expert_output,
        _validate_and_sanitize_cli,
        FilterResult,
    )
except ImportError as e:
    pass


class TestRagExpertPipeline(unittest.TestCase):

    def test_decompose_expert_queries_pbha_draid(self):
        """測試 PBHA 與硬碟擴充問題的查詢分解矩陣"""
        q = "我的环境是 FlashSystem 5600 两座，做 PBHA，上面有 100TB 的资料。现在我要针对每一座去新增硬盘的空间，做法会是先把原来的硬盘全部换成两倍容量的硬盘。要在不中断系统服务的状况下该怎么进行？"
        queries = _decompose_expert_queries(q)
        self.assertGreaterEqual(len(queries), 3)
        joined = " ".join(queries).lower()
        self.assertTrue(any(k in joined for k in ["pbha", "policy-based", "replication", "high availability"]))
        self.assertTrue(any(k in joined for k in ["draid", "array", "drive", "expand", "capacity"]))
        self.assertTrue(any(k in joined for k in ["migrate", "extent", "pool", "rmarray"]))

    def test_filter_noisy_chunks(self):
        """測試過濾與主題無關的物理警告雜訊"""
        noisy_doc = Document(
            page_content="CAUTION: An open expansion enclosure lid reduces airflow and might cause the enclosure or its components to overheat and shut down. Do not leave the lid open for more than 15 minutes.",
            metadata={"source": "fs5600_pdfguide.pdf", "page": 181}
        )
        relevant_doc = Document(
            page_content="Policy-based replication provides active-active high availability between two independent systems. When maintenance is required on one cluster, host I/O continues seamlessly on the alternate system.",
            metadata={"source": "redp5704.pdf", "page": 35}
        )
        docs = [noisy_doc, relevant_doc]
        filtered = _filter_noisy_chunks(docs, "FlashSystem 5600 PBHA 擴充硬碟容量")
        self.assertIn(relevant_doc, filtered)
        self.assertNotIn(noisy_doc, filtered)

    def test_reciprocal_rank_fusion(self):
        """測試 RRF 演算法多路排名融合"""
        doc1 = Document(page_content="Doc 1", metadata={"source": "s1"})
        doc2 = Document(page_content="Doc 2", metadata={"source": "s2"})
        doc3 = Document(page_content="Doc 3", metadata={"source": "s3"})

        list_a = [doc1, doc2]
        list_b = [doc2, doc3]

        fused = _reciprocal_rank_fusion([list_a, list_b], k=60)
        self.assertEqual(fused[0].page_content, "Doc 2")
        self.assertEqual(len(fused), 3)

    def test_expert_prompt_contract(self):
        """測試 Prompt 為通用骨架：有輸出契約與查證原則，且不含寫死的單一情境範例或指令白名單"""
        # 輸出契約
        for tag in ("[DIAGNOSIS]", "[STEPS]", "[RISKS]", "[SUMMARY]", "[SOURCES]"):
            self.assertIn(tag, _EXPERT_PROMPT_TEMPLATE)
        self.assertIn("{context}", _EXPERT_PROMPT_TEMPLATE)
        self.assertIn("{question}", _EXPERT_PROMPT_TEMPLATE)
        # 通用骨架的關鍵要求
        self.assertIn("前提", _EXPERT_PROMPT_TEMPLATE)
        self.assertIn("驗收標準", _EXPERT_PROMPT_TEMPLATE)
        self.assertIn("頁碼", _EXPERT_PROMPT_TEMPLATE)
        self.assertIn("逐字出自【參考資料】", _EXPERT_PROMPT_TEMPLATE)
        # 不得寫死特定情境或特定指令
        for hardcoded in ("PBHA", "DRAID", "mkmdiskgrp", "mkarray", "mkpool", "chvolumegroupreplication", "100TB", "```mermaid"):
            self.assertNotIn(hardcoded, _EXPERT_PROMPT_TEMPLATE)

    def test_post_processing_cli_validator(self):
        """測試 CLI 防幻覺校驗器能夠自動攔截並修正幻覺指令"""
        hallucinated_text = "使用 mkpool -name new_pool 建立儲存池，並下達 lsvolumegroup -replicationpolicy 檢查狀態。"
        sanitized = _validate_and_sanitize_cli(hallucinated_text)
        self.assertNotIn("mkpool", sanitized)
        self.assertIn("mkmdiskgrp", sanitized)
        self.assertNotIn("-replicationpolicy", sanitized)

    def test_parse_expert_output_with_mermaid_and_cli(self):
        """測試解析包含 Mermaid 與 CLI 指令的專家回答並通過校驗器"""
        raw_llm = """
[DIAGNOSIS]
FlashSystem 5600 採用 DRAID 架構，無法原地更換大硬碟擴容。但具備 PBHA 雙活架構，可透過站點輪替升級。

[STEPS]
```mermaid
flowchart TD
    A[檢查 PBHA] --> B[切換 I/O 至 Site B]
```

1. 事前健康檢查：確認 PBHA 同步狀態與 Quorum 在線。
2. 切換主機 I/O 偏好路徑至 Site B。
3. 執行指令重建 Site A：
```bash
mkmdiskgrp -ext 1024 -name pool_A
mkarray -level raid6 -driveclass 100 -drivecount 12 pool_A
chvolumegroupreplication -replicationpolicy pbha_policy -pool pool_A vg0
```

[RISKS]
- 必須確認 replication_status 為 consistent_synchronized。
- 必須維持 IP Quorum 在線。

[SUMMARY]
透過 PBHA 雙活的跨站保護，以站點輪替升級可達成零停機硬碟翻倍擴容。

[SOURCES]
[1] REDP-5704: Policy-Based Replication with IBM Storage FlashSystem, Chapter 4.
"""
        filter_res = FilterResult(
            question="test",
            chunks=[],
            context_text="",
            source_refs=["[1] redp5704.pdf"],
        )
        ans = _parse_expert_output(raw_llm, "test", filter_res)
        self.assertIn("DRAID", ans.diagnosis)
        self.assertGreaterEqual(len(ans.steps), 2)
        self.assertTrue(any("mermaid" in s.lower() for s in ans.steps))
        self.assertTrue(any("mkmdiskgrp" in s for s in ans.steps))
        self.assertGreaterEqual(len(ans.risks), 2)
        self.assertIn("REDP-5704", ans.sources[0])


if __name__ == "__main__":
    unittest.main()

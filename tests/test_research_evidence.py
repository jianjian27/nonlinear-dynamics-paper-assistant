import unittest
from unittest.mock import Mock, call

from agent_tools.retrieval_tools import SearchPaperEvidenceTool
from framework_agent.research_evidence import ResearchEvidenceNode


class TestResearchEvidenceNode(unittest.TestCase):
    def setUp(self):
        self.tool = Mock(spec=SearchPaperEvidenceTool)

        self.node = ResearchEvidenceNode(
            self.tool,
            top_k_per_paper=3,
        )

        self.state = {
            "question": "比较动力学分析方法",
            "selected_paper_ids": ["paper-001", "paper-002"],
            "approved": True,
        }

    def make_output(self, paper_id, *, with_evidence=True):
        evidences = []

        if with_evidence:
            evidences.append({
                "rank": 1,
                "score": 0.85,
                "paper_id": paper_id,
                "paper_title": f"论文 {paper_id}",
                "chunk_id": f"chunk-{paper_id}",
                "section_path": ["动力学分析"],
                "page_start": 3,
                "page_end": 4,
                "content": "使用Lyapunov指数和分岔图分析系统。",
            })

        return {
            "database_available": True,
            "scope_paper_ids": [paper_id],
            "indexed_count": 5,
            "result_count": len(evidences),
            "evidences": evidences,
        }

    def test_searches_each_paper_separately(self):
        self.tool.invoke.side_effect = [
            self.make_output("paper-001"),
            self.make_output("paper-002"),
        ]

        update = self.node.collect(self.state)

        self.assertEqual(
            self.tool.invoke.call_args_list,
            [
                call({
                    "query": "比较动力学分析方法",
                    "paper_ids": ["paper-001"],
                    "top_k": 3,
                }),
                call({
                    "query": "比较动力学分析方法",
                    "paper_ids": ["paper-002"],
                    "top_k": 3,
                }),
            ],
        )

        results = update["evidence_results"]

        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0]["evidences"][0]["page_start"],
            3,
        )
        self.assertEqual(
            results[1]["evidences"][0]["paper_id"],
            "paper-002",
        )
        self.assertTrue(
            all(item["error"] is None for item in results)
        )

    def test_continues_after_one_paper_fails(self):
        self.tool.invoke.side_effect = [
            RuntimeError("模拟检索失败"),
            self.make_output("paper-002"),
        ]

        update = self.node.collect(self.state)
        results = update["evidence_results"]

        self.assertEqual(self.tool.invoke.call_count, 2)
        self.assertEqual(results[0]["evidences"], [])
        self.assertIn("模拟检索失败", results[0]["error"])
        self.assertIsNone(results[1]["error"])
        self.assertEqual(len(results[1]["evidences"]), 1)

    def test_preserves_empty_result_without_error(self):
        self.tool.invoke.side_effect = [
            self.make_output("paper-001", with_evidence=False),
            self.make_output("paper-002"),
        ]

        update = self.node.collect(self.state)
        first_result = update["evidence_results"][0]

        self.assertEqual(first_result["evidences"], [])
        self.assertIsNone(first_result["error"])

    def test_rejects_evidence_from_another_paper(self):
        self.tool.invoke.side_effect = [
            self.make_output("outside-paper"),
            self.make_output("paper-002"),
        ]

        update = self.node.collect(self.state)
        first_result = update["evidence_results"][0]

        self.assertEqual(first_result["evidences"], [])
        self.assertIn("研究范围外", first_result["error"])

    def test_records_unavailable_database_as_failure(self):
        missing_output = {
            "database_available": False,
            "scope_paper_ids": ["paper-001"],
            "indexed_count": 0,
            "result_count": 0,
            "evidences": [],
        }

        self.tool.invoke.return_value = missing_output

        update = self.node.collect(self.state)

        self.assertTrue(
            all(
                "数据库不可用" in item["error"]
                for item in update["evidence_results"]
            )
        )

    def test_requires_confirmation_before_search(self):
        self.state["approved"] = False

        with self.assertRaisesRegex(ValueError, "先确认"):
            self.node.collect(self.state)

        self.tool.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
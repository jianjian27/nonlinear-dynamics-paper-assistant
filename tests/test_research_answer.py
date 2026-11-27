import json
import unittest
from unittest.mock import Mock

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from framework_agent.research_answer import ResearchAnswerNode


class TestResearchAnswerNode(unittest.TestCase):
    def setUp(self):
        self.model = Mock(spec=BaseChatModel)
        self.model.invoke.return_value = AIMessage(
            content="两篇论文均使用分岔图分析，具体差异需要结合证据说明。"
        )

        self.node = ResearchAnswerNode(self.model)

        self.state = {
            "question": "比较动力学分析方法",
            "approved": True,
            "selected_paper_ids": ["paper-001", "paper-002"],
            "candidate_papers": [
                {"paper_id": "paper-001", "title": "论文A"},
                {"paper_id": "paper-002", "title": "论文B"},
            ],
            "evidence_results": [
                self.make_result("paper-001"),
                self.make_result("paper-002"),
            ],
        }

    def make_result(self, paper_id):
        return {
            "paper_id": paper_id,
            "error": None,
            "evidences": [
                {
                    "paper_id": paper_id,
                    "paper_title": f"标题-{paper_id}",
                    "chunk_id": f"chunk-{paper_id}",
                    "section_path": ["动力学分析"],
                    "page_start": 3,
                    "page_end": 4,
                    "content": "使用分岔图分析系统。",
                },
            ],
        }

    def test_generates_answer_with_source_information(self):
        update = self.node.generate(self.state)

        self.assertIn("分岔图", update["answer"])
        self.model.invoke.assert_called_once()

        messages = self.model.invoke.call_args.args[0]
        payload = json.loads(messages[1].content)

        self.assertEqual(payload["question"], self.state["question"])
        self.assertEqual(len(payload["papers"]), 2)

        evidence = payload["papers"][0]["evidences"][0]
        self.assertEqual(evidence["chunk_id"], "chunk-paper-001")
        self.assertEqual(evidence["page_start"], 3)

    def test_skips_model_when_only_one_paper_has_evidence(self):
        self.state["evidence_results"][1]["evidences"] = []

        update = self.node.generate(self.state)

        self.assertIn("不足两篇", update["answer"])
        self.model.invoke.assert_not_called()

    def test_reports_missing_results(self):
        self.state["evidence_results"] = []

        update = self.node.generate(self.state)

        self.assertIn("不足两篇", update["answer"])
        self.model.invoke.assert_not_called()

    def test_preserves_failure_status_in_model_input(self):
        self.state["selected_paper_ids"].append("paper-003")
        self.state["evidence_results"].append({
            "paper_id": "paper-003",
            "evidences": [],
            "error": "模拟检索失败",
        })

        self.node.generate(self.state)

        messages = self.model.invoke.call_args.args[0]
        payload = json.loads(messages[1].content)

        self.assertEqual(
            payload["papers"][2]["status"],
            "检索失败",
        )

    def test_ignores_unselected_paper_results(self):
        self.state["evidence_results"].append(
            self.make_result("outside-paper")
        )

        self.node.generate(self.state)

        messages = self.model.invoke.call_args.args[0]
        payload = json.loads(messages[1].content)

        self.assertEqual(
            [paper["paper_id"] for paper in payload["papers"]],
            ["paper-001", "paper-002"],
        )

    def test_rejects_empty_model_answer(self):
        self.model.invoke.return_value = AIMessage(content="  ")

        with self.assertRaisesRegex(RuntimeError, "空回答"):
            self.node.generate(self.state)

    def test_rejects_truncated_model_answer(self):
        self.model.invoke.return_value = AIMessage(
            content="未完成的比较",
            response_metadata={"finish_reason": "length"},
        )

        with self.assertRaisesRegex(RuntimeError, "截断"):
            self.node.generate(self.state)


if __name__ == "__main__":
    unittest.main()
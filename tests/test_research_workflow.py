import unittest
from unittest.mock import Mock
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_tools.paper_tools import ListPapersTool
from agent_tools.retrieval_tools import SearchPaperEvidenceTool
from framework_agent.research_evidence import ResearchEvidenceNode
from framework_agent.research_nodes import ResearchScopeNodes
from framework_agent.research_workflow import create_research_workflow


class TestResearchWorkflow(unittest.TestCase):
    def setUp(self):
        self.list_tool = Mock(spec=ListPapersTool)
        self.search_tool = Mock(spec=SearchPaperEvidenceTool)

        self.list_tool.invoke.return_value = {
            "database_available": True,
            "total": 2,
            "papers": [
                {
                    "paper_id": "paper-001",
                    "title": "忆阻器研究",
                    "chunk_count": 5,
                },
                {
                    "paper_id": "paper-002",
                    "title": "Jerk系统研究",
                    "chunk_count": 8,
                },
            ],
        }

        # 模拟检索工具正常完成但没有返回证据。
        self.search_tool.invoke.return_value = {
            "database_available": True,
            "scope_paper_ids": None,
            "indexed_count": 5,
            "result_count": 0,
            "evidences": [],
        }

        scope_nodes = ResearchScopeNodes(self.list_tool)

        evidence_node = ResearchEvidenceNode(
            self.search_tool,
            top_k_per_paper=3,
        )

        self.workflow = create_research_workflow(
            scope_nodes,
            evidence_node,
            checkpointer=InMemorySaver(),
        )

        self.config = {
            "configurable": {
                "thread_id": str(uuid4()),
            }
        }

    def start_workflow(self):
        return self.workflow.invoke(
            {"question": "比较两篇论文的动力学分析方法"},
            config=self.config,
        )

    def test_waits_for_confirmation_before_search(self):
        result = self.start_workflow()

        self.assertIn("__interrupt__", result)
        self.assertEqual(result["evidence_results"], [])
        self.search_tool.invoke.assert_not_called()

        snapshot = self.workflow.get_state(self.config)

        self.assertEqual(
            snapshot.next,
            ("confirm_scope",),
        )

    def test_collects_evidence_after_approval(self):
        self.start_workflow()

        result = self.workflow.invoke(
            Command(resume={
                "approved": True,
                "selected_paper_ids": [
                    "paper-001",
                    "paper-002"
                ],
            }),
            config=self.config,
        )

        self.assertTrue(result["approved"])
        self.assertEqual(
            [
                item["paper_id"]
                for item in result["evidence_results"]
            ],
            ["paper-001", "paper-002"],
        )
        self.assertEqual(
            self.search_tool.invoke.call_count,
            2,
        )

        # 恢复工作流不应重新执行候选论文查询。
        self.list_tool.invoke.assert_called_once_with({})

        # 没有执行待执行节点，表示这一阶段流程结束。
        snapshot = self.workflow.get_state(self.config)
        self.assertEqual(snapshot.next, ())

    def test_cancel_finishes_without_search(self):
        self.start_workflow()

        result = self.workflow.invoke(
            Command(resume={
                "approved": False,
            }),
            config=self.config,
        )

        self.assertFalse(result["approved"])
        self.assertEqual(result["evidence_results"], [])
        self.search_tool.invoke.assert_not_called()

        snapshot = self.workflow.get_state(self.config)
        self.assertEqual(snapshot.next, ())

    def test_insufficient_papers_finish_without_interrupt(self):
        self.list_tool.invoke.return_value["papers"] = [
            {
                "paper_id": "paper-001",
                "title": "唯一的论文",
                "chunk_count": 5,
            },
        ]
        self.list_tool.invoke.return_value["total"] = 1

        result = self.start_workflow()

        self.assertNotIn("__interrupt__", result)
        self.assertFalse(result["approved"])
        self.assertIn("不足两篇", result["message"])
        self.search_tool.invoke.assert_not_called()

        snapshot = self.workflow.get_state(self.config)
        self.assertEqual(snapshot.next, ())


if __name__ == "__main__":
    unittest.main()
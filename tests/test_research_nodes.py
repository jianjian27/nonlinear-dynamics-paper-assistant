import unittest
from unittest.mock import Mock
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent_tools.paper_tools import ListPapersTool
from framework_agent.research_nodes import ResearchScopeNodes
from framework_agent.research_state import ResearchState


class TestResearchScopeNodes(unittest.TestCase):
    def setUp(self):
        self.tool = Mock(spec=ListPapersTool)

        self.tool.invoke.return_value = {
            "database_available": True,
            "total": 3,
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
                {
                    "paper_id": "paper-003",
                    "title": "没有正文的论文",
                    "chunk_count": 0,
                },
            ],
        }

        self.nodes = ResearchScopeNodes(self.tool)

        builder = StateGraph(ResearchState)
        builder.add_node(
            "load_candidates",
            self.nodes.load_candidates,
        )
        builder.add_node(
            "confirm_scope",
            self.nodes.confirm_scope,
        )

        builder.add_edge(START, "load_candidates")
        builder.add_edge("load_candidates", "confirm_scope")
        builder.add_edge("confirm_scope", END)

        self.workflow = builder.compile(
            checkpointer=InMemorySaver(),
        )

        self.config = {
            "configurable": {
                "thread_id": str(uuid4()),
            },
        }

    def start_workflow(self):
        return self.workflow.invoke(
            {"question": "比较两篇论文的动力学分析方法"},
            config=self.config,
        )

    def resume_workflow(self, decision):
        return self.workflow.invoke(
            Command(resume=decision),
            config=self.config,
        )

    def test_pauses_with_available_candidates(self):
        result = self.start_workflow()

        pending = result["__interrupt__"][0].value

        self.assertEqual(
            [
                paper["paper_id"]
                for paper in pending["candidate_papers"]
            ],
            ["paper-001", "paper-002"],
        )
        self.assertFalse(result["approved"])
        self.tool.invoke.assert_called_once_with({})

    def test_accepts_selected_papers(self):
        self.start_workflow()

        result = self.resume_workflow({
            "approved": True,
            "selected_paper_ids": [
                "paper-001",
                "paper-002",
                "paper-001",
            ],
        })

        self.assertTrue(result["approved"])
        self.assertEqual(
            result["selected_paper_ids"],
            ["paper-001", "paper-002"],
        )

        # 恢复确认节点，不应重新读取论文列表。
        self.tool.invoke.assert_called_once_with({})

    def test_cancels_research(self):
        self.start_workflow()

        result = self.resume_workflow({
            "approved": False,
        })

        self.assertFalse(result["approved"])
        self.assertEqual(result["selected_paper_ids"], [])
        self.assertEqual(result["message"], "已取消本次研究。")

    def test_rejects_unknown_paper(self):
        self.start_workflow()

        with self.assertRaisesRegex(ValueError, "候选列表"):
            self.resume_workflow({
                "approved": True,
                "selected_paper_ids": [
                    "paper-001",
                    "unknown-paper",
                ],
            })

    def test_requires_two_different_papers(self):
        self.start_workflow()

        with self.assertRaisesRegex(ValueError, "至少选择两篇"):
            self.resume_workflow({
                "approved": True,
                "selected_paper_ids": [
                    "paper-001",
                    "paper-001",
                ],
            })

    def test_skips_confirmation_when_database_is_missing(self):
        self.tool.invoke.return_value = {
            "database_available": False,
            "total": 0,
            "papers": [],
        }

        result = self.start_workflow()

        self.assertFalse(result["approved"])
        self.assertNotIn("__interrupt__", result)
        self.assertIn("数据库不存在", result["message"])


if __name__ == "__main__":
    unittest.main()
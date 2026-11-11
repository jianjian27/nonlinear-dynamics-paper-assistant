import unittest
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_runtime import AgentProtocolError
from evaluation.trace_evaluator import extract_tool_names
from framework_agent import LangChainPaperAgentRunner


class TestLangChainPaperAgentRunner(unittest.TestCase):
    def test_converts_multiple_tool_calls_for_existing_evaluator(self):
        agent = Mock()
        agent.invoke.return_value = {
            "messages": [
                HumanMessage(content="比较两篇论文的结构"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "get_paper_overview",
                            "args": {"paper_id": "paper-1"},
                        },
                        {
                            "id": "call-2",
                            "name": "get_paper_overview",
                            "args": {"paper_id": "paper-2"},
                        },
                    ],
                ),
                ToolMessage(content="论文一的章节", tool_call_id="call-1"),
                ToolMessage(content="论文二的章节", tool_call_id="call-2"),
                AIMessage(content="两篇论文的章节结构不同。"),
            ],
        }

        result = LangChainPaperAgentRunner(agent).run(" 比较两篇论文的结构 ")

        agent.invoke.assert_called_once_with({
            "messages": [{"role": "user", "content": "比较两篇论文的结构"}],
        })
        self.assertEqual(result.answer, "两篇论文的章节结构不同。")
        self.assertEqual(result.model_rounds, 2)
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(
            extract_tool_names(result.messages),
            ("get_paper_overview", "get_paper_overview"),
        )

    def test_rejects_run_without_final_model_message(self):
        agent = Mock()
        agent.invoke.return_value = {
            "messages": [HumanMessage(content="你好")],
        }

        with self.assertRaises(AgentProtocolError):
            LangChainPaperAgentRunner(agent).run("你好")

    def test_rejects_blank_question_before_invoking(self):
        agent = Mock()

        with self.assertRaises(ValueError):
            LangChainPaperAgentRunner(agent).run("  ")

        agent.invoke.assert_not_called()
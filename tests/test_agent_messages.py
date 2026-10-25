import unittest

from pydantic import ValidationError

from agent_runtime import (
    AgentAssistantMessage,
    AgentToolCall,
)


class TestAgentToolCall(
    unittest.TestCase
):
    """测试模型请求的工具调用。"""

    def test_normalizes_id_and_name(
        self,
    ) -> None:
        """调用ID和工具名应去除两侧空格。"""

        tool_call = AgentToolCall(
            id="  call-001  ",
            name="  list_papers  ",
            arguments_json="{}",
        )

        self.assertEqual(
            tool_call.id,
            "call-001",
        )

        self.assertEqual(
            tool_call.name,
            "list_papers",
        )

    def test_converts_to_openai_structure(
        self,
    ) -> None:
        """工具调用应转换为兼容字典。"""

        tool_call = AgentToolCall(
            id="call-002",
            name="get_paper_overview",
            arguments_json=(
                '{"paper_id": "paper-001"}'
            ),
        )

        self.assertEqual(
            tool_call.to_openai_dict(),
            {
                "id": "call-002",
                "type": "function",
                "function": {
                    "name": (
                        "get_paper_overview"
                    ),
                    "arguments": (
                        '{"paper_id": '
                        '"paper-001"}'
                    ),
                },
            },
        )

    def test_rejects_blank_id_or_name(
        self,
    ) -> None:
        """调用ID和工具名称不能为空。"""

        with self.assertRaises(
            ValidationError
        ):
            AgentToolCall(
                id="   ",
                name="list_papers",
                arguments_json="{}",
            )

        with self.assertRaises(
            ValidationError
        ):
            AgentToolCall(
                id="call-003",
                name="   ",
                arguments_json="{}",
            )

    def test_preserves_invalid_json(
        self,
    ) -> None:
        """参数JSON应交给调度器验证。"""

        tool_call = AgentToolCall(
            id="call-004",
            name="list_papers",
            arguments_json='{"broken":',
        )

        self.assertEqual(
            tool_call.arguments_json,
            '{"broken":',
        )


class TestAgentAssistantMessage(
    unittest.TestCase
):
    """测试Agent内部的Assistant消息。"""

    def test_represents_final_answer(
        self,
    ) -> None:
        """纯文本消息应被识别为最终回答。"""

        message = AgentAssistantMessage(
            content=(
                "  知识库中共有两篇论文。  "
            ),
            finish_reason="stop",
        )

        self.assertEqual(
            message.content,
            "知识库中共有两篇论文。",
        )

        self.assertFalse(
            message.requests_tools
        )

        self.assertTrue(
            message.is_final_answer
        )

    def test_represents_tool_request(
        self,
    ) -> None:
        """包含tool_calls时不应视为最终回答。"""

        tool_call = AgentToolCall(
            id="call-005",
            name="list_papers",
            arguments_json="{}",
        )

        message = AgentAssistantMessage(
            content=None,
            tool_calls=(
                tool_call,
            ),
            finish_reason="tool_calls",
        )

        self.assertTrue(
            message.requests_tools
        )

        self.assertFalse(
            message.is_final_answer
        )

    def test_converts_to_assistant_message(
        self,
    ) -> None:
        """应生成可追加到对话历史的消息。"""

        message = AgentAssistantMessage(
            content=None,
            tool_calls=(
                AgentToolCall(
                    id="call-006",
                    name=(
                        "search_paper_evidence"
                    ),
                    arguments_json=(
                        '{"query": "稳定性"}'
                    ),
                ),
            ),
            finish_reason="tool_calls",
        )

        converted = (
            message.to_openai_dict()
        )

        self.assertEqual(
            converted["role"],
            "assistant",
        )

        self.assertIsNone(
            converted["content"]
        )

        self.assertEqual(
            converted["tool_calls"][0][
                "id"
            ],
            "call-006",
        )

        self.assertEqual(
            converted["tool_calls"][0][
                "function"
            ]["name"],
            "search_paper_evidence",
        )

    def test_allows_content_with_tool_calls(
        self,
    ) -> None:
        """模型可以同时返回说明文字和工具调用。"""

        message = AgentAssistantMessage(
            content="我先查询论文库。",
            tool_calls=(
                AgentToolCall(
                    id="call-007",
                    name="list_papers",
                    arguments_json="{}",
                ),
            ),
        )

        self.assertTrue(
            message.requests_tools
        )

        self.assertFalse(
            message.is_final_answer
        )

        self.assertEqual(
            message.content,
            "我先查询论文库。",
        )

    def test_rejects_empty_message(
        self,
    ) -> None:
        """没有文本和工具调用的消息无效。"""

        with self.assertRaises(
            ValidationError
        ):
            AgentAssistantMessage(
                content="   ",
                tool_calls=(),
            )

    def test_rejects_unknown_fields(
        self,
    ) -> None:
        """未声明字段不应被静默接受。"""

        with self.assertRaises(
            ValidationError
        ):
            AgentAssistantMessage(
                content="回答",
                unknown="value",
            )


if __name__ == "__main__":
    unittest.main()
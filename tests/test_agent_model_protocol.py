import unittest
from collections.abc import (
    Sequence,
)
from typing import Any

from agent_runtime import (
    AgentAssistantMessage,
    AgentChatModel,
    AgentMessage,
    ToolDefinition,
)


class FakeAgentModel:
    """测试使用的模型替身。"""

    def __init__(self) -> None:
        self.received_messages: list[
            dict[str, Any]
        ] = []

        self.received_tools: list[
            dict[str, Any]
        ] = []

    def complete(
        self,
        *,
        messages: Sequence[
            AgentMessage
        ],
        tools: Sequence[
            ToolDefinition
        ],
    ) -> AgentAssistantMessage:
        self.received_messages = [
            dict(message)
            for message in messages
        ]

        self.received_tools = [
            dict(tool)
            for tool in tools
        ]

        return AgentAssistantMessage(
            content="测试回答",
            finish_reason="stop",
        )


class IncompleteModel:
    """没有实现complete方法的对象。"""


class TestAgentChatModelProtocol(
    unittest.TestCase
):
    """测试Agent模型协议。"""

    def test_accepts_structural_implementation(
        self,
    ) -> None:
        """不继承协议也可通过结构化检查。"""

        model = FakeAgentModel()

        self.assertIsInstance(
            model,
            AgentChatModel,
        )

    def test_rejects_missing_complete_method(
        self,
    ) -> None:
        """缺少complete方法的对象不符合协议。"""

        model = IncompleteModel()

        self.assertNotIsInstance(
            model,
            AgentChatModel,
        )

    def test_fake_model_receives_context(
        self,
    ) -> None:
        """模型应收到消息历史和工具定义。"""

        model = FakeAgentModel()

        messages: list[
            AgentMessage
        ] = [
            {
                "role": "user",
                "content": (
                    "知识库中有哪些论文？"
                ),
            }
        ]

        tools: list[
            ToolDefinition
        ] = [
            {
                "type": "function",
                "function": {
                    "name": "list_papers",
                },
            }
        ]

        response = model.complete(
            messages=messages,
            tools=tools,
        )

        self.assertEqual(
            response.content,
            "测试回答",
        )

        self.assertEqual(
            model.received_messages,
            messages,
        )

        self.assertEqual(
            model.received_tools,
            tools,
        )


if __name__ == "__main__":
    unittest.main()
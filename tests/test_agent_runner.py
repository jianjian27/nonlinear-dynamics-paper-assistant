import json
import unittest
from collections.abc import (
    Sequence,
)
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
)

from agent_runtime import (
    AgentAssistantMessage,
    AgentMessage,
    AgentProtocolError,
    AgentRunLimitError,
    AgentRunner,
    AgentToolCall,
    ToolDefinition,
    ToolRegistry,
)
from agent_tools.base import AgentTool


class PaperLookupInput(BaseModel):
    """测试论文查询工具输入。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    paper_id: str


class PaperLookupOutput(BaseModel):
    """测试论文查询工具输出。"""

    found: bool
    paper_id: str
    title: str


class PaperLookupTool(AgentTool):
    """测试使用的论文查询工具。"""

    name = "lookup_paper"
    description = "根据论文ID查询论文。"
    input_model = PaperLookupInput

    def run(
        self,
        arguments: BaseModel,
    ) -> PaperLookupOutput:
        if not isinstance(
            arguments,
            PaperLookupInput,
        ):
            raise TypeError(
                "lookup_paper收到错误参数。"
            )

        return PaperLookupOutput(
            found=True,
            paper_id=arguments.paper_id,
            title=(
                f"测试论文{arguments.paper_id}"
            ),
        )


class ScriptedAgentModel:
    """按照预设顺序返回消息的模型替身。"""

    def __init__(
        self,
        responses: Sequence[
            AgentAssistantMessage
        ],
    ) -> None:
        self.responses = list(
            responses
        )

        self.call_count = 0

        self.received_messages: list[
            list[dict[str, Any]]
        ] = []

        self.received_tools: list[
            list[dict[str, Any]]
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
        self.received_messages.append(
            [
                dict(message)
                for message in messages
            ]
        )

        self.received_tools.append(
            [
                dict(tool)
                for tool in tools
            ]
        )

        if (
            self.call_count
            >= len(self.responses)
        ):
            raise AssertionError(
                "脚本化模型没有更多预设响应。"
            )

        response = self.responses[
            self.call_count
        ]

        self.call_count += 1

        return response


def make_tool_request(
    *,
    call_id: str,
    paper_id: str | None = None,
) -> AgentAssistantMessage:
    """创建一次lookup_paper调用请求。"""

    if paper_id is None:
        arguments_json = "{}"
    else:
        arguments_json = json.dumps(
            {
                "paper_id": paper_id,
            },
            ensure_ascii=False,
        )

    return AgentAssistantMessage(
        content=None,
        tool_calls=(
            AgentToolCall(
                id=call_id,
                name="lookup_paper",
                arguments_json=(
                    arguments_json
                ),
            ),
        ),
        finish_reason="tool_calls",
    )


class TestAgentRunner(
    unittest.TestCase
):
    """测试Agent多轮运行循环。"""

    def create_runner(
        self,
        responses: Sequence[
            AgentAssistantMessage
        ],
        *,
        max_model_rounds: int = 6,
        max_tool_calls: int = 12,
    ) -> tuple[
        AgentRunner,
        ScriptedAgentModel,
    ]:
        model = ScriptedAgentModel(
            responses
        )

        registry = ToolRegistry(
            [
                PaperLookupTool(),
            ]
        )

        runner = AgentRunner(
            model=model,
            registry=registry,
            system_prompt="测试系统提示词",
            max_model_rounds=(
                max_model_rounds
            ),
            max_tool_calls=(
                max_tool_calls
            ),
        )

        return runner, model

    def test_returns_direct_answer(
        self,
    ) -> None:
        """模型不调用工具时应直接结束。"""

        runner, model = self.create_runner(
            [
                AgentAssistantMessage(
                    content="直接回答。",
                    finish_reason="stop",
                )
            ]
        )

        result = runner.run(
            "  用户问题  "
        )

        self.assertEqual(
            result.answer,
            "直接回答。",
        )

        self.assertEqual(
            result.model_rounds,
            1,
        )

        self.assertEqual(
            result.tool_call_count,
            0,
        )

        self.assertEqual(
            [
                message["role"]
                for message
                in result.messages
            ],
            [
                "system",
                "user",
                "assistant",
            ],
        )

        self.assertEqual(
            model.received_messages[0][1],
            {
                "role": "user",
                "content": "用户问题",
            },
        )

    def test_executes_tool_then_returns_answer(
        self,
    ) -> None:
        """工具结果应进入下一轮模型上下文。"""

        runner, model = self.create_runner(
            [
                make_tool_request(
                    call_id="call-001",
                    paper_id="paper-001",
                ),
                AgentAssistantMessage(
                    content=(
                        "查询到测试论文。"
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        result = runner.run(
            "查询paper-001"
        )

        self.assertEqual(
            result.answer,
            "查询到测试论文。",
        )

        self.assertEqual(
            result.model_rounds,
            2,
        )

        self.assertEqual(
            result.tool_call_count,
            1,
        )

        second_round_messages = (
            model.received_messages[1]
        )

        self.assertEqual(
            [
                message["role"]
                for message
                in second_round_messages
            ],
            [
                "system",
                "user",
                "assistant",
                "tool",
            ],
        )

        tool_message = (
            second_round_messages[-1]
        )

        self.assertEqual(
            tool_message["tool_call_id"],
            "call-001",
        )

        payload = json.loads(
            tool_message["content"]
        )

        self.assertTrue(
            payload["success"]
        )

        self.assertEqual(
            payload["result"][
                "paper_id"
            ],
            "paper-001",
        )

    def test_executes_parallel_tool_calls(
        self,
    ) -> None:
        """一轮中的多个工具调用都应执行。"""

        parallel_request = (
            AgentAssistantMessage(
                content=None,
                tool_calls=(
                    AgentToolCall(
                        id="call-002",
                        name="lookup_paper",
                        arguments_json=(
                            '{"paper_id": '
                            '"paper-001"}'
                        ),
                    ),
                    AgentToolCall(
                        id="call-003",
                        name="lookup_paper",
                        arguments_json=(
                            '{"paper_id": '
                            '"paper-002"}'
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            )
        )

        runner, model = self.create_runner(
            [
                parallel_request,
                AgentAssistantMessage(
                    content=(
                        "两篇论文均已查询。"
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        result = runner.run(
            "比较两篇论文"
        )

        self.assertEqual(
            result.tool_call_count,
            2,
        )

        second_round_messages = (
            model.received_messages[1]
        )

        tool_messages = [
            message
            for message
            in second_round_messages
            if message["role"] == "tool"
        ]

        self.assertEqual(
            len(tool_messages),
            2,
        )

        self.assertEqual(
            [
                message["tool_call_id"]
                for message
                in tool_messages
            ],
            [
                "call-002",
                "call-003",
            ],
        )

    def test_returns_tool_failure_to_model(
        self,
    ) -> None:
        """参数验证失败后模型仍可继续回答。"""

        runner, model = self.create_runner(
            [
                make_tool_request(
                    call_id="call-004",
                    paper_id=None,
                ),
                AgentAssistantMessage(
                    content=(
                        "工具参数缺失，"
                        "无法完成查询。"
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        result = runner.run(
            "查询论文"
        )

        self.assertEqual(
            result.model_rounds,
            2,
        )

        tool_message = (
            model
            .received_messages[1][-1]
        )

        payload = json.loads(
            tool_message["content"]
        )

        self.assertFalse(
            payload["success"]
        )

        self.assertEqual(
            payload["error"]["code"],
            "validation_error",
        )

    def test_stops_at_model_round_limit(
        self,
    ) -> None:
        """模型持续调用工具时应触发轮数限制。"""

        runner, _ = self.create_runner(
            [
                make_tool_request(
                    call_id="call-005",
                    paper_id="paper-001",
                ),
                make_tool_request(
                    call_id="call-006",
                    paper_id="paper-002",
                ),
            ],
            max_model_rounds=2,
        )

        with self.assertRaisesRegex(
            AgentRunLimitError,
            "最大模型轮数",
        ):
            runner.run(
                "持续查询"
            )

    def test_stops_at_tool_call_limit(
        self,
    ) -> None:
        """并行调用数超过限制时不应执行。"""

        request = AgentAssistantMessage(
            content=None,
            tool_calls=(
                AgentToolCall(
                    id="call-007",
                    name="lookup_paper",
                    arguments_json=(
                        '{"paper_id": '
                        '"paper-001"}'
                    ),
                ),
                AgentToolCall(
                    id="call-008",
                    name="lookup_paper",
                    arguments_json=(
                        '{"paper_id": '
                        '"paper-002"}'
                    ),
                ),
            ),
        )

        runner, _ = self.create_runner(
            [
                request,
            ],
            max_tool_calls=1,
        )

        with self.assertRaisesRegex(
            AgentRunLimitError,
            "工具调用次数",
        ):
            runner.run(
                "查询两篇论文"
            )

    def test_rejects_duplicate_call_id(
        self,
    ) -> None:
        """不同轮次不能重复使用调用ID。"""

        runner, _ = self.create_runner(
            [
                make_tool_request(
                    call_id="call-009",
                    paper_id="paper-001",
                ),
                make_tool_request(
                    call_id="call-009",
                    paper_id="paper-002",
                ),
            ]
        )

        with self.assertRaisesRegex(
            AgentProtocolError,
            "重复使用",
        ):
            runner.run(
                "查询论文"
            )

    def test_rejects_truncated_final_answer(
        self,
    ) -> None:
        """被截断的回答不能视为完成。"""

        runner, _ = self.create_runner(
            [
                AgentAssistantMessage(
                    content="未完成回答",
                    finish_reason="length",
                )
            ]
        )

        with self.assertRaisesRegex(
            AgentProtocolError,
            "长度上限",
        ):
            runner.run(
                "生成详细分析"
            )

    def test_includes_previous_history(
        self,
    ) -> None:
        """已有对话历史应放在新问题之前。"""

        runner, model = self.create_runner(
            [
                AgentAssistantMessage(
                    content="继续回答。",
                    finish_reason="stop",
                )
            ]
        )

        history = [
            {
                "role": "user",
                "content": "上一轮问题",
            },
            {
                "role": "assistant",
                "content": "上一轮回答",
            },
        ]

        runner.run(
            "当前问题",
            history=history,
        )

        first_request = (
            model.received_messages[0]
        )

        self.assertEqual(
            [
                message["content"]
                for message
                in first_request[1:]
            ],
            [
                "上一轮问题",
                "上一轮回答",
                "当前问题",
            ],
        )


if __name__ == "__main__":
    unittest.main()
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from models.aliyun_agent_model import (
    AliyunAgentChatModel,
)
from models.deepseek_api import (
    AliyunChatClient,
    ModelServiceError,
)


def make_tool_definition(
    name: str = "list_papers",
) -> dict[str, object]:
    """创建测试工具定义。"""

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "测试工具",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


def make_raw_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
    tool_type: str = "function",
) -> SimpleNamespace:
    """模拟OpenAI SDK工具调用对象。"""

    return SimpleNamespace(
        id=call_id,
        type=tool_type,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def make_response(
    *,
    content: str | None = "模型回答",
    tool_calls: list[
        SimpleNamespace
    ] | None = None,
    finish_reason: str = "stop",
    has_choices: bool = True,
) -> SimpleNamespace:
    """模拟OpenAI SDK响应。"""

    if not has_choices:
        return SimpleNamespace(
            choices=[]
        )

    message = SimpleNamespace(
        content=content,
        tool_calls=(
            tool_calls
            if tool_calls is not None
            else []
        ),
    )

    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )

    return SimpleNamespace(
        choices=[choice]
    )


class TestAliyunAgentChatModel(
    unittest.TestCase
):
    """测试阿里云Agent模型适配器。"""

    def setUp(self) -> None:
        self.chat_client = Mock(
            spec=AliyunChatClient
        )

        self.model = AliyunAgentChatModel(
            self.chat_client
        )

        self.messages = [
            {
                "role": "system",
                "content": "你是科研助手。",
            },
            {
                "role": "user",
                "content": (
                    "知识库中有哪些论文？"
                ),
            },
        ]

        self.tools = [
            make_tool_definition()
        ]

    def test_returns_text_answer(
        self,
    ) -> None:
        """普通模型回答应转换为最终消息。"""

        (
            self.chat_client
            .create_completion
            .return_value
        ) = make_response(
            content="知识库中有两篇论文。",
            finish_reason="stop",
        )

        result = self.model.complete(
            messages=self.messages,
            tools=self.tools,
        )

        self.assertEqual(
            result.content,
            "知识库中有两篇论文。",
        )

        self.assertTrue(
            result.is_final_answer
        )

        self.assertFalse(
            result.requests_tools
        )

        self.assertEqual(
            result.finish_reason,
            "stop",
        )

    def test_returns_single_tool_call(
        self,
    ) -> None:
        """单个SDK工具调用应转换为内部模型。"""

        raw_tool_call = (
            make_raw_tool_call(
                call_id="call-001",
                name="list_papers",
                arguments="{}",
            )
        )

        (
            self.chat_client
            .create_completion
            .return_value
        ) = make_response(
            content=None,
            tool_calls=[
                raw_tool_call,
            ],
            finish_reason="tool_calls",
        )

        result = self.model.complete(
            messages=self.messages,
            tools=self.tools,
        )

        self.assertTrue(
            result.requests_tools
        )

        self.assertFalse(
            result.is_final_answer
        )

        self.assertEqual(
            len(result.tool_calls),
            1,
        )

        tool_call = (
            result.tool_calls[0]
        )

        self.assertEqual(
            tool_call.id,
            "call-001",
        )

        self.assertEqual(
            tool_call.name,
            "list_papers",
        )

        self.assertEqual(
            tool_call.arguments_json,
            "{}",
        )

    def test_returns_parallel_tool_calls(
        self,
    ) -> None:
        """一轮返回的多个工具调用都应保留。"""

        raw_tool_calls = [
            make_raw_tool_call(
                call_id="call-002",
                name="list_papers",
                arguments="{}",
            ),
            make_raw_tool_call(
                call_id="call-003",
                name="get_paper_overview",
                arguments=(
                    '{"paper_id": "paper-001"}'
                ),
            ),
        ]

        (
            self.chat_client
            .create_completion
            .return_value
        ) = make_response(
            content=None,
            tool_calls=raw_tool_calls,
            finish_reason="tool_calls",
        )

        result = self.model.complete(
            messages=self.messages,
            tools=self.tools,
        )

        self.assertEqual(
            len(result.tool_calls),
            2,
        )

        self.assertEqual(
            [
                tool_call.name
                for tool_call
                in result.tool_calls
            ],
            [
                "list_papers",
                "get_paper_overview",
            ],
        )

    def test_sends_tools_with_auto_choice(
        self,
    ) -> None:
        """有工具时应使用自动工具选择。"""

        (
            self.chat_client
            .create_completion
            .return_value
        ) = make_response()

        self.model.complete(
            messages=self.messages,
            tools=self.tools,
        )

        (
            self.chat_client
            .create_completion
            .assert_called_once_with(
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
            )
        )

    def test_omits_tool_choice_without_tools(
        self,
    ) -> None:
        """没有工具时不应设置auto。"""

        (
            self.chat_client
            .create_completion
            .return_value
        ) = make_response()

        self.model.complete(
            messages=self.messages,
            tools=[],
        )

        (
            self.chat_client
            .create_completion
            .assert_called_once_with(
                messages=self.messages,
                tools=[],
                tool_choice=None,
            )
        )

    def test_accepts_tool_result_history(
        self,
    ) -> None:
        """后续请求应接受assistant和tool消息。"""

        messages = [
            *self.messages,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-004",
                        "type": "function",
                        "function": {
                            "name": (
                                "list_papers"
                            ),
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-004",
                "content": (
                    '{"success": true}'
                ),
            },
        ]

        (
            self.chat_client
            .create_completion
            .return_value
        ) = make_response(
            content="查询完成。"
        )

        result = self.model.complete(
            messages=messages,
            tools=self.tools,
        )

        self.assertEqual(
            result.content,
            "查询完成。",
        )

    def test_rejects_empty_choices(
        self,
    ) -> None:
        """空候选响应应转换为服务异常。"""

        (
            self.chat_client
            .create_completion
            .return_value
        ) = make_response(
            has_choices=False
        )

        with self.assertRaisesRegex(
            ModelServiceError,
            "没有返回任何候选结果",
        ):
            self.model.complete(
                messages=self.messages,
                tools=self.tools,
            )

    def test_rejects_empty_model_message(
        self,
    ) -> None:
        """模型既无文本也无调用时应失败。"""

        (
            self.chat_client
            .create_completion
            .return_value
        ) = make_response(
            content=None,
            tool_calls=[],
        )

        with self.assertRaisesRegex(
            ModelServiceError,
            "既没有返回文本",
        ):
            self.model.complete(
                messages=self.messages,
                tools=self.tools,
            )

    def test_rejects_unsupported_call_type(
        self,
    ) -> None:
        """目前只接受function工具调用。"""

        raw_tool_call = (
            make_raw_tool_call(
                call_id="call-005",
                name="list_papers",
                arguments="{}",
                tool_type="custom",
            )
        )

        (
            self.chat_client
            .create_completion
            .return_value
        ) = make_response(
            content=None,
            tool_calls=[
                raw_tool_call,
            ],
        )

        with self.assertRaisesRegex(
            ModelServiceError,
            "不支持的工具调用类型",
        ):
            self.model.complete(
                messages=self.messages,
                tools=self.tools,
            )

    def test_rejects_invalid_tool_call(
        self,
    ) -> None:
        """缺少调用ID的响应应失败。"""

        raw_tool_call = (
            make_raw_tool_call(
                call_id="   ",
                name="list_papers",
                arguments="{}",
            )
        )

        (
            self.chat_client
            .create_completion
            .return_value
        ) = make_response(
            content=None,
            tool_calls=[
                raw_tool_call,
            ],
        )

        with self.assertRaisesRegex(
            ModelServiceError,
            "工具调用结构不完整",
        ):
            self.model.complete(
                messages=self.messages,
                tools=self.tools,
            )


class TestAgentRequestForwarding(
    unittest.TestCase
):
    """测试工具定义是否真正传给SDK。"""

    def test_forwards_tools_to_openai_client(
        self,
    ) -> None:
        sdk_client = Mock()

        (
            sdk_client
            .chat
            .completions
            .create
            .return_value
        ) = make_response(
            content="最终回答"
        )

        chat_client = AliyunChatClient(
            model_name="test-model",
            max_tokens=1000,
            enable_thinking=False,
            client=sdk_client,
        )

        model = AliyunAgentChatModel(
            chat_client
        )

        messages = [
            {
                "role": "user",
                "content": "列出论文。",
            }
        ]

        tools = [
            make_tool_definition()
        ]

        result = model.complete(
            messages=messages,
            tools=tools,
        )

        self.assertEqual(
            result.content,
            "最终回答",
        )

        create = (
            sdk_client
            .chat
            .completions
            .create
        )

        arguments = (
            create.call_args.kwargs
        )

        self.assertEqual(
            arguments["model"],
            "test-model",
        )

        self.assertEqual(
            arguments["messages"],
            messages,
        )

        self.assertEqual(
            arguments["tools"],
            tools,
        )

        self.assertEqual(
            arguments["tool_choice"],
            "auto",
        )

        self.assertEqual(
            arguments["max_tokens"],
            1000,
        )

        self.assertEqual(
            arguments["extra_body"],
            {
                "enable_thinking": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
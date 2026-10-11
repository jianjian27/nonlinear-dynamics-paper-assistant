import unittest
from unittest.mock import (
    Mock,
    patch,
)

from models.deepseek_api import (
    AliyunChatClient,
    ModelServiceError,
    ask_deepseek,
)


def make_mock_client(
    *,
    answer: str | None = "模型回答",
    finish_reason: str = "stop",
    choices: bool = True,
) -> Mock:
    client = Mock()

    response = Mock()

    if choices:
        choice = Mock()
        choice.message.content = answer
        choice.finish_reason = finish_reason
        response.choices = [choice]
    else:
        response.choices = []

    (
        client
        .chat
        .completions
        .create
        .return_value
    ) = response

    return client


class TestAliyunChatClient(
    unittest.TestCase
):
    def test_sends_openai_compatible_messages(
        self,
    ) -> None:
        mock_client = make_mock_client()

        service = AliyunChatClient(
            model_name="test-model",
            max_tokens=1000,
            enable_thinking=False,
            client=mock_client,
        )

        messages = [
            {
                "role": "system",
                "content": "系统规则",
            },
            {
                "role": "user",
                "content": "用户问题",
            },
        ]

        answer = service.ask(messages)

        self.assertEqual(
            answer,
            "模型回答",
        )

        create = (
            mock_client
            .chat
            .completions
            .create
        )

        create.assert_called_once()

        arguments = create.call_args.kwargs

        self.assertEqual(
            arguments["model"],
            "test-model",
        )

        self.assertEqual(
            arguments["messages"],
            messages,
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

    def test_marks_truncated_answer(
        self,
    ) -> None:
        mock_client = make_mock_client(
            answer="未完成回答",
            finish_reason="length",
        )

        service = AliyunChatClient(
            client=mock_client,
        )

        answer = service.ask(
            [
                {
                    "role": "user",
                    "content": "问题",
                }
            ]
        )

        self.assertIn(
            "未完成回答",
            answer,
        )

        self.assertIn(
            "达到输出长度上限",
            answer,
        )

    def test_rejects_empty_messages(
        self,
    ) -> None:
        service = AliyunChatClient(
            client=make_mock_client(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "消息不能为空",
        ):
            service.ask([])

    def test_rejects_invalid_role(
        self,
    ) -> None:
        service = AliyunChatClient(
            client=make_mock_client(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "不支持的消息角色",
        ):
            service.ask(
                [
                    {
                        "role": "tool",
                        "content": "工具消息",
                    }
                ]
            )

    def test_rejects_empty_content(
        self,
    ) -> None:
        service = AliyunChatClient(
            client=make_mock_client(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "内容不能为空",
        ):
            service.ask(
                [
                    {
                        "role": "user",
                        "content": "   ",
                    }
                ]
            )

    def test_rejects_empty_response_choices(
        self,
    ) -> None:
        service = AliyunChatClient(
            client=make_mock_client(
                choices=False
            ),
        )

        with self.assertRaisesRegex(
            ModelServiceError,
            "没有返回任何候选结果",
        ):
            service.ask(
                [
                    {
                        "role": "user",
                        "content": "问题",
                    }
                ]
            )

    def test_rejects_empty_answer(
        self,
    ) -> None:
        service = AliyunChatClient(
            client=make_mock_client(
                answer=None
            ),
        )

        with self.assertRaisesRegex(
            ModelServiceError,
            "空内容",
        ):
            service.ask(
                [
                    {
                        "role": "user",
                        "content": "问题",
                    }
                ]
            )

    @patch(
        "models.deepseek_api.ask_messages"
    )
    def test_old_ask_deepseek_interface_is_kept(
        self,
        mock_ask_messages: Mock,
    ) -> None:
        mock_ask_messages.return_value = (
            "兼容接口回答"
        )

        answer = ask_deepseek(
            "  分析这篇论文  "
        )

        self.assertEqual(
            answer,
            "兼容接口回答",
        )

        messages = (
            mock_ask_messages
            .call_args
            .args[0]
        )

        self.assertEqual(
            messages[1],
            {
                "role": "user",
                "content": "分析这篇论文",
            },
        )


if __name__ == "__main__":
    unittest.main()
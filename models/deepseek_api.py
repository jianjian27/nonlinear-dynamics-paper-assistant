import os
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)


load_dotenv()


DEFAULT_MODEL_NAME = os.getenv(
    "DASHSCOPE_MODEL",
    "qwen3.8-max",
)

DEFAULT_REQUEST_TIMEOUT = float(
    os.getenv(
        "DASHSCOPE_TIMEOUT",
        "180",
    )
)

DEFAULT_MAX_TOKENS = int(
    os.getenv(
        "DASHSCOPE_MAX_TOKENS",
        "5000",
    )
)

DEFAULT_ENABLE_THINKING = (
    os.getenv(
        "DASHSCOPE_ENABLE_THINKING",
        "false",
    ).lower()
    == "true"
)


class ModelServiceError(RuntimeError):
    """模型服务调用失败。"""


class AliyunChatClient:
    """阿里云百炼OpenAI兼容聊天客户端。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        request_timeout: float = (
            DEFAULT_REQUEST_TIMEOUT
        ),
        max_tokens: int = DEFAULT_MAX_TOKENS,
        enable_thinking: bool = (
            DEFAULT_ENABLE_THINKING
        ),
        client: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "模型名称不能为空。"
            )

        if request_timeout <= 0:
            raise ValueError(
                "请求超时时间必须大于0。"
            )

        if max_tokens <= 0:
            raise ValueError(
                "最大输出Token数必须大于0。"
            )

        self.model_name = model_name
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

        if client is not None:
            self.client = client
            return

        resolved_api_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY")
        )

        resolved_base_url = (
            base_url
            or os.getenv("DASHSCOPE_BASE_URL")
        )

        if not resolved_api_key:
            raise ModelServiceError(
                "缺少环境变量 "
                "DASHSCOPE_API_KEY。"
            )

        if not resolved_base_url:
            raise ModelServiceError(
                "缺少环境变量 "
                "DASHSCOPE_BASE_URL。"
            )

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=request_timeout,
        )

    def ask(
        self,
        messages: Sequence[
            Mapping[str, str]
        ],
    ) -> str:
        """发送OpenAI兼容消息并返回回答。"""

        normalized_messages = (
            self._validate_messages(messages)
        )

        try:
            response = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model_name,
                    messages=normalized_messages,
                    max_tokens=self.max_tokens,
                    extra_body={
                        "enable_thinking": (
                            self.enable_thinking
                        ),
                    },
                )
            )
        except AuthenticationError as exc:
            raise ModelServiceError(
                "模型服务认证失败，请检查"
                "API Key及业务空间。"
            ) from exc
        except APITimeoutError as exc:
            raise ModelServiceError(
                "模型请求超过"
                f" {self.request_timeout:g} 秒，"
                "请稍后重试或减少输入内容。"
            ) from exc
        except RateLimitError as exc:
            raise ModelServiceError(
                "模型请求过于频繁或额度受限，"
                "请稍后重试并检查阿里云额度。"
            ) from exc
        except APIConnectionError as exc:
            raise ModelServiceError(
                "无法连接模型服务，请检查"
                "网络、代理和Base URL。"
            ) from exc
        except BadRequestError as exc:
            raise ModelServiceError(
                "模型拒绝了当前请求，请检查"
                "模型名称、输入长度和请求参数。"
            ) from exc
        except APIStatusError as exc:
            raise ModelServiceError(
                "模型服务返回异常状态码："
                f"{exc.status_code}。"
            ) from exc

        if not response.choices:
            raise ModelServiceError(
                "模型没有返回任何候选结果。"
            )

        choice = response.choices[0]
        answer = choice.message.content

        if not answer:
            raise ModelServiceError(
                "模型返回了空内容。"
            )

        if choice.finish_reason == "length":
            answer += (
                "\n\n---\n"
                "⚠️ 本次回答达到输出长度上限，"
                "结果可能不完整。"
            )

        return answer

    @staticmethod
    def _validate_messages(
        messages: Sequence[
            Mapping[str, str]
        ],
    ) -> list[dict[str, str]]:
        """检查并复制聊天消息。"""

        if not messages:
            raise ValueError(
                "模型消息不能为空。"
            )

        normalized_messages = []

        allowed_roles = {
            "system",
            "user",
            "assistant",
        }

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role not in allowed_roles:
                raise ValueError(
                    f"不支持的消息角色：{role}"
                )

            if (
                not isinstance(content, str)
                or not content.strip()
            ):
                raise ValueError(
                    "消息内容不能为空。"
                )

            normalized_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return normalized_messages


_default_client: AliyunChatClient | None = None


def get_default_chat_client(
) -> AliyunChatClient:
    """延迟创建默认模型客户端。"""

    global _default_client

    if _default_client is None:
        _default_client = AliyunChatClient()

    return _default_client


def ask_messages(
    messages: Sequence[
        Mapping[str, str]
    ],
) -> str:
    """使用默认客户端发送消息。"""

    return get_default_chat_client().ask(
        messages
    )


def ask_deepseek(prompt: str) -> str:
    """
    使用默认客户端回答单个Prompt。

    保留原函数名，兼容app.py和main.py。
    """

    cleaned_prompt = prompt.strip()

    if not cleaned_prompt:
        raise ValueError(
            "Prompt不能为空。"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名非线性动力系统领域"
                "科研助手。你的回答需要保持"
                "学术严谨。"
            ),
        },
        {
            "role": "user",
            "content": cleaned_prompt,
        },
    ]

    return ask_messages(messages)
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Any

from pydantic import ValidationError

from agent_runtime.messages import (
    AgentAssistantMessage,
    AgentToolCall,
)
from agent_runtime.model_protocol import (
    AgentMessage,
    ToolDefinition,
)
from models.deepseek_api import (
    AliyunChatClient,
    ModelServiceError,
)


class AliyunAgentChatModel:
    """将阿里云兼容响应转换为Agent内部消息。"""

    def __init__(
        self,
        chat_client: (
            AliyunChatClient | None
        ) = None,
    ) -> None:
        self.chat_client = (
            chat_client
            if chat_client is not None
            else AliyunChatClient()
        )

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
        """生成文本回答或工具调用请求。"""

        normalized_messages = (
            self._validate_messages(
                messages
            )
        )

        normalized_tools = (
            self._validate_tools(
                tools
            )
        )

        response = (
            self.chat_client
            .create_completion(
                messages=normalized_messages,
                tools=normalized_tools,
                tool_choice=(
                    "auto"
                    if normalized_tools
                    else None
                ),
            )
        )

        if not response.choices:
            raise ModelServiceError(
                "模型没有返回任何候选结果。"
            )

        choice = response.choices[0]
        message = choice.message

        tool_calls: list[
            AgentToolCall
        ] = []

        for raw_tool_call in (
            message.tool_calls or []
        ):
            if (
                raw_tool_call.type
                != "function"
            ):
                raise ModelServiceError(
                    "模型返回了不支持的"
                    "工具调用类型。"
                )

            function = (
                raw_tool_call.function
            )

            try:
                tool_call = AgentToolCall(
                    id=raw_tool_call.id,
                    name=function.name,
                    arguments_json=(
                        function.arguments
                    ),
                )

            except (
                ValidationError,
                TypeError,
                AttributeError,
            ) as exc:
                raise ModelServiceError(
                    "模型返回的工具调用"
                    "结构不完整。"
                ) from exc

            tool_calls.append(
                tool_call
            )

        content = message.content

        if (
            content is not None
            and not isinstance(
                content,
                str,
            )
        ):
            raise ModelServiceError(
                "模型返回了不支持的"
                "文本内容格式。"
            )

        try:
            return AgentAssistantMessage(
                content=content,
                tool_calls=tuple(
                    tool_calls
                ),
                finish_reason=(
                    choice.finish_reason
                ),
            )

        except ValidationError as exc:
            raise ModelServiceError(
                "模型既没有返回文本，"
                "也没有返回工具调用。"
            ) from exc

    @staticmethod
    def _validate_messages(
        messages: Sequence[
            AgentMessage
        ],
    ) -> list[dict[str, Any]]:
        """验证并复制Agent消息历史。"""

        if not messages:
            raise ValueError(
                "Agent消息不能为空。"
            )

        normalized_messages: list[
            dict[str, Any]
        ] = []

        allowed_roles = {
            "system",
            "user",
            "assistant",
            "tool",
        }

        for message in messages:
            normalized = dict(message)

            role = normalized.get(
                "role"
            )

            if role not in allowed_roles:
                raise ValueError(
                    "不支持的Agent消息角色："
                    f"{role}"
                )

            content = normalized.get(
                "content"
            )

            if role in {
                "system",
                "user",
            }:
                if (
                    not isinstance(
                        content,
                        str,
                    )
                    or not content.strip()
                ):
                    raise ValueError(
                        f"{role}消息内容不能为空。"
                    )

            elif role == "assistant":
                tool_calls = normalized.get(
                    "tool_calls"
                )

                has_content = (
                    isinstance(
                        content,
                        str,
                    )
                    and bool(
                        content.strip()
                    )
                )

                has_tool_calls = (
                    isinstance(
                        tool_calls,
                        list,
                    )
                    and bool(tool_calls)
                )

                if (
                    not has_content
                    and not has_tool_calls
                ):
                    raise ValueError(
                        "assistant消息必须包含"
                        "文本或工具调用。"
                    )

            elif role == "tool":
                tool_call_id = (
                    normalized.get(
                        "tool_call_id"
                    )
                )

                if (
                    not isinstance(
                        tool_call_id,
                        str,
                    )
                    or not tool_call_id.strip()
                ):
                    raise ValueError(
                        "tool消息必须包含"
                        "tool_call_id。"
                    )

                if not isinstance(
                    content,
                    str,
                ):
                    raise ValueError(
                        "tool消息内容必须是字符串。"
                    )

            normalized_messages.append(
                normalized
            )

        return normalized_messages

    @staticmethod
    def _validate_tools(
        tools: Sequence[
            ToolDefinition
        ],
    ) -> list[dict[str, Any]]:
        """验证并复制工具定义。"""

        normalized_tools: list[
            dict[str, Any]
        ] = []

        for tool in tools:
            normalized = dict(tool)

            if (
                normalized.get("type")
                != "function"
            ):
                raise ValueError(
                    "目前只支持function工具。"
                )

            function = normalized.get(
                "function"
            )

            if not isinstance(
                function,
                Mapping,
            ):
                raise ValueError(
                    "工具定义缺少function。"
                )

            name = function.get("name")

            if (
                not isinstance(name, str)
                or not name.strip()
            ):
                raise ValueError(
                    "工具定义缺少有效名称。"
                )

            normalized_tools.append(
                normalized
            )

        return normalized_tools
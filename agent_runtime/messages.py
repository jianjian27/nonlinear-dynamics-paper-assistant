from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class AgentToolCall(BaseModel):
    """模型请求执行的一次工具调用。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    id: str
    name: str
    arguments_json: str

    @field_validator("id", "name")
    @classmethod
    def validate_non_empty_text(cls, value: str,) -> str:
        """清理并验证调用ID和工具名称。"""

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "工具调用的ID和名称不能为空。"
            )

        return cleaned_value

    def to_openai_dict(self,) -> dict[str, Any]:
        """转换为OpenAI兼容的工具调用结构。"""

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": (
                    self.arguments_json
                ),
            },
        }


class AgentAssistantMessage(BaseModel):
    """模型返回给Agent运行时的统一消息。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    content: str | None = None

    tool_calls: tuple[
        AgentToolCall,
        ...
    ] = Field(
        default_factory=tuple,
    )

    finish_reason: str | None = None

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str | None,) -> str | None:
        """把空白回答统一处理成None。"""

        if value is None:
            return None

        cleaned_value = value.strip()

        if not cleaned_value:
            return None

        return cleaned_value

    @model_validator(mode="after")
    def validate_message_has_content(self,) -> "AgentAssistantMessage": # 加引号是“前向引用”写法。因为此时还处于 AgentAssistantMessage 类的定义内部，类尚未完全定义结束
        """消息必须包含文本或至少一次工具调用。"""

        if (
            self.content is None
            and not self.tool_calls
        ):
            raise ValueError(
                "Assistant消息必须包含"
                "文本内容或工具调用。"
            )

        return self

    @property
    def requests_tools(self,) -> bool:
        """模型是否请求调用工具。"""

        return bool(self.tool_calls)

    @property
    def is_final_answer(self,) -> bool:
        """当前消息是否可以作为最终回答。"""

        return (
            self.content is not None
            and not self.tool_calls
        )

    def to_openai_dict(self,) -> dict[str, Any]:
        """转换为OpenAI兼容的assistant消息。"""

        message:dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }

        if self.tool_calls:
            message["tool_calls"] = [
                tool_call.to_openai_dict()
                for tool_call in self.tool_calls
            ]


        return message

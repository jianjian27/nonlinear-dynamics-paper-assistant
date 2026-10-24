# ToolDispatcher。模型工具调用和 Python 工具之间的执行边界。
# 模型返回：
# 工具名 + JSON字符串参数
#        ↓
# ToolDispatcher
#        ├── 解析JSON
#        ├── 查找工具
#        ├── Pydantic验证
#        ├── 执行工具
#        └── 统一包装成功或失败结果

import json
from typing import Any

from pydantic import (
    BaseModel,
    ValidationError,
    ConfigDict,
)

from agent_runtime.tool_registry import (
    ToolNotFoundError,
    ToolRegistry,
)

class ToolExecutionError(BaseModel):
    """一次工具调用的结构化错误。"""

    model_config = ConfigDict(
        frozen=True,
    )

    code: str
    message: str

class ToolExecutionResult(BaseModel):
    """一次工具调用的统一执行结果。"""

    model_config = ConfigDict(
        frozen=True,
    )

    tool_call_id: str
    tool_name: str
    success: bool

    output: dict[str, Any] | None = None
    error: ToolExecutionError | None = None

    def to_tool_message(self) -> dict[str, str]:
        """转换为可以返回给大模型的tool消息。"""

        if self.success:
            payload: dict[str, Any] = {
                "success": True,
                "result": self.output,
            } # payload是常用的变量名，表示准备传递给另一个系统的数据，在 API、网络通信和 Tool Calling 中，习惯使用 payload 表示“准备发送的数据主体”。

        else:
            payload = {
                "success": False,
                "error": (
                    self.error.model_dump(mode="json")
                    if self.error is not None
                    else {
                        "code": "unknown_error",
                        "message": (
                            "工具执行失败，"
                            "但没有错误详情。"
                        ),
                    }
                ),
            }

        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": json.dumps(payload, ensure_ascii=False),
        }


class ToolDispatcher:
    """解析并执行大模型发出的工具调用。"""

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        self.registry = registry

    def dispatch(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments_json: str,
    ) -> ToolExecutionResult:
        """执行一次工具调用并返回结构化结果。"""

        cleaned_call_id = tool_call_id.strip()

        cleaned_tool_name = tool_name.strip()

        if not cleaned_call_id:
            return self._failure(
                tool_call_id=tool_call_id,
                tool_name=cleaned_tool_name,
                code="invalid_tool_call_id",
                message="工具调用ID不能为空。",
            )

        if not cleaned_tool_name:
            return self._failure(
                tool_call_id=cleaned_call_id,
                tool_name=tool_name,
                code="invalid_tool_name",
                message="工具名称不能为空。",
            )

        try:
            arguments = json.loads(arguments_json)


        except (

                json.JSONDecodeError,

                TypeError,

        ) as exc:

            return self._failure(

                tool_call_id=cleaned_call_id,

                tool_name=cleaned_tool_name,

                code="invalid_json",

                message=(

                    "工具参数不是有效的JSON："

                    f"{exc}"

                ),

            )

        if not isinstance(arguments, dict):
            return self._failure(

                tool_call_id=cleaned_call_id,

                tool_name=cleaned_tool_name,

                code="invalid_arguments",

                message=(

                    "工具参数必须是JSON对象。"

                ),

            )

        try:
            tool = self.registry.require(
                cleaned_tool_name
            )

            output = tool.invoke(arguments)

        except ToolNotFoundError as exc:
            return self._failure(
                tool_call_id=cleaned_call_id,
                tool_name=cleaned_tool_name,
                code="tool_not_found",
                message=str(exc),
            )

        except ValidationError as exc:
            return self._failure(
                tool_call_id=cleaned_call_id,
                tool_name=cleaned_tool_name,
                code="validation_error",
                message=self._format_validation_error(
                    exc
                ),
            )

        except Exception as exc:
            return self._failure(
                tool_call_id=cleaned_call_id,
                tool_name=cleaned_tool_name,
                code="tool_execution_error",
                message=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        return ToolExecutionResult(
            tool_call_id=cleaned_call_id,
            tool_name=cleaned_tool_name,
            success=True,
            output=output,
            error=None,
        )

    @staticmethod
    def _failure(
        *,
        tool_call_id: str,
        tool_name: str,
        code: str,
        message: str,
    ) -> ToolExecutionResult:
        """构造统一的工具失败结果。"""

        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=False,
            output=None,
            error=ToolExecutionError(
                code=code,
                message=message,
            ),
        )

    @staticmethod
    def _format_validation_error(
        error: ValidationError,
    ) -> str:
        """将Pydantic错误转换为简洁文本。"""

        messages: list[str] = []

        for item in error.errors():
            location = ".".join(
                str(part)
                for part in item["loc"]
            )

            error_message = item["msg"]

            if location:
                messages.append(
                    f"{location}: "
                    f"{error_message}"
                )
            else:
                messages.append(
                    error_message
                )

        return "；".join(messages)



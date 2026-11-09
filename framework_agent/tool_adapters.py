from collections.abc import Iterable

from typing import Any

from langchain_core.tools import (
    StructuredTool,
)

from agent_tools import (
    AgentTool,
)


def adapt_agent_tool(
    tool: AgentTool,
) -> StructuredTool:
    """将现有AgentTool适配为LangChain工具。"""

    if not isinstance(tool, AgentTool):
        raise TypeError(
            "只能适配AgentTool实例。"
        )

    def invoke_existing_tool(
        **arguments: Any,
    ) -> dict[str, Any]:
        """调用原有工具实现。"""

        return tool.invoke(arguments)

    return StructuredTool.from_function(
        func=invoke_existing_tool,
        name=tool.name,
        description=tool.description,
        args_schema=tool.input_model,
        infer_schema=False, # 默认情况下，LangChain 可以尝试从 Python 函数签名推断参数。这里关闭。
    )


def adapt_agent_tools(
    tools: Iterable[AgentTool],
) -> tuple[
    StructuredTool,
    ...
]:
    """批量转换并检查工具名称冲突。"""

    adapted_tools: list[StructuredTool] = []

    seen_names: set[str] = set()

    for tool in tools:
        adapted_tool = adapt_agent_tool(tool)

        if adapted_tool.name in seen_names:
            raise ValueError(
                "不能适配名称重复的工具："
                f"{adapted_tool.name}"
            )

        seen_names.add(adapted_tool.name)

        adapted_tools.append(adapted_tool)

    return tuple(adapted_tools)
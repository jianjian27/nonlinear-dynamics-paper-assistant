from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Any

from agent_runtime import (AgentRunResult)
from evaluation.models import (
    AgentEvalCase,
    ToolTraceEvaluation,
)


class ToolTraceFormatError(
    ValueError
):
    """Agent消息中的工具轨迹格式不合法。"""


def extract_tool_names(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """按照实际调用顺序提取工具名称。"""

    tool_names: list[str] = []

    for message_index, message in enumerate(
        messages,
        start=1,
    ):
        if (
            message.get("role")
            != "assistant"
        ):
            continue

        tool_calls = message.get("tool_calls")

        if tool_calls is None:
            continue

        if not isinstance(
            tool_calls,
            (list, tuple),
        ):
            raise ToolTraceFormatError(
                "第"
                f"{message_index}"
                "条assistant消息的"
                "tool_calls必须是列表或者元组。"
            )

        for call_index, tool_call in enumerate(tool_calls, start=1,):
            if not isinstance(
                tool_call,
                Mapping,
            ):
                raise ToolTraceFormatError(
                    "第"
                    f"{message_index}"
                    "条消息中的第"
                    f"{call_index}"
                    "个工具调用必须是JSON对象。"
                )

            function = tool_call.get("function")

            if not isinstance(
                function,
                Mapping,
            ):
                raise ToolTraceFormatError(
                    "工具调用缺少"
                    "function对象。"
                )

            tool_name = function.get("name")

            if not isinstance(
                tool_name,
                str,
            ):
                raise ToolTraceFormatError(
                    "工具调用缺少字符串类型"
                    "的工具名称。"
                )

            cleaned_name = tool_name.strip()

            if not cleaned_name:
                raise ToolTraceFormatError(
                    "工具名称不能为空。"
                )

            tool_names.append(cleaned_name)

    return tuple(tool_names)


def evaluate_tool_trace(
    case: AgentEvalCase,
    result: AgentRunResult,
) -> ToolTraceEvaluation:
    """比较一个评估案例与Agent实际工具轨迹。"""

    actual_tools = extract_tool_names(result.messages)

    if (
        len(actual_tools)
        != result.tool_call_count
    ):
        raise ToolTraceFormatError(
            "消息轨迹中的工具调用数量"
            "与AgentRunResult.tool_call_count"
            "不一致。"
        )

    actual_tools_set = set(actual_tools)

    missing_expected_tools = tuple(
        tool_name
        for tool_name in case.expected_tools
        if tool_name not in actual_tools_set
    )

    used_forbidden_tools = tuple(
        tool_name
        for tool_name in case.forbidden_tools
        if tool_name in actual_tools_set
    )

    passed = (
        not missing_expected_tools
        and not used_forbidden_tools
    )

    return ToolTraceEvaluation(
        case_id=case.case_id,
        actual_tools=actual_tools,
        missing_expected_tools=(
            missing_expected_tools
        ),
        used_forbidden_tools=(
            used_forbidden_tools
        ),
        tool_call_count=(
            result.tool_call_count
        ),
        model_rounds=(
            result.model_rounds
        ),
        passed=passed,
    )

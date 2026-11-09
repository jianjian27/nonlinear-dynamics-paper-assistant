import unittest

from agent_runtime import (
    AgentRunResult,
)
from evaluation import (
    AgentEvalCase,
    ToolTraceFormatError,
    evaluate_tool_trace,
    extract_tool_names,
)


def make_run_result(
    *tool_names: str,
    tool_call_count: int | None = None,
    model_rounds: int = 2,
) -> AgentRunResult:
    """创建包含指定工具调用的测试结果。"""

    messages: list[
        dict
    ] = [
        {
            "role": "system",
            "content": "测试系统提示词",
        },
        {
            "role": "user",
            "content": "测试问题",
        },
    ]

    if tool_names:
        tool_calls = []

        for index, tool_name in enumerate(
            tool_names,
            start=1,
        ):
            call_id = (
                f"call-{index:03d}"
            )

            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": "{}",
                    },
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
        )

        for index in range(
            1,
            len(tool_names) + 1,
        ):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": (
                        f"call-{index:03d}"
                    ),
                    "content": (
                        '{"success": true}'
                    ),
                }
            )

    messages.append(
        {
            "role": "assistant",
            "content": "最终回答",
        }
    )

    actual_count = (
        len(tool_names)
        if tool_call_count is None
        else tool_call_count
    )

    return AgentRunResult(
        answer="最终回答",
        messages=tuple(messages),
        model_rounds=model_rounds,
        tool_call_count=actual_count,
    )


class TestExtractToolNames(
    unittest.TestCase
):
    """测试工具名称提取。"""

    def test_extracts_tools_in_order(
        self,
    ) -> None:
        """应保持工具实际调用顺序。"""

        result = make_run_result(
            "list_papers",
            "get_paper_overview",
        )

        names = extract_tool_names(
            result.messages
        )

        self.assertEqual(
            names,
            (
                "list_papers",
                "get_paper_overview",
            ),
        )

    def test_returns_empty_tuple(
        self,
    ) -> None:
        """没有工具调用时返回空元组。"""

        result = make_run_result(
            model_rounds=1,
        )

        self.assertEqual(
            extract_tool_names(
                result.messages
            ),
            (),
        )

    def test_rejects_malformed_trace(
        self,
    ) -> None:
        """非法tool_calls不能被静默忽略。"""

        result = AgentRunResult(
            answer="回答",
            messages=(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": "错误格式",
                },
                {
                    "role": "assistant",
                    "content": "回答",
                },
            ),
            model_rounds=2,
            tool_call_count=1,
        )

        with self.assertRaisesRegex(
            ToolTraceFormatError,
            "tool_calls必须是列表",
        ):
            extract_tool_names(
                result.messages
            )


class TestEvaluateToolTrace(
    unittest.TestCase
):
    """测试工具轨迹评估。"""

    def test_passes_expected_tools(
        self,
    ) -> None:
        """调用全部必需工具时应通过。"""

        case = AgentEvalCase(
            case_id="tool-001",
            question="查询论文结构",
            expected_tools=(
                "list_papers",
                "get_paper_overview",
            ),
        )

        result = make_run_result(
            "list_papers",
            "get_paper_overview",
        )

        evaluation = evaluate_tool_trace(
            case,
            result,
        )

        self.assertTrue(
            evaluation.passed
        )

        self.assertEqual(
            evaluation.missing_expected_tools,
            (),
        )

        self.assertEqual(
            evaluation.actual_tools,
            (
                "list_papers",
                "get_paper_overview",
            ),
        )

    def test_fails_missing_tool(
        self,
    ) -> None:
        """缺少必需工具时应失败。"""

        case = AgentEvalCase(
            case_id="tool-002",
            question="查询论文结构",
            expected_tools=(
                "list_papers",
                "get_paper_overview",
            ),
        )

        result = make_run_result(
            "list_papers",
        )

        evaluation = evaluate_tool_trace(
            case,
            result,
        )

        self.assertFalse(
            evaluation.passed
        )

        self.assertEqual(
            evaluation.missing_expected_tools,
            (
                "get_paper_overview",
            ),
        )

    def test_fails_forbidden_tool(
        self,
    ) -> None:
        """调用禁止工具时应失败。"""

        case = AgentEvalCase(
            case_id="tool-003",
            question="知识库中有哪些论文？",
            expected_tools=(
                "list_papers",
            ),
            forbidden_tools=(
                "search_paper_evidence",
            ),
        )

        result = make_run_result(
            "list_papers",
            "search_paper_evidence",
        )

        evaluation = evaluate_tool_trace(
            case,
            result,
        )

        self.assertFalse(
            evaluation.passed
        )

        self.assertEqual(
            evaluation.used_forbidden_tools,
            (
                "search_paper_evidence",
            ),
        )

    def test_passes_direct_answer(
        self,
    ) -> None:
        """不需要工具的案例可以直接回答。"""

        case = AgentEvalCase(
            case_id="direct-001",
            question="你好",
            forbidden_tools=(
                "list_papers",
                "get_paper_overview",
                "search_paper_evidence",
            ),
        )

        result = make_run_result(
            model_rounds=1,
        )

        evaluation = evaluate_tool_trace(
            case,
            result,
        )

        self.assertTrue(
            evaluation.passed
        )

        self.assertEqual(
            evaluation.tool_call_count,
            0,
        )

    def test_allows_repeated_tool_names(
        self,
    ) -> None:
        """仅凭名称重复不能判断调用无效。"""

        case = AgentEvalCase(
            case_id="compare-001",
            question="比较两篇论文",
            expected_tools=(
                "get_paper_overview",
            ),
        )

        result = make_run_result(
            "get_paper_overview",
            "get_paper_overview",
        )

        evaluation = evaluate_tool_trace(
            case,
            result,
        )

        self.assertTrue(
            evaluation.passed
        )

        self.assertEqual(
            evaluation.tool_call_count,
            2,
        )

    def test_rejects_count_mismatch(
        self,
    ) -> None:
        """结果统计必须与消息轨迹一致。"""

        case = AgentEvalCase(
            case_id="invalid-001",
            question="测试",
        )

        result = make_run_result(
            "list_papers",
            tool_call_count=2,
        )

        with self.assertRaisesRegex(
            ToolTraceFormatError,
            "工具调用数量",
        ):
            evaluate_tool_trace(
                case,
                result,
            )


if __name__ == "__main__":
    unittest.main()
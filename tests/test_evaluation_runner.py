import unittest

from agent_runtime import (
    AgentRunResult,
)
from evaluation import (
    AgentEvalCase,
    evaluate_agent_case,
    find_missing_answer_keywords,
    run_evaluation_cases,
    summarize_evaluations,
)


def make_result(
    *,
    answer: str = "论文使用了Lyapunov指数和分岔图。",
    tool_name: str = (
        "search_paper_evidence"
    ),
) -> AgentRunResult:
    """创建测试用Agent结果。"""

    messages = (
        {
            "role": "user",
            "content": "测试问题",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-001",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-001",
            "content": (
                '{"success": true}'
            ),
        },
        {
            "role": "assistant",
            "content": answer,
        },
    )

    return AgentRunResult(
        answer=answer,
        messages=messages,
        model_rounds=2,
        tool_call_count=1,
    )


class TestAnswerKeywords(
    unittest.TestCase
):
    def test_finds_missing_keywords(
        self,
    ) -> None:
        missing = (
            find_missing_answer_keywords(
                answer="使用了Lyapunov指数。",
                expected_keywords=(
                    "Lyapunov",
                    "分岔图",
                ),
            )
        )

        self.assertEqual(
            missing,
            ("分岔图",),
        )

    def test_ignores_english_case(
        self,
    ) -> None:
        missing = (
            find_missing_answer_keywords(
                answer="LYAPUNOV exponent",
                expected_keywords=(
                    "Lyapunov",
                ),
            )
        )

        self.assertEqual(
            missing,
            (),
        )


class TestEvaluationRunner(
    unittest.TestCase
):
    def test_passes_valid_case(
        self,
    ) -> None:
        case = AgentEvalCase(
            case_id="method-001",
            question="使用了什么方法？",
            expected_tools=(
                "search_paper_evidence",
            ),
            expected_answer_keywords=(
                "Lyapunov",
                "分岔图",
            ),
        )

        evaluation = evaluate_agent_case(
            case,
            lambda question: make_result(),
        )

        self.assertTrue(
            evaluation.passed
        )

        self.assertIsNone(
            evaluation.error
        )

    def test_fails_missing_keyword(
        self,
    ) -> None:
        case = AgentEvalCase(
            case_id="method-002",
            question="使用了什么方法？",
            expected_tools=(
                "search_paper_evidence",
            ),
            expected_answer_keywords=(
                "Lyapunov",
                "分岔图",
            ),
        )

        evaluation = evaluate_agent_case(
            case,
            lambda question: make_result(
                answer="使用了Lyapunov指数。"
            ),
        )

        self.assertFalse(
            evaluation.passed
        )

        self.assertEqual(
            evaluation
            .missing_answer_keywords,
            ("分岔图",),
        )

    def test_records_agent_error(
        self,
    ) -> None:
        case = AgentEvalCase(
            case_id="error-001",
            question="测试错误",
        )

        def failing_agent(
            question: str,
        ) -> AgentRunResult:
            raise RuntimeError(
                "模型服务不可用"
            )

        evaluation = evaluate_agent_case(
            case,
            failing_agent,
        )

        self.assertFalse(
            evaluation.passed
        )

        self.assertIsNone(
            evaluation.trace
        )

        self.assertIn(
            "RuntimeError",
            evaluation.error or "",
        )

    def test_runs_multiple_cases(
        self,
    ) -> None:
        cases = (
            AgentEvalCase(
                case_id="batch-001",
                question="问题一",
                expected_tools=(
                    "search_paper_evidence",
                ),
            ),
            AgentEvalCase(
                case_id="batch-002",
                question="问题二",
                expected_tools=(
                    "search_paper_evidence",
                ),
            ),
        )

        evaluations = run_evaluation_cases(
            cases,
            lambda question: make_result(),
        )

        self.assertEqual(
            len(evaluations),
            2,
        )

        self.assertTrue(
            all(
                evaluation.passed
                for evaluation in evaluations
            )
        )

    def test_summarizes_results(
        self,
    ) -> None:
        cases = (
            AgentEvalCase(
                case_id="summary-001",
                question="问题一",
                expected_tools=(
                    "search_paper_evidence",
                ),
            ),
            AgentEvalCase(
                case_id="summary-002",
                question="问题二",
                expected_answer_keywords=(
                    "不存在的关键词",
                ),
            ),
        )

        evaluations = run_evaluation_cases(
            cases,
            lambda question: make_result(),
        )

        summary = summarize_evaluations(
            evaluations
        )

        self.assertEqual(
            summary.total_cases,
            2,
        )

        self.assertEqual(
            summary.passed_cases,
            1,
        )

        self.assertEqual(
            summary.failed_cases,
            1,
        )

        self.assertEqual(
            summary.pass_rate,
            0.5,
        )

        self.assertEqual(
            summary.failed_case_ids,
            ("summary-002",),
        )


if __name__ == "__main__":
    unittest.main()
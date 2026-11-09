import tempfile
import unittest
from pathlib import Path

from evaluation import (
    AgentCaseEvaluation,
    EvaluationSummary,
    render_evaluation_report,
    save_evaluation_report,
)


class TestEvaluationReport(
    unittest.TestCase
):
    def test_renders_summary(
        self,
    ) -> None:
        evaluations = (
            AgentCaseEvaluation(
                case_id="case-001",
                question="测试问题",
                duration_seconds=0.5,
                error=None,
                passed=True,
            ),
        )

        summary = EvaluationSummary(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            pass_rate=1.0,
            average_duration_seconds=0.5,
            total_model_rounds=0,
            total_tool_calls=0,
        )

        report = render_evaluation_report(
            evaluations,
            summary,
        )

        self.assertIn(
            "# Agent评估报告",
            report,
        )

        self.assertIn(
            "通过率：100.0%",
            report,
        )

        self.assertIn(
            "case-001",
            report,
        )

    def test_saves_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "report.md"
            )

            saved_path = (
                save_evaluation_report(
                    "# 测试报告\n",
                    path,
                )
            )

            self.assertEqual(
                saved_path,
                path,
            )

            self.assertEqual(
                path.read_text(
                    encoding="utf-8"
                ),
                "# 测试报告\n",
            )


if __name__ == "__main__":
    unittest.main()
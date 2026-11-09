from collections.abc import (
    Sequence,
)
from pathlib import Path

from evaluation.models import (
    AgentCaseEvaluation,
    EvaluationSummary,
)


def escape_markdown_cell(
    value: str,
) -> str:
    """转义Markdown表格中的特殊字符。"""

    return (
        value
        .replace("|", r"\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def render_evaluation_report(
    evaluations: Sequence[
        AgentCaseEvaluation
    ],
    summary: EvaluationSummary,
) -> str:
    """生成Agent评估Markdown报告。"""

    lines = [
        "# Agent评估报告",
        "",
        "## 汇总",
        "",
        f"- 案例总数：{summary.total_cases}",
        f"- 通过数量：{summary.passed_cases}",
        f"- 失败数量：{summary.failed_cases}",
        (
            "- 通过率："
            f"{summary.pass_rate:.1%}"
        ),
        (
            "- 平均耗时："
            f"{summary.average_duration_seconds:.3f} 秒"
        ),
        (
            "- 模型总轮数："
            f"{summary.total_model_rounds}"
        ),
        (
            "- 工具调用总数："
            f"{summary.total_tool_calls}"
        ),
        "",
        "## 案例结果",
        "",
        (
            "| 案例 | 结果 | 实际工具 | "
            "缺少关键词 | 错误 |"
        ),
        "|---|---|---|---|---|",
    ]

    for evaluation in evaluations:
        status = (
            "通过"
            if evaluation.passed
            else "失败"
        )

        if evaluation.trace is None:
            actual_tools = ""
        else:
            actual_tools = ", ".join(
                evaluation.trace.actual_tools
            )

        missing_keywords = ", ".join(
            evaluation
            .missing_answer_keywords
        )

        error = evaluation.error or ""

        lines.append(
            "| "
            f"{escape_markdown_cell(evaluation.case_id)}"
            " | "
            f"{status}"
            " | "
            f"{escape_markdown_cell(actual_tools)}"
            " | "
            f"{escape_markdown_cell(missing_keywords)}"
            " | "
            f"{escape_markdown_cell(error)}"
            " |"
        )

    return "\n".join(lines) + "\n"


def save_evaluation_report(
    report: str,
    path: str | Path,
) -> Path:
    """将评估报告保存到Markdown文件。"""

    report_path = Path(path)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    return report_path
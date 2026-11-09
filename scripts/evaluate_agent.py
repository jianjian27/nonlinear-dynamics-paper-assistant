import argparse
import sys
from collections.abc import (
    Sequence,
)
from pathlib import Path

from agent_app import (
    create_paper_agent_runner,
)
from embedding.sentence_transformer_encoder import (
    DEFAULT_MODEL_NAME,
)
from evaluation import (
    evaluate_agent_case,
    load_evaluation_cases,
    render_evaluation_report,
    save_evaluation_report,
    summarize_evaluations,
)


DEFAULT_DATASET_PATH = (
    "evaluation/datasets/"
    "agent_baseline.jsonl"
)

DEFAULT_REPORT_PATH = (
    "evaluation/reports/"
    "agent_baseline.md"
)


def create_argument_parser(
) -> argparse.ArgumentParser:
    """创建Agent评估命令行解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用固定评估集批量运行"
            "科研论文Agent。"
        )
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help="JSONL评估数据集路径。",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_REPORT_PATH,
        help="Markdown评估报告输出路径。",
    )

    parser.add_argument(
        "--database",
        default="data/papers.db",
        help="SQLite论文数据库路径。",
    )

    parser.add_argument(
        "--vector-cache-directory",
        default="data/vector_indexes",
        help="Embedding向量缓存目录。",
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL_NAME,
        help="Embedding模型名称。",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Embedding运行设备，例如cuda或cpu。",
    )

    parser.add_argument(
        "--max-model-rounds",
        type=int,
        default=6,
        help="每个案例最大模型轮数。",
    )

    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=12,
        help="每个案例最大工具调用次数。",
    )

    parser.add_argument(
        "--fail-on-failure",
        action="store_true",
        help=(
            "存在未通过案例时返回"
            "非零退出码，供CI质量门使用。"
        ),
    )

    return parser


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """运行真实科研Agent评估。"""

    parser = create_argument_parser()

    options = parser.parse_args(
        arguments
    )

    database_path = Path(
        options.database
    ).resolve()

    if not database_path.is_file():
        print(
            "错误：找不到论文数据库："
            f"{database_path}",
            file=sys.stderr,
        )
        return 1

    if options.max_model_rounds <= 0:
        print(
            "错误：最大模型轮数必须大于0。",
            file=sys.stderr,
        )
        return 1

    if options.max_tool_calls <= 0:
        print(
            "错误：最大工具调用次数必须大于0。",
            file=sys.stderr,
        )
        return 1

    try:
        cases = load_evaluation_cases(
            options.dataset
        )

        runner = create_paper_agent_runner(
            database_path=database_path,
            vector_cache_directory=(
                options
                .vector_cache_directory
            ),
            embedding_model_name=(
                options.embedding_model
            ),
            embedding_device=(
                options.device
            ),
            max_model_rounds=(
                options.max_model_rounds
            ),
            max_tool_calls=(
                options.max_tool_calls
            ),
        )

    except Exception as error:
        print(
            "评估初始化失败："
            f"{type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )
        return 1

    evaluations = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"[{index}/{len(cases)}] "
            f"正在运行：{case.case_id}"
        )

        evaluation = evaluate_agent_case(
            case,
            runner.run,
        )

        evaluations.append(
            evaluation
        )

        status = (
            "通过"
            if evaluation.passed
            else "失败"
        )

        print(
            f"结果：{status}，"
            f"耗时："
            f"{evaluation.duration_seconds:.2f}秒"
        )

        if evaluation.error:
            print(
                f"错误：{evaluation.error}"
            )

    evaluation_tuple = tuple(
        evaluations
    )

    summary = summarize_evaluations(
        evaluation_tuple
    )

    report = render_evaluation_report(
        evaluation_tuple,
        summary,
    )

    report_path = save_evaluation_report(
        report,
        options.output,
    )

    print()
    print("=== Agent评估汇总 ===")
    print(
        f"案例总数：{summary.total_cases}"
    )
    print(
        f"通过数量：{summary.passed_cases}"
    )
    print(
        f"失败数量：{summary.failed_cases}"
    )
    print(
        f"通过率：{summary.pass_rate:.1%}"
    )
    print(
        "模型总轮数："
        f"{summary.total_model_rounds}"
    )
    print(
        "工具调用总数："
        f"{summary.total_tool_calls}"
    )

    if summary.failed_case_ids:
        print(
            "失败案例："
            + ", ".join(
                summary.failed_case_ids
            )
        )

    print(
        f"报告已保存："
        f"{report_path.resolve()}"
    )

    if (
        options.fail_on_failure
        and summary.failed_cases
    ):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
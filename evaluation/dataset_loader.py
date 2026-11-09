import json
from pathlib import Path

from pydantic import (
    ValidationError,
)

from evaluation.models import (
    AgentEvalCase,
)


class EvaluationDatasetError(
    ValueError
):
    """评估数据集格式或内容不合法。"""


def load_evaluation_cases(
    path: str | Path,
) -> tuple[AgentEvalCase, ...]:
    """从JSONL文件读取并验证Agent评估案例。"""

    dataset_path = Path(path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            "找不到评估数据集："
            f"{dataset_path}"
        )

    cases: list[AgentEvalCase] = []

    seen_case_ids: set[str] = set()

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            cleaned_line = raw_line.strip()

            if not cleaned_line:
                continue

            try:
                raw_case = json.loads(cleaned_line)
            except json.JSONDecodeError as error:
                raise EvaluationDatasetError(
                    f"{dataset_path.name}"
                    f"第{line_number}行不是合法JSON："
                    f"{error.msg}"
                ) from error

            if not isinstance(raw_case, dict):
                raise EvaluationDatasetError(
                    f"{dataset_path.name}"
                    f"第{line_number}行"
                    "必须是JSON对象。"
                )

            try:
                case = AgentEvalCase.model_validate(raw_case)
            except ValidationError as error:
                raise EvaluationDatasetError(
                    f"{dataset_path.name}"
                    f"第{line_number}行"
                    "不符合评估案例格式："
                    f"{error}"
                ) from error

            if case.case_id in seen_case_ids:
                raise EvaluationDatasetError(
                    f"{dataset_path.name}"
                    f"第{line_number}行存在"
                    "重复case_id："
                    f"{case.case_id}"
                )

            seen_case_ids.add(case.case_id)

            cases.append(case)

    if not cases:
        raise EvaluationDatasetError(
            "评估数据集不能为空。"
        )

    return tuple(cases)





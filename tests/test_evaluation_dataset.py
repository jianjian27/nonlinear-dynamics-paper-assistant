import json
import unittest
from pathlib import Path
from tempfile import (
    TemporaryDirectory,
)

from pydantic import (
    ValidationError,
)

from evaluation import (
    AgentEvalCase,
    EvaluationDatasetError,
    load_evaluation_cases,
)


class TestAgentEvalCase(
    unittest.TestCase
):
    """测试Agent评估案例模型。"""

    def test_creates_valid_case(
        self,
    ) -> None:
        """合法数据应转换为不可变对象。"""

        case = AgentEvalCase(
            case_id="tool-001",
            question=(
                "  知识库中有哪些论文？  "
            ),
            expected_tools=[
                " list_papers ",
            ],
            expected_answer_keywords=[
                " 论文 ",
            ],
        )

        self.assertEqual(
            case.question,
            "知识库中有哪些论文？",
        )

        self.assertEqual(
            case.expected_tools,
            (
                "list_papers",
            ),
        )

        self.assertEqual(
            case.expected_answer_keywords,
            (
                "论文",
            ),
        )

    def test_uses_empty_defaults(
        self,
    ) -> None:
        """未指定工具时应使用空元组。"""

        case = AgentEvalCase(
            case_id="direct-001",
            question="你好",
        )

        self.assertEqual(
            case.expected_tools,
            (),
        )

        self.assertEqual(
            case.forbidden_tools,
            (),
        )

    def test_rejects_blank_question(
        self,
    ) -> None:
        """问题不能只包含空白。"""

        with self.assertRaises(
            ValidationError
        ):
            AgentEvalCase(
                case_id="invalid-001",
                question="   ",
            )

    def test_rejects_duplicate_items(
        self,
    ) -> None:
        """同一列表不能包含重复项目。"""

        with self.assertRaisesRegex(
            ValidationError,
            "不能包含重复值",
        ):
            AgentEvalCase(
                case_id="invalid-002",
                question="查询论文",
                expected_tools=[
                    "list_papers",
                    "list_papers",
                ],
            )

    def test_rejects_tool_conflict(
        self,
    ) -> None:
        """同一工具不能既要求又禁止。"""

        with self.assertRaisesRegex(
            ValidationError,
            "不能同时出现",
        ):
            AgentEvalCase(
                case_id="invalid-003",
                question="查询论文",
                expected_tools=[
                    "list_papers",
                ],
                forbidden_tools=[
                    "list_papers",
                ],
            )


class TestEvaluationDatasetLoader(
    unittest.TestCase
):
    """测试JSONL评估集加载器。"""

    def setUp(self) -> None:
        self.temporary_directory = (
            TemporaryDirectory()
        )

        self.directory = Path(
            self.temporary_directory.name
        )

        self.dataset_path = (
            self.directory
            / "cases.jsonl"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_lines(
        self,
        *records: object,
    ) -> None:
        """把测试记录写入临时JSONL文件。"""

        lines = [
            (
                record
                if isinstance(
                    record,
                    str,
                )
                else json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            for record in records
        ]

        self.dataset_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    def test_loads_valid_cases(
        self,
    ) -> None:
        """加载器应返回经过验证的案例。"""

        self.write_lines(
            {
                "case_id": "tool-001",
                "question": (
                    "知识库中有哪些论文？"
                ),
                "expected_tools": [
                    "list_papers",
                ],
            },
            "",
            {
                "case_id": "direct-001",
                "question": "你好",
                "forbidden_tools": [
                    "list_papers",
                ],
            },
        )

        cases = load_evaluation_cases(
            self.dataset_path
        )

        self.assertEqual(
            len(cases),
            2,
        )

        self.assertIsInstance(
            cases[0],
            AgentEvalCase,
        )

        self.assertEqual(
            cases[0].case_id,
            "tool-001",
        )

    def test_rejects_missing_file(
        self,
    ) -> None:
        """文件不存在时应明确报错。"""

        with self.assertRaisesRegex(
            FileNotFoundError,
            "找不到评估数据集",
        ):
            load_evaluation_cases(
                self.directory
                / "missing.jsonl"
            )

    def test_rejects_invalid_json(
        self,
    ) -> None:
        """JSON语法错误应包含行号。"""

        self.write_lines(
            '{"case_id": "broken"',
        )

        with self.assertRaisesRegex(
            EvaluationDatasetError,
            "第1行不是合法JSON",
        ):
            load_evaluation_cases(
                self.dataset_path
            )

    def test_rejects_non_object(
        self,
    ) -> None:
        """每行必须是JSON对象。"""

        self.write_lines(
            [
                "not",
                "an",
                "object",
            ]
        )

        with self.assertRaisesRegex(
            EvaluationDatasetError,
            "必须是JSON对象",
        ):
            load_evaluation_cases(
                self.dataset_path
            )

    def test_rejects_invalid_case(
        self,
    ) -> None:
        """字段验证错误应被转换为数据集错误。"""

        self.write_lines(
            {
                "case_id": "invalid-001",
                "question": "   ",
            }
        )

        with self.assertRaisesRegex(
            EvaluationDatasetError,
            "不符合评估案例格式",
        ):
            load_evaluation_cases(
                self.dataset_path
            )

    def test_rejects_duplicate_case_id(
        self,
    ) -> None:
        """同一数据集中的case_id必须唯一。"""

        record = {
            "case_id": "duplicate-001",
            "question": "测试问题",
        }

        self.write_lines(
            record,
            record,
        )

        with self.assertRaisesRegex(
            EvaluationDatasetError,
            "重复case_id",
        ):
            load_evaluation_cases(
                self.dataset_path
            )

    def test_rejects_empty_dataset(
        self,
    ) -> None:
        """没有任何案例的数据集没有评估意义。"""

        self.dataset_path.write_text(
            "\n\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            EvaluationDatasetError,
            "数据集不能为空",
        ):
            load_evaluation_cases(
                self.dataset_path
            )


if __name__ == "__main__":
    unittest.main()
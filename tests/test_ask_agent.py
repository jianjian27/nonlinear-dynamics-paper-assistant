import io
import json
import unittest
from contextlib import (
    redirect_stderr,
    redirect_stdout,
)
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import (
    Mock,
    patch,
)

from agent_runtime import (
    AgentProtocolError,
    AgentRunLimitError,
    AgentRunResult,
)
from models.deepseek_api import (
    ModelServiceError,
)
from scripts.ask_agent import main


class TestAskAgentCommand(
    unittest.TestCase
):
    """测试科研Agent命令行入口。"""

    def setUp(self) -> None:
        self.temporary_directory = (
            TemporaryDirectory()
        )

        self.directory = Path(
            self.temporary_directory.name
        )

        self.database_path = (
            self.directory
            / "papers.db"
        )

        self.database_path.touch()

        self.vector_cache_directory = (
            self.directory
            / "vector_indexes"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def make_direct_result(
    ) -> AgentRunResult:
        """创建没有调用工具的运行结果。"""

        return AgentRunResult(
            answer="知识库中有两篇论文。",
            messages=(
                {
                    "role": "system",
                    "content": "系统提示词",
                },
                {
                    "role": "user",
                    "content": "有哪些论文？",
                },
                {
                    "role": "assistant",
                    "content": (
                        "知识库中有两篇论文。"
                    ),
                },
            ),
            model_rounds=1,
            tool_call_count=0,
        )

    @staticmethod
    def make_tool_result(
    ) -> AgentRunResult:
        """创建包含一次工具调用的结果。"""

        tool_payload = json.dumps(
            {
                "success": True,
                "result": {
                    "total": 2,
                },
            },
            ensure_ascii=False,
        )

        return AgentRunResult(
            answer="知识库中有两篇论文。",
            messages=(
                {
                    "role": "system",
                    "content": "系统提示词",
                },
                {
                    "role": "user",
                    "content": "有哪些论文？",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-001",
                            "type": "function",
                            "function": {
                                "name": (
                                    "list_papers"
                                ),
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": (
                        "call-001"
                    ),
                    "content": tool_payload,
                },
                {
                    "role": "assistant",
                    "content": (
                        "知识库中有两篇论文。"
                    ),
                },
            ),
            model_rounds=2,
            tool_call_count=1,
        )

    @patch(
        "scripts.ask_agent."
        "create_paper_agent_runner"
    )
    def test_runs_agent_command(
        self,
        mock_create_runner: Mock,
    ) -> None:
        """有效参数应运行Agent并输出回答。"""

        runner = Mock()
        runner.run.return_value = (
            self.make_direct_result()
        )

        mock_create_runner.return_value = (
            runner
        )

        output = io.StringIO()
        error_output = io.StringIO()

        with (
            redirect_stdout(output),
            redirect_stderr(error_output),
        ):
            exit_code = main(
                [
                    "  有哪些论文？  ",
                    "--database",
                    str(self.database_path),
                    "--vector-cache-directory",
                    str(
                        self
                        .vector_cache_directory
                    ),
                    "--embedding-model",
                    "test-embedding",
                    "--device",
                    "cpu",
                    "--max-model-rounds",
                    "4",
                    "--max-tool-calls",
                    "8",
                ]
            )

        self.assertEqual(
            exit_code,
            0,
        )

        self.assertEqual(
            error_output.getvalue(),
            "",
        )

        self.assertIn(
            "知识库中有两篇论文。",
            output.getvalue(),
        )

        self.assertIn(
            "模型轮数=1",
            output.getvalue(),
        )

        self.assertNotIn(
            "Agent执行轨迹",
            output.getvalue(),
        )

        runner.run.assert_called_once_with(
            "有哪些论文？"
        )

        (
            mock_create_runner
            .assert_called_once_with(
                database_path=(
                    self.database_path
                    .resolve()
                ),
                vector_cache_directory=str(
                    self.vector_cache_directory
                ),
                embedding_model_name=(
                    "test-embedding"
                ),
                embedding_device="cpu",
                max_model_rounds=4,
                max_tool_calls=8,
            )
        )

    @patch(
        "scripts.ask_agent."
        "create_paper_agent_runner"
    )
    def test_displays_execution_trace(
        self,
        mock_create_runner: Mock,
    ) -> None:
        """show-trace应显示工具名称和状态。"""

        runner = Mock()
        runner.run.return_value = (
            self.make_tool_result()
        )

        mock_create_runner.return_value = (
            runner
        )

        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "有哪些论文？",
                    "--database",
                    str(self.database_path),
                    "--show-trace",
                ]
            )

        rendered = output.getvalue()

        self.assertEqual(
            exit_code,
            0,
        )

        self.assertIn(
            "Agent执行轨迹",
            rendered,
        )

        self.assertIn(
            "模型请求：list_papers",
            rendered,
        )

        self.assertIn(
            "[call-001] 成功",
            rendered,
        )

    @patch(
        "scripts.ask_agent."
        "create_paper_agent_runner"
    )
    def test_reports_direct_answer_trace(
        self,
        mock_create_runner: Mock,
    ) -> None:
        """未调用工具时轨迹应明确说明。"""

        runner = Mock()
        runner.run.return_value = (
            self.make_direct_result()
        )

        mock_create_runner.return_value = (
            runner
        )

        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "有哪些论文？",
                    "--database",
                    str(self.database_path),
                    "--show-trace",
                ]
            )

        self.assertEqual(
            exit_code,
            0,
        )

        self.assertIn(
            "模型未调用工具，直接回答",
            output.getvalue(),
        )

    @patch(
        "scripts.ask_agent."
        "create_paper_agent_runner"
    )
    def test_rejects_missing_database(
        self,
        mock_create_runner: Mock,
    ) -> None:
        """数据库不存在时不应创建Agent。"""

        missing_path = (
            self.directory
            / "missing.db"
        )

        error_output = io.StringIO()

        with redirect_stderr(
            error_output
        ):
            exit_code = main(
                [
                    "有哪些论文？",
                    "--database",
                    str(missing_path),
                ]
            )

        self.assertEqual(
            exit_code,
            1,
        )

        self.assertIn(
            "找不到论文数据库",
            error_output.getvalue(),
        )

        mock_create_runner.assert_not_called()

    @patch(
        "scripts.ask_agent."
        "create_paper_agent_runner"
    )
    def test_rejects_blank_question(
        self,
        mock_create_runner: Mock,
    ) -> None:
        """空白问题不应创建Agent。"""

        error_output = io.StringIO()

        with redirect_stderr(
            error_output
        ):
            exit_code = main(
                [
                    "   ",
                    "--database",
                    str(self.database_path),
                ]
            )

        self.assertEqual(
            exit_code,
            1,
        )

        self.assertIn(
            "问题不能为空",
            error_output.getvalue(),
        )

        mock_create_runner.assert_not_called()

    @patch(
        "scripts.ask_agent."
        "create_paper_agent_runner"
    )
    def test_rejects_invalid_limits(
        self,
        mock_create_runner: Mock,
    ) -> None:
        """轮数和工具次数必须大于0。"""

        cases = [
            (
                [
                    "--max-model-rounds",
                    "0",
                ],
                "最大模型轮数",
            ),
            (
                [
                    "--max-tool-calls",
                    "0",
                ],
                "最大工具调用次数",
            ),
        ]

        for extra_arguments, message in cases:
            with self.subTest(
                arguments=extra_arguments
            ):
                error_output = io.StringIO()

                with redirect_stderr(
                    error_output
                ):
                    exit_code = main(
                        [
                            "问题",
                            "--database",
                            str(
                                self.database_path
                            ),
                            *extra_arguments,
                        ]
                    )

                self.assertEqual(
                    exit_code,
                    1,
                )

                self.assertIn(
                    message,
                    error_output.getvalue(),
                )

        mock_create_runner.assert_not_called()

    @patch(
        "scripts.ask_agent."
        "create_paper_agent_runner"
    )
    def test_reports_expected_agent_errors(
        self,
        mock_create_runner: Mock,
    ) -> None:
        """已知Agent异常应显示对应错误分类。"""

        runner = Mock()

        mock_create_runner.return_value = (
            runner
        )

        cases = [
            (
                ModelServiceError(
                    "模拟模型错误"
                ),
                "模型服务错误",
            ),
            (
                AgentRunLimitError(
                    "模拟运行限制"
                ),
                "Agent达到运行限制",
            ),
            (
                AgentProtocolError(
                    "模拟协议错误"
                ),
                "Agent协议错误",
            ),
        ]

        for exception, expected in cases:
            with self.subTest(
                exception=type(
                    exception
                ).__name__
            ):
                runner.run.side_effect = (
                    exception
                )

                error_output = (
                    io.StringIO()
                )

                with (
                    redirect_stdout(
                        io.StringIO()
                    ),
                    redirect_stderr(
                        error_output
                    ),
                ):
                    exit_code = main(
                        [
                            "问题",
                            "--database",
                            str(
                                self.database_path
                            ),
                        ]
                    )

                self.assertEqual(
                    exit_code,
                    1,
                )

                self.assertIn(
                    expected,
                    error_output.getvalue(),
                )

    @patch(
        "scripts.ask_agent."
        "create_paper_agent_runner"
    )
    def test_reports_unexpected_error(
        self,
        mock_create_runner: Mock,
    ) -> None:
        """未知异常应转换为命令行错误。"""

        runner = Mock()
        runner.run.side_effect = (
            RuntimeError(
                "模拟未知错误"
            )
        )

        mock_create_runner.return_value = (
            runner
        )

        error_output = io.StringIO()

        with (
            redirect_stdout(
                io.StringIO()
            ),
            redirect_stderr(
                error_output
            ),
        ):
            exit_code = main(
                [
                    "问题",
                    "--database",
                    str(self.database_path),
                ]
            )

        self.assertEqual(
            exit_code,
            1,
        )

        self.assertIn(
            "Agent运行失败",
            error_output.getvalue(),
        )

        self.assertIn(
            "RuntimeError",
            error_output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
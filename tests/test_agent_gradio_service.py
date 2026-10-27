import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from agent_app import (
    GradioAgentService,
)
from agent_runtime import (
    AgentRunResult,
)


def make_direct_result(
) -> AgentRunResult:
    """创建直接回答结果。"""

    return AgentRunResult(
        answer="直接回答",
        messages=(
            {
                "role": "system",
                "content": "系统提示",
            },
            {
                "role": "user",
                "content": "问题",
            },
            {
                "role": "assistant",
                "content": "直接回答",
            },
        ),
        model_rounds=1,
        tool_call_count=0,
    )


def make_tool_result(
    *,
    success: bool = True,
) -> AgentRunResult:
    """创建包含工具调用的结果。"""

    if success:
        payload = {
            "success": True,
            "result": {
                "total": 2,
            },
        }
    else:
        payload = {
            "success": False,
            "error": {
                "code": (
                    "validation_error"
                ),
                "message": (
                    "paper_id不能为空。"
                ),
            },
        }

    return AgentRunResult(
        answer="Agent最终回答",
        messages=(
            {
                "role": "system",
                "content": "系统提示",
            },
            {
                "role": "user",
                "content": "问题",
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
                                "get_paper_overview"
                            ),
                            "arguments": (
                                '{"paper_id": '
                                '"paper-001"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-001",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Agent最终回答"
                ),
            },
        ),
        model_rounds=2,
        tool_call_count=1,
    )


class TestGradioAgentService(
    unittest.TestCase
):
    """测试Gradio科研Agent服务。"""

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

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_asks_runner_and_formats_result(
        self,
    ) -> None:
        """服务应运行Agent并转换展示结果。"""

        runner = Mock()
        runner.run.return_value = (
            make_tool_result()
        )

        runner_factory = Mock(
            return_value=runner
        )

        service = GradioAgentService(
            database_path=(
                self.database_path
            ),
            runner_factory=runner_factory,
        )

        result = service.ask(
            "  查询论文  "
        )

        runner.run.assert_called_once_with(
            "查询论文"
        )

        self.assertEqual(
            result.answer_markdown,
            "Agent最终回答",
        )

        self.assertIn(
            "get_paper_overview",
            result.trace_markdown,
        )

        self.assertIn(
            "paper-001",
            result.trace_markdown,
        )

        self.assertIn(
            "✅ 成功",
            result.trace_markdown,
        )

        self.assertIn(
            "模型轮数：2",
            result.status_text,
        )

        self.assertIn(
            "工具调用数：1",
            result.status_text,
        )

    def test_reuses_runner(
        self,
    ) -> None:
        """多次提问应复用同一个Runner。"""

        runner = Mock()
        runner.run.return_value = (
            make_direct_result()
        )

        runner_factory = Mock(
            return_value=runner
        )

        service = GradioAgentService(
            database_path=(
                self.database_path
            ),
            runner_factory=runner_factory,
        )

        service.ask("第一个问题")
        service.ask("第二个问题")

        runner_factory.assert_called_once_with()

        self.assertEqual(
            runner.run.call_count,
            2,
        )

    def test_reset_recreates_runner(
        self,
    ) -> None:
        """reset后应重新创建Runner。"""

        first_runner = Mock()
        first_runner.run.return_value = (
            make_direct_result()
        )

        second_runner = Mock()
        second_runner.run.return_value = (
            make_direct_result()
        )

        runner_factory = Mock(
            side_effect=[
                first_runner,
                second_runner,
            ]
        )

        service = GradioAgentService(
            database_path=(
                self.database_path
            ),
            runner_factory=runner_factory,
        )

        service.ask("第一个问题")

        service.reset()

        service.ask("第二个问题")

        self.assertEqual(
            runner_factory.call_count,
            2,
        )

        first_runner.run.assert_called_once()
        second_runner.run.assert_called_once()

    def test_renders_direct_answer_trace(
        self,
    ) -> None:
        """没有工具调用时也应返回轨迹文字。"""

        runner = Mock()
        runner.run.return_value = (
            make_direct_result()
        )

        service = GradioAgentService(
            database_path=(
                self.database_path
            ),
            runner_factory=Mock(
                return_value=runner
            ),
        )

        result = service.ask(
            "普通问题"
        )

        self.assertIn(
            "模型没有调用工具",
            result.trace_markdown,
        )

    def test_renders_tool_failure(
        self,
    ) -> None:
        """工具失败状态应展示错误信息。"""

        runner = Mock()
        runner.run.return_value = (
            make_tool_result(
                success=False
            )
        )

        service = GradioAgentService(
            database_path=(
                self.database_path
            ),
            runner_factory=Mock(
                return_value=runner
            ),
        )

        result = service.ask(
            "错误参数测试"
        )

        self.assertIn(
            "validation_error",
            result.trace_markdown,
        )

        self.assertIn(
            "paper_id不能为空",
            result.trace_markdown,
        )

    def test_rejects_blank_question(
        self,
    ) -> None:
        """空问题不应创建Runner。"""

        runner_factory = Mock()

        service = GradioAgentService(
            database_path=(
                self.database_path
            ),
            runner_factory=runner_factory,
        )

        with self.assertRaisesRegex(
            ValueError,
            "请输入",
        ):
            service.ask("   ")

        runner_factory.assert_not_called()

    def test_requires_database(
        self,
    ) -> None:
        """数据库不存在时不应创建Runner。"""

        runner_factory = Mock()

        service = GradioAgentService(
            database_path=(
                self.directory
                / "missing.db"
            ),
            runner_factory=runner_factory,
        )

        with self.assertRaisesRegex(
            FileNotFoundError,
            "找不到论文数据库",
        ):
            service.ask(
                "有哪些论文？"
            )

        runner_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
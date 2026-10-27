import unittest
from unittest.mock import (
    Mock,
    patch,
)

import app
from agent_app import (
    GradioAgentResult,
)
from agent_runtime import (
    AgentProtocolError,
    AgentRunLimitError,
)
from models.deepseek_api import (
    ModelServiceError,
)
from pathlib import Path
from types import SimpleNamespace


class TestAgentWebCallback(
    unittest.TestCase
):
    """测试科研Agent网页回调。"""

    def test_returns_agent_result(
        self,
    ) -> None:
        """正常运行时应返回回答、轨迹和状态。"""

        service_result = GradioAgentResult(
            answer_markdown="## Agent回答",
            trace_markdown="## Agent执行轨迹",
            status_text="Agent运行完成。",
        )

        progress = Mock()

        with patch.object(
            app.agent_service,
            "ask",
            return_value=service_result,
        ) as ask_mock:
            result = app.ask_research_agent(
                "  知识库中有哪些论文？  ",
                progress=progress,
            )

        self.assertEqual(
            result,
            (
                "## Agent回答",
                "## Agent执行轨迹",
                "Agent运行完成。",
            ),
        )

        ask_mock.assert_called_once_with(
            "知识库中有哪些论文？"
        )

        self.assertEqual(
            progress.call_count,
            2,
        )

    def test_rejects_blank_question(
        self,
    ) -> None:
        """空问题不应调用Agent服务。"""

        progress = Mock()

        with patch.object(
            app.agent_service,
            "ask",
        ) as ask_mock:
            answer, trace, status = (
                app.ask_research_agent(
                    "   ",
                    progress=progress,
                )
            )

        ask_mock.assert_not_called()
        progress.assert_not_called()

        self.assertIn(
            "请输入问题",
            answer,
        )

        self.assertIn(
            "尚未运行Agent",
            trace,
        )

        self.assertEqual(
            status,
            "等待输入问题。",
        )

    def test_handles_model_service_error(
        self,
    ) -> None:
        """模型服务错误应转换为网页提示。"""

        with patch.object(
            app.agent_service,
            "ask",
            side_effect=ModelServiceError(
                "模型服务暂时不可用。"
            ),
        ):
            answer, trace, status = (
                app.ask_research_agent(
                    "测试问题",
                    progress=Mock(),
                )
            )

        self.assertIn(
            "模型服务调用失败",
            answer,
        )

        self.assertIn(
            "模型服务暂时不可用",
            answer,
        )

        self.assertIn(
            "模型服务调用失败",
            trace,
        )

        self.assertEqual(
            status,
            "Agent运行失败。",
        )

    def test_handles_run_limit_error(
        self,
    ) -> None:
        """Agent超过运行限制时应给出明确提示。"""

        with patch.object(
            app.agent_service,
            "ask",
            side_effect=AgentRunLimitError(
                "超过最大模型轮数。"
            ),
        ):
            answer, trace, status = (
                app.ask_research_agent(
                    "复杂研究任务",
                    progress=Mock(),
                )
            )

        self.assertIn(
            "Agent达到运行限制",
            answer,
        )

        self.assertIn(
            "超过最大模型轮数",
            answer,
        )

        self.assertIn(
            "未能在限制内完成",
            trace,
        )

        self.assertEqual(
            status,
            "Agent达到运行限制。",
        )

    def test_handles_protocol_error(
        self,
    ) -> None:
        """模型返回异常消息时应显示协议错误。"""

        with patch.object(
            app.agent_service,
            "ask",
            side_effect=AgentProtocolError(
                "工具调用缺少名称。"
            ),
        ):
            answer, trace, status = (
                app.ask_research_agent(
                    "测试协议错误",
                    progress=Mock(),
                )
            )

        self.assertIn(
            "Agent协议错误",
            answer,
        )

        self.assertIn(
            "工具调用缺少名称",
            answer,
        )

        self.assertIn(
            "异常的Agent消息",
            trace,
        )

        self.assertEqual(
            status,
            "Agent协议错误。",
        )

    def test_handles_missing_database(
        self,
    ) -> None:
        """数据库不存在时应提示先导入论文。"""

        with patch.object(
            app.agent_service,
            "ask",
            side_effect=FileNotFoundError(
                "找不到论文数据库。"
            ),
        ):
            answer, trace, status = (
                app.ask_research_agent(
                    "有哪些论文？",
                    progress=Mock(),
                )
            )

        self.assertIn(
            "找不到论文数据库",
            answer,
        )

        self.assertIn(
            "先向知识库导入论文",
            trace,
        )

        self.assertEqual(
            status,
            "知识库不可用。",
        )

    def test_handles_unexpected_error(
        self,
    ) -> None:
        """未知异常也不应导致网页直接崩溃。"""

        with patch.object(
            app.agent_service,
            "ask",
            side_effect=RuntimeError(
                "未知测试错误"
            ),
        ):
            answer, trace, status = (
                app.ask_research_agent(
                    "测试未知异常",
                    progress=Mock(),
                )
            )

        self.assertIn(
            "Agent运行失败",
            answer,
        )

        self.assertIn(
            "RuntimeError",
            answer,
        )

        self.assertIn(
            "未知测试错误",
            answer,
        )

        self.assertIn(
            "未成功完成",
            trace,
        )

        self.assertEqual(
            status,
            "Agent运行失败。",
        )


class TestAgentCacheReset(
    unittest.TestCase
):
    """测试论文库变化后的Agent缓存重置。"""

    def test_resets_agent_after_new_paper(
        self,
    ) -> None:
        """新论文入库后应清除Agent旧索引。"""

        ingest_result = SimpleNamespace(
            created=True,
            paper=SimpleNamespace(
                id="paper-001",
                title="混沌系统研究",
            ),
            sections=(
                object(),
                object(),
            ),
            chunks=(
                object(),
                object(),
                object(),
            ),
        )

        progress = Mock()

        with (
            patch.object(
                app.knowledge_service,
                "ingest_pdf",
                return_value=ingest_result,
            ) as ingest_mock,
            patch.object(
                app.agent_service,
                "reset",
            ) as reset_mock,
            patch.object(
                app,
                "load_knowledge_scope_choices",
                return_value=[
                    (
                        "混沌系统研究",
                        "paper-001",
                    ),
                ],
            ),
            patch.object(
                app,
                "load_knowledge_library_rows",
                return_value=[],
            ),
        ):
            result = app.ingest_paper_from_web(
                "paper.pdf",
                "混沌系统研究",
                "张三，李四",
                "2025",
                "10.1234/example",
                progress=progress,
            )

        ingest_mock.assert_called_once_with(
            Path("paper.pdf"),
            title="混沌系统研究",
            authors=(
                "张三",
                "李四",
            ),
            year=2025,
            doi="10.1234/example",
        )

        reset_mock.assert_called_once_with()

        self.assertIn(
            "新论文已成功入库",
            result[0],
        )

    def test_does_not_reset_for_duplicate_paper(
        self,
    ) -> None:
        """重复PDF没有改变知识库，不需要重置。"""

        ingest_result = SimpleNamespace(
            created=False,
            paper=SimpleNamespace(
                id="paper-001",
                title="混沌系统研究",
            ),
            sections=(),
            chunks=(),
        )

        with (
            patch.object(
                app.knowledge_service,
                "ingest_pdf",
                return_value=ingest_result,
            ),
            patch.object(
                app.agent_service,
                "reset",
            ) as reset_mock,
            patch.object(
                app,
                "load_knowledge_scope_choices",
                return_value=[],
            ),
            patch.object(
                app,
                "load_knowledge_library_rows",
                return_value=[],
            ),
        ):
            result = app.ingest_paper_from_web(
                "paper.pdf",
                "",
                "",
                "",
                "",
                progress=Mock(),
            )

        reset_mock.assert_not_called()

        self.assertIn(
            "已经存在于知识库",
            result[0],
        )

    def test_resets_agent_after_deletion(
        self,
    ) -> None:
        """删除论文后应清除Agent旧索引。"""

        deletion_result = SimpleNamespace(
            paper=SimpleNamespace(
                id="paper-001",
                title="混沌系统研究",
            ),
            stored_file_deleted=False,
            file_warning=None,
        )

        with (
            patch.object(
                app.knowledge_service,
                "delete_paper",
                return_value=deletion_result,
            ) as delete_mock,
            patch.object(
                app.agent_service,
                "reset",
            ) as reset_mock,
            patch.object(
                app,
                "load_knowledge_scope_choices",
                return_value=[],
            ),
            patch.object(
                app,
                "load_knowledge_library_rows",
                return_value=[],
            ),
        ):
            result = app.delete_paper_from_web(
                "paper-001",
                True,
            )

        delete_mock.assert_called_once_with(
            "paper-001"
        )

        reset_mock.assert_called_once_with()

        self.assertIn(
            "论文已从知识库删除",
            result[0],
        )

    def test_does_not_delete_without_confirmation(
        self,
    ) -> None:
        """没有确认时不得删除论文或重置缓存。"""

        with (
            patch.object(
                app.knowledge_service,
                "delete_paper",
            ) as delete_mock,
            patch.object(
                app.agent_service,
                "reset",
            ) as reset_mock,
        ):
            result = app.delete_paper_from_web(
                "paper-001",
                False,
            )

        delete_mock.assert_not_called()
        reset_mock.assert_not_called()

        self.assertIn(
            "请先确认删除",
            result[0],
        )


if __name__ == "__main__":
    unittest.main()
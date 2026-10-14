import unittest
from unittest.mock import (
    MagicMock,
    patch,
)

from rag.answer_service import (
    RagAnswer,
)
from rag.context_builder import (
    ContextEvidence,
    RagContext,
)
from rag.gradio_service import (
    GradioRagService,
    RetrieverRuntime,
)
from pathlib import Path
from tempfile import TemporaryDirectory

from domain.paper import (
    Paper,
)


class TestGradioRagService(
    unittest.TestCase
):
    """测试Gradio使用的RAG服务。"""

    @staticmethod
    def create_answer() -> RagAnswer:
        """创建测试使用的RAG回答。"""

        evidence = ContextEvidence(
            citation_number=1,
            paper_id="paper-001",
            paper_title="混沌系统研究",
            chunk_id="chunk-001",
            section_path=(
                "2 动力学分析",
                "2.1 Lyapunov指数",
            ),
            page_start=3,
            page_end=4,
            score=0.8765,
            content=(
                "论文使用最大Lyapunov指数"
                "分析系统的混沌特性。"
            ),
        )

        return RagAnswer(
            question="论文使用了什么方法？",
            answer=(
                "论文使用最大Lyapunov指数"
                "分析混沌特性。[证据1]"
            ),
            context=RagContext(
                evidences=(evidence,)
            ),
        )

    def test_lists_paper_choices(
        self,
    ) -> None:
        """论文应转换成显示文字和ID组成的选项。"""

        with TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "papers.db"
            )

            # 创建空文件，使数据库路径检查通过。
            database_path.touch()

            service = GradioRagService(
                database_path=database_path
            )

            first_paper = Paper(
                id="paper-001",
                title="分数阶混沌系统研究",
                authors=(
                    "作者甲",
                    "作者乙",
                ),
                year=2026,
                file_name="first.pdf",
                file_path="papers/first.pdf",
                file_hash="hash-001",
            )

            second_paper = Paper(
                id="paper-002",
                title="忆阻神经网络研究",
                authors=(),
                file_name="second.pdf",
                file_path="papers/second.pdf",
                file_hash="hash-002",
            )

            fake_connection = MagicMock()

            with patch(
                "rag.gradio_service.create_connection",
                return_value=fake_connection,
            ), patch(
                "rag.gradio_service.PaperRepository",
            ) as repository_class:
                repository = (
                    repository_class.return_value
                )

                repository.list_all.return_value = [
                    first_paper,
                    second_paper,
                ]

                choices = (
                    service.list_paper_choices()
                )

        self.assertEqual(
            choices,
            [
                (
                    (
                        "分数阶混沌系统研究"
                        "（2026；作者甲、作者乙）"
                    ),
                    "paper-001",
                ),
                (
                    "忆阻神经网络研究",
                    "paper-002",
                ),
            ],
        )

        repository.list_all.assert_called_once_with()
        fake_connection.close.assert_called_once_with()

    def test_paper_choices_require_database(
        self,
    ) -> None:
        """数据库不存在时应给出明确错误。"""

        with TemporaryDirectory() as directory:
            missing_path = (
                Path(directory)
                / "missing.db"
            )

            service = GradioRagService(
                database_path=missing_path
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "找不到论文数据库",
            ):
                service.list_paper_choices()

    def test_rejects_blank_question(
        self,
    ) -> None:
        """空问题应当被拒绝。"""

        service = GradioRagService()

        with self.assertRaisesRegex(
            ValueError,
            "请输入",
        ):
            service.ask("   ")

    def test_rejects_invalid_top_k(
        self,
    ) -> None:
        """Top-K必须大于0。"""

        service = GradioRagService()

        with self.assertRaisesRegex(
            ValueError,
            "Top-K必须大于0",
        ):
            service.ask(
                "什么是混沌？",
                top_k=0,
            )

    def test_renders_evidence_markdown(
        self,
    ) -> None:
        """证据信息应转换为Markdown。"""

        rag_answer = self.create_answer()

        markdown = (
            GradioRagService
            ._render_evidence(rag_answer)
        )

        self.assertIn(
            "[证据1]",
            markdown,
        )

        self.assertIn(
            "混沌系统研究",
            markdown,
        )

        self.assertIn(
            "2 动力学分析 > 2.1 Lyapunov指数",
            markdown,
        )

        self.assertIn(
            "第3—4页",
            markdown,
        )

        self.assertIn(
            "0.8765",
            markdown,
        )

        self.assertIn(
            "最大Lyapunov指数",
            markdown,
        )

    def test_renders_empty_evidence(
        self,
    ) -> None:
        """没有证据时应显示提示。"""

        rag_answer = RagAnswer(
            question="未知问题",
            answer=(
                "当前论文库中没有足够信息"
                "回答这个问题。"
            ),
            context=RagContext(
                evidences=()
            ),
        )

        markdown = (
            GradioRagService
            ._render_evidence(rag_answer)
        )

        self.assertIn(
            "没有检索到可用证据",
            markdown,
        )

    def test_ask_returns_gradio_result(
        self,
    ) -> None:
        """ask应返回回答、证据和状态。"""

        service = GradioRagService(
            database_path="test.db"
        )


        fake_retriever = MagicMock()
        fake_runtime = RetrieverRuntime(
            retriever=fake_retriever,
            paper_id="paper-001",
            indexed_count=10,
            used_cache=True,
        )
        fake_connection = MagicMock()
        fake_answer = self.create_answer()

        with patch.object(
            service,
            "_get_or_build_retriever",
            return_value=fake_runtime,
        ) as get_retriever, patch(
            "rag.gradio_service.create_connection",
            return_value=fake_connection,
        ), patch(
            "rag.gradio_service.AliyunChatClient",
        ), patch(
            "rag.gradio_service.RagAnswerService",
        ) as answer_service_class:
            answer_service = (
                answer_service_class.return_value
            )

            answer_service.answer.return_value = (
                fake_answer
            )

            result = service.ask(
                "论文使用了什么方法？",
                top_k=3,
                paper_id="paper-001",
            )

        answer_service.answer.assert_called_once_with(
            "论文使用了什么方法？",
            top_k=3,
        )

        fake_connection.close.assert_called_once_with()

        self.assertIn(
            "最大Lyapunov指数",
            result.answer_markdown,
        )

        self.assertIn(
            "[证据1]",
            result.evidence_markdown,
        )

        self.assertIn(
            "10 个Chunk",
            result.status_text,
        )

        self.assertIn(
            "1 条证据",
            result.status_text,
        )

        self.assertIn(
            "已从磁盘缓存加载",
            result.status_text,
        )

        get_retriever.assert_called_once_with(
            paper_id="paper-001"
        )

        self.assertIn(
            "检索范围：指定论文",
            result.status_text,
        )

    def test_reuses_retriever_for_same_scope(
        self,
    ) -> None:
        """相同论文范围应复用已经构建的索引。"""

        with TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "papers.db"
            )

            database_path.touch()

            service = GradioRagService(
                database_path=database_path
            )

            fake_encoder = MagicMock()
            fake_encoder.identifier = (
                "test-embedding-model"
            )

            service._encoder = fake_encoder

            fake_connection = MagicMock()
            fake_retriever = MagicMock()

            fake_retriever.build_index.return_value = 4
            fake_retriever.last_build_used_cache = True

            with patch(
                "rag.gradio_service.create_connection",
                return_value=fake_connection,
            ), patch(
                "rag.gradio_service.VectorCache.for_index",
            ), patch(
                "rag.gradio_service.SemanticRetriever",
                return_value=fake_retriever,
            ) as retriever_class:
                first_runtime = (
                    service
                    ._get_or_build_retriever(
                        paper_id="paper-001"
                    )
                )

                second_runtime = (
                    service
                    ._get_or_build_retriever(
                        paper_id="paper-001"
                    )
                )

        self.assertIs(
            first_runtime,
            second_runtime,
        )

        fake_retriever.build_index.assert_called_once_with(
            paper_id="paper-001"
        )

        retriever_class.assert_called_once()

        self.assertEqual(
            first_runtime.indexed_count,
            4,
        )

        self.assertTrue(
            first_runtime.used_cache
        )

if __name__ == "__main__":
    unittest.main()
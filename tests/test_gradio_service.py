import unittest
from unittest.mock import (
    MagicMock,
    patch,
    sentinel,
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

from database.connection import (
    create_connection,
)
from database.paper_repository import (
    PaperRepository,
)
from database.schema import (
    initialize_database,
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
            paper_ids=(
                "paper-001",
                "paper-002",
            ),
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
                paper_ids=[
                    "paper-002",
                    "paper-001",
                ],
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
            paper_ids=(
                "paper-001",
                "paper-002",
            )
        )

        self.assertIn(
            "检索范围：2 篇指定论文",
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
                        paper_ids=(
                            "paper-001",
                            "paper-002",
                        )
                    )
                )

                second_runtime = (
                    service
                    ._get_or_build_retriever(
                        paper_ids=(
                            "paper-001",
                            "paper-002",
                        )
                    )
                )

        self.assertIs(
            first_runtime,
            second_runtime,
        )

        fake_retriever.build_index.assert_called_once_with(
            paper_ids=(
                "paper-001",
                "paper-002",
            )
        )

        retriever_class.assert_called_once()

        self.assertEqual(
            first_runtime.indexed_count,
            4,
        )

        self.assertTrue(
            first_runtime.used_cache
        )

    def test_empty_selection_uses_all_papers(
        self,
    ) -> None:
        """空论文选择应表示整个知识库。"""

        service = GradioRagService(
            database_path="test.db"
        )

        fake_retriever = MagicMock()

        fake_runtime = RetrieverRuntime(
            retriever=fake_retriever,
            paper_ids=None,
            indexed_count=20,
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
                paper_ids=[],
            )

        get_retriever.assert_called_once_with(
            paper_ids=None
        )

        self.assertIn(
            "检索范围：整个论文库",
            result.status_text,
        )

    def test_ingests_uploaded_pdf(
        self,
    ) -> None:
        """网页上传文件应保存并进入入库管道。"""

        with TemporaryDirectory() as directory:
            root_directory = Path(directory)

            source_path = (
                root_directory
                / "uploaded.pdf"
            )

            source_path.write_bytes(
                b"test pdf content"
            )

            database_path = (
                root_directory
                / "data"
                / "papers.db"
            )

            storage_directory = (
                root_directory
                / "stored_papers"
            )

            service = GradioRagService(
                database_path=database_path,
                paper_storage_directory=(
                    storage_directory
                ),
            )

            service._retriever_runtimes[
                "all"
            ] = MagicMock()

            fake_connection = MagicMock()
            fake_pipeline = MagicMock()

            fake_result = MagicMock()
            fake_result.created = True

            fake_pipeline.ingest.return_value = (
                fake_result
            )

            with patch(
                (
                    "rag.gradio_service."
                    "calculate_file_sha256"
                ),
                return_value="hash-001",
            ), patch(
                (
                    "rag.gradio_service."
                    "create_connection"
                ),
                return_value=fake_connection,
            ), patch(
                (
                    "rag.gradio_service."
                    "initialize_database"
                ),
            ) as initialize_database_mock, patch(
                (
                    "rag.gradio_service."
                    "create_pdf_ingestion_pipeline"
                ),
                return_value=fake_pipeline,
            ):
                result = service.ingest_pdf(
                    source_path,
                    title="测试论文",
                    authors=(
                        "作者甲",
                        "作者乙",
                    ),
                    year=2026,
                    doi="10.0000/example",
                )

            stored_path = (
                storage_directory
                / "hash-001.pdf"
            )

            self.assertTrue(
                stored_path.is_file()
            )

            self.assertEqual(
                stored_path.read_bytes(),
                b"test pdf content",
            )

            initialize_database_mock.assert_called_once_with(
                fake_connection
            )

            fake_pipeline.ingest.assert_called_once_with(
                pdf_path=stored_path,
                title="测试论文",
                authors=(
                    "作者甲",
                    "作者乙",
                ),
                year=2026,
                doi="10.0000/example",
            )

            fake_connection.close.assert_called_once_with()

            self.assertIs(
                result,
                fake_result,
            )

            self.assertEqual(
                service._retriever_runtimes,
                {},
            )

    def test_rejects_non_pdf_upload(
        self,
    ) -> None:
        """网页入库服务只接受PDF文件。"""

        with TemporaryDirectory() as directory:
            file_path = (
                Path(directory)
                / "paper.txt"
            )

            file_path.write_text(
                "not a pdf",
                encoding="utf-8",
            )

            service = GradioRagService()

            with self.assertRaisesRegex(
                ValueError,
                "只能导入PDF",
            ):
                service.ingest_pdf(
                    file_path
                )

    def test_deletes_paper_and_managed_pdf(
        self,
    ) -> None:
        """删除论文时应清理受管PDF和内存索引。"""

        with TemporaryDirectory() as directory:
            root_directory = Path(directory)

            database_path = (
                root_directory
                / "papers.db"
            )

            storage_directory = (
                root_directory
                / "papers"
            )

            storage_directory.mkdir()

            pdf_path = (
                storage_directory
                / "hash-001.pdf"
            )

            pdf_path.write_bytes(
                b"pdf content"
            )

            connection = create_connection(
                database_path
            )

            initialize_database(
                connection
            )

            repository = PaperRepository(
                connection
            )

            repository.add(
                Paper(
                    id="paper-001",
                    title="测试论文",
                    authors=(),
                    file_name="paper.pdf",
                    file_path=str(pdf_path),
                    file_hash="hash-001",
                )
            )

            connection.close()

            service = GradioRagService(
                database_path=database_path,
                paper_storage_directory=(
                    storage_directory
                ),
            )

            service._retriever_runtimes[
                "all"
            ] = MagicMock()

            result = service.delete_paper(
                "paper-001"
            )

            self.assertEqual(
                result.paper.id,
                "paper-001",
            )

            self.assertTrue(
                result.stored_file_deleted
            )

            self.assertFalse(
                pdf_path.exists()
            )

            self.assertEqual(
                service._retriever_runtimes,
                {},
            )

            verification_connection = (
                create_connection(
                    database_path
                )
            )

            try:
                remaining_paper = (
                    PaperRepository(
                        verification_connection
                    )
                    .get_by_id(
                        "paper-001"
                    )
                )
            finally:
                verification_connection.close()

            self.assertIsNone(
                remaining_paper
            )

    def test_rejects_deleting_missing_paper(
        self,
    ) -> None:
        """删除不存在的论文应给出错误。"""

        with TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "papers.db"
            )

            connection = create_connection(
                database_path
            )

            initialize_database(
                connection
            )

            connection.close()

            service = GradioRagService(
                database_path=database_path
            )

            with self.assertRaisesRegex(
                ValueError,
                "不存在该论文",
            ):
                service.delete_paper(
                    "missing-paper"
                )

if __name__ == "__main__":
    unittest.main()
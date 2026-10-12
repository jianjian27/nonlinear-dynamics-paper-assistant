import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import (
    Mock,
    patch,
)

from rag.answer_service import RagAnswer
from rag.context_builder import (
    ContextEvidence,
    RagContext,
)
from scripts.ask_papers import main


def make_rag_answer() -> RagAnswer:
    evidence = ContextEvidence(
        citation_number=1,
        paper_id="paper-1",
        paper_title="混沌系统研究",
        chunk_id="chunk-1",
        section_path=(
            "动力学分析",
            "Lyapunov指数",
        ),
        page_start=3,
        page_end=4,
        score=0.91,
        content=(
            "系统的最大Lyapunov指数大于零，"
            "表明系统处于混沌状态。"
        ),
    )

    context = RagContext(
        evidences=(evidence,)
    )

    return RagAnswer(
        question="系统使用了什么分析方法？",
        answer=(
            "论文计算了最大Lyapunov指数。"
            "[证据1]"
        ),
        context=context,
    )


class TestAskPapersCommand(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.temporary_directory = (
            TemporaryDirectory()
        )

        self.database_path = (
            Path(
                self.temporary_directory.name
            )
            / "papers.db"
        )

        self.database_path.touch()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @patch(
        "scripts.ask_papers.RagAnswerService"
    )
    @patch(
        "scripts.ask_papers.AliyunChatClient"
    )
    @patch(
        "scripts.ask_papers.SemanticRetriever"
    )
    @patch(
        "scripts.ask_papers."
        "SentenceTransformerEncoder"
    )
    @patch(
        "scripts.ask_papers.create_connection"
    )
    def test_runs_complete_rag_command(
        self,
        mock_create_connection: Mock,
        mock_encoder_class: Mock,
        mock_retriever_class: Mock,
        mock_chat_client_class: Mock,
        mock_answer_service_class: Mock,
    ) -> None:
        connection = Mock()

        mock_create_connection.return_value = (
            connection
        )

        encoder = (
            mock_encoder_class.return_value
        )

        retriever = (
            mock_retriever_class.return_value
        )

        retriever.build_index.return_value = 10

        chat_client = (
            mock_chat_client_class.return_value
        )

        answer_service = (
            mock_answer_service_class
            .return_value
        )

        answer_service.answer.return_value = (
            make_rag_answer()
        )

        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "系统使用了什么分析方法？",
                    "--database",
                    str(self.database_path),
                    "--top-k",
                    "3",
                    "--device",
                    "cpu",
                ]
            )

        self.assertEqual(exit_code, 0)

        mock_encoder_class.assert_called_once_with(
            model_name=mock_encoder_class
            .call_args
            .kwargs["model_name"],
            device="cpu",
        )

        mock_retriever_class.assert_called_once()

        retriever.build_index.assert_called_once_with(
            paper_id=None
        )

        mock_chat_client_class.assert_called_once()

        mock_answer_service_class.assert_called_once()

        answer_service.answer.assert_called_once_with(
            "系统使用了什么分析方法？",
            top_k=3,
        )

        self.assertIn(
            "=== RAG回答 ===",
            output.getvalue(),
        )

        self.assertIn(
            "[证据1]",
            output.getvalue(),
        )

        self.assertIn(
            "混沌系统研究",
            output.getvalue(),
        )

        connection.close.assert_called_once()

        self.assertIsNotNone(encoder)
        self.assertIsNotNone(chat_client)

    @patch(
        "scripts.ask_papers.SemanticRetriever"
    )
    @patch(
        "scripts.ask_papers."
        "SentenceTransformerEncoder"
    )
    @patch(
        "scripts.ask_papers.create_connection"
    )
    def test_rejects_database_without_chunks(
        self,
        mock_create_connection: Mock,
        mock_encoder_class: Mock,
        mock_retriever_class: Mock,
    ) -> None:
        connection = Mock()

        mock_create_connection.return_value = (
            connection
        )

        retriever = (
            mock_retriever_class.return_value
        )

        retriever.build_index.return_value = 0

        exit_code = main(
            [
                "论文结论是什么？",
                "--database",
                str(self.database_path),
            ]
        )

        self.assertEqual(exit_code, 1)

        connection.close.assert_called_once()

    def test_rejects_missing_database(
        self,
    ) -> None:
        missing_database = (
            Path(
                self.temporary_directory.name
            )
            / "missing.db"
        )

        exit_code = main(
            [
                "论文结论是什么？",
                "--database",
                str(missing_database),
            ]
        )

        self.assertEqual(exit_code, 1)

    def test_rejects_invalid_top_k(
        self,
    ) -> None:
        exit_code = main(
            [
                "论文结论是什么？",
                "--database",
                str(self.database_path),
                "--top-k",
                "0",
            ]
        )

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
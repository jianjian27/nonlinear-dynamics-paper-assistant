import unittest
from unittest.mock import Mock

from database.paper_repository import (
    PaperRepository,
)
from domain.chunk import Chunk
from domain.paper import Paper
from embedding.vector_index import (
    VectorSearchResult,
)
from models.deepseek_api import (
    AliyunChatClient,
)
from rag.answer_service import (
    INSUFFICIENT_INFORMATION_ANSWER,
    RagAnswerService,
)
from rag.context_builder import (
    ContextBuilder,
)
from rag.prompt import (
    RagPromptBuilder,
)
from retrieval.semantic_retriever import (
    SemanticRetriever,
)


def make_result(
    index: int,
    content: str,
    *,
    paper_id: str = "paper-1",
    score: float = 0.9,
) -> VectorSearchResult:
    chunk = Chunk(
        id=f"chunk-{index}",
        paper_id=paper_id,
        chunk_index=index,
        section_title=f"章节{index}",
        section_path=(
            "稳定性分析",
            f"章节{index}",
        ),
        content=content,
        page_start=index + 1,
        page_end=index + 1,
        pages=(index + 1,),
        source_blocks=(f"{index + 1}:1",),
        content_hash=f"hash-{index}",
    )

    return VectorSearchResult(
        chunk=chunk,
        score=score,
    )


def make_paper() -> Paper:
    return Paper(
        id="paper-1",
        title="混沌系统稳定性研究",
        authors=("张三",),
        file_name="paper.pdf",
        file_path="papers/paper.pdf",
        file_hash="file-hash",
        year=2025,
    )


class TestRagAnswerService(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.retriever = Mock(
            spec=SemanticRetriever
        )

        self.paper_repository = Mock(
            spec=PaperRepository
        )

        self.chat_client = Mock(
            spec=AliyunChatClient
        )

        self.service = RagAnswerService(
            retriever=self.retriever,
            paper_repository=(
                self.paper_repository
            ),
            context_builder=ContextBuilder(),
            prompt_builder=RagPromptBuilder(),
            chat_client=self.chat_client,
        )

    def test_answers_with_retrieved_evidence(
        self,
    ) -> None:
        result = make_result(
            0,
            "作者构造李雅普诺夫函数，"
            "证明同步误差渐近收敛。",
        )

        self.retriever.search.return_value = [
            result
        ]

        (
            self.paper_repository
            .get_by_id
            .return_value
        ) = make_paper()

        self.chat_client.ask.return_value = (
            "论文通过李雅普诺夫函数"
            "证明系统稳定。[证据1]"
        )

        rag_answer = self.service.answer(
            "  论文如何证明系统稳定？  ",
            top_k=3,
        )

        self.retriever.search.assert_called_once_with(
            "论文如何证明系统稳定？",
            top_k=3,
        )

        self.assertEqual(
            rag_answer.question,
            "论文如何证明系统稳定？",
        )

        self.assertEqual(
            rag_answer.retrieved_count,
            1,
        )

        self.assertIn(
            "[证据1]",
            rag_answer.answer,
        )

        self.assertEqual(
            rag_answer
            .context
            .evidences[0]
            .paper_title,
            "混沌系统稳定性研究",
        )

        sent_messages = (
            self.chat_client
            .ask
            .call_args
            .args[0]
        )

        self.assertEqual(
            sent_messages[0]["role"],
            "system",
        )

        self.assertEqual(
            sent_messages[1]["role"],
            "user",
        )

        self.assertIn(
            "李雅普诺夫函数",
            sent_messages[1]["content"],
        )

    def test_queries_each_paper_only_once(
        self,
    ) -> None:
        self.retriever.search.return_value = [
            make_result(
                0,
                "第一条证据",
            ),
            make_result(
                1,
                "第二条证据",
            ),
        ]

        (
            self.paper_repository
            .get_by_id
            .return_value
        ) = make_paper()

        self.chat_client.ask.return_value = (
            "综合回答"
        )

        self.service.answer(
            "论文有哪些结论？"
        )

        (
            self.paper_repository
            .get_by_id
            .assert_called_once_with(
                "paper-1"
            )
        )

    def test_returns_fixed_answer_when_no_results(
        self,
    ) -> None:
        self.retriever.search.return_value = []

        rag_answer = self.service.answer(
            "数据库中没有涉及的问题"
        )

        self.assertEqual(
            rag_answer.answer,
            INSUFFICIENT_INFORMATION_ANSWER,
        )

        self.assertEqual(
            rag_answer.retrieved_count,
            0,
        )

        self.assertTrue(
            rag_answer.context.is_empty
        )

        self.chat_client.ask.assert_not_called()

        (
            self.paper_repository
            .get_by_id
            .assert_not_called()
        )

    def test_uses_paper_id_when_paper_missing(
        self,
    ) -> None:
        self.retriever.search.return_value = [
            make_result(
                0,
                "论文证据",
                paper_id="missing-paper",
            )
        ]

        (
            self.paper_repository
            .get_by_id
            .return_value
        ) = None

        self.chat_client.ask.return_value = (
            "模型回答"
        )

        rag_answer = self.service.answer(
            "论文内容是什么？"
        )

        evidence = (
            rag_answer
            .context
            .evidences[0]
        )

        self.assertEqual(
            evidence.paper_title,
            "missing-paper",
        )

    def test_rejects_blank_question(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "问题不能为空",
        ):
            self.service.answer("   ")

        self.retriever.search.assert_not_called()

    def test_rejects_invalid_top_k(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "top_k",
        ):
            self.service.answer(
                "论文结论是什么？",
                top_k=0,
            )

        self.retriever.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
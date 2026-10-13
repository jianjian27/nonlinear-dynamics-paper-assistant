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

        service._indexed_count = 10
        service._used_cache = True

        fake_retriever = MagicMock()
        fake_connection = MagicMock()
        fake_answer = self.create_answer()

        with patch.object(
            service,
            "_get_or_build_retriever",
            return_value=fake_retriever,
        ), patch(
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


if __name__ == "__main__":
    unittest.main()
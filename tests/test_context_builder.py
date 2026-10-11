import unittest

from domain.chunk import Chunk
from embedding.vector_index import (
    VectorSearchResult,
)
from rag.context_builder import (
    ContextBuilder,
)


def make_result(
    index: int,
    content: str,
    *,
    paper_id: str = "paper-1",
    score: float = 0.8,
) -> VectorSearchResult:
    chunk = Chunk(
        id=f"chunk-{index}",
        paper_id=paper_id,
        chunk_index=index,
        section_title=f"章节{index}",
        section_path=(
            "数学模型",
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


class TestContextBuilder(
    unittest.TestCase
):
    def test_builds_traceable_context(
        self,
    ) -> None:
        results = [
            make_result(
                0,
                "使用李雅普诺夫函数证明系统稳定。",
                score=0.91,
            ),
            make_result(
                1,
                "通过数值仿真验证理论结果。",
                score=0.82,
            ),
        ]

        builder = ContextBuilder()

        context = builder.build(
            results,
            paper_titles={
                "paper-1": "混沌系统稳定性研究",
            },
        )

        self.assertEqual(
            len(context.evidences),
            2,
        )

        self.assertFalse(context.is_empty)

        self.assertIn(
            "[证据1]",
            context.text,
        )

        self.assertIn(
            "论文：混沌系统稳定性研究",
            context.text,
        )

        self.assertIn(
            "章节：数学模型 > 章节0",
            context.text,
        )

        self.assertIn(
            "页码：第1页",
            context.text,
        )

        self.assertIn(
            "相似度：0.9100",
            context.text,
        )

    def test_uses_paper_id_when_title_missing(
        self,
    ) -> None:
        builder = ContextBuilder()

        context = builder.build(
            [
                make_result(
                    0,
                    "论文原文",
                    paper_id="unknown-paper",
                )
            ]
        )

        self.assertIn(
            "论文：unknown-paper",
            context.text,
        )

    def test_limits_each_chunk_and_total_content(
        self,
    ) -> None:
        builder = ContextBuilder(
            max_total_content_chars=10,
            max_chunk_chars=6,
        )

        context = builder.build(
            [
                make_result(
                    0,
                    "abcdefgh",
                ),
                make_result(
                    1,
                    "ijklmnop",
                ),
            ]
        )

        contents = [
            evidence.content
            for evidence in context.evidences
        ]

        self.assertEqual(
            contents,
            [
                "abcde…",
                "ijk…",
            ],
        )

        self.assertEqual(
            sum(map(len, contents)),
            10,
        )

    def test_empty_results_create_empty_context(
        self,
    ) -> None:
        builder = ContextBuilder()

        context = builder.build([])

        self.assertTrue(context.is_empty)
        self.assertEqual(context.text, "")

    def test_rejects_invalid_limits(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "总字符数",
        ):
            ContextBuilder(
                max_total_content_chars=0
            )

        with self.assertRaisesRegex(
            ValueError,
            "单个Chunk",
        ):
            ContextBuilder(
                max_chunk_chars=0
            )


if __name__ == "__main__":
    unittest.main()
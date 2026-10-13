import unittest

from domain.chunk import Chunk
from retrieval.content_filter import (
    is_reference_chunk,
    is_reference_section_title,
    normalize_text,
    query_requests_references,
)


def make_chunk(
    section_title: str,
    section_path: tuple[str, ...],
) -> Chunk:
    return Chunk(
        id="chunk-1",
        paper_id="paper-1",
        chunk_index=0,
        section_title=section_title,
        section_path=section_path,
        content="测试内容",
        page_start=1,
        page_end=1,
        pages=(1,),
        source_blocks=("1:1",),
        content_hash="hash-1",
    )


class TestContentFilter(
    unittest.TestCase
):
    def test_normalizes_title_text(
        self,
    ) -> None:
        self.assertEqual(
            normalize_text(
                "  References： "
            ),
            "references",
        )

    def test_recognizes_chinese_reference_title(
        self,
    ) -> None:
        self.assertTrue(
            is_reference_section_title(
                "参考文献："
            )
        )

    def test_recognizes_english_reference_title(
        self,
    ) -> None:
        self.assertTrue(
            is_reference_section_title(
                "REFERENCES"
            )
        )

    def test_rejects_normal_section_title(
        self,
    ) -> None:
        self.assertFalse(
            is_reference_section_title(
                "稳定性分析"
            )
        )

    def test_recognizes_reference_chunk_by_title(
        self,
    ) -> None:
        chunk = make_chunk(
            "参考文献：",
            ("参考文献：",),
        )

        self.assertTrue(
            is_reference_chunk(chunk)
        )

    def test_recognizes_reference_chunk_by_path(
        self,
    ) -> None:
        chunk = make_chunk(
            "文献条目",
            (
                "论文附录",
                "Bibliography",
                "文献条目",
            ),
        )

        self.assertTrue(
            is_reference_chunk(chunk)
        )

    def test_rejects_normal_content_chunk(
        self,
    ) -> None:
        chunk = make_chunk(
            "Lyapunov指数分析",
            (
                "动力学分析",
                "Lyapunov指数分析",
            ),
        )

        self.assertFalse(
            is_reference_chunk(chunk)
        )

    def test_detects_reference_query_intent(
        self,
    ) -> None:
        self.assertTrue(
            query_requests_references(
                "这篇论文引用了哪些参考文献？"
            )
        )

        self.assertTrue(
            query_requests_references(
                "Please list the references."
            )
        )

    def test_rejects_normal_research_query(
        self,
    ) -> None:
        self.assertFalse(
            query_requests_references(
                "论文使用了什么稳定性分析方法？"
            )
        )


if __name__ == "__main__":
    unittest.main()
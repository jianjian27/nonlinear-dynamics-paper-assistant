import unittest

from ingestion.chunk_builder import build_chunks
from utils.pdf_structure import Section, TextBlock, TextLine


def make_block(page: int, block_id: int, text: str) -> TextBlock:
    line = TextLine(
        page=page,
        block_id=block_id,
        text=text,
        bbox=(0.0, 0.0, 100.0, 10.0),
        font_size=10.0,
        bold=False,
        centered=False,
        column="single",
    )
    return TextBlock(
        page=page,
        block_id=block_id,
        bbox=line.bbox,
        lines=[line],
        column="single",
    )


class ChunkBuilderTests(unittest.TestCase):
    def test_preserves_section_path_and_does_not_duplicate_children(self):
        child = Section(
            title="1.1 子章节",
            level=2,
            blocks=[make_block(2, 1, "子章节正文")],
        )
        parent = Section(
            title="1 父章节",
            level=1,
            blocks=[make_block(1, 1, "父章节正文")],
            children=[child],
        )

        chunks = build_chunks(
            [parent],
            "paper-1",
            max_chars=100,
            overlap_chars=0,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].content, "父章节正文")
        self.assertEqual(chunks[0].section_path, ("1 父章节",))
        self.assertEqual(chunks[1].content, "子章节正文")
        self.assertEqual(chunks[1].section_path, ("1 父章节", "1.1 子章节"))

    def test_tracks_exact_pages_and_source_blocks(self):
        section = Section(
            title="2 模型",
            level=1,
            blocks=[
                make_block(3, 7, "第一段"),
                make_block(4, 2, "第二段"),
            ],
        )

        chunk = build_chunks(
            [section],
            "paper-2",
            max_chars=100,
            overlap_chars=0,
        )[0]

        self.assertEqual(chunk.pages, (3, 4))
        self.assertEqual(chunk.page_start, 3)
        self.assertEqual(chunk.page_end, 4)
        self.assertEqual(chunk.source_blocks, ("3:7", "4:2"))

    def test_splits_on_block_boundaries_and_keeps_small_overlap(self):
        section = Section(
            title="3 实验",
            level=1,
            blocks=[
                make_block(5, 1, "A" * 20),
                make_block(5, 2, "B" * 20),
                make_block(6, 1, "C" * 20),
            ],
        )

        chunks = build_chunks(
            [section],
            "paper-3",
            max_chars=45,
            overlap_chars=20,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_blocks, ("5:1", "5:2"))
        self.assertEqual(chunks[1].source_blocks, ("5:2", "6:1"))
        self.assertLessEqual(len(chunks[0].content), 45)
        self.assertLessEqual(len(chunks[1].content), 45)

    def test_ids_are_deterministic(self):
        section = Section(
            title="结论",
            level=1,
            blocks=[make_block(7, 1, "研究结论")],
        )

        first = build_chunks([section], "paper-4")
        second = build_chunks([section], "paper-4")

        self.assertEqual(first[0].id, second[0].id)
        self.assertEqual(first[0].content_hash, second[0].content_hash)

    def test_rejects_invalid_size_configuration(self):
        with self.assertRaises(ValueError):
            build_chunks([], "paper-5", max_chars=100, overlap_chars=100)


if __name__ == "__main__":
    unittest.main()
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from database.chunk_repository import ChunkRepository
from database.connection import create_connection
from database.paper_repository import PaperRepository
from database.schema import initialize_database
from database.section_repository import SectionRepository
from domain.paper import Paper
from ingestion.chunk_builder import build_chunks
from utils.pdf_structure import (
    Section,
    TextBlock,
    TextLine,
)


def make_block(
    page: int,
    block_id: int,
    text: str,
) -> TextBlock:
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


class TestChunkRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()

        database_path = (
            Path(self.temporary_directory.name)
            / "test.db"
        )

        self.connection = create_connection(
            database_path
        )
        initialize_database(self.connection)

        PaperRepository(
            self.connection
        ).add(
            Paper(
                id="paper-1",
                title="测试论文",
                authors=("渐渐",),
                file_name="paper.pdf",
                file_path="papers/paper.pdf",
                file_hash="file-hash-1",
            )
        )

        self.section_repository = (
            SectionRepository(
                self.connection
            )
        )

        self.chunk_repository = ChunkRepository(
            self.connection
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.temporary_directory.cleanup()

    @staticmethod
    def create_section_tree() -> list[Section]:
        child = Section(
            title="1.1 系统模型",
            level=2,
            blocks=[
                make_block(
                    page=2,
                    block_id=1,
                    text="系统模型正文",
                )
            ],
        )

        parent = Section(
            title="1 数学模型",
            level=1,
            blocks=[
                make_block(
                    page=1,
                    block_id=1,
                    text="数学模型正文",
                )
            ],
            children=[child],
        )

        conclusion = Section(
            title="2 结论",
            level=1,
            blocks=[
                make_block(
                    page=3,
                    block_id=1,
                    text="研究结论正文",
                )
            ],
        )

        return [parent, conclusion]

    def test_adds_and_reads_chunks(self) -> None:
        section_tree = self.create_section_tree()

        section_records = (
            self.section_repository.add_tree(
                paper_id="paper-1",
                root_sections=section_tree,
            )
        )

        original_chunks = build_chunks(
            root_sections=section_tree,
            paper_id="paper-1",
            max_chars=100,
            overlap_chars=0,
        )

        self.chunk_repository.add_many(
            chunks=original_chunks,
            sections=section_records,
        )

        saved_chunks = (
            self.chunk_repository.list_by_paper(
                "paper-1"
            )
        )

        self.assertEqual(
            saved_chunks,
            original_chunks,
        )

    def test_links_chunks_to_sections(
        self,
    ) -> None:
        section_tree = self.create_section_tree()

        section_records = (
            self.section_repository.add_tree(
                paper_id="paper-1",
                root_sections=section_tree,
            )
        )

        chunks = build_chunks(
            root_sections=section_tree,
            paper_id="paper-1",
            max_chars=100,
            overlap_chars=0,
        )

        self.chunk_repository.add_many(
            chunks=chunks,
            sections=section_records,
        )

        rows = self.connection.execute(
            """
            SELECT
                chunks.section_title,
                chunks.section_id,
                sections.title
            FROM chunks
            JOIN sections
                ON chunks.section_id = sections.id
            ORDER BY chunks.chunk_index
            """
        ).fetchall()

        self.assertEqual(len(rows), 3)

        for row in rows:
            self.assertEqual(
                row["section_title"],
                row["title"],
            )

            self.assertIsNotNone(
                row["section_id"]
            )

    def test_gets_chunk_by_id(self) -> None:
        section_tree = self.create_section_tree()

        section_records = (
            self.section_repository.add_tree(
                paper_id="paper-1",
                root_sections=section_tree,
            )
        )

        chunks = build_chunks(
            root_sections=section_tree,
            paper_id="paper-1",
            max_chars=100,
            overlap_chars=0,
        )

        self.chunk_repository.add_many(
            chunks=chunks,
            sections=section_records,
        )

        saved_chunk = (
            self.chunk_repository.get_by_id(
                chunks[1].id
            )
        )

        self.assertIsNotNone(saved_chunk)
        self.assertEqual(
            saved_chunk,
            chunks[1],
        )

    def test_rejects_chunk_without_saved_section(
        self,
    ) -> None:
        saved_section_tree = [
            Section(
                title="已保存章节",
                level=1,
                blocks=[
                    make_block(
                        page=1,
                        block_id=1,
                        text="已保存正文",
                    )
                ],
            )
        ]

        section_records = (
            self.section_repository.add_tree(
                paper_id="paper-1",
                root_sections=saved_section_tree,
            )
        )

        unsaved_section_tree = [
            Section(
                title="未保存章节",
                level=1,
                blocks=[
                    make_block(
                        page=2,
                        block_id=1,
                        text="未保存正文",
                    )
                ],
            )
        ]

        chunks = build_chunks(
            root_sections=unsaved_section_tree,
            paper_id="paper-1",
            max_chars=100,
            overlap_chars=0,
        )

        with self.assertRaises(ValueError):
            self.chunk_repository.add_many(
                chunks=chunks,
                sections=section_records,
            )

        row = self.connection.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()

        self.assertEqual(row[0], 0)

    def test_lists_all_chunks(self) -> None:
        section_tree = self.create_section_tree()

        section_records = (
            self.section_repository.add_tree(
                paper_id="paper-1",
                root_sections=section_tree,
            )
        )

        chunks = build_chunks(
            root_sections=section_tree,
            paper_id="paper-1",
            max_chars=100,
            overlap_chars=0,
        )

        self.chunk_repository.add_many(
            chunks=chunks,
            sections=section_records,
        )

        saved_chunks = (
            self.chunk_repository.list_all()
        )

        self.assertEqual(
            saved_chunks,
            chunks,
        )

    def test_empty_paper_ids_return_empty_list(
        self,
    ) -> None:
        """没有论文ID时应返回空列表。"""

        chunks = (
            self.chunk_repository
            .list_by_paper_ids([])
        )

        self.assertEqual(
            chunks,
            [],
        )

    def test_lists_chunks_by_paper_ids(
        self,
    ) -> None:
        """只返回指定论文的Chunk。"""

        first_section_tree = (
            self.create_section_tree()
        )

        first_section_records = (
            self.section_repository.add_tree(
                paper_id="paper-1",
                root_sections=(
                    first_section_tree
                ),
            )
        )

        first_chunks = build_chunks(
            root_sections=(
                first_section_tree
            ),
            paper_id="paper-1",
            max_chars=100,
            overlap_chars=0,
        )

        self.chunk_repository.add_many(
            chunks=first_chunks,
            sections=first_section_records,
        )

        PaperRepository(
            self.connection
        ).add(
            Paper(
                id="paper-2",
                title="第二篇测试论文",
                authors=("作者乙",),
                file_name="paper-2.pdf",
                file_path="papers/paper-2.pdf",
                file_hash="file-hash-2",
            )
        )

        second_section_tree = (
            self.create_section_tree()
        )

        second_section_records = (
            self.section_repository.add_tree(
                paper_id="paper-2",
                root_sections=(
                    second_section_tree
                ),
            )
        )

        second_chunks = build_chunks(
            root_sections=(
                second_section_tree
            ),
            paper_id="paper-2",
            max_chars=100,
            overlap_chars=0,
        )

        self.chunk_repository.add_many(
            chunks=second_chunks,
            sections=second_section_records,
        )

        saved_chunks = (
            self.chunk_repository
            .list_by_paper_ids(
                [
                    " paper-2 ",
                    "paper-1",
                    "paper-2",
                ]
            )
        )

        self.assertEqual(
            len(saved_chunks),
            (
                len(first_chunks)
                + len(second_chunks)
            ),
        )

        self.assertEqual(
            {
                chunk.paper_id
                for chunk in saved_chunks
            },
            {
                "paper-1",
                "paper-2",
            },
        )

        self.assertEqual(
            saved_chunks,
            (
                first_chunks
                + second_chunks
            ),
        )

if __name__ == "__main__":
    unittest.main()
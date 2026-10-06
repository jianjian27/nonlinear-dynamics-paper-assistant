import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from database.connection import create_connection
from database.paper_repository import PaperRepository
from database.schema import initialize_database
from database.section_repository import SectionRepository
from domain.paper import Paper
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


class TestSectionRepository(unittest.TestCase):
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

        paper_repository = PaperRepository(
            self.connection
        )

        paper_repository.add(
            Paper(
                id="paper-1",
                title="测试论文",
                authors=("渐渐",),
                file_name="paper.pdf",
                file_path="papers/paper.pdf",
                file_hash="file-hash-1",
            )
        )

        self.repository = SectionRepository(
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
                    text="子章节正文",
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
                    text="父章节正文",
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
                    text="结论正文",
                )
            ],
        )

        return [parent, conclusion]

    def test_adds_and_lists_section_tree(self,) -> None:
        sections = self.create_section_tree()

        added_records = self.repository.add_tree(
            paper_id="paper-1",
            root_sections=sections,
        )

        saved_records = (
            self.repository.list_by_paper(
                "paper-1"
            )
        )

        self.assertEqual(len(added_records), 3)
        self.assertEqual(len(saved_records), 3)

        self.assertEqual(
            [record.title for record in saved_records],
            [
                "1 数学模型",
                "1.1 系统模型",
                "2 结论",
            ],
        )

        self.assertEqual(
            [
                record.section_index
                for record in saved_records
            ],
            [0, 1, 2],
        )

    def test_preserves_parent_and_path(self,) -> None:
        self.repository.add_tree(
            paper_id="paper-1",
            root_sections=self.create_section_tree(),
        )

        records = self.repository.list_by_paper(
            "paper-1"
        )

        parent = records[0]
        child = records[1]
        conclusion = records[2]

        self.assertIsNone(parent.parent_id)

        self.assertEqual(
            child.parent_id,
            parent.id,
        )

        self.assertEqual(
            child.section_path,
            (
                "1 数学模型",
                "1.1 系统模型",
            ),
        )

        self.assertIsNone(conclusion.parent_id)

    def test_preserves_content_and_pages(self,) -> None:
        self.repository.add_tree(
            paper_id="paper-1",
            root_sections=self.create_section_tree(),
        )

        records = self.repository.list_by_paper(
            "paper-1"
        )

        parent = records[0]
        child = records[1]

        self.assertEqual(
            parent.content,
            "父章节正文",
        )
        self.assertEqual(parent.page_start, 1)
        self.assertEqual(parent.page_end, 1)

        self.assertEqual(
            child.content,
            "子章节正文",
        )
        self.assertEqual(child.page_start, 2)
        self.assertEqual(child.page_end, 2)

    def test_rejects_unknown_paper(self,) -> None:
        with self.assertRaises(
            sqlite3.IntegrityError
        ):
            self.repository.add_tree(
                paper_id="missing-paper",
                root_sections=(
                    self.create_section_tree()
                ),
            )

if __name__ == "__main__":
    unittest.main()
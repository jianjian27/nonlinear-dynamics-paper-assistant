import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from database.chunk_repository import ChunkRepository
from database.connection import create_connection
from database.paper_repository import PaperRepository
from database.schema import initialize_database
from database.section_repository import SectionRepository
from ingestion.paper_content_ingestor import (
    PaperContentIngestor,
)
from ingestion.paper_ingestor import PaperIngestor
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


class TestPaperContentIngestor(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()

        self.directory = Path(
            self.temporary_directory.name
        )

        self.connection = create_connection(
            self.directory / "test.db"
        )
        initialize_database(self.connection)

        paper_repository = PaperRepository(
            self.connection
        )

        self.section_repository = (
            SectionRepository(self.connection)
        )
        self.chunk_repository = ChunkRepository(
            self.connection
        )

        self.ingestor = PaperContentIngestor(
            connection=self.connection,
            paper_ingestor=PaperIngestor(
                paper_repository
            ),
            section_repository=(
                self.section_repository
            ),
            chunk_repository=self.chunk_repository,
        )

        self.file_path = (
            self.directory / "paper.pdf"
        )
        self.file_path.write_bytes(
            b"paper content"
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.temporary_directory.cleanup()

    @staticmethod
    def create_sections() -> list[Section]:
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

        return [parent]

    def test_ingests_complete_paper_content(
        self,
    ) -> None:
        result = self.ingestor.ingest(
            file_path=self.file_path,
            title="测试论文",
            root_sections=self.create_sections(),
            authors=("渐渐",),
            max_chars=100,
            overlap_chars=0,
        )

        self.assertTrue(result.created)
        self.assertEqual(len(result.sections), 2)
        self.assertEqual(len(result.chunks), 2)

        paper_count = self.connection.execute(
            "SELECT COUNT(*) FROM papers"
        ).fetchone()[0]

        section_count = self.connection.execute(
            "SELECT COUNT(*) FROM sections"
        ).fetchone()[0]

        chunk_count = self.connection.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]

        self.assertEqual(paper_count, 1)
        self.assertEqual(section_count, 2)
        self.assertEqual(chunk_count, 2)

    def test_duplicate_file_is_not_inserted_again(
        self,
    ) -> None:
        first_result = self.ingestor.ingest(
            file_path=self.file_path,
            title="测试论文",
            root_sections=self.create_sections(),
            max_chars=100,
            overlap_chars=0,
        )

        second_result = self.ingestor.ingest(
            file_path=self.file_path,
            title="重复论文",
            root_sections=self.create_sections(),
            max_chars=100,
            overlap_chars=0,
        )

        self.assertTrue(first_result.created)
        self.assertFalse(second_result.created)

        paper_count = self.connection.execute(
            "SELECT COUNT(*) FROM papers"
        ).fetchone()[0]

        self.assertEqual(paper_count, 1)

    def test_rolls_back_all_data_on_failure(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            self.ingestor.ingest(
                file_path=self.file_path,
                title="事务测试论文",
                root_sections=self.create_sections(),
                max_chars=100,
                overlap_chars=100,
            )

        for table_name in (
            "papers",
            "sections",
            "chunks",
        ):
            count = self.connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]

            self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
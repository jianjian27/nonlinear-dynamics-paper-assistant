import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from database.connection import create_connection
from database.paper_repository import PaperRepository
from database.schema import initialize_database
from ingestion.paper_ingestor import PaperIngestor


class TestPaperIngestor(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()

        self.directory = Path(
            self.temporary_directory.name
        )

        database_path = (
            self.directory / "test.db"
        )

        connection = create_connection(
            database_path
        )

        initialize_database(connection)

        self.connection = connection
        self.repository = PaperRepository(
            connection
        )
        self.ingestor = PaperIngestor(
            self.repository
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.temporary_directory.cleanup()

    def test_ingests_new_paper(self) -> None:
        file_path = self.directory / "paper.pdf"
        file_path.write_bytes(b"paper content")

        result = self.ingestor.ingest(
            file_path=file_path,
            title="测试论文",
            authors=("红眼睛", "渐渐"),
            year=2026,
        )

        self.assertTrue(result.created)
        self.assertEqual(
            result.paper.title,
            "测试论文",
        )
        self.assertEqual(
            result.paper.authors,
            ("红眼睛", "渐渐"),
        )
        self.assertEqual(
            result.paper.file_name,
            "paper.pdf",
        )
        self.assertTrue(
            result.paper.id.startswith("paper-")
        )

    def test_ingesting_same_file_returns_existing(
        self,
    ) -> None:
        file_path = self.directory / "paper.pdf"
        file_path.write_bytes(b"paper content")

        first_result = self.ingestor.ingest(
            file_path=file_path,
            title="测试论文",
        )

        second_result = self.ingestor.ingest(
            file_path=file_path,
            title="另一个标题",
        )

        self.assertTrue(first_result.created)
        self.assertFalse(second_result.created)
        self.assertEqual(
            first_result.paper.id,
            second_result.paper.id,
        )
        self.assertEqual(
            second_result.paper.title,
            "测试论文",
        )

    def test_same_content_with_different_names_is_duplicate(
        self,
    ) -> None:
        first_path = self.directory / "first.pdf"
        second_path = self.directory / "second.pdf"

        first_path.write_bytes(b"same content")
        second_path.write_bytes(b"same content")

        first_result = self.ingestor.ingest(
            file_path=first_path,
            title="第一份文件",
        )

        second_result = self.ingestor.ingest(
            file_path=second_path,
            title="第二份文件",
        )

        self.assertTrue(first_result.created)
        self.assertFalse(second_result.created)
        self.assertEqual(
            first_result.paper.id,
            second_result.paper.id,
        )

    def test_different_content_creates_two_papers(
        self,
    ) -> None:
        first_path = self.directory / "first.pdf"
        second_path = self.directory / "second.pdf"

        first_path.write_bytes(b"content one")
        second_path.write_bytes(b"content two")

        first_result = self.ingestor.ingest(
            file_path=first_path,
            title="第一篇论文",
        )

        second_result = self.ingestor.ingest(
            file_path=second_path,
            title="第二篇论文",
        )

        self.assertTrue(first_result.created)
        self.assertTrue(second_result.created)
        self.assertNotEqual(
            first_result.paper.id,
            second_result.paper.id,
        )

        row = self.connection.execute(
            "SELECT COUNT(*) FROM papers"
        ).fetchone()

        self.assertEqual(row[0], 2)


if __name__ == "__main__":
    unittest.main()
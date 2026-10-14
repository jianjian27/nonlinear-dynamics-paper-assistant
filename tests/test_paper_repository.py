import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from database.connection import create_connection
from database.paper_repository import PaperRepository
from database.schema import initialize_database
from domain.paper import Paper


class TestPaperRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()

        database_path = (
            Path(self.temporary_directory.name)
            / "test.db"
        )

        self.connection = create_connection(database_path)

        initialize_database(self.connection)

        self.repository = PaperRepository(self.connection)

    def tearDown(self) -> None:
        self.connection.close()
        self.temporary_directory.cleanup()

    @staticmethod
    def create_test_paper() -> Paper:
        return Paper(
            id="paper-1",
            title="忆阻神经网络同步控制研究",
            authors=("红眼睛", "渐渐"),
            year=2026,
            file_name="paper.pdf",
            file_path="papers/paper.pdf",
            file_hash="file-hash-1",
            doi="10.0000/example",
        )

    def test_list_all_returns_empty_list(
        self,
    ) -> None:
        """空数据库应返回空列表。"""

        papers = self.repository.list_all()

        self.assertEqual(
            papers,
            [],
        )

    def test_list_all_returns_papers_by_title(
        self,
    ) -> None:
        """论文列表应按照标题排序。"""

        second_paper = Paper(
            id="paper-2",
            title="忆阻神经网络研究",
            authors=("作者乙",),
            year=2025,
            file_name="second.pdf",
            file_path="papers/second.pdf",
            file_hash="file-hash-2",
        )

        first_paper = Paper(
            id="paper-1",
            title="分数阶混沌系统研究",
            authors=("作者甲",),
            year=2026,
            file_name="first.pdf",
            file_path="papers/first.pdf",
            file_hash="file-hash-1",
        )

        # 故意按照与标题排序相反的顺序写入。
        self.repository.add(second_paper)
        self.repository.add(first_paper)

        papers = self.repository.list_all()

        self.assertEqual(
            len(papers),
            2,
        )

        self.assertEqual(
            [
                paper.id
                for paper in papers
            ],
            [
                "paper-1",
                "paper-2",
            ],
        )

        self.assertEqual(
            papers[0].title,
            "分数阶混沌系统研究",
        )

        self.assertEqual(
            papers[1].title,
            "忆阻神经网络研究",
        )

    def test_add_and_get_paper_by_id(self) -> None:
        original_paper = self.create_test_paper()

        self.repository.add(original_paper)

        saved_paper = self.repository.get_by_id(
            "paper-1"
        )

        self.assertIsNotNone(saved_paper)
        self.assertEqual(
            saved_paper.id,
            original_paper.id,
        )
        self.assertEqual(
            saved_paper.title,
            original_paper.title,
        )
        self.assertEqual(
            saved_paper.authors,
            original_paper.authors,
        )
        self.assertEqual(saved_paper.year, 2026)
        self.assertIsNotNone(
            saved_paper.created_at
        )

    def test_get_missing_paper_returns_none(self) -> None:
        paper = self.repository.get_by_id("missing-paper")

        self.assertIsNone(paper)

    def test_get_paper_by_file_hash(self) -> None:
        original_paper = self.create_test_paper()
        self.repository.add(original_paper)

        saved_paper = (
            self.repository.get_by_file_hash(
                "file-hash-1"
            )
        )

        self.assertIsNotNone(saved_paper)
        self.assertEqual(
            saved_paper.id,
            "paper-1",
        )

    def test_exists_by_file_hash(self) -> None:
        self.assertFalse(
            self.repository.exists_by_file_hash("file-hash-1")
        )

        self.repository.add(self.create_test_paper())

        self.assertTrue(
            self.repository.exists_by_file_hash("file-hash-1")
        )

    def test_duplicate_file_hash_is_rejected(self) -> None:
        first_paper = self.create_test_paper()

        second_paper = Paper(
            id="paper-2",
            title="重复导入的论文",
            authors=("jianjian",),
            file_name="copy.pdf",
            file_path="papers/copy.pdf",
            file_hash="file-hash-1",
        )

        self.repository.add(first_paper)

        with self.assertRaises(sqlite3.IntegrityError):
            self.repository.add(second_paper)


if __name__ == "__main__":
    unittest.main()

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

    def test_lists_papers_with_content_counts(
        self,
    ) -> None:
        """论文概览应包含章节和Chunk数量。"""

        paper = self.create_test_paper()

        self.repository.add(paper)

        self.connection.execute(
            """
            INSERT INTO sections (
                id,
                paper_id,
                parent_id,
                title,
                level,
                section_index,
                content
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "section-1",
                paper.id,
                None,
                "1 引言",
                1,
                0,
                "章节内容",
            ),
        )

        self.connection.execute(
            """
            INSERT INTO chunks (
                id,
                paper_id,
                section_id,
                chunk_index,
                section_title,
                section_path,
                content,
                pages,
                source_blocks,
                metadata,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chunk-1",
                paper.id,
                "section-1",
                0,
                "1 引言",
                '["1 引言"]',
                "Chunk内容",
                "[1]",
                "[]",
                "{}",
                "content-hash-1",
            ),
        )

        self.connection.commit()

        overviews = (
            self.repository
            .list_all_with_counts()
        )

        self.assertEqual(
            len(overviews),
            1,
        )

        self.assertEqual(
            overviews[0].paper.id,
            paper.id,
        )

        self.assertEqual(
            overviews[0].section_count,
            1,
        )

        self.assertEqual(
            overviews[0].chunk_count,
            1,
        )

    def test_deleting_paper_cascades_content(
        self,
    ) -> None:
        """删除论文后章节和Chunk也应被删除。"""

        paper = self.create_test_paper()

        self.repository.add(paper)

        self.connection.execute(
            """
            INSERT INTO sections (
                id,
                paper_id,
                parent_id,
                title,
                level,
                section_index,
                content
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "section-1",
                paper.id,
                None,
                "1 引言",
                1,
                0,
                "章节内容",
            ),
        )

        self.connection.execute(
            """
            INSERT INTO chunks (
                id,
                paper_id,
                section_id,
                chunk_index,
                section_title,
                section_path,
                content,
                pages,
                source_blocks,
                metadata,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chunk-1",
                paper.id,
                "section-1",
                0,
                "1 引言",
                '["1 引言"]',
                "Chunk内容",
                "[1]",
                "[]",
                "{}",
                "content-hash-1",
            ),
        )

        self.connection.commit()

        deleted_paper = (
            self.repository.delete_by_id(
                paper.id
            )
        )

        self.assertIsNotNone(
            deleted_paper
        )

        paper_count = self.connection.execute(
            "SELECT COUNT(*) FROM papers"
        ).fetchone()[0]

        section_count = self.connection.execute(
            "SELECT COUNT(*) FROM sections"
        ).fetchone()[0]

        chunk_count = self.connection.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]

        self.assertEqual(
            paper_count,
            0,
        )

        self.assertEqual(
            section_count,
            0,
        )

        self.assertEqual(
            chunk_count,
            0,
        )


    def test_deleting_missing_paper_returns_none(
        self,
    ) -> None:
        """删除不存在的论文应返回None。"""

        result = (
            self.repository.delete_by_id(
                "missing-paper"
            )
        )

        self.assertIsNone(
            result
        )

    def test_gets_overview_by_paper_id(
        self,
    ) -> None:
        """应根据ID查询论文概览。"""

        paper = self.create_test_paper()

        self.repository.add(
            paper
        )

        overview = (
            self.repository
            .get_overview_by_id(
                paper.id
            )
        )

        self.assertIsNotNone(
            overview
        )

        self.assertEqual(
            overview.paper.id,
            paper.id,
        )

        self.assertEqual(
            overview.paper.title,
            paper.title,
        )

        self.assertEqual(
            overview.section_count,
            0,
        )

        self.assertEqual(
            overview.chunk_count,
            0,
        )

        missing_overview = (
            self.repository
            .get_overview_by_id(
                "missing-paper"
            )
        )

        self.assertIsNone(
            missing_overview
        )

if __name__ == "__main__":
    unittest.main()

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from database.connection import create_connection
from database.schema import initialize_database


class TestDatabaseSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()

        self.database_path = (
            Path(self.temporary_directory.name) / "test.db"
        )

        self.connection = create_connection(
            self.database_path
        )

        initialize_database(self.connection)

    def tearDown(self) -> None:
        self.connection.close()
        self.temporary_directory.cleanup()

    def test_required_tables_are_created(self) -> None:
        rows = self.connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()

        table_names = {
            row["name"]
            for row in rows
        }

        self.assertIn("papers", table_names)
        self.assertIn("sections", table_names)
        self.assertIn("chunks", table_names)

    def test_database_schema_version_is_one(self) -> None:
        row = self.connection.execute(
            "PRAGMA user_version"
        ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)

    def test_deleting_paper_deletes_related_data(
        self,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO papers (
                id,
                title,
                file_name,
                file_path,
                file_hash
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "paper-1",
                "测试论文",
                "paper.pdf",
                "papers/paper.pdf",
                "file-hash-1",
            ),
        )

        self.connection.execute(
            """
            INSERT INTO sections (
                id,
                paper_id,
                title,
                level,
                section_index
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "section-1",
                "paper-1",
                "引言",
                1,
                0,
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
                content,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chunk-1",
                "paper-1",
                "section-1",
                0,
                "引言",
                "这是一段测试文本。",
                "content-hash-1",
            ),
        )

        self.connection.commit()

        self.connection.execute(
            "DELETE FROM papers WHERE id = ?",
            ("paper-1",),
        )
        self.connection.commit()

        section_count = self.connection.execute(
            "SELECT COUNT(*) FROM sections"
        ).fetchone()[0]

        chunk_count = self.connection.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]

        self.assertEqual(section_count, 0)
        self.assertEqual(chunk_count, 0)

    def test_duplicate_file_hash_is_rejected(
        self,
    ) -> None:
        paper_values = (
            "paper-1",
            "测试论文",
            "paper.pdf",
            "papers/paper.pdf",
            "same-file-hash",
        )

        self.connection.execute(
            """
            INSERT INTO papers (
                id,
                title,
                file_name,
                file_path,
                file_hash
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            paper_values,
        )

        with self.assertRaises(
            sqlite3.IntegrityError
        ):
            self.connection.execute(
                """
                INSERT INTO papers (
                    id,
                    title,
                    file_name,
                    file_path,
                    file_hash
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "paper-2",
                    "重复论文",
                    "copy.pdf",
                    "papers/copy.pdf",
                    "same-file-hash",
                ),
            )


if __name__ == "__main__":
    unittest.main()
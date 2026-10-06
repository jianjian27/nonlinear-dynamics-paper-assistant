import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from database.connection import create_connection


class TestCreateConnection(unittest.TestCase):
    def test_create_database_and_access_row_by_name(
            self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = (
                Path(temporary_directory) / "test.db"
            )

            connection = create_connection(database_path)

            try:
                row = connection.execute(
                    "SELECT 1 AS value"
                ).fetchone()

                self.assertIsNotNone(row)
                self.assertEqual(row["value"], 1)
            finally:
                connection.close()

    def test_foreign_keys_are_enabled(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = (
                Path(temporary_directory) / "test.db"
            )

            connection = create_connection(database_path)

            try:
                row = connection.execute(
                    "PRAGMA foreign_keys"
                ).fetchone()

                self.assertIsNotNone(row)
                self.assertEqual(row[0], 1)
            finally:
                connection.close()



if __name__ == "__main__":
    unittest.main()
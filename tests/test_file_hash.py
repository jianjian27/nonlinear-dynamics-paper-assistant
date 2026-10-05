import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ingestion.file_hash import calculate_file_sha256


class TestCalculateFileSha256(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()

        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_calculates_known_sha256(self) -> None:
        file_path = self.directory / "hello.txt"
        file_path.write_bytes(b"hello")

        result = calculate_file_sha256(file_path)

        self.assertEqual(
            result,
            (
                "2cf24dba5fb0a30e26e83b2ac5b9e29e"
                "1b161e5c1fa7425e73043362938b9824"
            )
        )

    def test_same_content_has_same_hash(self) -> None:
        first_path = self.directory / "first.pdf"
        second_path = self.directory / "second.pdf"

        content = b"same paper content"

        first_path.write_bytes(content)
        second_path.write_bytes(content)

        first_hash = calculate_file_sha256(first_path)
        second_hash = calculate_file_sha256(second_path)

        self.assertEqual(first_hash, second_hash)

    def test_different_content_has_different_hash(self) -> None:
        first_path = self.directory / "first.pdf"
        second_path = self.directory / "second.pdf"

        first_path.write_bytes(b"first paper content")
        second_path.write_bytes(b"second paper content")

        first_hash = calculate_file_sha256(first_path)
        second_hash = calculate_file_sha256(second_path)

        self.assertNotEqual(first_hash, second_hash,)

    def test_missing_file_raises_error(self) -> None:
        missing_path = (self.directory / "missing.pdf")

        with self.assertRaises(FileNotFoundError):
            calculate_file_sha256(missing_path)


if __name__ == "__main__":
    unittest.main()

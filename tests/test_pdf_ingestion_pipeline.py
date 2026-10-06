import unittest
from pathlib import Path
from unittest.mock import Mock, patch, sentinel

from ingestion.pdf_ingestion_pipeline import (
    PdfIngestionPipeline,
)
from utils.pdf_structure import Section


class TestPdfIngestionPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.content_ingestor = Mock()

        self.pipeline = PdfIngestionPipeline(
            self.content_ingestor
        )

        self.sections = [
            Section(
                title="1 引言",
                level=1,
            )
        ]

    def test_parses_pdf_and_sends_sections_to_ingestor(
        self,
    ) -> None:
        pdf_path = Path("papers/test.pdf")

        self.content_ingestor.ingest.return_value = (
            sentinel.result
        )

        with patch(
            (
                "ingestion.pdf_ingestion_pipeline."
                "parse_pdf_sections"
            ),
            return_value=self.sections,
        ) as parser:
            result = self.pipeline.ingest(
                pdf_path=pdf_path,
                title="测试论文",
                authors=("红眼睛", "渐渐"),
                year=2026,
                doi="10.0000/example",
                max_chars=2000,
                overlap_chars=200,
            )

        parser.assert_called_once_with(
            str(pdf_path)
        )

        self.content_ingestor.ingest.assert_called_once_with(
            file_path=pdf_path,
            title="测试论文",
            root_sections=self.sections,
            authors=("红眼睛", "渐渐"),
            year=2026,
            doi="10.0000/example",
            max_chars=2000,
            overlap_chars=200,
        )

        self.assertIs(
            result,
            sentinel.result,
        )

    def test_uses_file_name_when_title_is_empty(
        self,
    ) -> None:
        pdf_path = Path(
            "papers/chaos_system.pdf"
        )

        with patch(
            (
                "ingestion.pdf_ingestion_pipeline."
                "parse_pdf_sections"
            ),
            return_value=self.sections,
        ):
            self.pipeline.ingest(
                pdf_path=pdf_path,
                title="   ",
            )

        call_arguments = (
            self.content_ingestor.ingest.call_args
        )

        self.assertEqual(
            call_arguments.kwargs["title"],
            "chaos_system",
        )

    def test_rejects_pdf_without_sections(
        self,
    ) -> None:
        pdf_path = Path("papers/empty.pdf")

        with patch(
            (
                "ingestion.pdf_ingestion_pipeline."
                "parse_pdf_sections"
            ),
            return_value=[],
        ):
            with self.assertRaisesRegex(
                ValueError,
                "没有生成可入库的章节结构",
            ):
                self.pipeline.ingest(pdf_path)

        self.content_ingestor.ingest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
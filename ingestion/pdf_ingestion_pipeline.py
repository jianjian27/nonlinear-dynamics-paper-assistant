from collections.abc import Sequence
from pathlib import Path

from ingestion.paper_content_ingestor import (
    PaperContentIngestor,
    PaperContentIngestorResult,
)
from utils.paper_parser import parse_pdf_sections


class PdfIngestionPipeline:
    """连接PDF结构解析与数据库入库。"""

    def __init__(
        self,
        content_ingestor: PaperContentIngestor,
    ) -> None:
        self.content_ingestor = content_ingestor

    def ingest(
        self,
        pdf_path: str | Path,
        title: str | None = None,
        authors: Sequence[str] = (),
        year: int | None = None,
        doi: str | None = None,
        *,
        max_chars: int = 3000,
        overlap_chars: int = 300,
    ) -> PaperContentIngestorResult:
        """解析PDF章节，并保存论文结构化内容。"""

        path = Path(pdf_path)

        paper_title = (
            title.strip()
            if title and title.strip()
            else path.stem
        )

        root_sections = parse_pdf_sections(
            str(path)
        )

        if not root_sections:
            raise ValueError(
                "PDF没有生成可入库的章节结构。"
            )

        return self.content_ingestor.ingest(
            file_path=path,
            title=paper_title,
            root_sections=root_sections,
            authors=tuple(authors),
            year=year,
            doi=doi,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
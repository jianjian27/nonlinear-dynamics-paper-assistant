import sqlite3

from database.chunk_repository import (
    ChunkRepository,
)
from database.paper_repository import (
    PaperRepository,
)
from database.section_repository import (
    SectionRepository,
)
from ingestion.paper_content_ingestor import (
    PaperContentIngestor,
)
from ingestion.paper_ingestor import (
    PaperIngestor,
)
from ingestion.pdf_ingestion_pipeline import (
    PdfIngestionPipeline,
)



def create_pdf_ingestion_pipeline(
    connection: sqlite3.Connection,
) -> PdfIngestionPipeline:
    """组装PDF结构化入库管道。"""

    paper_repository = PaperRepository(
        connection
    )

    section_repository = SectionRepository(
        connection
    )

    chunk_repository = ChunkRepository(
        connection
    )

    paper_ingestor = PaperIngestor(
        paper_repository
    )

    content_ingestor = PaperContentIngestor(
        connection=connection,
        paper_ingestor=paper_ingestor,
        section_repository=section_repository,
        chunk_repository=chunk_repository,
    )

    return PdfIngestionPipeline(
        content_ingestor
    )
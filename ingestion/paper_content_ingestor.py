import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from database.chunk_repository import ChunkRepository
from database.section_repository import SectionRepository
from domain.chunk import Chunk
from domain.paper import Paper
from domain.section_record import SectionRecord
from ingestion.chunk_builder import build_chunks
from ingestion.paper_ingestor import PaperIngestor
from utils.pdf_structure import Section


@dataclass(frozen=True)
class PaperContentIngestorResult:
    """论文及其结构化内容的入库结果。"""

    paper: Paper
    sections: tuple[SectionRecord, ...]
    chunks: tuple[Chunk, ...]
    created: bool


class PaperContentIngestor:
    """以一个数据库事务的方式保存论文、章节和 Chunk。"""

    def __init__(
            self,
            connection: sqlite3.Connection,
            paper_ingestor: PaperIngestor,
            section_repository: SectionRepository,
            chunk_repository: ChunkRepository,
    ) -> None:
        self.connection = connection
        self.paper_ingestor = paper_ingestor
        self.section_repository = section_repository
        self.chunk_repository = chunk_repository

    def ingest(
            self,
            file_path: str | Path,
            title: str,
            root_sections: list[Section],
            authors: Sequence[str] = (),
            year: int | None = None,
            doi: str | None = None,
            *,
            max_chars: int = 3000,
            overlap_chars: int = 300,
    ) -> PaperContentIngestorResult:
        """保存论文、章节和 Chunk。"""

        try:
            paper_result = self.paper_ingestor.ingest(
                file_path=file_path,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                commit=False,
            )

            if not paper_result.created:
                existing_sections = (
                    self.section_repository.list_by_paper(
                        paper_result.paper.id
                    )
                )

                existing_chunks = (
                    self.chunk_repository.list_by_paper(
                        paper_result.paper.id
                    )
                )

                return PaperContentIngestorResult(
                    paper=paper_result.paper,
                    sections=tuple(existing_sections),
                    chunks=tuple(existing_chunks),
                    created=False,
                )

            section_records = (
                self.section_repository.add_tree(
                    paper_id=paper_result.paper.id,
                    root_sections=root_sections,
                    commit=False,
                )
            )

            chunks = build_chunks(
                root_sections=root_sections,
                paper_id=paper_result.paper.id,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )

            self.chunk_repository.add_many(
                chunks=chunks,
                sections=section_records,
                commit=False,
            )

            self.connection.commit()

            return PaperContentIngestorResult(
                paper=paper_result.paper,
                sections=tuple(section_records),
                chunks=tuple(chunks),
                created=True,
            )

        except Exception:
            self.connection.rollback()
            raise
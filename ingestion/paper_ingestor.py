from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from database.paper_repository import PaperRepository
from domain.paper import Paper
from ingestion.file_hash import calculate_file_sha256


@dataclass(frozen=True)
class PaperIngestionResult:
    """论文入库操作的结果。"""

    paper: Paper
    created: bool


class PaperIngestor:
    """负责计算文件哈希并注册论文。"""

    def __init__(
            self,
            repository: PaperRepository,
    ) -> None:
        self.repository = repository

    def ingest(
            self,
            file_path: str | Path,
            title: str,
            authors: Sequence[str] = (),
            year: int | None = None,
            doi: str | None = None,
            *,
            commit: bool = True,
    ) -> PaperIngestionResult:
        """导入论文元数据，并避免重复文件入库。"""

        path = Path(file_path)
        file_hash = calculate_file_sha256(path)

        existing_paper = (
            self.repository.get_by_file_hash(
                file_hash
            )
        )

        if existing_paper is not None:
            return PaperIngestionResult(
                paper=existing_paper,
                created=False,
            )

        paper = Paper(
            id=f"paper-{file_hash}",
            title=title,
            authors=tuple(authors),
            file_name=path.name,
            file_path=str(path.resolve()),
            file_hash=file_hash,
            year=year,
            doi=doi,
        )

        self.repository.add(
            paper,
            commit=commit,
        )

        saved_paper = self.repository.get_by_id(
            paper.id
        )

        if saved_paper is None:
            raise RuntimeError(
                "论文保存成功后无法重新查询。"
            )

        return PaperIngestionResult(
            paper=saved_paper,
            created=True,
        )
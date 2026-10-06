from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    """论文在业务代码中的数据模型。"""

    id: str
    title: str
    authors: tuple[str, ...]
    file_name: str
    file_path: str
    file_hash: str
    year: int | None = None
    doi: str | None = None
    created_at: str | None = None
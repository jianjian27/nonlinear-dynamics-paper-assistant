from dataclasses import dataclass


@dataclass(frozen=True)
class SectionRecord:
    """保存在数据库中的章节记录"""

    id: str
    paper_id: str
    parent_id: str | None
    title: str
    level: int
    section_index: int
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    content: str
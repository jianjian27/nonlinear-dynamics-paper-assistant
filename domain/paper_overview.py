from dataclasses import dataclass

from domain.paper import Paper


@dataclass(frozen=True)
class PaperOverview:
    """知识库中的论文及其内容统计。"""

    paper: Paper
    section_count: int
    chunk_count: int
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass

from embedding.vector_index import (
    VectorSearchResult,
)


@dataclass(frozen=True)
class ContextEvidence:
    """提供给大模型的一条论文证据。"""

    citation_number: int
    paper_id: str
    paper_title: str
    chunk_id: str
    section_path: tuple[str, ...]
    page_start: int
    page_end: int
    score: float
    content: str

    @property
    def citation_label(self) -> str:
        """生成证据引用编号。"""

        return f"证据{self.citation_number}"

    @property
    def section_text(self) -> str:
        """生成完整章节路径。"""

        if not self.section_path:
            return "未识别章节"

        return " > ".join(
            self.section_path
        )

    @property
    def page_text(self) -> str:
        """生成人类可读的页码范围。"""

        if self.page_start == self.page_end:
            return f"第{self.page_start}页"

        return (
            f"第{self.page_start}"
            f"—{self.page_end}页"
        )

    def render(self) -> str:
        """把一条证据转换为Prompt文本。"""

        return "\n".join(
            [
                f"[{self.citation_label}]",
                f"论文：{self.paper_title}",
                f"论文ID：{self.paper_id}",
                f"章节：{self.section_text}",
                f"页码：{self.page_text}",
                f"相似度：{self.score:.4f}",
                f"Chunk ID：{self.chunk_id}",
                f"原文：{self.content}",
            ]
        )


@dataclass(frozen=True)
class RagContext:
    """一次RAG检索得到的完整证据上下文。"""

    evidences: tuple[ContextEvidence, ...]

    @property
    def is_empty(self) -> bool:
        """上下文是否没有任何证据。"""

        return not self.evidences

    @property
    def text(self) -> str:
        """生成可以放入Prompt的证据文本。"""

        return "\n\n".join(
            evidence.render()
            for evidence in self.evidences
        )


class ContextBuilder:
    """将向量检索结果转换为可引用上下文。"""

    def __init__(
        self,
        *,
        max_total_content_chars: int = 12000,
        max_chunk_chars: int = 3000,
    ) -> None:
        if max_total_content_chars <= 0:
            raise ValueError(
                "上下文总字符数必须大于0。"
            )

        if max_chunk_chars <= 0:
            raise ValueError(
                "单个Chunk字符数必须大于0。"
            )

        self.max_total_content_chars = (
            max_total_content_chars
        )

        self.max_chunk_chars = (
            max_chunk_chars
        )

    def build(
        self,
        results: Sequence[VectorSearchResult],
        *,
        paper_titles: Mapping[str, str] | None = None,
    ) -> RagContext:
        """根据检索结果生成有长度限制的上下文。"""

        titles = (
            paper_titles
            if paper_titles is not None
            else {}
        )

        evidences: list[ContextEvidence] = []

        remaining_chars = (
            self.max_total_content_chars
        )

        for result in results:
            if remaining_chars <= 0:
                break

            chunk = result.chunk
            content = chunk.content.strip()

            if not content:
                continue

            allowed_chars = min(
                self.max_chunk_chars,
                remaining_chars,
            )

            content = self._truncate_text(
                content,
                allowed_chars,
            )

            paper_title = titles.get(
                chunk.paper_id,
                chunk.paper_id,
            )

            evidence = ContextEvidence(
                citation_number=len(evidences) + 1,
                paper_id=chunk.paper_id,
                paper_title=paper_title,
                chunk_id=chunk.id,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                score=result.score,
                content=content,
            )

            evidences.append(evidence)

            remaining_chars -= len(content)

        return RagContext(
            evidences=tuple(evidences)
        )

    @staticmethod
    def _truncate_text(
        text: str,
        max_chars: int,
    ) -> str:
        """在不超过限制的情况下截取文本。"""

        if len(text) <= max_chars:
            return text

        if max_chars == 1:
            return "…"

        return (
            text[:max_chars - 1]
            + "…"
        )
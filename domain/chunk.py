from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of paper text with traceable source metadata."""

    id: str
    paper_id: str
    chunk_index: int
    section_title: str
    section_path: tuple[str, ...]
    content: str
    page_start: int
    page_end: int
    pages: tuple[int, ...]
    source_blocks: tuple[str, ...]
    content_hash: str

    @property
    def embedding_text(self) -> str:
        """Text that should later be sent to the embedding model."""
        path = " > ".join(self.section_path)
        return f"章节：{path}\n\n{self.content}" if path else self.content


def hash_chunk_content(content: str) -> str:
    """Return a stable hash for change detection and index rebuilding."""
    return sha256(content.encode("utf-8")).hexdigest()
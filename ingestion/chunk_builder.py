from dataclasses import dataclass
from hashlib import sha256
from typing import Iterator

from domain.chunk import Chunk, hash_chunk_content
from utils.pdf_structure import Section, TextBlock


@dataclass(frozen=True)
class _TextUnit:
    """Smallest unit used internally while assembling a chunk."""

    text: str
    page: int
    source_block: str


def iter_sections_with_paths(
    sections: list[Section],
    parent_path: tuple[str, ...] = (),
) -> Iterator[tuple[Section, tuple[str, ...]]]:
    """Yield every section once in document order with its full title path."""
    for section in sections:
        section_path = (*parent_path, section.title)
        yield section, section_path
        yield from iter_sections_with_paths(section.children, section_path)


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split an oversized line as a last resort."""
    return [
        text[start : start + max_chars]
        for start in range(0, len(text), max_chars)
    ]


def _block_to_units(block: TextBlock, max_chars: int) -> list[_TextUnit]:
    """Keep block/line boundaries whenever they fit within the size limit."""
    source_block = f"{block.page}:{block.block_id}"
    block_text = block.text.strip()

    if not block_text:
        return []

    if len(block_text) <= max_chars:
        return [_TextUnit(block_text, block.page, source_block)]

    units = []
    for line in block.lines:
        line_text = line.text.strip()
        if not line_text:
            continue

        for part in _split_long_text(line_text, max_chars):
            units.append(_TextUnit(part, block.page, source_block))

    return units


def _content_length(units: list[_TextUnit]) -> int:
    if not units:
        return 0
    return sum(len(unit.text) for unit in units) + 2 * (len(units) - 1)


def _select_overlap(units: list[_TextUnit], overlap_chars: int) -> list[_TextUnit]:
    """Reuse complete trailing units without exceeding the overlap budget."""
    if overlap_chars <= 0:
        return []

    selected = []
    selected_length = 0

    for unit in reversed(units):
        added_length = len(unit.text) + (2 if selected else 0)
        if selected_length + added_length > overlap_chars:
            break
        selected.append(unit)
        selected_length += added_length

    selected.reverse()
    return selected


def _make_chunk(
    *,
    paper_id: str,
    chunk_index: int,
    section_title: str,
    section_path: tuple[str, ...],
    units: list[_TextUnit],
) -> Chunk:
    content = "\n\n".join(unit.text for unit in units)
    pages = tuple(sorted({unit.page for unit in units}))
    source_blocks = tuple(dict.fromkeys(unit.source_block for unit in units))
    content_hash = hash_chunk_content(content)

    identity = "\x1f".join(
        (
            paper_id,
            str(chunk_index),
            *section_path,
            content_hash,
        )
    )
    chunk_id = sha256(identity.encode("utf-8")).hexdigest()

    return Chunk(
        id=chunk_id,
        paper_id=paper_id,
        chunk_index=chunk_index,
        section_title=section_title,
        section_path=section_path,
        content=content,
        page_start=pages[0],
        page_end=pages[-1],
        pages=pages,
        source_blocks=source_blocks,
        content_hash=content_hash,
    )


def build_chunks(
    root_sections: list[Section],
    paper_id: str,
    *,
    max_chars: int = 3000,
    overlap_chars: int = 300,
) -> list[Chunk]:
    """Build traceable chunks from each section's direct text blocks.

    Children are traversed separately. This is important because ``Section.full_text``
    already contains child content and would otherwise duplicate evidence.
    """
    if not paper_id.strip():
        raise ValueError("paper_id 不能为空。")
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0。")
    if overlap_chars < 0:
        raise ValueError("overlap_chars 不能小于 0。")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars 必须小于 max_chars。")

    chunks = []
    chunk_index = 0

    for section, section_path in iter_sections_with_paths(root_sections):
        units = [
            unit
            for block in section.blocks
            for unit in _block_to_units(block, max_chars)
        ]
        current_units: list[_TextUnit] = []

        for unit in units:
            candidate = [*current_units, unit]
            if current_units and _content_length(candidate) > max_chars:
                chunks.append(
                    _make_chunk(
                        paper_id=paper_id,
                        chunk_index=chunk_index,
                        section_title=section.title,
                        section_path=section_path,
                        units=current_units,
                    )
                )
                chunk_index += 1
                current_units = _select_overlap(current_units, overlap_chars)

                # Do not allow overlap alone to prevent the next unit from fitting.
                if current_units and _content_length([*current_units, unit]) > max_chars:
                    current_units = []

            current_units.append(unit)

        if current_units:
            chunks.append(
                _make_chunk(
                    paper_id=paper_id,
                    chunk_index=chunk_index,
                    section_title=section.title,
                    section_path=section_path,
                    units=current_units,
                )
            )
            chunk_index += 1

    return chunks
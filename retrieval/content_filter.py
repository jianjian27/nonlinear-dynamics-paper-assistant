from domain.chunk import Chunk


REFERENCE_SECTION_PREFIXES = (
    "参考文献",
    "引用文献",
    "references",
    "bibliography",
)

REFERENCE_QUERY_KEYWORDS = (
    "参考文献",
    "引用文献",
    "引用了哪些",
    "引用哪些",
    "文献列表",
    "参考资料",
    "bibliography",
    "references",
    "cited papers",
    "cited works",
)


def normalize_text(text: str) -> str:
    """生成用于规则判断的规范化文本。"""

    return (
        "".join(text.lower().split())
        .strip("：:。.-—_")
    )


def is_reference_section_title(
    title: str,
) -> bool:
    """判断标题是否表示参考文献章节。"""

    normalized_title = normalize_text(
        title
    )

    return any(
        normalized_title.startswith(prefix)
        for prefix in(
            normalize_text(item)
            for item in (
                REFERENCE_SECTION_PREFIXES
            )
        )
    )


def is_reference_chunk(
    chunk: Chunk,
) -> bool:
    """判断Chunk是否来自参考文献章节。"""

    possible_titles = (
        chunk.section_title,
        *chunk.section_path,
    )

    return any(
        is_reference_section_title(title)
        for title in possible_titles
    )


def query_requests_references(
    query: str,
) -> bool:
    """判断用户是否明确要求查询参考文献。"""

    normalized_query = normalize_text(query)

    return any(
        normalize_text(keyword)
        in normalized_query
        for keyword in REFERENCE_QUERY_KEYWORDS
    )
from utils.heading_detector import (
    estimate_body_font_size,
    find_headings,
)
from utils.pdf_structure import (
    extract_text_blocks,
    order_document_blocks,
)
from utils.section_builder import (
    build_section_hierarchy,
    build_sections,
)


def parse_pdf_sections(
    pdf_path,
    heading_threshold=6,
):
    """
    读取 PDF 并构建章节树。

    完整流程：
        提取文本块；
        恢复阅读顺序；
        估计正文字号；
        识别标题；
        构建章节；
        建立章节层级。
    """
    blocks = extract_text_blocks(
        pdf_path
    )

    if not blocks:
        return []

    ordered_blocks = (
        order_document_blocks(
            blocks
        )
    )

    body_size = (
        estimate_body_font_size(
            ordered_blocks
        )
    )

    headings = find_headings(
        ordered_blocks,
        body_size=body_size,
        threshold=heading_threshold,
    )

    flat_sections = build_sections(
        ordered_blocks,
        headings,
    )

    root_sections = (
        build_section_hierarchy(
            flat_sections
        )
    )

    return root_sections
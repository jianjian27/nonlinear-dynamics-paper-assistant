from utils.heading_detector import (
    Heading,
)
from utils.pdf_structure import (
    Section,
    TextBlock,
)


def build_sections(
    blocks: list[TextBlock],
    headings: list[Heading],
):
    """
    根据有序文本块和标题，
    将论文划分为章节。

    标题块本身保存到 Section.title，
    标题后面的普通文本块保存到
    Section.blocks。
    """
    heading_starts = {
        id(heading.blocks[0]): heading
        for heading in headings
        if heading.blocks
    }

    all_heading_block_ids = {
        id(block)
        for heading in headings
        for block in heading.blocks
    }

    sections = []
    front_matter_blocks = []
    current_section = None

    for block in blocks:
        block_identity = id(
            block
        )

        heading = heading_starts.get(
            block_identity
        )

        if heading is not None:
            current_section = Section(
                title=heading.text,
                level=heading.level,
                number=heading.number,
                heading_blocks=list(
                    heading.blocks
                ),
            )

            sections.append(
                current_section
            )

            continue

        if (
            block_identity
            in all_heading_block_ids
        ):
            # 如果标题由多个块组成，
            # 后续标题块不能加入正文。
            continue

        if current_section is None:
            front_matter_blocks.append(
                block
            )
        else:
            current_section.blocks.append(
                block
            )

    if front_matter_blocks:
        front_matter_section = Section(
            title="文档前置内容",
            level=0,
            blocks=front_matter_blocks,
        )

        sections.insert(
            0,
            front_matter_section,
        )

    return sections


def build_section_hierarchy(
    sections,
):
    """
    根据章节级别建立父子关系。

    示例：
        1
        ├── 1.1
        └── 1.2
        2
        └── 2.1
    """
    root_sections = []
    section_stack = []

    for section in sections:
        # 防止重复调用函数时，
        # 子章节被重复加入。
        section.children.clear()

        if section.level <= 0:
            root_sections.append(
                section
            )

            section_stack.clear()
            continue

        while (
            section_stack
            and section_stack[-1].level
            >= section.level
        ):
            section_stack.pop()

        if section_stack:
            parent_section = (
                section_stack[-1]
            )

            parent_section.children.append(
                section
            )
        else:
            root_sections.append(
                section
            )

        section_stack.append(
            section
        )

    return root_sections


def flatten_section_hierarchy(
    sections,
):
    """
    将章节树展开为阅读顺序列表。

    父章节排在子章节之前。
    """
    flattened_sections = []

    def visit(section):
        flattened_sections.append(
            section
        )

        for child in section.children:
            visit(child)

    for section in sections:
        visit(section)

    return flattened_sections
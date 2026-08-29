from dataclasses import dataclass, field
from typing import Literal

import pymupdf

BBox = tuple[
    float,
    float,
    float,
    float,
]

ColumnName = Literal[
    "left",
    "right",
    "full",
    "single",
]


@dataclass
class TextLine:
    """
    PDF 中的一行文本。
    """

    page: int
    block_id: int
    text: str
    bbox: BBox
    font_size: float
    bold: bool
    centered: bool
    column: ColumnName | None = None

    @property
    def x0(self):
        return self.bbox[0]

    @property
    def y0(self):
        return self.bbox[1]

    @property
    def x1(self):
        return self.bbox[2]

    @property
    def y1(self):
        return self.bbox[3]

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0


@dataclass
class TextBlock:
    """
    PDF 中由多行文字组成的文本块。
    """

    page: int
    block_id: int
    bbox: BBox
    lines: list[TextLine] = field(
        default_factory=list
    )
    column: ColumnName | None = None

    def add_line(self, line):
        """
        向文本块中加入一个 TextLine 对象。
        """
        self.lines.append(line)

    @property
    def text(self):
        return "\n".join(
            line.text
            for line in self.lines
        )

    @property
    def x0(self):
        return self.bbox[0]

    @property
    def y0(self):
        return self.bbox[1]

    @property
    def x1(self):
        return self.bbox[2]

    @property
    def y1(self):
        return self.bbox[3]

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def font_size(self):
        """
        返回文本块中的最大字号。
        """
        if not self.lines:
            return 0.0

        return max(
            line.font_size
            for line in self.lines
        )

    @property
    def bold(self):
        """
        只要其中一行使用粗体，
        就认为文本块包含粗体。
        """
        return any(
            line.bold
            for line in self.lines
        )

    @property
    def centered(self):
        """
        所有文本行都大致居中时，
        才认为整个文本块居中。
        """
        if not self.lines:
            return False

        return all(
            line.centered
            for line in self.lines
        )

    def show_info(self):
        """
        显示文本块的基本信息。
        """
        print(
            f"第 {self.page} 页，"
            f"文本块编号 {self.block_id}，"
            f"栏位 {self.column}，"
            f"包含 {len(self.lines)} 行"
        )

        print(self.text)


@dataclass
class Section:
    """
    识别出的论文章节。
    """

    title: str
    level: int
    number: str | None = None
    blocks: list[TextBlock] = field(
        default_factory=list
    )

    @property
    def text(self):
        return "\n\n".join(
            block.text
            for block in self.blocks
        )

    @property
    def pages(self):
        return sorted(
            {
                block.page
                for block in self.blocks
            }
        )


def is_line_centered(
    bbox,
    page_width,
    tolerance=0.12,
):
    """
    判断一行文字是否大致居中。
    """
    x0 = bbox[0]
    x1 = bbox[2]

    line_center = (
        x0 + x1
    ) / 2

    page_center = (
        page_width / 2
    )

    distance = abs(
        line_center - page_center
    )

    allowed_distance = (
        page_width * tolerance
    )

    return (
        distance
        <= allowed_distance
    )


def classify_block_column(
    bbox,
    page_width,
):
    """
    根据文本块横坐标判断栏位。

    返回：
        left：左栏
        right：右栏
        full：横跨页面中部
    """
    x0 = bbox[0]
    x1 = bbox[2]

    page_center = (
        page_width / 2
    )

    tolerance = (
        page_width * 0.04
    )

    left_boundary = (
        page_center - tolerance
    )

    right_boundary = (
        page_center + tolerance
    )

    crosses_center = (
        x0 < left_boundary
        and x1 > right_boundary
    )

    if crosses_center:
        return "full"

    if x1 <= right_boundary:
        return "left"

    if x0 >= left_boundary:
        return "right"

    return "full"


def split_block_by_column(text_block):
    """
    如果一个文本块同时包含不同栏位的行，
    就按照栏位把它拆成多个文本块。
    """
    column_groups = {}

    for line in text_block.lines:
        column = line.column

        if column not in column_groups:
            column_groups[column] = []

        column_groups[column].append(
            line
        )

    # 所有行都在同一个栏位，不需要拆分
    if len(column_groups) == 1:
        only_column = next(
            iter(column_groups)
        )

        text_block.column = only_column

        return [text_block]

    split_blocks = []

    for column, lines in column_groups.items():
        new_bbox = (
            min(line.x0 for line in lines),
            min(line.y0 for line in lines),
            max(line.x1 for line in lines),
            max(line.y1 for line in lines),
        )

        new_block = TextBlock(
            page=text_block.page,
            block_id=text_block.block_id,
            bbox=new_bbox,
            lines=lines,
            column=column,
        )

        split_blocks.append(
            new_block
        )

    return split_blocks


def order_column_blocks(blocks):
    """
    对普通的左右栏文本块进行排序。

    阅读顺序：
        先读左栏，从上到下；
        再读右栏，从上到下。

    暂时忽略 full 类型的跨栏块。
    """
    left_blocks = [
        block
        for block in blocks
        if block.column == "left"
    ]

    right_blocks = [
        block
        for block in blocks
        if block.column == "right"
    ]

    left_blocks.sort(
        key=lambda block: (
            block.y0,
            block.x0,
        )
    )

    right_blocks.sort(
        key=lambda block: (
            block.y0,
            block.x0,
        )
    )

    return (
        left_blocks
        + right_blocks
    )


def blocks_share_visual_row(
    first,
    second,
    y_tolerance=3.0,
):
    """
    判断两个文本块是否处于同一视觉行。

    判断依据：
        1. 顶部坐标足够接近；
        2. 或者纵向范围有较大重叠。
    """
    close_top = (
        abs(first.y0 - second.y0)
        <= y_tolerance
    )

    overlap_height = max(
        0,
        min(first.y1, second.y1)
        - max(first.y0, second.y0),
    )

    smaller_height = max(
        min(
            first.height,
            second.height,
        ),
        0.1,
    )

    overlap_ratio = (
        overlap_height
        / smaller_height
    )

    enough_overlap = (
        overlap_ratio >= 0.5
    )

    return (
        first.page == second.page
        and (
            close_top
            or enough_overlap
        )
    )


def merge_full_width_rows(blocks):
    """
    如果一个视觉行中包含 full 块，
    就将该行的所有文本块合并为一个 full 块。
    """
    remaining_blocks = list(
        blocks
    )

    merged_blocks = []

    while True:
        full_block = next(
            (
                block
                for block in remaining_blocks
                if block.column == "full"
            ),
            None,
        )

        if full_block is None:
            break

        row_blocks = [
            block
            for block in remaining_blocks
            if blocks_share_visual_row(
                block,
                full_block,
            )
        ]

        row_lines = [
            line
            for block in row_blocks
            for line in block.lines
        ]

        row_lines.sort(
            key=lambda line: (
                line.x0,
                line.y0,
            )
        )

        for line in row_lines:
            line.column = "full"

        merged_bbox = (
            min(
                block.x0
                for block in row_blocks
            ),
            min(
                block.y0
                for block in row_blocks
            ),
            max(
                block.x1
                for block in row_blocks
            ),
            max(
                block.y1
                for block in row_blocks
            ),
        )

        merged_block = TextBlock(
            page=full_block.page,
            block_id=min(
                block.block_id
                for block in row_blocks
            ),
            bbox=merged_bbox,
            lines=row_lines,
            column="full",
        )

        merged_blocks.append(
            merged_block
        )

        remaining_blocks = [
            block
            for block in remaining_blocks
            if block not in row_blocks
        ]

    return (
        remaining_blocks
        + merged_blocks
    )


def group_full_blocks(
    full_blocks,
    y_tolerance=3.0,
):
    """
    将纵向位置接近的跨栏块分为一组。

    例如，一个公式可能被 PDF 拆成多个
    位于相同高度的文本块。
    """
    sorted_blocks = sorted(
        full_blocks,
        key=lambda block: (
            block.y0,
            block.x0,
        ),
    )

    groups = []

    for block in sorted_blocks:
        if not groups:
            groups.append([block])
            continue

        previous_group = groups[-1]

        previous_y = min(
            item.y0
            for item in previous_group
        )

        if abs(block.y0 - previous_y) <= y_tolerance:
            previous_group.append(
                block
            )
        else:
            groups.append([block])

    return groups


def order_page_blocks(blocks):
    """
    对同一页中的文本块进行阅读顺序排序。

    每遇到一组跨栏块，就先处理它上方区域的
    左右栏内容，再插入该组跨栏块。
    """
    blocks = merge_full_width_rows(
        blocks
    )

    regular_blocks = [
        block
        for block in blocks
        if block.column in (
            "left",
            "right",
        )
    ]

    full_blocks = [
        block
        for block in blocks
        if block.column == "full"
    ]

    full_groups = group_full_blocks(
        full_blocks
    )

    ordered_blocks = []
    remaining_blocks = list(
        regular_blocks
    )

    for full_group in full_groups:
        group_y = min(
            block.y0
            for block in full_group
        )

        blocks_above = [
            block
            for block in remaining_blocks
            if block.y0 < group_y
        ]

        ordered_blocks.extend(
            order_column_blocks(
                blocks_above
            )
        )

        ordered_blocks.extend(
            sorted(
                full_group,
                key=lambda block: (
                    block.y0,
                    block.x0,
                ),
            )
        )

        remaining_blocks = [
            block
            for block in remaining_blocks
            if block not in blocks_above
        ]

    ordered_blocks.extend(
        order_column_blocks(
            remaining_blocks
        )
    )

    return ordered_blocks


def order_document_blocks(blocks):
    """
    按照页码和页面版式，
    对整篇文档的文本块进行排序。
    """
    page_numbers = sorted(
        {
            block.page
            for block in blocks
        }
    )

    ordered_blocks = []

    for page_number in page_numbers:
        page_blocks = [
            block
            for block in blocks
            if block.page == page_number
        ]

        ordered_page_blocks = (
            order_page_blocks(
                page_blocks
            )
        )

        ordered_blocks.extend(
            ordered_page_blocks
        )

    return ordered_blocks


def extract_text_blocks(pdf_path):
    """
    从 PDF 中提取 TextBlock 对象。

    每个 TextBlock 内部包含多个
    TextLine 对象。
    """
    text_blocks = []

    with pymupdf.open(
        pdf_path
    ) as document:

        for page_index in range(
            len(document)
        ):
            page = document[page_index]

            page_number = (
                page_index + 1
            )

            page_width = (
                page.rect.width
            )

            page_data = page.get_text(
                "dict",
                sort=False,
            )

            block_id = 0

            for block_data in page_data.get(
                "blocks",
                [],
            ):
                # type 为 0 才是文本块
                if block_data.get("type") != 0:
                    continue

                block_id += 1

                block_bbox = tuple(
                    block_data.get(
                        "bbox",
                        (0, 0, 0, 0),
                    )
                )
                block_column = (
                    classify_block_column(
                        block_bbox,
                        page_width,
                    )
                )
                text_block = TextBlock(
                    page=page_number,
                    block_id=block_id,
                    column=block_column,
                    bbox=block_bbox,
                )

                for line_data in block_data.get(
                    "lines",
                    [],
                ):
                    spans = line_data.get(
                        "spans",
                        [],
                    )

                    if not spans:
                        continue

                    text_parts = []
                    font_sizes = []
                    bold = False

                    for span in spans:
                        span_text = span.get(
                            "text",
                            "",
                        )

                        if span_text:
                            text_parts.append(
                                span_text
                            )

                        font_size = float(
                            span.get(
                                "size",
                                0,
                            )
                        )

                        font_sizes.append(
                            font_size
                        )

                        flags = span.get(
                            "flags",
                            0,
                        )

                        font_name = span.get(
                            "font",
                            "",
                        ).lower()

                        if (
                            flags & 16
                            or "bold" in font_name
                        ):
                            bold = True

                    line_text = "".join(
                        text_parts
                    ).strip()

                    if not line_text:
                        continue

                    if not font_sizes:
                        continue

                    line_bbox = tuple(
                        line_data.get(
                            "bbox",
                            (0, 0, 0, 0),
                        )
                    )

                    line_column = (
                        classify_block_column(
                            line_bbox,
                            page_width,
                        )
                    )

                    font_size = max(
                        font_sizes
                    )

                    centered = (
                        is_line_centered(
                            line_bbox,
                            page_width,
                        )
                    )

                    text_line = TextLine(
                        page=page_number,
                        block_id=block_id,
                        text=line_text,
                        bbox=line_bbox,
                        font_size=font_size,
                        bold=bold,
                        centered=centered,
                        column=line_column
                    )

                    text_block.add_line(
                        text_line
                    )

                # 只有真正包含文字行的块才保留
                if text_block.lines:
                    split_blocks = (
                        split_block_by_column(
                            text_block
                        )
                    )

                    text_blocks.extend(
                        split_blocks
                    )

    return text_blocks
import argparse
import re
from collections import Counter

import pymupdf
from utils.pdf_structure import (
    extract_text_blocks,
    order_document_blocks,
)

# 中文论文中常见的章节标题
CHINESE_HEADING_KEYWORDS = {
    "摘要",
    "关键词",
    "引言",
    "研究背景",
    "相关工作",
    "研究方法",
    "数学模型",
    "系统模型",
    "模型建立",
    "稳定性分析",
    "动力学分析",
    "数值仿真",
    "数值模拟",
    "实验结果",
    "结果与讨论",
    "讨论",
    "结论",
    "总结",
    "参考文献",
    "附录",
}


# 英文论文中常见的章节标题
ENGLISH_HEADING_KEYWORDS = {
    "abstract",
    "keywords",
    "introduction",
    "background",
    "related work",
    "literature review",
    "method",
    "methods",
    "methodology",
    "materials and methods",
    "mathematical model",
    "model formulation",
    "system model",
    "stability analysis",
    "dynamical analysis",
    "dynamic analysis",
    "numerical simulation",
    "numerical simulations",
    "experiment",
    "experiments",
    "experimental results",
    "results",
    "results and discussion",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "appendix",
}


# 中文章节编号，例如：
# 一、引言
# 二. 数学模型
CHINESE_NUMBERING_PATTERN = re.compile(
    r"^[一二三四五六七八九十百]+"
    r"[、.．)]\s*"
)


# 阿拉伯数字章节编号前缀，例如：
# 1 Introduction
# 2. Mathematical Model
# 2.1 Stability Analysis
# 1. 2 分数阶高阶多涡卷Jerk系统模型
ARABIC_NUMBERING_PATTERN = re.compile(
    r"^\s*"
    r"\d+"
    r"(?:\s*[.．]\s*\d+){0,3}"
    r"[、.．)]?"
    r"\s+"
)

NUMBERED_HEADING_PATTERN = re.compile(
    r"^\s*"
    r"(?P<number>"
    r"\d+"
    r"(?:\s*[.．]\s*\d+){0,3}"
    r")"
    r"[、.．)]?"
    r"\s+"
    r"(?P<title>"
    r"[A-Za-z\u4e00-\u9fff]"
    r".{1,79}"
    r")"
    r"\s*$"
)


# 罗马数字章节编号，例如：
# I. INTRODUCTION
# II. MATHEMATICAL MODEL
ROMAN_NUMBERING_PATTERN = re.compile(
    r"^[IVXLCDM]+"
    r"[、.．)\s]+\s*",
    flags=re.IGNORECASE,
)


NUMBERING_PATTERNS = (
    CHINESE_NUMBERING_PATTERN,
    ARABIC_NUMBERING_PATTERN,
    ROMAN_NUMBERING_PATTERN,
)


def normalize_text(text):
    """
    统一文本格式，方便比较和匹配。
    """
    text = text.strip().lower()

    # 将连续空白统一替换为一个空格
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # 去掉标题末尾常见标点
    text = text.strip(
        " \t\r\n"
        ":："
        ".．"
        "、"
        ";；"
    )

    return text


def remove_numbering(text):
    """
    删除标题开头的章节编号。

    示例：
        2.1 Mathematical Model
        -> Mathematical Model
    """
    result = text.strip()

    for pattern in NUMBERING_PATTERNS:
        result = pattern.sub(
            "",
            result,
            count=1,
        )

    return result.strip()

def parse_numbered_heading(text):
    """
    解析阿拉伯数字章节标题。

    返回示例：
        {
            "number": "1.2",
            "title": "Mathematical Model",
            "level": 2,
        }

    如果不是章节标题，则返回 None。
    """
    match = NUMBERED_HEADING_PATTERN.match(
        text
    )

    if match is None:
        return None

    number = re.sub(
        r"\s+",
        "",
        match.group("number"),
    )

    number = number.replace(
        "．",
        ".",
    )

    title = match.group(
        "title"
    ).strip()

    level = number.count(".") + 1

    return {
        "number": number,
        "title": title,
        "level": level,
    }


def matches_numbering(text):
    """
    判断文本是否以有效章节编号开头。
    """
    if parse_numbered_heading(text):
        return True

    return bool(
        CHINESE_NUMBERING_PATTERN.match(
            text.strip()
        )
        or ROMAN_NUMBERING_PATTERN.match(
            text.strip()
        )
    )


def matches_heading_keyword(text):
    """
    判断文本是否匹配常见的中英文标题词。
    """
    text_without_numbering = remove_numbering(
        text
    )

    normalized = normalize_text(
        text_without_numbering
    )

    if normalized in CHINESE_HEADING_KEYWORDS:
        return True

    if normalized in ENGLISH_HEADING_KEYWORDS:
        return True

    # 处理类似：
    # Introduction and Motivation
    # Results and Analysis
    all_keywords = (
        CHINESE_HEADING_KEYWORDS
        | ENGLISH_HEADING_KEYWORDS
    )

    return any(
        normalized.startswith(
            keyword + " "
        )
        for keyword in all_keywords
    )


def is_english_all_caps_heading(text):
    """
    判断是否为英文全大写标题。
    """
    letters = "".join(
        character
        for character in text
        if (
            "A" <= character <= "Z"
            or "a" <= character <= "z"
        )
    )

    return (
        len(letters) >= 4
        and letters.isupper()
        and len(text) <= 100
    )


def is_centered(
    bbox,
    page_width,
    tolerance=0.12,
):
    """
    判断文本行是否大致位于页面水平中央。

    tolerance=0.12 表示允许中心位置存在
    页面宽度 12% 的偏差。
    """
    x0, _, x1, _ = bbox

    line_center = (
        x0 + x1
    ) / 2

    page_center = (
        page_width / 2
    )

    return (
        abs(line_center - page_center)
        <= page_width * tolerance
    )


def extract_lines(pdf_path):
    """
    提取 PDF 中每一行文本及其版式信息。

    返回：
        lines:
            文本行信息列表

        size_char_counts:
            各字号对应的字符数量

        page_count:
            PDF 总页数
    """
    lines = []
    size_char_counts = Counter()

    with pymupdf.open(pdf_path) as document:
        page_count = len(document)

        for page_number, page in enumerate(
            document,
            start=1,
        ):
            page_width = page.rect.width
            page_height = page.rect.height

            page_data = page.get_text(
                "dict",
                sort=True,
            )

            for block in page_data.get(
                "blocks",
                [],
            ):
                # type == 0 表示文本块
                if block.get("type") != 0:
                    continue

                for line in block.get(
                    "lines",
                    [],
                ):
                    spans = line.get(
                        "spans",
                        [],
                    )

                    if not spans:
                        continue

                    text = "".join(
                        span.get(
                            "text",
                            "",
                        )
                        for span in spans
                    ).strip()

                    if not text:
                        continue

                    for span in spans:
                        span_text = span.get(
                            "text",
                            "",
                        ).strip()

                        if not span_text:
                            continue

                        font_size = round(
                            float(
                                span.get(
                                    "size",
                                    0,
                                )
                            ),
                            1,
                        )

                        size_char_counts[
                            font_size
                        ] += len(span_text)

                    max_size = max(
                        float(
                            span.get(
                                "size",
                                0,
                            )
                        )
                        for span in spans
                    )

                    is_bold = any(
                        (
                            span.get(
                                "flags",
                                0,
                            )
                            & 16
                        )
                        or (
                            "bold"
                            in span.get(
                                "font",
                                "",
                            ).lower()
                        )
                        for span in spans
                    )

                    bbox = line.get(
                        "bbox",
                        (0, 0, 0, 0),
                    )

                    lines.append(
                        {
                            "page": page_number,
                            "page_width": page_width,
                            "page_height": page_height,
                            "bbox": bbox,
                            "y": bbox[1],
                            "text": text,
                            "size": max_size,
                            "bold": is_bold,
                            "centered": is_centered(
                                bbox,
                                page_width,
                            ),
                        }
                    )

    return (
        lines,
        size_char_counts,
        page_count,
    )


def find_repeated_margin_texts(
    lines,
    page_count,
):
    """
    找出可能的重复页眉或页脚。

    如果同一段短文本出现在多页的顶部或底部，
    就可能是页眉、页脚或页码。
    """
    text_pages = {}

    for line in lines:
        page_height = line[
            "page_height"
        ]

        is_top_margin = (
            line["y"]
            <= page_height * 0.12
        )

        is_bottom_margin = (
            line["y"]
            >= page_height * 0.88
        )

        if not (
            is_top_margin
            or is_bottom_margin
        ):
            continue

        normalized = normalize_text(
            line["text"]
        )

        if not normalized:
            continue

        if len(normalized) > 100:
            continue

        text_pages.setdefault(
            normalized,
            set(),
        ).add(line["page"])

    # 至少在两页出现，并且不能只出现在极少比例页面
    minimum_pages = max(
        2,
        round(page_count * 0.25),
    )

    return {
        text
        for text, pages
        in text_pages.items()
        if len(pages) >= minimum_pages
    }


def calculate_heading_score(
    line,
    body_size,
    repeated_margin_texts,
):
    """
    根据版式和内容特征计算标题分数。

    返回：
        score:
            标题得分

        reasons:
            得分或扣分原因
    """
    score = 0
    reasons = []

    text = line["text"]
    normalized = normalize_text(text)

    size_difference = (
        line["size"] - body_size
    )

    if size_difference >= 2.0:
        score += 3
        reasons.append("字号明显较大")
    elif size_difference >= 0.8:
        score += 2
        reasons.append("字号较大")

    if line["bold"]:
        score += 2
        reasons.append("粗体")

    numbered_heading = parse_numbered_heading(
        text
    )

    if numbered_heading is not None:
        level = numbered_heading["level"]

        if level >= 2:
            score += 6
            reasons.append(
                f"{level}级章节编号"
            )
        else:
            score += 3
            reasons.append(
                "一级章节编号"
            )

    elif (
            CHINESE_NUMBERING_PATTERN.match(
                text.strip()
            )
            or ROMAN_NUMBERING_PATTERN.match(
        text.strip()
    )
    ):
        score += 3
        reasons.append(
            "章节编号"
        )

    if matches_heading_keyword(text):
        score += 4
        reasons.append("标题关键词")

    if is_english_all_caps_heading(
        text
    ):
        score += 1
        reasons.append("英文全大写")

    if line["centered"]:
        score += 1
        reasons.append("居中")

    if len(text) <= 50:
        score += 1
        reasons.append("文本较短")
    elif len(text) > 100:
        score -= 3
        reasons.append("文本过长")

    if text.endswith(
        (
            "。",
            ".",
            "；",
            ";",
        )
    ):
        score -= 2
        reasons.append("以句末标点结尾")

    if normalized in repeated_margin_texts:
        score -= 5
        reasons.append("疑似重复页眉页脚")

    return score, reasons


def find_heading_candidates(
    lines,
    body_size,
    repeated_margin_texts,
    threshold,
):
    """
    根据标题评分筛选候选标题。
    """
    candidates = []

    for line in lines:
        score, reasons = (
            calculate_heading_score(
                line,
                body_size,
                repeated_margin_texts,
            )
        )

        if score < threshold:
            continue

        candidate = dict(line)
        candidate["score"] = score
        candidate["reasons"] = reasons

        candidates.append(candidate)

    return candidates


def find_block_heading_candidates(
    blocks,
    body_size,
    repeated_margin_texts,
    threshold,
):
    """
    根据完整文本块判断标题候选。

    与旧函数的区别：
        旧函数逐行判断；
        新函数保留文本块中的多行标题。
    """
    candidates = []

    for block in blocks:
        combined_text = " ".join(
            block.text.splitlines()
        )

        item = {
            "page": block.page,
            "block_id": block.block_id,
            "column": block.column,
            "bbox": block.bbox,
            "y": block.y0,
            "text": combined_text,
            "size": block.font_size,
            "bold": block.bold,
            "centered": block.centered,
        }

        score, reasons = (
            calculate_heading_score(
                item,
                body_size,
                repeated_margin_texts,
            )
        )

        if score < threshold:
            continue

        item["score"] = score
        item["reasons"] = reasons

        candidates.append(
            item
        )

    return candidates


def print_font_statistics(
    size_char_counts,
):
    """
    打印不同字号对应的字符数量。
    """
    print("各字号对应的字符数量：")

    for size, count in sorted(
        size_char_counts.items(),
        reverse=True,
    ):
        print(
            f"  {size:>5.1f}: {count}"
        )


def print_candidates(
    candidates,
    limit,
):
    """
    打印标题候选。
    """
    print()
    print(
        f"标题候选共 {len(candidates)} 条，"
        f"显示前 {limit} 条："
    )

    for item in candidates[:limit]:
        reasons = "、".join(
            item["reasons"]
        )

        print(
            f"p{item['page']:02d} "
            f"y={item['y']:7.1f} "
            f"size={item['size']:5.1f} "
            f"score={item['score']:2d} "
            f"bold={str(item['bold']):5} "
            f"center={str(item['centered']):5} "
            f"[{reasons}] "
            f"{item['text']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "检查中英文 PDF 的字号、版式"
            "及章节标题候选。"
        )
    )

    parser.add_argument(
        "pdf_path",
        help="需要检查的 PDF 文件路径。",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="最多显示多少个标题候选。",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="标题候选的最低分数。",
    )

    args = parser.parse_args()

    (
        lines,
        size_char_counts,
        page_count,
    ) = extract_lines(args.pdf_path)

    if not size_char_counts:
        print(
            "没有提取到可用文本。"
            "该 PDF 可能是扫描版。"
        )
        return

    # 将字符数量最多的字号视为正文主要字号
    body_size = max(
        size_char_counts,
        key=size_char_counts.get,
    )

    repeated_margin_texts = (
        find_repeated_margin_texts(
            lines,
            page_count,
        )
    )

    text_blocks = extract_text_blocks(
        args.pdf_path
    )

    ordered_blocks = order_document_blocks(
        text_blocks
    )

    candidates = (
        find_block_heading_candidates(
            ordered_blocks,
            body_size,
            repeated_margin_texts,
            args.threshold,
        )
    )

    print(f"PDF 页数：{page_count}")
    print(f"提取文本行数：{len(lines)}")
    print(f"正文主要字号：{body_size}")

    print_font_statistics(
        size_char_counts
    )

    print()
    print(
        "检测到的重复页眉页脚文本："
        f"{len(repeated_margin_texts)} 条"
    )

    for text in sorted(
        repeated_margin_texts
    ):
        print(f"  {text}")

    print_candidates(
        candidates,
        args.limit,
    )


if __name__ == "__main__":
    main()
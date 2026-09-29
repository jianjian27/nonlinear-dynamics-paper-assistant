import re

from collections import Counter
from dataclasses import dataclass, field
from utils.pdf_structure import (
    ColumnName,
    TextBlock,
)

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
    r".{1,199}"
    r")"
    r"\s*$"
)


@dataclass
class Heading:
    """
    识别出的论文标题。

    一个标题可能来自一个或多个
    TextBlock。
    """

    text: str
    level: int
    blocks: list[TextBlock] = field(
        default_factory=list
    )
    number: str | None = None
    score: int = 0
    reasons: list[str] = field(
        default_factory=list
    )

    @property
    def page(self):
        """
        返回标题所在的起始页。
        """
        if not self.blocks:
            return None

        return self.blocks[0].page

    @property
    def column(
        self,
    ) -> ColumnName | None:
        """
        返回标题起始块的栏位。
        """
        if not self.blocks:
            return None

        return self.blocks[0].column

    @property
    def y0(self):
        """
        返回标题起始块的纵向位置。
        """
        if not self.blocks:
            return None

        return self.blocks[0].y0


def parse_numbered_heading(text):
    """
    解析阿拉伯数字编号的中英文标题。

    返回示例：
        {
            "number": "1.2",
            "title": "Mathematical Model",
            "level": 2,
        }

    如果不是编号标题，则返回 None。
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

    level = (
        number.count(".")
        + 1
    )

    return {
        "number": number,
        "title": title,
        "level": level,
    }


def normalize_text(text):
    """
    统一标题文本格式，方便比较。

    处理内容：
        转为小写；
        合并连续空白；
        去掉首尾常见标点。
    """
    normalized = text.strip().lower()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    normalized = normalized.strip(
        " \t\r\n"
        ":："
        ".．"
        "、"
        ";；"
    )

    return normalized


def remove_numbering(text):
    """
    删除阿拉伯数字章节编号。

    示例：
        2.1 Mathematical Model
        -> Mathematical Model
    """
    parsed = parse_numbered_heading(
        text
    )

    if parsed is None:
        return text.strip()

    return parsed["title"]


def matches_heading_keyword(text):
    """
    判断文本是否精确匹配常见的
    中英文论文结构标题。

    关键词只作为辅助证据，
    不负责识别所有标题。
    """
    text_without_numbering = (
        remove_numbering(
            text
        )
    )

    normalized = normalize_text(
        text_without_numbering
    )

    all_keywords = (
        CHINESE_HEADING_KEYWORDS
        | ENGLISH_HEADING_KEYWORDS
    )

    return (
        normalized
        in all_keywords
    )


def is_english_all_caps_heading(text):
    """
    判断英文字符是否全部为大写。
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


def estimate_body_font_size(
    blocks,
):
    """
    根据字符数量估计论文的主要正文字号。

    使用字符总量最多的字号，
    而不是文本行数量最多的字号。
    """
    size_character_counts = Counter()

    for block in blocks:
        for line in block.lines:
            compact_text = re.sub(
                r"\s+",
                "",
                line.text,
            )

            character_count = len(
                compact_text
            )

            if character_count == 0:
                continue

            rounded_size = round(
                line.font_size,
                1,
            )

            size_character_counts[
                rounded_size
            ] += character_count

    if not size_character_counts:
        raise ValueError(
            "无法估计正文字号："
            "文档中没有可用文本。"
        )

    body_size = max(
        size_character_counts,
        key=size_character_counts.get,
    )

    return body_size


def calculate_heading_score(
    block,
    body_size,
    repeated_margin_texts=None,
):
    """
    根据文本块的内容和版式计算标题分数。
    """
    if repeated_margin_texts is None:
        repeated_margin_texts = set()

    text = " ".join(
        block.text.splitlines()
    )

    normalized = normalize_text(
        text
    )

    score = 0
    reasons = []

    size_difference = (
        block.font_size
        - body_size
    )

    if size_difference >= 2.0:
        score += 3
        reasons.append(
            "字号明显较大"
        )
    elif size_difference >= 0.8:
        score += 2
        reasons.append(
            "字号较大"
        )

    if block.bold:
        score += 2
        reasons.append(
            "粗体"
        )

    parsed_heading = (
        parse_numbered_heading(
            text
        )
    )

    if parsed_heading is not None:
        level = parsed_heading[
            "level"
        ]

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

    if matches_heading_keyword(
        text
    ):
        score += 4
        reasons.append(
            "标题关键词"
        )

    if is_english_all_caps_heading(
        text
    ):
        score += 1
        reasons.append(
            "英文全大写"
        )

    if block.centered:
        score += 1
        reasons.append(
            "居中"
        )

    if len(text) <= 50:
        score += 1
        reasons.append(
            "文本较短"
        )
    elif len(text) > 150:
        score -= 3
        reasons.append(
            "文本过长"
        )

    if text.endswith(
        (
            "。",
            ".",
            "；",
            ";",
        )
    ):
        score -= 2
        reasons.append(
            "以句末标点结尾"
        )

    if normalized in repeated_margin_texts:
        score -= 5
        reasons.append(
            "疑似重复页眉页脚"
        )

    return score, reasons


def detect_heading(
    block,
    body_size,
    threshold=6,
    repeated_margin_texts=None,
):
    """
    判断一个 TextBlock 是否为标题。

    是标题时返回 Heading，
    否则返回 None。
    """
    score, reasons = (
        calculate_heading_score(
            block,
            body_size,
            repeated_margin_texts,
        )
    )

    if score < threshold:
        return None

    text = " ".join(
        block.text.splitlines()
    )

    parsed_heading = (
        parse_numbered_heading(
            text
        )
    )

    if parsed_heading is None:
        number = None
        level = 1
    else:
        number = parsed_heading[
            "number"
        ]
        level = parsed_heading[
            "level"
        ]

    return Heading(
        text=text,
        level=level,
        number=number,
        blocks=[block],
        score=score,
        reasons=reasons,
    )


def find_headings(
    blocks,
    body_size,
    threshold=6,
    repeated_margin_texts=None,
):
    """
    从有序文本块中识别标题。
    """
    headings = []

    for block in blocks:
        heading = detect_heading(
            block,
            body_size,
            threshold,
            repeated_margin_texts,
        )

        if heading is not None:
            headings.append(
                heading
            )

    return headings
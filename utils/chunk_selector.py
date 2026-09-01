import re

def select_chunks(
        chunks,
        analysis_type
):
    """
    根据分析任务选择论文片段

    参数:
        chunks:
            文本块列表

        analysis_type:
            分析类型

    返回:
        需要发送给模型的文本
    """


    if analysis_type == "快速阅读":

        selected = chunks[:3]


    elif analysis_type == "数学模型分析":

        selected = chunks[1:4]


    elif analysis_type == "混沌动力学分析":

        selected = chunks[2:5]


    elif analysis_type == "创新点评价":

        selected = chunks[:2] + chunks[-2:]


    else:

        selected = chunks


    return selected


ANALYSIS_TITLE_TERMS = {
    "数学模型分析": {
        "数学模型",
        "系统模型",
        "模型建立",
        "model formulation",
        "mathematical model",
        "system model",
    },
    "混沌动力学分析": {
        "混沌",
        "动力学",
        "稳定性",
        "分岔",
        "吸引子",
        "chaos",
        "dynamical",
        "dynamic analysis",
        "stability",
        "bifurcation",
        "attractor",
    },
    "创新点评价": {
        "方法",
        "设计",
        "实验结果",
        "结果与讨论",
        "结论",
        "method",
        "design",
        "results",
        "discussion",
        "conclusion",
    },
}


def normalize_section_title(
    title,
):
    """
    统一章节标题格式。
    """
    normalized = title.lower()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def calculate_section_relevance(
    section,
    analysis_type,
):
    """
    根据章节标题计算其与分析任务的相关度。

    当前版本使用标题词语匹配，
    后续可以升级为语义相关度判断。
    """
    terms = ANALYSIS_TITLE_TERMS.get(
        analysis_type,
        set(),
    )

    normalized_title = (
        normalize_section_title(
            section.title
        )
    )

    matched_terms = [
        term
        for term in terms
        if term in normalized_title
    ]

    return len(
        matched_terms
    )


def select_sections(
    root_sections,
    analysis_type,
):
    """
    根据分析模式选择相关章节。

    如果父章节已经匹配，
    直接选择父章节及其完整子树，
    避免再次选择子章节造成正文重复。
    """
    if analysis_type == "快速阅读":
        return list(
            root_sections
        )

    selected_sections = []

    def visit(section):
        relevance = (
            calculate_section_relevance(
                section,
                analysis_type,
            )
        )

        if relevance > 0:
            selected_sections.append(
                section
            )

            # 父章节的 full_text 已经包含子章节，
            # 因此不再向下重复选择。
            return

        for child in section.children:
            visit(child)

    for section in root_sections:
        visit(section)

    # 如果标题关键词没有匹配成功，
    # 暂时回退到全部根章节，
    # 避免返回空内容。
    if not selected_sections:
        return list(
            root_sections
        )

    return selected_sections


def format_section(
    section,
):
    """
    将一个章节及其子章节格式化为
    带 Markdown 标题的文本。
    """
    markdown_level = min(
        max(section.level, 1),
        6,
    )

    heading_prefix = (
        "#"
        * markdown_level
    )

    text_parts = [
        f"{heading_prefix} "
        f"{section.title}"
    ]

    own_text = (
        section.text.strip()
    )

    if own_text:
        text_parts.append(
            own_text
        )

    for child in section.children:
        child_text = format_section(
            child
        )

        if child_text:
            text_parts.append(
                child_text
            )

    return "\n\n".join(
        text_parts
    )


def combine_selected_sections(
    sections,
):
    """
    将多个已选择章节组合成
    一段供模型分析的完整文本。
    """
    formatted_sections = [
        format_section(section)
        for section in sections
    ]

    return "\n\n".join(
        formatted_sections
    )
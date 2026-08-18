import re


def clean_markdown(text):
    """
    清理LLM输出中的Markdown/LaTeX格式问题

    目标：
    1. 修复行内公式格式
    2. 修复LaTeX转义问题
    3. 保持原始文本内容不变
    """


    # =========================
    # 1. 修复错误转义的公式定界符
    # =========================

    # 模型有时会输出：
    # \\(x\\)
    # \\[公式\\]

    text = text.replace(
        "\\\\(",
        "\\("
    )

    text = text.replace(
        "\\\\)",
        "\\)"
    )

    text = text.replace(
        "\\\\[",
        "\\["
    )

    text = text.replace(
        "\\\\]",
        "\\]"
    )


    # =========================
    # 2. 将 $...$ 转换为 \(...\)
    # =========================

    # 例如：
    # $f(x)$
    #
    # 转换：
    # \(f(x)\)

    text = re.sub(
        r"\$([^$\n]+?)\$",
        r"\\(\1\\)",
        text
    )


    # =========================
    # 3. 修复数学下标转义
    # =========================

    # 模型经常输出：
    # x\_i
    #
    # LaTeX中应该：
    # x_i

    text = re.sub(
        r"\\_([a-zA-Z0-9]+)",
        r"_\1",
        text
    )


    # =========================
    # 4. 修复中文环境中常见公式包裹
    # =========================

    # 例如：
    # [ \begin{cases}
    #
    # 改成：
    # \[
    # \begin{cases}

    text = text.replace(
        "[ \\begin",
        "\\[\\begin"
    )


    text = text.replace(
        "\\end{cases} ]",
        "\\end{cases}\\]"
    )

    def fix_inline_math_parentheses(text):
        """
        修复模型使用普通括号包裹LaTeX行内公式的问题
        """

        pattern = r"\(([^()\n]*\\[a-zA-Z]+[^()\n]*)\)"

        def repl(match):
            content = match.group(1)

            return r"\(" + content + r"\)"

        return re.sub(
            pattern,
            repl,
            text
        )

    text = fix_inline_math_parentheses(text)
    return text
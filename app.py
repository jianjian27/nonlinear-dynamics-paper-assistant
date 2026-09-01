import gradio as gr

from models.deepseek_api import (
    ModelServiceError,
    ask_deepseek,
)

from utils.pdf_reader import extract_pages_from_pdf
from utils.text_splitter import split_pages
from utils.chunk_selector import (
    combine_selected_sections,
    select_chunks,
    select_sections,
)
from utils.markdown_cleaner import clean_markdown
from utils.paper_parser import (
    parse_pdf_sections,
)

from prompts.paper_summary_prompt import PAPER_SUMMARY_PROMPT
from prompts.model_analysis_prompt import MODEL_ANALYSIS_PROMPT
from prompts.chaos_analysis_prompt import CHAOS_ANALYSIS_PROMPT
from prompts.innovation_prompt import INNOVATION_PROMPT

# -----------------------------
def analyze_paper(
    pdf_file,
    abstract,
    analysis_type,
    progress=gr.Progress(),
):
    """
    根据 PDF 或摘要执行论文分析。
    """
    abstract = (abstract or "").strip()

    if pdf_file is None and not abstract:
        return (
            "## ⚠️ 缺少论文内容\n\n"
            "请上传 PDF，或者在摘要输入框中粘贴论文摘要。"
        )

    try:
        if pdf_file is not None:
            progress(
                0.1,
                desc="正在读取 PDF……",
            )

            pages = extract_pages_from_pdf(
                pdf_file.name
            )

            has_text = any(
                page["text"].strip()
                for page in pages
            )

            if not pages or not has_text:
                return (
                    "## ⚠️ 无法提取 PDF 文本\n\n"
                    "该 PDF 可能是扫描版，或者没有可读取的文本层。"
                )

            progress(
                0.3,
                desc="正在切分论文文本……",
            )

            chunks = split_pages(pages)

            if not chunks:
                return (
                    "## ⚠️ 论文内容为空\n\n"
                    "PDF 已读取，但没有生成可分析的文本块。"
                )

            progress(
                0.5,
                desc="正在选择相关论文片段……",
            )

            selected_chunks = select_chunks(
                chunks,
                analysis_type,
            )

            if not selected_chunks:
                return (
                    "## ⚠️ 没有找到可分析内容\n\n"
                    "当前分析模式没有选出任何论文片段。"
                )

            paper_content = "\n\n".join(
                chunk["text"]
                for chunk in selected_chunks
            )

            if analysis_type != "快速阅读":
                progress(
                    0.55,
                    desc="正在识别论文章节结构……",
                )

                root_sections = (
                    parse_pdf_sections(
                        pdf_file.name
                    )
                )

                selected_sections = (
                    select_sections(
                        root_sections,
                        analysis_type,
                    )
                )

                structured_content = (
                    combine_selected_sections(
                        selected_sections
                    )
                ).strip()

                if structured_content:
                    paper_content = (
                        structured_content
                    )

        else:
            paper_content = abstract

        if analysis_type == "快速阅读":
            prompt = PAPER_SUMMARY_PROMPT
        elif analysis_type == "数学模型分析":
            prompt = MODEL_ANALYSIS_PROMPT
        elif analysis_type == "混沌动力学分析":
            prompt = CHAOS_ANALYSIS_PROMPT
        else:
            prompt = INNOVATION_PROMPT

        prompt = prompt.replace(
            "{paper_text}",
            paper_content,
        )

        progress(
            0.7,
            desc="正在调用模型分析论文……",
        )

        answer = ask_deepseek(prompt)

        progress(
            0.95,
            desc="正在整理分析结果……",
        )

        answer = clean_markdown(answer)

        progress(
            1.0,
            desc="分析完成",
        )

        return answer

    except ModelServiceError as exc:
        return (
            "## ❌ 模型调用失败\n\n"
            f"{exc}"
        )
    except Exception as exc:
        return (
            "## ❌ 论文处理失败\n\n"
            f"{type(exc).__name__}: {exc}"
        )



# -----------------------------
# Gradio界面
# -----------------------------

with gr.Blocks(
    title="Nonlinear Dynamics Paper Analyst"
) as demo:


    # 标题
    gr.Markdown(
        """
# Nonlinear Dynamics Paper Analyst

## 非线性动力系统论文智能分析助手

帮助研究人员快速理解：
- 混沌动力系统
- 非线性系统
- 忆阻神经网络
- 动力学模型分析
"""
    )


    with gr.Row():


        # 左侧输入区域
        with gr.Column(scale=1):

            gr.Markdown(
                "## 📄 论文输入"
            )


            pdf_input = gr.File(
                label="上传论文PDF",
                file_types=[".pdf"]
            )


            abstract_input = gr.Textbox(
                label="论文摘要（可选）",
                placeholder=
                "如果暂时不上传PDF，可以粘贴论文摘要...",
                lines=8
            )


            analysis_type = gr.Radio(
                choices=[
                    "快速阅读",
                    "数学模型分析",
                    "混沌动力学分析",
                    "创新点评价"
                ],
                value="快速阅读",
                label="分析模式"
            )


            analyze_button = gr.Button(
                "🚀 开始分析",
                variant="primary"
            )



        # 右侧输出区域
        with gr.Column(scale=2):

            gr.Markdown(
                "## 📑 分析结果"
            )


            output = gr.Markdown(
                """
等待分析...
"""
            )



    # 按钮连接函数

    analyze_button.click(
        fn=analyze_paper,
        inputs=[
            pdf_input,
            abstract_input,
            analysis_type
        ],
        outputs=output
    )



# 启动

if __name__ == "__main__":

    demo.launch()
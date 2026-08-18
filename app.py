import gradio as gr

from models.deepseek_api import ask_deepseek

from utils.pdf_reader import extract_pages_from_pdf
from utils.text_splitter import split_pages
from utils.chunk_selector import select_chunks
from utils.markdown_cleaner import clean_markdown

from prompts.paper_summary_prompt import PAPER_SUMMARY_PROMPT
from prompts.model_analysis_prompt import MODEL_ANALYSIS_PROMPT
from prompts.chaos_analysis_prompt import CHAOS_ANALYSIS_PROMPT
from prompts.innovation_prompt import INNOVATION_PROMPT

# -----------------------------
def analyze_paper(pdf_file, abstract, analysis_type):

    if pdf_file:

        pages = extract_pages_from_pdf(
            pdf_file.name
        )


        chunks = split_pages(
            pages
        )


        selected_chunks = select_chunks(
            chunks,
            analysis_type
        )


        paper_content = "\n\n".join(
            [
                chunk["text"]
                for chunk in selected_chunks
            ]
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



    # 替换变量

    prompt = prompt.replace(
        "{paper_text}",
        paper_content
    )

    answer = ask_deepseek(prompt)

    answer = clean_markdown(
        answer
    )

    return answer



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
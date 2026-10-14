import gradio as gr
import re
from pathlib import Path

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

from rag.gradio_service import (
    GradioRagService,
)

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


knowledge_service = GradioRagService()



def load_knowledge_scope_choices(
) -> list[tuple[str, str]]:
    """加载知识库中可选择的论文。"""

    try:
        return (
            knowledge_service
            .list_paper_choices()
        )
    except FileNotFoundError:
        # 数据库尚未创建时，
        # 网页仍然可以正常启动。
        return []

def load_knowledge_library_rows(
) -> list[list[str]]:
    """生成知识库管理表格的数据。"""

    overviews = (
        knowledge_service
        .list_paper_overviews()
    )

    rows: list[list[str]] = []

    for overview in overviews:
        paper = overview.paper

        authors_text = (
            "、".join(paper.authors)
            if paper.authors
            else "—"
        )

        year_text = (
            str(paper.year)
            if paper.year is not None
            else "—"
        )

        rows.append(
            [
                paper.title,
                authors_text,
                year_text,
                str(
                    overview.section_count
                ),
                str(
                    overview.chunk_count
                ),
                paper.id,
            ]
        )

    return rows

def refresh_knowledge_library():
    """刷新知识库列表和论文选择组件。"""

    choices = (
        load_knowledge_scope_choices()
    )

    rows = (
        load_knowledge_library_rows()
    )

    return (
        gr.update(
            choices=choices,
            value=[],
        ),
        gr.update(
            choices=choices,
            value=None,
        ),
        gr.update(
            value=rows,
        ),
        (
            "## ✅ 知识库列表已刷新\n\n"
            f"当前共 {len(rows)} 篇论文。"
        ),
    )

def delete_paper_from_web(
    paper_id,
    confirmed,
):
    """从网页删除一篇已入库论文。"""

    if not paper_id:
        return (
            "## ⚠️ 请选择需要删除的论文",
            gr.update(),
            gr.update(),
            gr.update(),
            confirmed,
        )

    if not confirmed:
        return (
            (
                "## ⚠️ 请先确认删除\n\n"
                "删除操作会同时移除该论文的"
                "章节、Chunk和检索数据。"
            ),
            gr.update(),
            gr.update(),
            gr.update(),
            confirmed,
        )

    try:
        result = (
            knowledge_service.delete_paper(
                str(paper_id)
            )
        )

        choices = (
            load_knowledge_scope_choices()
        )

        rows = (
            load_knowledge_library_rows()
        )

        if result.file_warning:
            file_message = (
                "\n\n"
                f"⚠️ {result.file_warning}"
            )
        elif result.stored_file_deleted:
            file_message = (
                "\n\n"
                "本地受管PDF文件也已删除。"
            )
        else:
            file_message = (
                "\n\n"
                "数据库内容已删除；"
                "外部原始PDF文件未被删除。"
            )

        message = (
            "## ✅ 论文已从知识库删除\n\n"
            f"- 标题：{result.paper.title}\n"
            f"- 论文 ID：`{result.paper.id}`"
            f"{file_message}"
        )

        return (
            message,
            gr.update(
                choices=choices,
                value=[],
            ),
            gr.update(
                choices=choices,
                value=None,
            ),
            gr.update(
                value=rows,
            ),
            False,
        )

    except Exception as exc:
        return (
            (
                "## ❌ 删除论文失败\n\n"
                f"{type(exc).__name__}: {exc}"
            ),
            gr.update(),
            gr.update(),
            gr.update(),
            confirmed,
        )

def ingest_paper_from_web(
    pdf_file,
    title,
    authors_text,
    year_text,
    doi,
    progress=gr.Progress(),
):
    """把网页上传的PDF导入知识库。"""

    if pdf_file is None:
        return (
            "## ⚠️ 请先选择PDF文件",
            gr.update(),
            gr.update(),
            gr.update(),
        )

    try:
        pdf_path = Path(
            getattr(
                pdf_file,
                "name",
                pdf_file,
            )
        )

        authors = tuple(
            author.strip()
            for author in re.split(
                r"[,，;；\n]+",
                authors_text or "",
            )
            if author.strip()
        )

        cleaned_year = (
            str(year_text or "").strip()
        )

        if cleaned_year:
            try:
                year = int(cleaned_year)
            except ValueError as exc:
                raise ValueError(
                    "发表年份必须是整数。"
                ) from exc

            if not 1000 <= year <= 9999:
                raise ValueError(
                    "请输入有效的四位发表年份。"
                )
        else:
            year = None

        progress(
            0.1,
            desc="正在保存并解析PDF……",
        )

        result = (
            knowledge_service.ingest_pdf(
                pdf_path,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
            )
        )

        progress(
            1.0,
            desc="论文入库完成",
        )

        status = (
            "新论文已成功入库"
            if result.created
            else "该PDF已经存在于知识库中"
        )

        message = (
            f"## ✅ {status}\n\n"
            f"- 标题：{result.paper.title}\n"
            f"- 论文 ID：`{result.paper.id}`\n"
            f"- 章节数量：{len(result.sections)}\n"
            f"- Chunk 数量：{len(result.chunks)}"
        )

        choices = (
            load_knowledge_scope_choices()
        )

        rows = (
            load_knowledge_library_rows()
        )

        return (
            message,
            gr.update(
                choices=choices,
                value=[],
            ),
            gr.update(
                choices=choices,
                value=None,
            ),
            gr.update(
                value=rows,
            ),
        )

    except Exception as exc:
        return (
            (
                "## ❌ 论文入库失败\n\n"
                f"{type(exc).__name__}: {exc}"
            ),
            gr.update(),
            gr.update(),
            gr.update(),
        )

def ask_paper_library(
    question,
    paper_scopes,
    top_k,
    progress=gr.Progress(),
):
    """根据本地论文知识库回答问题。"""

    cleaned_question = (
        question or ""
    ).strip()

    if not cleaned_question:
        return (
            "## ⚠️ 请输入问题",
            (
                "## 检索证据\n\n"
                "尚未执行检索。"
            ),
            "等待输入问题。",
        )

    paper_ids = [
        str(paper_id)
        for paper_id in (
                paper_scopes or []
        )
    ]

    try:
        progress(
            0.1,
            desc="正在准备语义检索……",
        )

        result = knowledge_service.ask(
            cleaned_question,
            top_k=int(top_k),
            paper_ids=paper_ids,
        )

        progress(
            1.0,
            desc="问答完成",
        )

        return (
            result.answer_markdown,
            result.evidence_markdown,
            result.status_text,
        )

    except ModelServiceError as exc:
        return (
            (
                "## ❌ 模型调用失败\n\n"
                f"{exc}"
            ),
            (
                "## 检索证据\n\n"
                "模型调用失败，未生成完整结果。"
            ),
            "问答失败。",
        )

    except FileNotFoundError as exc:
        return (
            (
                "## ❌ 找不到论文数据库\n\n"
                f"{exc}"
            ),
            (
                "## 检索证据\n\n"
                "请先将论文导入数据库。"
            ),
            "知识库不可用。",
        )

    except Exception as exc:
        return (
            (
                "## ❌ 知识库问答失败\n\n"
                f"{type(exc).__name__}: {exc}"
            ),
            (
                "## 检索证据\n\n"
                "本次检索未成功完成。"
            ),
            "问答失败。",
        )



# -----------------------------
# Gradio界面
# -----------------------------

with gr.Blocks(
    title="Nonlinear Dynamics Paper Analyst"
) as demo:

    gr.Markdown(
        """
# Nonlinear Dynamics Paper Analyst

## 非线性动力系统论文智能分析助手

支持单篇论文分析与科研知识库问答。
"""
    )

    with gr.Tabs():

        with gr.Tab(
            "📄 单篇论文分析"
        ):
            with gr.Row():

                with gr.Column(scale=1):
                    gr.Markdown(
                        "## 论文输入"
                    )

                    pdf_input = gr.File(
                        label="上传论文PDF",
                        file_types=[".pdf"],
                    )

                    abstract_input = gr.Textbox(
                        label="论文摘要（可选）",
                        placeholder=(
                            "如果暂时不上传PDF，"
                            "可以粘贴论文摘要……"
                        ),
                        lines=8,
                    )

                    analysis_type = gr.Radio(
                        choices=[
                            "快速阅读",
                            "数学模型分析",
                            "混沌动力学分析",
                            "创新点评价",
                        ],
                        value="快速阅读",
                        label="分析模式",
                    )

                    analyze_button = gr.Button(
                        "🚀 开始分析",
                        variant="primary",
                    )

                with gr.Column(scale=2):
                    gr.Markdown(
                        "## 分析结果"
                    )

                    analysis_output = gr.Markdown(
                        "等待分析……"
                    )

            analyze_button.click(
                fn=analyze_paper,
                inputs=[
                    pdf_input,
                    abstract_input,
                    analysis_type,
                ],
                outputs=analysis_output,
            )

        with gr.Tab(
            "🔎 科研知识库问答"
        ):
            gr.Markdown(
                """
## 基于已入库论文进行问答

系统会先检索相关论文片段，再根据检索证据生成回答。
回答下方会展示论文、章节、页码和原文证据。
"""
            )

            with gr.Accordion(
                "➕ 上传论文到知识库",
                open=False,
            ):
                gr.Markdown(
                    (
                        "上传后系统会自动解析章节、"
                        "生成Chunk并写入本地知识库。"
                    )
                )

                with gr.Row():
                    knowledge_pdf_input = gr.File(
                        label="论文PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )

                    with gr.Column():
                        knowledge_title_input = (
                            gr.Textbox(
                                label="论文标题（可选）",
                                placeholder=(
                                    "留空时使用文件名"
                                ),
                            )
                        )

                        knowledge_authors_input = (
                            gr.Textbox(
                                label="作者（可选）",
                                placeholder=(
                                    "多位作者使用逗号分隔"
                                ),
                            )
                        )

                        knowledge_year_input = (
                            gr.Textbox(
                                label="发表年份（可选）",
                                placeholder="例如：2026",
                            )
                        )

                        knowledge_doi_input = (
                            gr.Textbox(
                                label="DOI（可选）",
                            )
                        )

                knowledge_ingest_button = gr.Button(
                    "📥 导入知识库",
                    variant="secondary",
                )

                knowledge_ingest_status = (
                    gr.Markdown(
                        "尚未导入论文。"
                    )
                )

            with gr.Accordion(
                    "📚 管理已入库论文",
                    open=False,
            ):
                knowledge_library_table = (
                    gr.Dataframe(
                        headers=[
                            "论文标题",
                            "作者",
                            "年份",
                            "章节数",
                            "Chunk数",
                            "论文ID",
                        ],
                        value=(
                            load_knowledge_library_rows()
                        ),
                        datatype=[
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                            "str",
                        ],
                        interactive=False,
                        wrap=True,
                        label="知识库论文列表",
                    )
                )

                knowledge_refresh_button = (
                    gr.Button(
                        "🔄 刷新论文列表",
                        variant="secondary",
                    )
                )

                knowledge_delete_scope = (
                    gr.Dropdown(
                        choices=(
                            load_knowledge_scope_choices()
                        ),
                        value=None,
                        multiselect=False,
                        label="选择需要删除的论文",
                    )
                )

                knowledge_delete_confirm = (
                    gr.Checkbox(
                        label=(
                            "我确认删除该论文及其"
                            "章节和Chunk数据"
                        ),
                        value=False,
                    )
                )

                knowledge_delete_button = (
                    gr.Button(
                        "🗑️ 删除选中论文",
                        variant="stop",
                    )
                )

                knowledge_management_status = (
                    gr.Markdown(
                        "尚未执行管理操作。"
                    )
                )

            with gr.Row():

                with gr.Column(scale=1):
                    knowledge_scope = (
                        gr.Dropdown(
                            choices=(
                                load_knowledge_scope_choices()
                            ),
                            value=[],
                            multiselect=True,
                            label="限定论文范围",
                            info=(
                                "可选择一篇或多篇论文；"
                                "不选择时检索整个论文库。"
                            ),
                        )
                    )

                    knowledge_question = (
                        gr.Textbox(
                            label="研究问题",
                            placeholder=(
                                "例如：论文使用了"
                                "哪些动力学分析方法？"
                            ),
                            lines=5,
                        )
                    )

                    knowledge_top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="检索证据数量 Top-K",
                    )


                    knowledge_ask_button = (
                        gr.Button(
                            "🔍 检索并回答",
                            variant="primary",
                        )
                    )

                    knowledge_status = (
                        gr.Textbox(
                            label="运行状态",
                            value="等待提问。",
                            interactive=False,
                        )
                    )


                with gr.Column(scale=2):
                    gr.Markdown(
                        "## 知识库回答"
                    )

                    knowledge_answer = (
                        gr.Markdown(
                            "等待提问……"
                        )
                    )

                    knowledge_evidence = (
                        gr.Markdown(
                            (
                                "## 检索证据\n\n"
                                "尚未执行检索。"
                            )
                        )
                    )

            knowledge_ingest_button.click(
                fn=ingest_paper_from_web,
                inputs=[
                    knowledge_pdf_input,
                    knowledge_title_input,
                    knowledge_authors_input,
                    knowledge_year_input,
                    knowledge_doi_input,
                ],
                outputs=[
                    knowledge_ingest_status,
                    knowledge_scope,
                    knowledge_delete_scope,
                    knowledge_library_table,
                ],
            )

            knowledge_refresh_button.click(
                fn=refresh_knowledge_library,
                inputs=[],
                outputs=[
                    knowledge_scope,
                    knowledge_delete_scope,
                    knowledge_library_table,
                    knowledge_management_status,
                ],
            )

            knowledge_delete_button.click(
                fn=delete_paper_from_web,
                inputs=[
                    knowledge_delete_scope,
                    knowledge_delete_confirm,
                ],
                outputs=[
                    knowledge_management_status,
                    knowledge_scope,
                    knowledge_delete_scope,
                    knowledge_library_table,
                    knowledge_delete_confirm,
                ],
            )

            knowledge_ask_button.click(
                fn=ask_paper_library,
                inputs=[
                    knowledge_question,
                    knowledge_scope,
                    knowledge_top_k,
                ],
                outputs=[
                    knowledge_answer,
                    knowledge_evidence,
                    knowledge_status,
                ],
            )



if __name__ == "__main__":
    demo.launch()
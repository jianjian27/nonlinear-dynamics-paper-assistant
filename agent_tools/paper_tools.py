from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_tools.base import AgentTool

from database.connection import create_connection

from database.paper_repository import PaperRepository

from database.section_repository import SectionRepository


class ListPapersInput(BaseModel):
    """list_papers不需要业务参数。"""

    model_config = ConfigDict(
        extra="forbid",
    )


class PaperToolItem(BaseModel):
    """一篇论文返回给Agent的信息。"""

    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    section_count: int
    chunk_count: int


class ListPapersOutput(BaseModel):
    """list_papers的结构化执行结果。"""

    database_available: bool
    total: int
    papers: list[PaperToolItem]


class GetPaperOverviewInput(BaseModel):
    """查询单篇论文概览的输入。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    paper_id: str = Field(
        description=(
            "需要查询的论文唯一ID，"
            "通常由list_papers工具获得。"
        ),
    )

    @field_validator("paper_id") # Pydantic 字段验证装饰器
    # 创建模型对象时，自动验证和处理 paper_id 字段。
    # 验证方法必须返回处理后的字段值；
    # 若字段不合法，可以抛出 ValueError。
    @classmethod
    def normalize_paper_id(
        cls,
        value: str,
    ) -> str:
        """清理并验证论文ID。"""

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "paper_id不能为空。"
            )

        return cleaned_value


class PaperSectionItem(BaseModel):
    """论文中的一个章节。"""

    section_index: int
    title: str
    level: int
    section_path: list[str]
    page_start: int | None
    page_end: int | None


class PaperOverviewItem(PaperToolItem):
    """一篇论文的详细概览。"""

    file_name: str
    sections: list[PaperSectionItem]


class GetPaperOverviewOutput(BaseModel):
    """查询论文概览工具的输出。"""

    database_available: bool
    found: bool
    paper: PaperOverviewItem | None



class ListPapersTool(AgentTool):
    """列出知识库中的论文。"""

    name = "list_papers"

    description = (
        "列出本地论文知识库中的全部论文，"
        "并返回论文ID、标题、作者、年份、"
        "章节数量和Chunk数量。"
        "当用户询问知识库中有哪些论文，"
        "或者需要先确定论文范围时使用。"
    )

    input_model = ListPapersInput

    def __init__(
        self,
        database_path: str | Path =(
            "data/papers.db"
        ),
    ) -> None:
        self.database_path = Path(
            database_path
        ).resolve()

    def run(
        self,
        arguments: BaseModel,
    ) -> ListPapersOutput:
        """读取论文列表并转换为结构化结果。"""

        if not isinstance(
            arguments,
            ListPapersInput
        ):
            raise TypeError(
                "List_papers收到错误的参数模型。"
            )

        if not self.database_path.is_file():
            return ListPapersOutput(
                database_available=False,
                total=0,
                papers=[],
            )

        connection = create_connection(self.database_path)

        try:
            repository = PaperRepository(connection)

            overviews = (repository.list_all_with_counts())
        finally:
            connection.close()

        papers = [
            PaperToolItem(
                paper_id=overview.paper.id,
                title=overview.paper.title,
                authors=list(overview.paper.authors),
                year=overview.paper.year,
                doi=overview.paper.doi,
                section_count=overview.section_count,
                chunk_count=overview.chunk_count,
            )
            for overview in overviews
        ]

        return ListPapersOutput(
            database_available=True,
            total=len(papers),
            papers=papers,
        )




class GetPaperOverviewTool(AgentTool):
    """查询一篇论文的详细概览。"""

    name = "get_paper_overview"

    description = (
        "根据论文ID查询一篇已入库论文的"
        "详细信息，包括标题、作者、年份、"
        "DOI、章节数量、Chunk数量和章节列表。"
        "paper_id通常应先通过list_papers获得。"
        "当用户要求查看某篇论文的结构、"
        "元数据或内容概况时使用。"
    )

    input_model = GetPaperOverviewInput

    def __init__(
        self,
        database_path: str | Path = (
            "data/papers.db"
        ),
    ) -> None:
        self.database_path = Path(
            database_path
        ).resolve()

    def run(
        self,
        arguments: BaseModel,
    ) -> GetPaperOverviewOutput:
        """查询论文和章节结构。"""

        if not isinstance(
            arguments,
            GetPaperOverviewInput,
        ):
            raise TypeError(
                "get_paper_overview收到"
                "错误的参数模型。"
            )

        if not self.database_path.is_file():
            return GetPaperOverviewOutput(
                database_available=False,
                found=False,
                paper=None,
            )

        connection = create_connection(self.database_path)

        try:
            paper_repository = (
                PaperRepository(
                    connection
                )
            )

            overview = (
                paper_repository
                .get_overview_by_id(
                    arguments.paper_id
                )
            )

            if overview is None:
                return (
                    GetPaperOverviewOutput(
                        database_available=True,
                        found=False,
                        paper=None,
                    )
                )

            section_repository = (
                SectionRepository(
                    connection
                )
            )

            sections = (
                section_repository
                .list_by_paper(
                    arguments.paper_id
                )
            )
        finally:
            connection.close()

        paper = overview.paper

        section_items = [
            PaperSectionItem(
                section_index=(
                    section.section_index
                ),
                title=section.title,
                level=section.level,
                section_path=list(
                    section.section_path
                ),
                page_start=(
                    section.page_start
                ),
                page_end=(
                    section.page_end
                ),
            )
            for section in sections
        ]

        return GetPaperOverviewOutput(
            database_available=True,
            found=True,
            paper=PaperOverviewItem(
                paper_id=paper.id,
                title=paper.title,
                authors=list(
                    paper.authors
                ),
                year=paper.year,
                doi=paper.doi,
                section_count=(
                    overview.section_count
                ),
                chunk_count=(
                    overview.chunk_count
                ),
                file_name=paper.file_name,
                sections=section_items,
            ),
        )


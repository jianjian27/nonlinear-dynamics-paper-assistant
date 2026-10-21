from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agent_tools.base import AgentTool

from database.connection import create_connection

from database.paper_repository import PaperRepository


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
    ) -> BaseModel:
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

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from agent_tools.base import AgentTool
from database.chunk_repository import (
    ChunkRepository,
)
from database.connection import (
    create_connection,
)
from database.paper_repository import (
    PaperRepository,
)
from embedding.base import TextEncoder
from embedding.sentence_transformer_encoder import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEncoder,
)
from embedding.vector_cache import (
    VectorCache,
)
from retrieval.semantic_retriever import (
    SemanticRetriever,
)

class SearchPaperEvidenceInput(BaseModel):
    """检索论文原文证据的输入参数。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str = Field(
        description=(
            "需要从论文知识库中检索的"
            "自然语言问题。"
        ),
    )

    paper_ids: list[str] | None = Field(
        default=None,
        description=(
            "需要限定的论文ID列表。"
            "不提供时检索整个论文库；"
            "论文ID通常由list_papers工具获得。"
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "最多返回多少条论文原文证据，"
            "取值范围为1到10。"
        ),
    )

    @field_validator("query")
    @classmethod
    def normalize_query(
            cls,
            value: str,
    ) -> str:
        """清理并验证检索问题。"""

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "query不能为空。"
            )

        return cleaned_value

    @field_validator("paper_ids")
    @classmethod
    def normalize_paper_ids(
            cls,
            value: list[str] | None,
    ) -> list[str] | None:
        """清理、去重并排序论文ID。"""

        if value is None:
            return None

        normalized_ids = sorted(
            {
                paper_id.strip()
                for paper_id in value
                if paper_id.strip()
            }
        )

        if not normalized_ids:
            raise ValueError(
                "paper_ids不能是空列表。"
            )

        return normalized_ids


class EvidenceToolItem(BaseModel):
    """一条返回给Agent的论文原文证据。"""

    rank: int
    score: float

    paper_id: str
    paper_title: str

    chunk_id: str

    section_path: list[str]

    page_start: int
    page_end: int

    content: str


class SearchPaperEvidenceOutput(BaseModel):
    """论文证据检索工具的结构化输出。"""

    database_available: bool

    scope_paper_ids: list[str] | None

    indexed_count: int
    result_count: int

    evidences: list[EvidenceToolItem]


@dataclass(frozen=True)
class _RetrieverRuntime:
    """一个论文检索范围对应的运行状态。"""

    retriever: SemanticRetriever

    paper_ids: tuple[str, ...] | None

    indexed_count: int


class SearchPaperEvidenceTool(AgentTool):
    """从论文知识库中检索可追踪原文证据。"""

    name = "search_paper_evidence"

    description = (
        "根据自然语言问题从论文知识库中"
        "检索相关原文证据，返回论文标题、"
        "章节路径、页码、相似度和原文内容。"
        "可检索整个知识库，也可通过paper_ids"
        "限定一篇或多篇论文。"
        "当用户询问论文内容、研究方法、"
        "数学模型、实验结果、结论或需要"
        "比较多篇论文时使用。"
    )

    input_model = SearchPaperEvidenceInput

    def __init__(
        self,
        database_path: str | Path = (
            "data/papers.db"
        ),
        *,
        vector_cache_directory: str | Path = (
            "data/vector_indexes"
        ),
        embedding_model: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        encoder: TextEncoder | None = None,
    ) -> None:
        self.database_path = Path(
            database_path
        ).resolve()

        self.vector_cache_directory = Path(
            vector_cache_directory
        )

        self.embedding_model = embedding_model
        self.device = device

        self._encoder = encoder

        self._runtimes: dict[
            str,
            _RetrieverRuntime,
        ] = {}

        self._lock = Lock()

    def run(
        self,
        arguments: BaseModel,
    ) -> SearchPaperEvidenceOutput:
        """执行论文证据检索。"""

        if not isinstance(
            arguments,
            SearchPaperEvidenceInput,
        ):
            raise TypeError(
                "search_paper_evidence收到"
                "错误的参数模型。"
            )

        if not self.database_path.is_file():
            return SearchPaperEvidenceOutput(
                database_available=False,
                scope_paper_ids=(
                    arguments.paper_ids
                ),
                indexed_count=0,
                result_count=0,
                evidences=[],
            )

        paper_ids = (
            tuple(arguments.paper_ids)
            if arguments.paper_ids is not None
            else None
        )

        with self._lock:
            runtime = (
                self._get_or_build_runtime(
                    paper_ids
                )
            )

            if runtime.indexed_count == 0:
                return SearchPaperEvidenceOutput(
                    database_available=True,
                    scope_paper_ids=(
                        list(paper_ids)
                        if paper_ids is not None
                        else None
                    ),
                    indexed_count=0,
                    result_count=0,
                    evidences=[],
                )

            results = runtime.retriever.search(
                arguments.query,
                top_k=arguments.top_k,
            )

            paper_titles = (
                self._load_paper_titles(
                    {
                        result.chunk.paper_id
                        for result in results
                    }
                )
            )

        evidences = [
            EvidenceToolItem(
                rank=rank,
                score=result.score,
                paper_id=(
                    result.chunk.paper_id
                ),
                paper_title=paper_titles.get(
                    result.chunk.paper_id,
                    result.chunk.paper_id,
                ),
                chunk_id=result.chunk.id,
                section_path=list(
                    result.chunk.section_path
                ),
                page_start=(
                    result.chunk.page_start
                ),
                page_end=(
                    result.chunk.page_end
                ),
                content=result.chunk.content,
            )
            for rank, result in enumerate(
                results,
                start=1,
            )
        ]

        return SearchPaperEvidenceOutput(
            database_available=True,
            scope_paper_ids=(
                list(paper_ids)
                if paper_ids is not None
                else None
            ),
            indexed_count=(
                runtime.indexed_count
            ),
            result_count=len(evidences),
            evidences=evidences,
        )

    def _get_or_build_runtime(
        self,
        paper_ids: tuple[str, ...] | None,
    ) -> _RetrieverRuntime:
        """获取或构建指定范围的检索器。"""

        scope_key = self._build_scope_key(
            paper_ids
        )

        cached_runtime = self._runtimes.get(
            scope_key
        )

        if cached_runtime is not None:
            return cached_runtime

        encoder = (
            self._get_or_create_encoder()
        )

        vector_cache = VectorCache.for_index(
            self.vector_cache_directory,
            model_name=encoder.identifier,
            scope=scope_key,
            source_id=str(
                self.database_path
            ),
        )

        connection = create_connection(
            self.database_path
        )

        try:
            retriever = SemanticRetriever(
                chunk_repository=ChunkRepository(
                    connection
                ),
                encoder=encoder,
                vector_cache=vector_cache,
            )

            if paper_ids is None:
                indexed_count = (
                    retriever.build_index()
                )

            elif len(paper_ids) == 1:
                indexed_count = (
                    retriever.build_index(
                        paper_id=paper_ids[0]
                    )
                )

            else:
                indexed_count = (
                    retriever.build_index(
                        paper_ids=paper_ids
                    )
                )

        finally:
            connection.close()

        runtime = _RetrieverRuntime(
            retriever=retriever,
            paper_ids=paper_ids,
            indexed_count=indexed_count,
        )

        self._runtimes[
            scope_key
        ] = runtime

        return runtime

    def _get_or_create_encoder(
        self,
    ) -> TextEncoder:
        """延迟创建并复用Embedding模型。"""

        if self._encoder is None:
            self._encoder = (
                SentenceTransformerEncoder(
                    model_name=(
                        self.embedding_model
                    ),
                    device=self.device,
                )
            )

        return self._encoder

    def _load_paper_titles(
        self,
        paper_ids: set[str],
    ) -> dict[str, str]:
        """查询检索结果对应的论文标题。"""

        connection = create_connection(
            self.database_path
        )

        try:
            repository = PaperRepository(
                connection
            )

            titles: dict[str, str] = {}

            for paper_id in paper_ids:
                paper = repository.get_by_id(
                    paper_id
                )

                if paper is not None:
                    titles[paper_id] = (
                        paper.title
                    )

            return titles

        finally:
            connection.close()

    @staticmethod
    def _build_scope_key(
        paper_ids: tuple[str, ...] | None,
    ) -> str:
        """生成稳定的向量缓存范围标识。"""

        if paper_ids is None:
            return "all"

        if len(paper_ids) == 1:
            return (
                f"paper:{paper_ids[0]}"
            )

        return (
            "papers:"
            + ",".join(paper_ids)
        )
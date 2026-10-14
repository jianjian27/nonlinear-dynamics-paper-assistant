from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from database.chunk_repository import (
    ChunkRepository,
)
from database.connection import (
    create_connection,
)
from database.paper_repository import (
    PaperRepository,
)
from embedding.sentence_transformer_encoder import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEncoder,
)
from embedding.vector_cache import (
    VectorCache,
)
from models.deepseek_api import (
    AliyunChatClient,
)
from rag.answer_service import (
    RagAnswer,
    RagAnswerService,
)
from rag.context_builder import (
    ContextBuilder,
)
from rag.prompt import (
    RagPromptBuilder,
)
from retrieval.semantic_retriever import (
    SemanticRetriever,
)
from domain.paper import (
    Paper,
)
from collections.abc import Sequence

@dataclass(frozen=True)
class GradioRagResult:
    """适合交给Gradio展示的RAG结果。"""

    answer_markdown: str
    evidence_markdown: str
    status_text: str

@dataclass(frozen=True)
class RetrieverRuntime:
    """某个检索范围对应的运行状态。"""

    retriever: SemanticRetriever
    paper_ids: tuple[str, ...] | None
    indexed_count: int
    used_cache: bool

    @property
    def scope_key(self) -> str:
        """返回当前检索范围的唯一标识。"""

        if self.paper_ids is None:
            return "all"

        if len(self.paper_ids) == 1:
            return (
                f"paper:{self.paper_ids[0]}"
            )

        return (
            "papers:"
            + ",".join(self.paper_ids)
        )

class GradioRagService:
    """为Gradio界面提供可复用的RAG问答服务。"""

    def __init__(
        self,
        *,
        database_path: str | Path = (
            "data/papers.db"
        ),
        vector_cache_directory: str | Path = (
            "data/vector_indexes"
        ),
        embedding_model: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        max_context_chars: int = 12000,
        max_chunk_chars: int = 3000,
    ) -> None:
        self.database_path = Path(
            database_path
        ).resolve()

        self.vector_cache_directory = Path(
            vector_cache_directory
        )

        self.embedding_model = embedding_model
        self.device = device

        self.max_context_chars = max_context_chars
        self.max_chunk_chars = max_chunk_chars

        self._encoder: (
                SentenceTransformerEncoder | None
        ) = None

        self._retriever_runtimes: dict[
            str,
            RetrieverRuntime,
        ] = {}

        # 防止多个网页请求同时初始化模型或调用检索器。
        self._lock = Lock()

    def list_paper_choices(
        self,
    ) -> list[tuple[str, str]]:
        """返回适合Gradio选择组件使用的论文选项。"""

        if not self.database_path.is_file():
            raise FileNotFoundError(
                "找不到论文数据库："
                f"{self.database_path}"
            )

        connection = create_connection(
            self.database_path
        )

        try:
            repository = PaperRepository(
                connection
            )

            papers = repository.list_all()
        finally:
            connection.close()

        return [
            (
                self._format_paper_label(
                    paper
                ),
                paper.id,
            )
            for paper in papers
        ]

    @staticmethod
    def _format_paper_label(
        paper: Paper,
    ) -> str:
        """生成论文选择组件中显示的文字。"""

        details: list[str] = []

        if paper.year is not None:
            details.append(
                str(paper.year)
            )

        if paper.authors:
            details.append(
                "、".join(paper.authors)
            )

        if not details:
            return paper.title

        detail_text = "；".join(details)

        return (
            f"{paper.title}"
            f"（{detail_text}）"
        )

    @staticmethod
    def _normalize_paper_ids(
        paper_ids: Sequence[str] | None,
    ) -> tuple[str, ...] | None:
        """清理、去重并排序论文ID。"""

        if paper_ids is None:
            return None

        normalized_ids = tuple(
            sorted(
                {
                    paper_id.strip()
                    for paper_id in paper_ids
                    if paper_id.strip()
                }
            )
        )

        if not normalized_ids:
            return None

        return normalized_ids

    def ask(
            self,
            question: str,
            *,
            top_k: int = 5,
            paper_ids: Sequence[str] | None = None,
    ) -> GradioRagResult:
        """在指定论文范围内执行RAG问答。"""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "请输入需要查询的问题。"
            )

        if top_k <= 0:
            raise ValueError(
                "Top-K必须大于0。"
            )

        normalized_paper_ids = (
            self._normalize_paper_ids(
                paper_ids
            )
        )

        with self._lock:
            runtime = self._get_or_build_retriever(
                paper_ids=normalized_paper_ids
            )

            connection = create_connection(
                self.database_path
            )

            try:
                answer_service = RagAnswerService(
                    retriever=runtime.retriever,
                    paper_repository=(
                        PaperRepository(connection)
                    ),
                    context_builder=ContextBuilder(
                        max_total_content_chars=(
                            self.max_context_chars
                        ),
                        max_chunk_chars=(
                            self.max_chunk_chars
                        ),
                    ),
                    prompt_builder=(
                        RagPromptBuilder()
                    ),
                    chat_client=(
                        AliyunChatClient()
                    ),
                )

                rag_answer = (
                    answer_service.answer(
                        cleaned_question,
                        top_k=top_k,
                    )
                )
            finally:
                connection.close()

        return GradioRagResult(
            answer_markdown=(
                rag_answer.answer
            ),
            evidence_markdown=(
                self._render_evidence(
                    rag_answer
                )
            ),
            status_text=(
                self._build_status_text(
                    rag_answer,
                    runtime,
                )
            ),
        )

    def _get_or_build_retriever(
            self,
            *,
            paper_ids: tuple[str, ...] | None,
    ) -> RetrieverRuntime:
        """获取或构建指定范围的检索器。"""

        if paper_ids is None:
            scope_key = "all"
        elif len(paper_ids) == 1:
            scope_key = (
                f"paper:{paper_ids[0]}"
            )
        else:
            scope_key = (
                    "papers:"
                    + ",".join(paper_ids)
            )

        cached_runtime = (
            self._retriever_runtimes.get(
                scope_key
            )
        )

        if cached_runtime is not None:
            return cached_runtime

        if not self.database_path.is_file():
            raise FileNotFoundError(
                "找不到论文数据库："
                f"{self.database_path}"
            )

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
                chunk_repository=(
                    ChunkRepository(connection)
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

        if indexed_count == 0:
            if paper_ids is None:
                message = (
                    "论文数据库中没有"
                    "可检索的Chunk。"
                )
            else:
                message = (
                    "所选论文中没有"
                    "可检索的Chunk。"
                )

            raise RuntimeError(message)

        runtime = RetrieverRuntime(
            retriever=retriever,
            paper_ids=paper_ids,
            indexed_count=indexed_count,
            used_cache=(
                retriever.last_build_used_cache
            ),
        )

        self._retriever_runtimes[
            scope_key
        ] = runtime

        return runtime

    def _get_or_create_encoder(
        self,
    ) -> SentenceTransformerEncoder:
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

    @staticmethod
    def _build_status_text(
            rag_answer: RagAnswer,
            runtime: RetrieverRuntime,
    ) -> str:
        """生成界面底部的运行状态。"""

        vector_status = (
            "已从磁盘缓存加载"
            if runtime.used_cache
            else "已重新计算并写入缓存"
        )

        if runtime.paper_ids is None:
            scope_text = "整个论文库"
        elif len(runtime.paper_ids) == 1:
            scope_text = "1 篇指定论文"
        else:
            scope_text = (
                f"{len(runtime.paper_ids)} "
                "篇指定论文"
            )

        return (
            f"检索范围：{scope_text}；"
            f"索引包含 {runtime.indexed_count} 个Chunk；"
            f"本次使用 {rag_answer.retrieved_count} 条证据；"
            f"向量状态：{vector_status}。"
        )

    @staticmethod
    def _render_evidence(rag_answer: RagAnswer) -> str:
        """"把结构化证据转换成Markdown。"""

        if rag_answer.context.is_empty:
            return (
                "## 检索证据\n\n"
                "没有检索到可用证据。"
            )

        sections =[
            "## 检索证据"
        ]

        for evidence in (
            rag_answer.context.evidences
        ):
            content = " ".join(
                evidence.content.split()
            )

            if len(content) > 800:
                content = (
                        content[:799] + "…"
                )

            quoted_content = (
                    "> "
                    + content.replace(
                "\n",
                "\n> ",
            )
            )

            sections.append(
                "\n".join(
                    [
                        (
                            "### "
                            f"[{evidence.citation_label}] "
                            f"{evidence.paper_title}"
                        ),
                        "",
                        (
                            f"- 章节："
                            f"{evidence.section_text}"
                        ),
                        (
                            f"- 页码："
                            f"{evidence.page_text}"
                        ),
                        (
                            f"- 相似度："
                            f"{evidence.score:.4f}"
                        ),
                        (
                            f"- Chunk ID："
                            f"`{evidence.chunk_id}`"
                        ),
                        "",
                        quoted_content,
                    ]
                )
            )

        return "\n\n".join(sections)
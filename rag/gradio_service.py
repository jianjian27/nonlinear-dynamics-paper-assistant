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

from shutil import copy2

from database.schema import (
    initialize_database,
)
from ingestion.file_hash import (
    calculate_file_sha256,
)
from ingestion.paper_content_ingestor import (
    PaperContentIngestorResult,
)
from ingestion.pipeline_factory import (
    create_pdf_ingestion_pipeline,
)
from domain.paper_overview import (
    PaperOverview,
)

@dataclass(frozen=True)
class GradioRagResult:
    """适合交给Gradio展示的RAG结果。"""

    answer_markdown: str
    evidence_markdown: str
    status_text: str

@dataclass(frozen=True)
class PaperDeletionResult:
    """删除论文及其本地文件的结果。"""

    paper: Paper
    stored_file_deleted: bool
    file_warning: str | None = None

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
        paper_storage_directory: str | Path = (
            "data/papers"
        ),
    ) -> None:
        self.database_path = Path(
            database_path
        ).resolve()

        self.vector_cache_directory = Path(
            vector_cache_directory
        )

        self.paper_storage_directory = Path(
            paper_storage_directory
        ).resolve()

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

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        *,
        title: str | None = None,
        authors: Sequence[str] = (),
        year: int | None = None,
        doi: str | None = None,
    ) -> PaperContentIngestorResult:
        """保存上传的PDF并执行结构化入库。"""

        source_path = Path(
            pdf_path
        ).resolve()

        if not source_path.is_file():
            raise FileNotFoundError(
                "找不到上传的PDF文件："
                f"{source_path}"
            )

        if source_path.suffix.lower() != ".pdf":
            raise ValueError(
                "只能导入PDF文件。"
            )

        cleaned_title = (
            title.strip()
            if title and title.strip()
            else source_path.stem
        )

        cleaned_authors = tuple(
            author.strip()
            for author in authors
            if author.strip()
        )

        cleaned_doi = (
            doi.strip()
            if doi and doi.strip()
            else None
        )

        file_hash = calculate_file_sha256(
            source_path
        )

        self.paper_storage_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        stored_path = (
            self.paper_storage_directory
            / f"{file_hash}.pdf"
        )

        if not stored_path.exists():
            copy2(
                source_path,
                stored_path,
            )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            connection = create_connection(
                self.database_path
            )

            try:
                initialize_database(
                    connection
                )

                pipeline = (
                    create_pdf_ingestion_pipeline(
                        connection
                    )
                )

                result = pipeline.ingest(
                    pdf_path=stored_path,
                    title=cleaned_title,
                    authors=cleaned_authors,
                    year=year,
                    doi=cleaned_doi,
                )
            finally:
                connection.close()

            if result.created:
                # 数据库内容已经变化，
                # 旧的内存索引不能继续使用。
                self._retriever_runtimes.clear()

        return result

    def list_paper_overviews(
        self,
    ) -> list[PaperOverview]:
        """返回知识库中的论文及内容统计。"""

        if not self.database_path.is_file():
            return []

        connection = create_connection(
            self.database_path
        )

        try:
            repository = PaperRepository(
                connection
            )

            return (
                repository
                .list_all_with_counts()
            )
        finally:
            connection.close()

    def delete_paper(
        self,
        paper_id: str,
    ) -> PaperDeletionResult:
        """删除论文、结构化内容和受管PDF文件。"""

        cleaned_paper_id = (
            paper_id.strip()
        )

        if not cleaned_paper_id:
            raise ValueError(
                "请选择需要删除的论文。"
            )

        if not self.database_path.is_file():
            raise FileNotFoundError(
                "找不到论文数据库："
                f"{self.database_path}"
            )

        with self._lock:
            connection = create_connection(
                self.database_path
            )

            try:
                repository = PaperRepository(
                    connection
                )

                deleted_paper = (
                    repository.delete_by_id(
                        cleaned_paper_id
                    )
                )
            finally:
                connection.close()

            if deleted_paper is None:
                raise ValueError(
                    "知识库中不存在该论文。"
                )

            # 删除论文后，所有已有内存索引都可能过期。
            self._retriever_runtimes.clear()

        stored_file_deleted = False
        file_warning = None

        paper_file_path = Path(
            deleted_paper.file_path
        ).resolve()

        managed_directory = (
            self.paper_storage_directory
            .resolve()
        )

        # 只允许删除 data/papers 管理范围内的文件。
        # 通过命令行导入的外部原文件不会被删除。
        if (
            paper_file_path.is_file()
            and paper_file_path.is_relative_to(
                managed_directory
            )
        ):
            try:
                paper_file_path.unlink()
                stored_file_deleted = True
            except OSError as exc:
                file_warning = (
                    "数据库记录已删除，"
                    "但本地PDF文件删除失败："
                    f"{exc}"
                )

        return PaperDeletionResult(
            paper=deleted_paper,
            stored_file_deleted=(
                stored_file_deleted
            ),
            file_warning=file_warning,
        )

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
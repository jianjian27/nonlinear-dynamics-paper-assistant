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

@dataclass(frozen=True)
class GradioRagResult:
    """适合交给Gradio展示的RAG结果。"""

    answer_markdown: str
    evidence_markdown: str
    status_text: str

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

        self._retriever: (
            SemanticRetriever | None
        ) = None

        self._indexed_count = 0
        self._used_cache = False

        # 防止多个网页请求同时初始化模型或调用检索器。
        self._lock = Lock()

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> GradioRagResult:
        """执行一次面向网页的RAG问答。"""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "请输入需要查询的问题。"
            )

        if top_k <= 0:
            raise ValueError(
                "Top-K必须大于0。"
            )

        with self._lock:
            retriever = (
                self._get_or_build_retriever()
            )

            connection = create_connection(
                self.database_path
            )

            try:
                answer_service = RagAnswerService(
                    retriever=retriever,
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
                    rag_answer
                )
            ),
        )

    def _get_or_build_retriever(self) -> SemanticRetriever:
        """第一次提问时构建检索器，之后直接复用。"""

        if self._retriever is not None:
            return self._retriever

        if not self.database_path.is_file():
            raise FileNotFoundError(
                "找不到论文数据库："
                f"{self.database_path}"
            )

        encoder = SentenceTransformerEncoder(
            model_name=self.embedding_model,
            device=self.device,
        )

        vector_cache = VectorCache.for_index(
            self.vector_cache_directory,
            model_name=encoder.identifier,
            scope="all",
            source_id=str(self.database_path),
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

            indexed_count = (
                retriever.build_index()
            )

        finally:
            connection.close()

        if indexed_count == 0:
            raise RuntimeError(
                "论文数据库中没有可检索的Chunk。"
            )

        self._retriever = retriever
        self._indexed_count = indexed_count
        self._used_cache = (
            retriever.last_build_used_cache
        )

        return retriever

    def _build_status_text(self, rag_answer: RagAnswer) -> str:
        """生成界面底部的运行状态。"""

        vector_status = (
            "已从磁盘缓存加载"
            if self._used_cache
            else "已重新计算并写入缓存"
        )

        return (
            f"索引包含 {self._indexed_count} 个Chunk；"
            f"本次使用 {rag_answer.retrieved_count} 条证据；"
            f"向量状态：{vector_status}。"
        )

    @staticmethod
    def _render_evidence(rag_answer: RagAnswer) -> str:
        """"把结构化证据转换成Mardown。"""

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
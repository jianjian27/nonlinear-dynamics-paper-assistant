import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

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
from models.deepseek_api import (
    AliyunChatClient,
    ModelServiceError,
)
from rag.answer_service import (
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


def shorten_text(
    text: str,
    max_chars: int,
) -> str:
    """生成适合命令行展示的原文预览。"""

    normalized = " ".join(
        text.split()
    )

    if len(normalized) <= max_chars:
        return normalized

    if max_chars == 1:
        return "…"

    return (
        normalized[:max_chars - 1]
        + "…"
    )


def create_argument_parser(
) -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "根据SQLite论文库中的证据，"
            "生成带引用的RAG回答。"
        )
    )

    parser.add_argument(
        "question",
        help="需要向论文知识库提出的问题。",
    )

    parser.add_argument(
        "--database",
        default="data/papers.db",
        help=(
            "SQLite数据库路径，默认是"
            " data/papers.db。"
        ),
    )

    parser.add_argument(
        "--paper-id",
        default=None,
        help=(
            "只检索指定论文；"
            "不填写时检索整个论文库。"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索多少个相关Chunk。",
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL_NAME,
        help="本地Embedding模型名称。",
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Embedding模型运行设备，"
            "例如cuda或cpu。"
        ),
    )

    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=12000,
        help="证据原文的总字符数上限。",
    )

    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=3000,
        help="单个证据的字符数上限。",
    )

    parser.add_argument(
        "--evidence-preview-chars",
        type=int,
        default=500,
        help=(
            "命令行中每条证据原文的"
            "最大显示字符数。"
        ),
    )

    return parser


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """执行一次完整的RAG论文问答。"""

    parser = create_argument_parser()
    options = parser.parse_args(arguments)

    question = options.question.strip()

    if not question:
        print(
            "错误：问题不能为空。",
            file=sys.stderr,
        )
        return 1

    if options.top_k <= 0:
        print(
            "错误：top-k必须大于0。",
            file=sys.stderr,
        )
        return 1

    if options.max_context_chars <= 0:
        print(
            "错误：max-context-chars必须大于0。",
            file=sys.stderr,
        )
        return 1

    if options.max_chunk_chars <= 0:
        print(
            "错误：max-chunk-chars必须大于0。",
            file=sys.stderr,
        )
        return 1

    if options.evidence_preview_chars <= 0:
        print(
            "错误：evidence-preview-chars必须大于0。",
            file=sys.stderr,
        )
        return 1

    database_path = Path(
        options.database
    ).resolve()

    if not database_path.is_file():
        print(
            f"错误：找不到数据库：{database_path}",
            file=sys.stderr,
        )
        return 1

    connection = create_connection(
        database_path
    )

    try:
        chunk_repository = ChunkRepository(
            connection
        )

        paper_repository = PaperRepository(
            connection
        )

        print("正在加载本地Embedding模型……")

        encoder = SentenceTransformerEncoder(
            model_name=(
                options.embedding_model
            ),
            device=options.device,
        )

        retriever = SemanticRetriever(
            chunk_repository=chunk_repository,
            encoder=encoder,
        )

        print("正在读取Chunk并构建向量索引……")

        indexed_count = retriever.build_index(
            paper_id=options.paper_id
        )

        if indexed_count == 0:
            print(
                "错误：检索范围中没有可用的Chunk。",
                file=sys.stderr,
            )
            return 1

        print(
            "索引构建完成，共包含"
            f" {indexed_count} 个Chunk。"
        )

        context_builder = ContextBuilder(
            max_total_content_chars=(
                options.max_context_chars
            ),
            max_chunk_chars=(
                options.max_chunk_chars
            ),
        )

        prompt_builder = RagPromptBuilder()

        print("正在连接阿里云模型服务……")

        chat_client = AliyunChatClient()

        answer_service = RagAnswerService(
            retriever=retriever,
            paper_repository=paper_repository,
            context_builder=context_builder,
            prompt_builder=prompt_builder,
            chat_client=chat_client,
        )

        print("正在检索证据并生成回答……")

        rag_answer = answer_service.answer(
            question,
            top_k=options.top_k,
        )

        print()
        print("=== RAG回答 ===")
        print()
        print(rag_answer.answer)

        print()
        print("=== 回答所用证据 ===")

        if rag_answer.context.is_empty:
            print("没有检索到可用证据。")

        for evidence in (
            rag_answer.context.evidences
        ):
            preview = shorten_text(
                evidence.content,
                options.evidence_preview_chars,
            )

            print()
            print(
                f"[{evidence.citation_label}] "
                f"相似度：{evidence.score:.4f}"
            )
            print(
                f"论文：{evidence.paper_title}"
            )
            print(
                f"论文ID：{evidence.paper_id}"
            )
            print(
                f"章节：{evidence.section_text}"
            )
            print(
                f"页码：{evidence.page_text}"
            )
            print(
                f"Chunk ID：{evidence.chunk_id}"
            )
            print(f"原文：{preview}")

    except ModelServiceError as exc:
        print(
            f"模型调用失败：{exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            (
                "RAG问答失败："
                f"{type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        return 1
    finally:
        connection.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
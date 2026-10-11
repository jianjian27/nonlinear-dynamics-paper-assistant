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
from retrieval.semantic_retriever import (
    SemanticRetriever,
)


def shorten_text(
    text: str,
    max_chars: int,
) -> str:
    """压缩空白，并生成指定长度的文本预览。"""

    normalized = " ".join(
        text.split()
    )

    if len(normalized) <= max_chars:
        return normalized

    return (
        normalized[:max_chars]
        + "……"
    )


def format_pages(
    page_start: int,
    page_end: int,
) -> str:
    """把起止页码转换为适合显示的文字。"""

    if page_start == page_end:
        return f"第{page_start}页"

    return (
        f"第{page_start}"
        f"—{page_end}页"
    )


def create_argument_parser(
) -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用本地Embedding模型，"
            "从SQLite论文库中检索相关原文。"
        )
    )

    parser.add_argument(
        "query",
        help="需要检索的自然语言问题。",
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
        help="返回最相关的几个Chunk。",
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="每条结果最多显示多少个字符。",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Sentence Transformer模型名称。",
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "模型运行设备，例如 cuda 或 cpu；"
            "不填写时由模型自动选择。"
        ),
    )

    return parser


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """执行一次论文语义检索。"""

    parser = create_argument_parser()
    options = parser.parse_args(arguments)

    query = options.query.strip()

    if not query:
        print(
            "错误：检索问题不能为空。",
            file=sys.stderr,
        )
        return 1

    if options.top_k <= 0:
        print(
            "错误：top-k必须大于0。",
            file=sys.stderr,
        )
        return 1

    if options.preview_chars <= 0:
        print(
            "错误：preview-chars必须大于0。",
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

        print("正在加载Embedding模型……")

        encoder = SentenceTransformerEncoder(
            model_name=options.model,
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
            f"索引构建完成，共包含"
            f" {indexed_count} 个Chunk。"
        )

        results = retriever.search(
            query,
            top_k=options.top_k,
        )

        print()
        print("=== 语义检索结果 ===")
        print(f"问题：{query}")
        print(f"结果数量：{len(results)}")

        paper_titles: dict[str, str] = {}

        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk = result.chunk

            if chunk.paper_id not in paper_titles:
                paper = (
                    paper_repository.get_by_id(
                        chunk.paper_id
                    )
                )

                paper_titles[chunk.paper_id] = (
                    paper.title
                    if paper is not None
                    else chunk.paper_id
                )

            paper_title = paper_titles[
                chunk.paper_id
            ]

            section_path = " > ".join(
                chunk.section_path
            )

            page_text = format_pages(
                chunk.page_start,
                chunk.page_end,
            )

            preview = shorten_text(
                chunk.content,
                options.preview_chars,
            )

            print()
            print(
                f"[第{rank}名] "
                f"相似度：{result.score:.4f}"
            )
            print(f"论文：{paper_title}")
            print(f"论文ID：{chunk.paper_id}")
            print(f"章节：{section_path}")
            print(f"页码：{page_text}")
            print(f"Chunk ID：{chunk.id}")
            print(f"原文：{preview}")

    except Exception as exc:
        print(
            (
                "检索失败："
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
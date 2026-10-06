import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from database.chunk_repository import ChunkRepository
from database.connection import create_connection
from database.paper_repository import PaperRepository
from database.schema import initialize_database
from database.section_repository import SectionRepository
from ingestion.paper_content_ingestor import (
    PaperContentIngestor,
)
from ingestion.paper_ingestor import PaperIngestor
from ingestion.pdf_ingestion_pipeline import (
    PdfIngestionPipeline,
)


def create_pipeline(
    connection: sqlite3.Connection,
) -> PdfIngestionPipeline:
    """组装PDF结构化入库所需的对象。"""

    paper_repository = PaperRepository(
        connection
    )

    section_repository = SectionRepository(
        connection
    )

    chunk_repository = ChunkRepository(
        connection
    )

    paper_ingestor = PaperIngestor(
        paper_repository
    )

    content_ingestor = PaperContentIngestor(
        connection=connection,
        paper_ingestor=paper_ingestor,
        section_repository=section_repository,
        chunk_repository=chunk_repository,
    )

    return PdfIngestionPipeline(
        content_ingestor
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "解析PDF章节结构，并将论文、章节和"
            "Chunk保存到SQLite。"
        )
    )

    parser.add_argument(
        "pdf_path",
        help="需要导入的PDF文件路径。",
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
        "--title",
        default=None,
        help="论文标题；不填写时使用文件名。",
    )

    parser.add_argument(
        "--author",
        action="append",
        default=[],
        help=(
            "论文作者，可以重复使用该参数。"
        ),
    )

    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="论文发表年份。",
    )

    parser.add_argument(
        "--doi",
        default=None,
        help="论文DOI。",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="单个Chunk的最大字符数。",
    )

    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=300,
        help="相邻Chunk的最大重叠字符数。",
    )

    return parser


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """执行一次PDF入库任务。"""

    parser = create_argument_parser()
    options = parser.parse_args(arguments)

    pdf_path = Path(
        options.pdf_path
    ).resolve()

    if not pdf_path.is_file():
        print(
            f"错误：找不到PDF文件：{pdf_path}",
            file=sys.stderr,
        )
        return 1

    if pdf_path.suffix.lower() != ".pdf":
        print(
            f"错误：文件不是PDF：{pdf_path}",
            file=sys.stderr,
        )
        return 1

    database_path = Path(
        options.database
    ).resolve()

    connection = None

    try:
        database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = create_connection(
            database_path
        )

        initialize_database(connection)

        pipeline = create_pipeline(
            connection
        )

        result = pipeline.ingest(
            pdf_path=pdf_path,
            title=options.title,
            authors=options.author,
            year=options.year,
            doi=options.doi,
            max_chars=options.max_chars,
            overlap_chars=options.overlap_chars,
        )

    except Exception as exc:
        print(
            (
                "入库失败："
                f"{type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        return 1
    finally:
        if connection is not None:
            connection.close()

    status = (
        "新论文已入库"
        if result.created
        else "论文已经存在"
    )

    print(f"状态：{status}")
    print(f"论文ID：{result.paper.id}")
    print(f"标题：{result.paper.title}")
    print(f"章节数量：{len(result.sections)}")
    print(f"Chunk数量：{len(result.chunks)}")
    print(f"数据库：{database_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
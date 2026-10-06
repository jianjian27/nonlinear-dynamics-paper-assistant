import argparse
import sqlite3
import sys
from pathlib import Path

from database.chunk_repository import ChunkRepository
from database.connection import create_connection
from database.section_repository import SectionRepository


def shorten_text(
    text: str,
    max_chars: int,
) -> str:
    """压缩空白并截取文本预览。"""

    normalized = " ".join(
        text.split()
    )

    if len(normalized) <= max_chars:
        return normalized

    return (
        normalized[:max_chars]
        + "……"
    )


def select_paper(
    connection: sqlite3.Connection,
    paper_id: str | None,
) -> sqlite3.Row | None:
    """选择指定论文，未指定时选择最近入库的论文。"""

    if paper_id:
        return connection.execute(
            """
            SELECT *
            FROM papers
            WHERE id = ?
            """,
            (paper_id,),
        ).fetchone()

    return connection.execute(
        """
        SELECT *
        FROM papers
        ORDER BY created_at DESC, rowid DESC
        LIMIT 1
        """
    ).fetchone()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "检查SQLite中论文、章节和Chunk的"
            "实际入库质量。"
        )
    )

    parser.add_argument(
        "--database",
        default="data/papers.db",
        help="SQLite数据库路径。",
    )

    parser.add_argument(
        "--paper-id",
        default=None,
        help=(
            "需要检查的论文ID；不填写时"
            "检查最近入库的论文。"
        ),
    )

    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=10,
        help="最多展示多少个Chunk。",
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=160,
        help="每个Chunk预览的最大字符数。",
    )

    options = parser.parse_args()

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
        paper = select_paper(
            connection,
            options.paper_id,
        )

        if paper is None:
            print(
                "错误：数据库中没有找到论文。",
                file=sys.stderr,
            )
            return 1

        section_repository = SectionRepository(
            connection
        )
        chunk_repository = ChunkRepository(
            connection
        )

        sections = (
            section_repository.list_by_paper(
                paper["id"]
            )
        )

        chunks = (
            chunk_repository.list_by_paper(
                paper["id"]
            )
        )

        print("=== 论文信息 ===")
        print(f"ID：{paper['id']}")
        print(f"标题：{paper['title']}")
        print(f"文件：{paper['file_name']}")
        print(f"作者JSON：{paper['authors']}")
        print(f"年份：{paper['year']}")
        print(f"DOI：{paper['doi']}")
        print(f"入库时间：{paper['created_at']}")

        print()
        print("=== 数据统计 ===")
        print(f"章节数量：{len(sections)}")
        print(f"Chunk数量：{len(chunks)}")

        if chunks:
            lengths = [
                len(chunk.content)
                for chunk in chunks
            ]

            average_length = (
                sum(lengths) / len(lengths)
            )

            print(
                f"最短Chunk：{min(lengths)} 字符"
            )
            print(
                f"最长Chunk：{max(lengths)} 字符"
            )
            print(
                "平均Chunk："
                f"{average_length:.1f} 字符"
            )

        print()
        print("=== 章节结构 ===")

        if not sections:
            print("警告：没有保存任何章节。")

        for section in sections:
            depth = max(
                len(section.section_path) - 1,
                0,
            )

            indentation = "  " * depth

            page_text = (
                f"第{section.page_start}"
                f"—{section.page_end}页"
                if section.page_start is not None
                else "无页码"
            )

            print(
                f"{indentation}- "
                f"{section.title} "
                f"({page_text})"
            )

        print()
        print("=== Chunk预览 ===")

        if not chunks:
            print("警告：没有生成任何Chunk。")

        for chunk in chunks[
            : options.chunk_limit
        ]:
            path = " > ".join(
                chunk.section_path
            )

            preview = shorten_text(
                chunk.content,
                options.preview_chars,
            )

            print()
            print(
                f"[Chunk {chunk.chunk_index}]"
            )
            print(f"章节：{path}")
            print(
                "页码："
                f"{chunk.page_start}"
                f"—{chunk.page_end}"
            )
            print(
                "来源块："
                f"{', '.join(chunk.source_blocks)}"
            )
            print(f"字符数：{len(chunk.content)}")
            print(f"正文：{preview}")

    finally:
        connection.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
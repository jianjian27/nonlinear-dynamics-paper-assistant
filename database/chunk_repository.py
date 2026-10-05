import json
import sqlite3

from domain.chunk import Chunk
from domain.section_record import SectionRecord


class ChunkRepository:
    """负责 Chunk 的数据库读写。"""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def add_many(
        self,
        chunks: list[Chunk],
        sections: list[SectionRecord],
    ) -> None:
        """批量保存 Chunk，并关联对应章节。"""

        section_ids = {
            (
                section.paper_id,
                section.section_path,
            ): section.id
            for section in sections
        }

        rows = []

        for chunk in chunks:
            section_key = (
                chunk.paper_id,
                chunk.section_path,
            )

            section_id = section_ids.get(
                section_key
            )

            if section_id is None:
                raise ValueError(
                    "找不到 Chunk 对应的章节："
                    f"{chunk.section_path}"
                )

            rows.append(
                (
                    chunk.id,
                    chunk.paper_id,
                    section_id,
                    chunk.chunk_index,
                    chunk.section_title,
                    json.dumps(
                        chunk.section_path,
                        ensure_ascii=False,
                    ),
                    chunk.content,
                    chunk.page_start,
                    chunk.page_end,
                    json.dumps(
                        chunk.pages,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        chunk.source_blocks,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {},
                        ensure_ascii=False,
                    ),
                    chunk.content_hash,
                )
            )

        try:
            self.connection.executemany(
                """
                INSERT INTO chunks (
                    id,
                    paper_id,
                    section_id,
                    chunk_index,
                    section_title,
                    section_path,
                    content,
                    page_start,
                    page_end,
                    pages,
                    source_blocks,
                    metadata,
                    content_hash
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,
                rows,
            )

            self.connection.commit()
        except sqlite3.Error:
            self.connection.rollback()
            raise

    def list_by_paper(
        self,
        paper_id: str,
    ) -> list[Chunk]:
        """按照原始顺序查询论文的所有 Chunk。"""

        rows = self.connection.execute(
            """
            SELECT *
            FROM chunks
            WHERE paper_id = ?
            ORDER BY chunk_index
            """,
            (paper_id,),
        ).fetchall()

        return [
            self._row_to_chunk(row)
            for row in rows
        ]

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """根据 Chunk ID 查询单个 Chunk。"""

        row = self.connection.execute(
            """
            SELECT *
            FROM chunks
            WHERE id = ?
            """,
            (chunk_id,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_chunk(row)

    @staticmethod
    def _row_to_chunk(
        row: sqlite3.Row,
    ) -> Chunk:
        """把数据库记录转换为 Chunk 对象。"""

        return Chunk(
            id=row["id"],
            paper_id=row["paper_id"],
            chunk_index=row["chunk_index"],
            section_title=row["section_title"],
            section_path=tuple(
                json.loads(
                    row["section_path"]
                )
            ),
            content=row["content"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            pages=tuple(
                json.loads(row["pages"])
            ),
            source_blocks=tuple(
                json.loads(
                    row["source_blocks"]
                )
            ),
            content_hash=row["content_hash"],
        )
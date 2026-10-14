import json
import sqlite3

from domain.paper import Paper
from domain.paper_overview import (
    PaperOverview,
)


class PaperRepository:
    """负责 papers 表的写入和查询。"""

    def __init__(
            self,
            connection: sqlite3.Connection
    ) -> None:
        self.connection = connection

    def add(self, paper: Paper, *, commit:bool = True,) -> None:
        """向数据库中保存一篇论文。"""

        authors_json = json.dumps(
            paper.authors,
            ensure_ascii=False
        )

        try:
            self.connection.execute(
                """
                INSERT INTO papers (
                    id,
                    title,
                    authors,
                    year,
                    file_name,
                    file_path,
                    file_hash,
                    doi
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper.id,
                    paper.title,
                    authors_json,
                    paper.year,
                    paper.file_name,
                    paper.file_path,
                    paper.file_hash,
                    paper.doi
                ),
            )

            if commit:
                self.connection.commit()
        except sqlite3.Error:
            self.connection.rollback()
            raise

    def list_all(
        self,
    ) -> list[Paper]:
        """按照标题查询所有论文。"""

        rows = self.connection.execute(
            """
            SELECT *
            FROM papers
            ORDER BY title COLLATE NOCASE, id
            """
        ).fetchall()

        return [
            self._row_to_paper(row)
            for row in rows
        ]

    def list_all_with_counts(
        self,
    ) -> list[PaperOverview]:
        """查询所有论文及其章节和Chunk数量。"""

        rows = self.connection.execute(
            """
            SELECT
                papers.*,

                (
                    SELECT COUNT(*)
                    FROM sections
                    WHERE sections.paper_id = papers.id
                ) AS section_count,

                (
                    SELECT COUNT(*)
                    FROM chunks
                    WHERE chunks.paper_id = papers.id
                ) AS chunk_count

            FROM papers
            ORDER BY title COLLATE NOCASE, id
            """
        ).fetchall()

        return [
            PaperOverview(
                paper=self._row_to_paper(
                    row
                ),
                section_count=(
                    row["section_count"]
                ),
                chunk_count=(
                    row["chunk_count"]
                ),
            )
            for row in rows
        ]

    def delete_by_id(
        self,
        paper_id: str,
        *,
        commit: bool = True,
    ) -> Paper | None:
        """删除论文并返回被删除的论文。"""

        paper = self.get_by_id(
            paper_id
        )

        if paper is None:
            return None

        try:
            self.connection.execute(
                """
                DELETE FROM papers
                WHERE id = ?
                """,
                (paper_id,),
            )

            if commit:
                self.connection.commit()

        except sqlite3.Error:
            self.connection.rollback()
            raise

        return paper

    def get_by_id(
            self,
            paper_id: str
    ) -> Paper | None:
        """根据论文 ID 查询论文。"""

        row = self.connection.execute(
            """
            SELECT *
            FROM papers
            WHERE id = ?
            """,
            (paper_id,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_paper(row)

    def get_by_file_hash(
            self,
            file_hash: str,
    ) -> Paper | None:
        """根据文件哈希查询论文。"""

        row = self.connection.execute(
            """
            SELECT *
            FROM papers
            WHERE file_hash = ?
            """,
            (file_hash,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_paper(row)

    def exists_by_file_hash(
            self,
            file_hash: str,
    ) -> bool:
        """判断相同文件是否已经入库。"""

        row = self.connection.execute(
            """
            SELECT 1
            FROM papers
            WHERE file_hash = ?
            LIMIT 1
            """,
            (file_hash,),
        ).fetchone()

        return row is not None

    @staticmethod
    def _row_to_paper(
            row: sqlite3.Row,
    ) -> Paper:
        """将数据库查询结果转换为 Paper 对象。"""

        authors = tuple(
            json.loads(row["authors"])
        )

        return Paper(
            id=row["id"],
            title=row["title"],
            authors=authors,
            year=row["year"],
            file_name=row["file_name"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            doi=row["doi"],
            created_at=row["created_at"],
        )
import json
import sqlite3

from domain.paper import Paper


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
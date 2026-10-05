import sqlite3
from hashlib import sha256

from domain.section_record import SectionRecord
from utils.pdf_structure import Section


class SectionRepository:
    """负责章节树的数据库读写。"""

    def __init__(
            self,
            connection: sqlite3.Connection,
    ) -> None:
        self.connection = connection

    def add_tree(
            self,
            paper_id: str,
            root_sections: list[Section],
    ) -> list[SectionRecord]:
        """按照文档顺序保存完整章节树。"""

        records = self._build_records(
            paper_id=paper_id,
            root_sections=root_sections,
        )

        try:
            self.connection.executemany(
                """
                INSERT INTO sections (
                    id,
                    paper_id,
                    parent_id,
                    title,
                    level,
                    section_index,
                    page_start,
                    page_end,
                    content
                ) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.id,
                        record.paper_id,
                        record.parent_id,
                        record.title,
                        record.level,
                        record.section_index,
                        record.page_start,
                        record.page_end,
                        record.content,
                    )
                    for record in records
                ],
            )

            self.connection.commit()
        except sqlite3.Error:
            self.connection.rollback()
            raise

        return records

    def list_by_paper(self, paper_id: str,) -> list[SectionRecord]:
        """按照原始文档顺序查询论文的章节。"""

        rows = self.connection.execute(
            """
            SELECT *
            FROM sections
            WHERE paper_id = ?
            ORDER BY section_index
            """,
            (paper_id,),
        ).fetchall()

        records: list[SectionRecord] = []
        paths_by_id: dict[
            str,
            tuple[str, ...],
        ] = {}

        for row in rows:
            parent_id = row["parent_id"]

            if parent_id is None:
                parent_path: tuple[str, ...] = ()
            else:
                parent_path = paths_by_id[parent_id]

            section_path = (
                *parent_path,
                row["title"],
            )

            paths_by_id[row["id"]] = section_path

            records.append(
                SectionRecord(
                    id=row["id"],
                    paper_id=row["paper_id"],
                    parent_id=parent_id,
                    title=row["title"],
                    level=row["level"],
                    section_index=row["section_index"],
                    section_path=section_path,
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    content=row["content"],
                )
            )

        return records

    @staticmethod
    def _build_records(
            paper_id: str,
            root_sections: list[Section],
    ) -> list[SectionRecord]:
        """把章节树转换为扁平的章节记录。"""

        records: list[SectionRecord] = []

        def visit(
                sections: list[Section],
                parent_id: str | None,
                parent_path: tuple[str, ...],
        ) -> None:
            for section in sections:
                section_index = len(records)

                section_path = (
                    *parent_path,
                    section.title,
                )

                identity = "\x1f".join(
                    (
                        paper_id,
                        str(section_index),
                        *section_path,
                    )
                )

                section_id = sha256(
                    identity.encode("utf-8")
                ).hexdigest()

                direct_blocks = (
                    section.heading_blocks
                    + section.blocks
                )

                pages = sorted(
                    {
                        block.page
                        for block in direct_blocks
                    }
                )

                if pages:
                    page_start = pages[0]
                    page_end = pages[-1]
                else:
                    page_start = None
                    page_end = None

                record = SectionRecord(
                    id=section_id,
                    paper_id=paper_id,
                    parent_id=parent_id,
                    title=section.title,
                    level=section.level,
                    section_index=section_index,
                    section_path=section_path,
                    page_start=page_start,
                    page_end=page_end,
                    content=section.text,
                )

                records.append(record)

                visit(
                    sections=section.children,
                    parent_id=section_id,
                    parent_path=section_path,
                )

        visit(
            sections=root_sections,
            parent_id=None,
            parent_path=(),
        )

        return records
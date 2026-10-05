import sqlite3


def initialize_database(
    connection: sqlite3.Connection,
) -> None:
    """创建论文知识库所需的数据表和索引。"""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT NOT NULL DEFAULT '[]',
            year INTEGER,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            doi TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sections (
            id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            parent_id TEXT,
            title TEXT NOT NULL,
            level INTEGER NOT NULL,
            section_index INTEGER NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            content TEXT NOT NULL DEFAULT '',

            FOREIGN KEY (paper_id)
                REFERENCES papers(id)
                ON DELETE CASCADE,

            FOREIGN KEY (parent_id)
                REFERENCES sections(id)
                ON DELETE CASCADE,

            CHECK (level >= 0),
            CHECK (section_index >= 0),
            CHECK (
                page_start IS NULL
                OR page_end IS NULL
                OR page_end >= page_start
            )
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            section_id TEXT,
            chunk_index INTEGER NOT NULL,
            section_title TEXT NOT NULL DEFAULT '',
            section_path TEXT NOT NULL DEFAULT '[]',
            content TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            pages TEXT NOT NULL DEFAULT '[]',
            source_blocks TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (paper_id)
                REFERENCES papers(id)
                ON DELETE CASCADE,

            FOREIGN KEY (section_id)
                REFERENCES sections(id)
                ON DELETE SET NULL,

            UNIQUE (paper_id, chunk_index),

            CHECK (chunk_index >= 0),
            CHECK (
                page_start IS NULL
                OR page_end IS NULL
                OR page_end >= page_start
            )
        );

        CREATE INDEX IF NOT EXISTS idx_sections_paper_id
        ON sections(paper_id);

        CREATE INDEX IF NOT EXISTS idx_sections_parent_id
        ON sections(parent_id);

        CREATE INDEX IF NOT EXISTS idx_chunks_paper_id
        ON chunks(paper_id);

        CREATE INDEX IF NOT EXISTS idx_chunks_section_id
        ON chunks(section_id);

        CREATE INDEX IF NOT EXISTS idx_chunks_content_hash
        ON chunks(content_hash);

        PRAGMA user_version = 1;
        """
    )

    connection.commit()
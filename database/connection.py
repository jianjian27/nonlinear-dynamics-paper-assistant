import sqlite3
from pathlib import Path

def create_connection(
        database_path: str | Path,
) -> sqlite3.Connection:
    """创建并配置 SQLite 数据库连接。"""

    connection = sqlite3.connect(database_path)

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection
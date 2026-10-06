import hashlib
from pathlib import Path


_FILE_READ_SIZE = 1024 * 1024


def calculate_file_sha256(
        file_path: str | Path,
) -> str:
    """分块读取文件并计算 SHA-256 哈希值。"""

    path = Path(file_path)
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            data = file.read(_FILE_READ_SIZE)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()
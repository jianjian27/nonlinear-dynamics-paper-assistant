import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from domain.chunk import Chunk
from hashlib import sha256


CACHE_FORMAT_VERSION = 1


class VectorCache:
    """在本地磁盘保存和加载Chunk向量。"""

    @classmethod
    def for_index(
        cls,
        root_directory: str | Path,
        *,
        model_name: str,
        scope: str,
        source_id: str,
    ) -> "VectorCache":
        """为指定数据源、模型和范围创建缓存。"""

        cleaned_model_name = (
            model_name.strip()
        )

        cleaned_scope = (
            scope.strip()
        )

        cleaned_source_id = (
            source_id.strip()
        )

        if not cleaned_model_name:
            raise ValueError(
                "Embedding模型名称不能为空。"
            )

        if not cleaned_scope:
            raise ValueError(
                "向量缓存范围不能为空。"
            )

        if not cleaned_source_id:
            raise ValueError(
                "向量缓存数据源不能为空。"
            )

        cache_identity = json.dumps(
            {
                "model_name": cleaned_model_name,
                "scope": cleaned_scope,
                "source_id": cleaned_source_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        cache_key = sha256(cache_identity.encode("utf-8")).hexdigest()[:24]

        return cls(
            Path(root_directory) / cache_key
        )


    def __init__(
        self,
        directory: str | Path,
    ) -> None:
        self.directory = Path(directory)

        self.vectors_path = (self.directory / "vectors.npz")

        self.manifest_path = (self.directory / "manifest.json")

    def load(
        self,
        *,
        chunks: Sequence[Chunk],
        model_name: str,
        dimension: int,
        scope: str,
    ) -> np.ndarray | None:
        """
        加载有效缓存。

        缓存不存在、损坏或与当前Chunk不一致时，
        返回None。
        """

        if (
            not self.vectors_path.is_file()
            or not self.manifest_path.is_file()
        ):
            return None

        expected_manifest = (
            self._build_manifest(
                chunks=chunks,
                model_name=model_name,
                dimension=dimension,
                scope=scope,
            )
        )

        try:
            manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )

            for key, expected_value in (
                expected_manifest.items()
            ):
                if (
                    manifest.get(key) != expected_value
                ):
                    return None

            with np.load(
                self.vectors_path,
                allow_pickle=False,
            ) as archive:
                vectors = np.asarray(
                    archive["vectors"], dtype=np.float32,
                )

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            return None

        expected_shape = (
            len(chunks),
            dimension,
        )

        if vectors.shape != expected_shape:
            return None

        if not np.all(
            np.isfinite(vectors)
        ):
            return None

        return vectors

    def save(
        self,
        *,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        model_name: str,
        dimension: int,
        scope: str,
    ) -> None:
        """将Chunk向量和校验清单保存到磁盘。"""

        self._validate_metadata(
            model_name=model_name,
            dimension=dimension,
            scope=scope,
        )

        array = np.asarray(
            vectors,
            dtype=np.float32,
        )

        expected_shape = (
            len(chunks),
            dimension,
        )

        if array.shape != expected_shape:
            raise ValueError(
                "待缓存向量形状不正确："
                f"期望 {expected_shape}，"
                f"实际 {array.shape}。"
            )

        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                "待缓存向量包含NaN或无穷值。"
        )

        chunk_ids = [
            chunk.id
            for chunk in chunks
        ]

        if len(chunk_ids) != len(
            set(chunk_ids)
        ):
            raise ValueError(
                "不能缓存存在重复ID的Chunk。"
            )

        manifest = self._build_manifest(
            chunks=chunks,
            model_name=model_name,
            dimension=dimension,
            scope=scope,
        )

        self.directory.mkdir(parents=True, exist_ok=True)

        temporary_vectors_path = (
            self.directory
            / "vectors.tmp.npz"
        )

        temporary_manifest_path = (
            self.directory
            / "manifest.tmp.json"
        )

        try:
            with temporary_vectors_path.open(
                "wb"
            ) as file:
                np.savez_compressed(
                    file,
                    vectors=array,
                )

            temporary_vectors_path.replace(
                self.vectors_path
            )

            temporary_manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            temporary_manifest_path.replace(
                self.manifest_path
            )

        finally:
            temporary_vectors_path.unlink(
                missing_ok=True
            )

            temporary_manifest_path.unlink(
                missing_ok=True
            )

    @staticmethod
    def _build_manifest(
        *,
        chunks: Sequence[Chunk],
        model_name: str,
        dimension: int,
        scope: str,
    ) -> dict[str, object]:
        """生成用于检查缓存有效性的清单。"""

        VectorCache._validate_metadata(
            model_name=model_name,
            dimension=dimension,
            scope=scope,
        )

        return {
            "format_version": (
                CACHE_FORMAT_VERSION
            ),
            "model_name": model_name,
            "dimension": dimension,
            "scope": scope,
            "chunk_count": len(chunks),
            "chunk_ids": [
                chunk.id
                for chunk in chunks
            ],
            "content_hashes": [
                chunk.content_hash
                for chunk in chunks
            ],
        }

    @staticmethod
    def _validate_metadata(
        *,
        model_name: str,
        dimension: int,
        scope: str,
    ) -> None:
        """检查缓存元数据参数。"""

        if not model_name.strip():
            raise ValueError(
                "Embedding模型名称不能为空。"
            )

        if dimension <= 0:
            raise ValueError(
                "Embedding维度必须大于0。"
            )

        if not scope.strip():
            raise ValueError(
                "向量缓存范围不能为空。"
            )


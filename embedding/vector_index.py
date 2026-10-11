from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from domain.chunk import Chunk


@dataclass(frozen=True)
class VectorSearchResult:
    """一次向量检索返回的Chunk和相似度。"""

    chunk: Chunk
    score: float


class InMemoryVectorIndex:
    """使用NumPy在内存中保存和检索向量。"""

    def __init__(
        self,
        dimension: int,
    ) -> None:
        if dimension <= 0:
            raise ValueError(
                "向量维度必须大于0。"
            )

        self.dimension = dimension
        self._chunks: list[Chunk] = []

        self._vectors = np.empty(
            (0, dimension),
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def add(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
    ) -> None:
        """把Chunk和对应的向量添加到索引中。"""

        array = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError(
                "Chunk向量必须是二维数组。"
            )

        if array.shape[0] != len(chunks):
            raise ValueError(
                "Chunk数量和向量行数不匹配。"
            )

        if array.shape[1] != self.dimension:
            raise ValueError(
                "Chunk向量维度与索引维度不一致。"
            )

        new_ids = [
            chunk.id
            for chunk in chunks
        ]

        if len(new_ids) != len(set(new_ids)):
            raise ValueError(
                "本次添加的Chunk中存在重复的ID。"
            )

        existing_ids = {
            chunk.id
            for chunk in self._chunks
        }

        duplicate_ids = (
            existing_ids.intersection(new_ids)
        )

        if duplicate_ids:
            raise ValueError(
                "索引中已经存在相同Chunk ID。 "
            )

        normalized_vectors = (
            self._normalize_rows(array)
        )

        self._chunks.extend(chunks)

        self._vectors = np.vstack(
            (
                self._vectors,
                normalized_vectors,
            )
        )

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """返回与查询向量最相似的Top-k个Chunk。"""

        if top_k <= 0:
            raise ValueError(
                "top_k必须大于0。"
            )

        vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                "查询向量必须是一维数组。"
            )

        if vector.shape[0] != self.dimension:
            raise ValueError(
                "查询向量维度与索引维度不一致。"
            )

        normalized_query = (
            self._normalize_vector(vector)
        )

        if not self._chunks:
            return []

        scores = (
            self._vectors
            @ normalized_query
        )

        result_count = min(
            top_k,
            len(self._chunks),
        )

        top_indices = np.argsort(
            -scores,
            kind="stable",
        )[:result_count]

        return [
            VectorSearchResult(
                chunk=self._chunks[index],
                score=float(scores[index]),
            )
            for index in top_indices
        ]

    @staticmethod
    def _normalize_rows(
        vectors: np.ndarray,
    ) -> np.ndarray:
        """将二维数组中的每一行归一化。"""

        if not np.all(np.isfinite(vectors)):
            raise ValueError(
                "向量包含NaN或无穷值。"
            )

        lengths = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        if np.any(lengths == 0):
            raise ValueError(
                "不能添加零向量。"
            )

        return (
            vectors / lengths
        ).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _normalize_vector(
        vector: np.ndarray,
    ) -> np.ndarray:
        """将一个查询向量归一化。"""

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "查询向量包含NaN或无穷值。"
            )

        length = np.linalg.norm(vector)

        if length == 0:
            raise ValueError(
                "查询向量不能为零向量。"
            )

        return (
            vector / length
        ).astype(
            np.float32,
            copy=False,
        )
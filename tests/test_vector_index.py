import unittest

import numpy as np

from domain.chunk import Chunk
from embedding.vector_index import (
    InMemoryVectorIndex,
)


def make_chunk(
    index: int,
    content: str,
) -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        paper_id="paper-1",
        chunk_index=index,
        section_title=f"章节{index}",
        section_path=(f"章节{index}",),
        content=content,
        page_start=index + 1,
        page_end=index + 1,
        pages=(index + 1,),
        source_blocks=(f"{index + 1}:1",),
        content_hash=f"hash-{index}",
    )


class TestInMemoryVectorIndex(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.chunks = [
            make_chunk(
                0,
                "李雅普诺夫稳定性证明",
            ),
            make_chunk(
                1,
                "四阶Runge-Kutta数值方法",
            ),
            make_chunk(
                2,
                "同步误差收敛分析",
            ),
        ]

        self.vectors = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )

    def test_adds_chunks_and_vectors(
        self,
    ) -> None:
        index = InMemoryVectorIndex(
            dimension=2
        )

        index.add(
            self.chunks,
            self.vectors,
        )

        self.assertEqual(len(index), 3)

    def test_returns_results_in_score_order(
        self,
    ) -> None:
        index = InMemoryVectorIndex(
            dimension=2
        )

        index.add(
            self.chunks,
            self.vectors,
        )

        results = index.search(
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            top_k=2,
        )

        self.assertEqual(len(results), 2)

        self.assertEqual(
            results[0].chunk.id,
            "chunk-0",
        )

        self.assertEqual(
            results[1].chunk.id,
            "chunk-2",
        )

        self.assertGreater(
            results[0].score,
            results[1].score,
        )

    def test_top_k_cannot_exceed_index_size(
        self,
    ) -> None:
        index = InMemoryVectorIndex(
            dimension=2
        )

        index.add(
            self.chunks,
            self.vectors,
        )

        results = index.search(
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            top_k=10,
        )

        self.assertEqual(len(results), 3)

    def test_empty_index_returns_empty_results(
        self,
    ) -> None:
        index = InMemoryVectorIndex(
            dimension=2
        )

        results = index.search(
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        )

        self.assertEqual(results, [])

    def test_rejects_mismatched_chunk_count(
        self,
    ) -> None:
        index = InMemoryVectorIndex(
            dimension=2
        )

        with self.assertRaisesRegex(
            ValueError,
            "Chunk数量和向量行数不匹配。",
        ):
            index.add(
                self.chunks,
                self.vectors[:2],
            )

    def test_rejects_wrong_dimension(
        self,
    ) -> None:
        index = InMemoryVectorIndex(
            dimension=3
        )

        with self.assertRaisesRegex(
            ValueError,
            "索引维度不一致",
        ):
            index.add(
                self.chunks,
                self.vectors,
            )

    def test_rejects_duplicate_chunk_ids(
        self,
    ) -> None:
        index = InMemoryVectorIndex(
            dimension=2
        )

        index.add(
            [self.chunks[0]],
            self.vectors[:1],
        )

        with self.assertRaisesRegex(
            ValueError,
            "已经存在",
        ):
            index.add(
                [self.chunks[0]],
                self.vectors[:1],
            )

    def test_rejects_zero_query_vector(
        self,
    ) -> None:
        index = InMemoryVectorIndex(
            dimension=2
        )

        index.add(
            self.chunks,
            self.vectors,
        )

        with self.assertRaisesRegex(
            ValueError,
            "查询向量不能为零向量。",
        ):
            index.search(
                np.array(
                    [0.0, 0.0],
                    dtype=np.float32,
                )
            )


if __name__ == "__main__":
    unittest.main()
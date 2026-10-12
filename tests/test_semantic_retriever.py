import unittest
from unittest.mock import Mock

import numpy as np

from database.chunk_repository import (
    ChunkRepository,
)
from domain.chunk import Chunk
from embedding.base import TextEncoder
from retrieval.semantic_retriever import (
    SemanticRetriever,
)
from embedding.vector_cache import (
    VectorCache,
)


def make_chunk(
    index: int,
    content: str,
    *,
    paper_id: str = "paper-1",
) -> Chunk:
    return Chunk(
        id=f"{paper_id}-chunk-{index}",
        paper_id=paper_id,
        chunk_index=index,
        section_title=f"章节{index}",
        section_path=(
            "数学模型",
            f"章节{index}",
        ),
        content=content,
        page_start=index + 1,
        page_end=index + 1,
        pages=(index + 1,),
        source_blocks=(f"{index + 1}:1",),
        content_hash=f"hash-{index}",
    )


def make_reference_chunk(
    index: int,
    content: str,
) -> Chunk:
    return Chunk(
        id=f"paper-1-reference-{index}",
        paper_id="paper-1",
        chunk_index=index,
        section_title="参考文献：",
        section_path=("参考文献：",),
        content=content,
        page_start=index + 1,
        page_end=index + 1,
        pages=(index + 1,),
        source_blocks=(f"{index + 1}:1",),
        content_hash=f"reference-hash-{index}",
    )


class TestSemanticRetriever(
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

        self.repository = Mock(
            spec=ChunkRepository
        )

        self.encoder = Mock(
            spec=TextEncoder
        )

        self.encoder.identifier = (
            "test-encoder"
        )

        self.encoder.dimension = 2

        self.retriever = SemanticRetriever(
            chunk_repository=self.repository,
            encoder=self.encoder,
        )

    def test_builds_index_from_all_chunks(
        self,
    ) -> None:
        self.repository.list_all.return_value = (
            self.chunks
        )

        self.encoder.encode_passages.return_value = (
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.8, 0.2],
                ],
                dtype=np.float32,
            )
        )

        count = self.retriever.build_index()

        self.assertEqual(count, 3)
        self.assertTrue(
            self.retriever.is_built
        )
        self.assertEqual(
            self.retriever.indexed_count,
            3,
        )

        expected_texts = [
            chunk.embedding_text
            for chunk in self.chunks
        ]

        self.encoder.encode_passages.assert_called_once_with(
            expected_texts
        )

    def test_builds_index_for_one_paper(
        self,
    ) -> None:
        self.repository.list_by_paper.return_value = (
            self.chunks[:2]
        )

        self.encoder.encode_passages.return_value = (
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )

        count = self.retriever.build_index(
            paper_id="paper-1"
        )

        self.assertEqual(count, 2)

        self.repository.list_by_paper.assert_called_once_with(
            "paper-1"
        )

        self.repository.list_all.assert_not_called()

    def test_searches_with_query_embedding(
        self,
    ) -> None:
        self.repository.list_all.return_value = (
            self.chunks
        )

        self.encoder.encode_passages.return_value = (
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.8, 0.2],
                ],
                dtype=np.float32,
            )
        )

        self.retriever.build_index()

        self.encoder.encode_queries.return_value = (
            np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            )
        )

        results = self.retriever.search(
            "  如何证明系统稳定？  ",
            top_k=2,
        )

        self.encoder.encode_queries.assert_called_once_with(
            ["如何证明系统稳定？"]
        )

        self.assertEqual(len(results), 2)

        self.assertEqual(
            results[0].chunk.id,
            "paper-1-chunk-0",
        )

        self.assertEqual(
            results[1].chunk.id,
            "paper-1-chunk-2",
        )

    def test_requires_index_before_search(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "build_index",
        ):
            self.retriever.search(
                "如何证明系统稳定？"
            )

    def test_rejects_blank_query(
        self,
    ) -> None:
        self.repository.list_all.return_value = []
        self.retriever.build_index()

        with self.assertRaisesRegex(
            ValueError,
            "不能为空",
        ):
            self.retriever.search("   ")

    def test_empty_database_builds_empty_index(
        self,
    ) -> None:
        self.repository.list_all.return_value = []

        count = self.retriever.build_index()

        self.assertEqual(count, 0)
        self.assertTrue(
            self.retriever.is_built
        )
        self.assertEqual(
            self.retriever.indexed_count,
            0,
        )

        self.encoder.encode_passages.assert_not_called()

    def test_excludes_reference_chunks_for_normal_query(
        self,
    ) -> None:
        chunks = [
            make_reference_chunk(
                0,
                "李雅普诺夫稳定性参考文献",
            ),
            make_chunk(
                1,
                "正文中的李雅普诺夫稳定性分析",
            ),
            make_chunk(
                2,
                "正文中的数值仿真结果",
            ),
        ]

        self.repository.list_all.return_value = (
            chunks
        )

        self.encoder.encode_passages.return_value = (
            np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )

        self.retriever.build_index()

        self.encoder.encode_queries.return_value = (
            np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            )
        )

        results = self.retriever.search(
            "论文如何分析系统稳定性？",
            top_k=2,
        )

        result_ids = [
            result.chunk.id
            for result in results
        ]

        self.assertEqual(
            result_ids,
            [
                "paper-1-chunk-1",
                "paper-1-chunk-2",
            ],
        )

        self.assertNotIn(
            "paper-1-reference-0",
            result_ids,
        )

    def test_returns_only_references_for_reference_query(
            self,
    ) -> None:
        chunks = [
            make_reference_chunk(
                0,
                "李雅普诺夫稳定性参考文献",
            ),
            make_chunk(
                1,
                "正文中的稳定性分析",
            ),
        ]

        self.repository.list_all.return_value = (
            chunks
        )

        self.encoder.encode_passages.return_value = (
            np.array(
                [
                    [0.8, 0.2],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )

        self.retriever.build_index()

        self.encoder.encode_queries.return_value = (
            np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            )
        )

        results = self.retriever.search(
            "论文引用了哪些参考文献？",
            top_k=2,
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].chunk.id,
            "paper-1-reference-0",
        )

    def test_uses_cached_passage_vectors(
        self,
    ) -> None:
        self.repository.list_all.return_value = (
            self.chunks
        )

        cached_vectors = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )

        vector_cache = Mock(
            spec=VectorCache
        )

        vector_cache.load.return_value = (
            cached_vectors
        )

        retriever = SemanticRetriever(
            chunk_repository=self.repository,
            encoder=self.encoder,
            vector_cache=vector_cache,
        )

        count = retriever.build_index()

        self.assertEqual(count, 3)

        self.assertTrue(
            retriever.last_build_used_cache
        )

        self.encoder.encode_passages.assert_not_called()

        vector_cache.load.assert_called_once_with(
            chunks=self.chunks,
            model_name="test-encoder",
            dimension=2,
            scope="all",
        )

        vector_cache.save.assert_not_called()

    def test_encodes_and_saves_on_cache_miss(
        self,
    ) -> None:
        self.repository.list_by_paper.return_value = (
            self.chunks
        )

        generated_vectors = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )

        self.encoder.encode_passages.return_value = (
            generated_vectors
        )

        vector_cache = Mock(
            spec=VectorCache
        )

        vector_cache.load.return_value = None

        retriever = SemanticRetriever(
            chunk_repository=self.repository,
            encoder=self.encoder,
            vector_cache=vector_cache,
        )

        count = retriever.build_index(
            paper_id="paper-1"
        )

        self.assertEqual(count, 3)

        self.assertFalse(
            retriever.last_build_used_cache
        )

        expected_texts = [
            chunk.embedding_text
            for chunk in self.chunks
        ]

        self.encoder.encode_passages.assert_called_once_with(
            expected_texts
        )

        vector_cache.load.assert_called_once_with(
            chunks=self.chunks,
            model_name="test-encoder",
            dimension=2,
            scope="paper:paper-1",
        )

        vector_cache.save.assert_called_once_with(
            chunks=self.chunks,
            vectors=generated_vectors,
            model_name="test-encoder",
            dimension=2,
            scope="paper:paper-1",
        )


if __name__ == "__main__":
    unittest.main()
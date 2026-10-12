import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from domain.chunk import Chunk
from embedding.vector_cache import (
    VectorCache,
)


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="chunk-1",
            paper_id="paper-1",
            chunk_index=0,
            section_title="数学模型",
            section_path=("数学模型",),
            content="系统数学模型。",
            page_start=1,
            page_end=1,
            pages=(1,),
            source_blocks=("1:1",),
            content_hash="hash-1",
        ),
        Chunk(
            id="chunk-2",
            paper_id="paper-1",
            chunk_index=1,
            section_title="稳定性分析",
            section_path=("稳定性分析",),
            content="系统稳定性分析。",
            page_start=2,
            page_end=2,
            pages=(2,),
            source_blocks=("2:1",),
            content_hash="hash-2",
        ),
    ]


class TestVectorCache(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.temporary_directory = (
            TemporaryDirectory()
        )

        self.cache = VectorCache(
            Path(
                self.temporary_directory.name
            )
        )

        self.chunks = make_chunks()

        self.vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_saves_and_loads_vectors(
        self,
    ) -> None:
        self.cache.save(
            chunks=self.chunks,
            vectors=self.vectors,
            model_name="test-model",
            dimension=3,
            scope="all",
        )

        loaded_vectors = self.cache.load(
            chunks=self.chunks,
            model_name="test-model",
            dimension=3,
            scope="all",
        )

        self.assertIsNotNone(
            loaded_vectors
        )

        np.testing.assert_array_equal(
            loaded_vectors,
            self.vectors,
        )

        self.assertTrue(
            self.cache.vectors_path.is_file()
        )

        self.assertTrue(
            self.cache.manifest_path.is_file()
        )

    def test_returns_none_when_content_changes(
        self,
    ) -> None:
        self.cache.save(
            chunks=self.chunks,
            vectors=self.vectors,
            model_name="test-model",
            dimension=3,
            scope="all",
        )

        changed_chunks = [
            self.chunks[0],
            replace(
                self.chunks[1],
                content_hash="changed-hash",
            ),
        ]

        loaded_vectors = self.cache.load(
            chunks=changed_chunks,
            model_name="test-model",
            dimension=3,
            scope="all",
        )

        self.assertIsNone(
            loaded_vectors
        )

    def test_returns_none_for_different_model(
        self,
    ) -> None:
        self.cache.save(
            chunks=self.chunks,
            vectors=self.vectors,
            model_name="model-a",
            dimension=3,
            scope="all",
        )

        loaded_vectors = self.cache.load(
            chunks=self.chunks,
            model_name="model-b",
            dimension=3,
            scope="all",
        )

        self.assertIsNone(
            loaded_vectors
        )

    def test_returns_none_for_different_scope(
        self,
    ) -> None:
        self.cache.save(
            chunks=self.chunks,
            vectors=self.vectors,
            model_name="test-model",
            dimension=3,
            scope="all",
        )

        loaded_vectors = self.cache.load(
            chunks=self.chunks,
            model_name="test-model",
            dimension=3,
            scope="paper:paper-1",
        )

        self.assertIsNone(
            loaded_vectors
        )

    def test_returns_none_for_corrupt_manifest(
        self,
    ) -> None:
        self.cache.save(
            chunks=self.chunks,
            vectors=self.vectors,
            model_name="test-model",
            dimension=3,
            scope="all",
        )

        self.cache.manifest_path.write_text(
            "not valid json",
            encoding="utf-8",
        )

        loaded_vectors = self.cache.load(
            chunks=self.chunks,
            model_name="test-model",
            dimension=3,
            scope="all",
        )

        self.assertIsNone(
            loaded_vectors
        )

    def test_rejects_wrong_vector_shape(
        self,
    ) -> None:
        wrong_vectors = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        with self.assertRaisesRegex(
            ValueError,
            "形状不正确",
        ):
            self.cache.save(
                chunks=self.chunks,
                vectors=wrong_vectors,
                model_name="test-model",
                dimension=3,
                scope="all",
            )


if __name__ == "__main__":
    unittest.main()
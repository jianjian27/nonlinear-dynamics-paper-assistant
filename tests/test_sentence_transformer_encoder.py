import unittest
from unittest.mock import Mock

import numpy as np

from embedding.sentence_transformer_encoder import (
    SentenceTransformerEncoder,
)


class TestSentenceTransformerEncoder(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.model = Mock()

        self.model.get_embedding_dimension.return_value = 3

        self.encoder = SentenceTransformerEncoder(
            model=self.model,
            batch_size=16,
        )

    def test_encodes_queries_with_prefix(
        self,
    ) -> None:
        self.model.encode.return_value = np.array(
            [[1.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        result = self.encoder.encode_queries(
            ["如何证明系统稳定？"]
        )

        self.model.encode.assert_called_once_with(
            ["query: 如何证明系统稳定？"],
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(
            result.dtype,
            np.float32,
        )

    def test_encodes_passages_with_prefix(
        self,
    ) -> None:
        self.model.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        result = self.encoder.encode_passages(
            [
                "李雅普诺夫稳定性分析。",
                "采用四阶龙格库塔方法。",
            ]
        )

        called_texts = (
            self.model.encode
            .call_args.args[0]
        )

        self.assertEqual(
            called_texts,
            [
                "passage: 李雅普诺夫稳定性分析。",
                "passage: 采用四阶龙格库塔方法。",
            ],
        )

        self.assertEqual(result.shape, (2, 3))

    def test_empty_sequence_returns_empty_matrix(
        self,
    ) -> None:
        result = self.encoder.encode_passages([])

        self.assertEqual(result.shape, (0, 3))
        self.model.encode.assert_not_called()

    def test_rejects_blank_text(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "不能为空",
        ):
            self.encoder.encode_queries(
                ["   "]
            )

        self.model.encode.assert_not_called()

    def test_rejects_invalid_batch_size(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "batch_size必须大于0",
        ):
            SentenceTransformerEncoder(
                model=self.model,
                batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
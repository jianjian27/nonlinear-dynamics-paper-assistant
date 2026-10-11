from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from embedding.base import TextEncoder


DEFAULT_MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)


class SentenceTransformerEncoder(TextEncoder):
    """使用本地 Sentence Transformer生成 Embedding。"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        batch_size: int = 32,
        model: SentenceTransformer | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                "batch_size必须大于0。"
            )

        self.batch_size = batch_size

        self.model = (
            model
            if model is not None
            else SentenceTransformer(
                model_name,
                device=device,
            )
        )

    @property
    def dimension(self) -> int:
        dimension = (
            self.model
            .get_embedding_dimension()
        )

        if dimension is None:
            raise RuntimeError(
                "模型没有提供Embedding维度。"
            )

        return dimension

    def encode_queries(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        return self._encode(
            texts=texts,
            prefix="query",
        )

    def encode_passages(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        return self._encode(
            texts=texts,
            prefix="passage",
        )

    def _encode(
        self,
        texts: Sequence[str],
        prefix: str,
    ) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        prepared_texts = []

        for text in texts:
            cleaned_text = text.strip()

            if not cleaned_text:
                raise ValueError(
                    "待编码文本不能为空。"
                )

            prepared_texts.append(
                f"{prefix}: {cleaned_text}"
            )

        embeddings = self.model.encode(
            prepared_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise RuntimeError(
                "模型返回的Embedding不是二维数组。"
            )

        if array.shape[1] != self.dimension:
            raise RuntimeError(
                "模型返回的Embedding维度不正确。"
            )

        return array
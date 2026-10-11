from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class TextEncoder(ABC):
    """将查询和候选文本转换为向量的接口。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度。"""

    @abstractmethod
    def encode_queries(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """编码用户查询。"""

    @abstractmethod
    def encode_passages(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """编码候选文本。"""
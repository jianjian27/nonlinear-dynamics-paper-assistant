import numpy as np
import torch

from embedding.sentence_transformer_encoder import (
    SentenceTransformerEncoder,
)


PASSAGES = [
    (
        "本文构造李雅普诺夫函数，并证明其导数"
        "负定，从而得到同步误差渐近收敛到零。"
    ),
    (
        "数值实验采用四阶Runge-Kutta方法，"
        "积分步长设置为0.001。"
    ),
    (
        "忆阻神经网络具有复杂的非线性动力学"
        "行为，可产生周期振荡和混沌现象。"
    ),
]

QUERIES = [
    "论文如何证明同步误差最终收敛？",
    (
        "Which numerical integration method "
        "is used in the simulation?"
    ),
]


def main() -> int:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"计算设备：{device}")
    print("正在加载Embedding模型……")

    encoder = SentenceTransformerEncoder(
        device=device,
        batch_size=16,
    )

    print(
        f"Embedding维度：{encoder.dimension}"
    )

    passage_vectors = (
        encoder.encode_passages(
            PASSAGES
        )
    )

    query_vectors = (
        encoder.encode_queries(
            QUERIES
        )
    )

    print(
        "Passage向量形状："
        f"{passage_vectors.shape}"
    )
    print(
        "Query向量形状："
        f"{query_vectors.shape}"
    )

    passage_lengths = np.linalg.norm(
        passage_vectors,
        axis=1,
    )

    query_lengths = np.linalg.norm(
        query_vectors,
        axis=1,
    )

    print(
        "Passage向量长度：",
        np.round(passage_lengths, 4),
    )
    print(
        "Query向量长度：",
        np.round(query_lengths, 4),
    )

    scores = (
        query_vectors
        @ passage_vectors.T
    )

    for query_index, query in enumerate(
        QUERIES
    ):
        sorted_indices = np.argsort(
            scores[query_index]
        )[::-1]

        print()
        print("=" * 60)
        print(f"问题：{query}")

        for rank, passage_index in enumerate(
            sorted_indices,
            start=1,
        ):
            score = scores[
                query_index,
                passage_index,
            ]

            print()
            print(
                f"第{rank}名，相似度："
                f"{score:.4f}"
            )
            print(
                f"原文：{PASSAGES[passage_index]}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
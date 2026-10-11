from database.chunk_repository import (
    ChunkRepository,
)
from domain.chunk import Chunk
from embedding.base import TextEncoder
from embedding.vector_index import (
    InMemoryVectorIndex,
    VectorSearchResult,
)


class SemanticRetriever:
    """连接Chunk数据库、Embedding和向量索引。"""

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        encoder: TextEncoder,
    ) -> None:
        self.chunk_repository = chunk_repository
        self.encoder = encoder

        self._index = InMemoryVectorIndex(
            dimension=encoder.dimension,
        )

        self._is_built = False
        self._paper_id: str | None = None

    @property
    def is_built(self) -> bool:
        """索引是否已经成功构建。"""

        return self._is_built

    @property
    def indexed_count(self) -> int:
        """当前索引中Chunk的数量。"""

        return len(self._index)

    def build_index(
        self,
        paper_id: str | None = None,
    ) -> int:
        """从SQLite读取Chunk并构建内存索引。"""

        if paper_id is None:
            chunks = (
                self.chunk_repository.list_all()
            )
        else:
            chunks = (
                self.chunk_repository.list_by_paper(
                    paper_id
                )
            )

        new_index = InMemoryVectorIndex(
            dimension=self.encoder.dimension
        )

        if chunks:
            passage_texts = [
                chunk.embedding_text
                for chunk in chunks
            ]

            passage_vectors = (
                self.encoder.encode_passages(passage_texts)
            )

            new_index.add(
                chunks=chunks,
                vectors=passage_vectors,
            )

        self._index = new_index
        self._paper_id = paper_id
        self._is_built = True

        return len(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """根据自然语言问题检索相关Chunk。"""

        if not self._is_built:
            raise RuntimeError(
                "请先调用 build_index() 构建索引。"
            )

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "查询问题不能为空。"
            )

        query_vectors = (
            self.encoder.encode_queries(
                [cleaned_query]
            )
        )

        expected_shape = (
            1,
            self.encoder.dimension,
        )

        if query_vectors.shape != expected_shape:
            raise RuntimeError(
                "查询Embedding形状不正确："
                f"期望 {expected_shape}，"
                f"实际 {query_vectors.shape}。"
            )

        return self._index.search(
            query_vector=query_vectors[0],
            top_k=top_k,
        )


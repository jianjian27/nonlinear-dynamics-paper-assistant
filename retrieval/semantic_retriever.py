from database.chunk_repository import (
    ChunkRepository,
)
from embedding.base import TextEncoder
from embedding.vector_index import (
    InMemoryVectorIndex,
    VectorSearchResult,
)
from retrieval.content_filter import (
    is_reference_chunk,
    query_requests_references,
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
        include_references: bool | None = None,
    ) -> list[VectorSearchResult]:
        """
        根据自然语言问题检索相关Chunk。

        include_references为None时，根据问题内容
        自动判断是否需要保留参考文献。
        """

        if not self._is_built:
            raise RuntimeError(
                "请先调用 build_index() 构建索引。"
            )

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "查询问题不能为空。"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k必须大于0。"
            )

        if include_references is None:
            reference_mode = (
                "only"
                if query_requests_references(
                    cleaned_query
                )
                else "exclude"
            )
        elif include_references:
            reference_mode = "include"
        else:
            reference_mode = "exclude"

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

        candidate_count = (
            top_k
            if reference_mode == "include"
            else max(
                self.indexed_count,
                1,
            )
        )

        candidates = self._index.search(
            query_vector=query_vectors[0],
            top_k=candidate_count,
        )

        if reference_mode == "only":
            filtered_results = [
                result
                for result in candidates
                if is_reference_chunk(
                    result.chunk
                )
            ]
        elif reference_mode == "exclude":
            filtered_results = [
                result
                for result in candidates
                if not is_reference_chunk(
                    result.chunk
                )
            ]
        else:
            filtered_results = candidates

        return filtered_results[:top_k]
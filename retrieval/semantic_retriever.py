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
from embedding.vector_cache import (
    VectorCache,
)
from collections.abc import (
    Sequence,
)


class SemanticRetriever:
    """连接Chunk数据库、Embedding和向量索引。"""

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        encoder: TextEncoder,
        *,
        vector_cache: VectorCache | None = None,
    ) -> None:
        self.chunk_repository = chunk_repository
        self.encoder = encoder
        self.vector_cache = vector_cache

        self._index = InMemoryVectorIndex(
            dimension=encoder.dimension,
        )

        self._is_built = False
        self._paper_ids: (
                tuple[str, ...] | None
        ) = None
        self._last_build_used_cache = False

    @property
    def is_built(self) -> bool:
        """索引是否已经成功构建。"""

        return self._is_built

    @property
    def indexed_count(self) -> int:
        """当前索引中Chunk的数量。"""

        return len(self._index)

    @property
    def last_build_used_cache(self) -> bool:
        """最近一次构建是否使用了磁盘缓存。"""

        return self._last_build_used_cache

    def build_index(
            self,
            paper_id: str | None = None,
            *,
            paper_ids: (
                    Sequence[str] | None
            ) = None,
    ) -> int:
        """从指定论文范围构建内存索引。"""

        if (
                paper_id is not None
                and paper_ids is not None
        ):
            raise ValueError(
                "paper_id和paper_ids"
                "不能同时使用。"
            )

        if paper_ids is not None:
            normalized_ids = tuple(
                sorted(
                    {
                        current_id.strip()
                        for current_id
                        in paper_ids
                        if current_id.strip()
                    }
                )
            )

            if not normalized_ids:
                chunks = []
                cache_scope = (
                    "papers:empty"
                )

            elif len(normalized_ids) == 1:
                only_paper_id = (
                    normalized_ids[0]
                )

                chunks = (
                    self.chunk_repository
                    .list_by_paper(
                        only_paper_id
                    )
                )

                cache_scope = (
                    f"paper:{only_paper_id}"
                )

            else:
                chunks = (
                    self.chunk_repository
                    .list_by_paper_ids(
                        normalized_ids
                    )
                )

                cache_scope = (
                        "papers:"
                        + ",".join(
                    normalized_ids
                )
                )

            active_paper_ids = (
                normalized_ids
            )

        elif paper_id is not None:
            cleaned_paper_id = (
                paper_id.strip()
            )

            if not cleaned_paper_id:
                raise ValueError(
                    "paper_id不能为空。"
                )

            chunks = (
                self.chunk_repository
                .list_by_paper(
                    cleaned_paper_id
                )
            )

            cache_scope = (
                f"paper:{cleaned_paper_id}"
            )

            active_paper_ids = (
                cleaned_paper_id,
            )

        else:
            chunks = (
                self.chunk_repository
                .list_all()
            )

            cache_scope = "all"
            active_paper_ids = None

        new_index = InMemoryVectorIndex(
            dimension=self.encoder.dimension
        )

        self._last_build_used_cache = False

        if chunks:
            passage_vectors = None

            if self.vector_cache is not None:
                passage_vectors = (
                    self.vector_cache.load(
                        chunks=chunks,
                        model_name=(
                            self.encoder.identifier
                        ),
                        dimension=(
                            self.encoder.dimension
                        ),
                        scope=cache_scope,
                    )
                )

            if passage_vectors is not None:
                self._last_build_used_cache = (
                    True
                )
            else:
                passage_texts = [
                    chunk.embedding_text
                    for chunk in chunks
                ]

                passage_vectors = (
                    self.encoder
                    .encode_passages(
                        passage_texts
                    )
                )

                if self.vector_cache is not None:
                    self.vector_cache.save(
                        chunks=chunks,
                        vectors=(
                            passage_vectors
                        ),
                        model_name=(
                            self.encoder.identifier
                        ),
                        dimension=(
                            self.encoder.dimension
                        ),
                        scope=cache_scope,
                    )

            new_index.add(
                chunks=chunks,
                vectors=passage_vectors,
            )

        self._index = new_index
        self._paper_ids = active_paper_ids
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
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pydantic import ValidationError

from agent_tools.retrieval_tools import (
    SearchPaperEvidenceInput,
    SearchPaperEvidenceTool,
)
from database.chunk_repository import (
    ChunkRepository,
)
from database.connection import (
    create_connection,
)
from database.paper_repository import (
    PaperRepository,
)
from database.schema import (
    initialize_database,
)
from database.section_repository import (
    SectionRepository,
)
from domain.chunk import (
    Chunk,
    hash_chunk_content,
)
from domain.paper import Paper
from embedding.base import TextEncoder
from utils.pdf_structure import Section


class FakeEncoder(TextEncoder):
    """测试专用的确定性Embedding编码器。"""

    def __init__(self) -> None:
        self.passage_call_count = 0
        self.query_call_count = 0

    @property
    def identifier(self) -> str:
        return "fake-agent-tool-encoder"

    @property
    def dimension(self) -> int:
        return 2

    def encode_passages(
        self,
        texts: list[str],
    ) -> np.ndarray:
        self.passage_call_count += 1

        return np.asarray(
            [
                self._encode_text(text)
                for text in texts
            ],
            dtype=np.float32,
        )

    def encode_queries(
        self,
        texts: list[str],
    ) -> np.ndarray:
        self.query_call_count += 1

        return np.asarray(
            [
                self._encode_text(text)
                for text in texts
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _encode_text(
        text: str,
    ) -> list[float]:
        """根据关键词生成简单的二维测试向量。"""

        if (
            "稳定" in text
            or "Lyapunov" in text
        ):
            return [1.0, 0.0]

        return [0.0, 1.0]


class TestSearchPaperEvidenceTool(
    unittest.TestCase
):
    """测试论文证据检索工具。"""

    def setUp(self) -> None:
        self.temporary_directory = (
            TemporaryDirectory()
        )

        self.directory = Path(
            self.temporary_directory.name
        )

        self.database_path = (
            self.directory
            / "papers.db"
        )

        self.vector_cache_directory = (
            self.directory
            / "vector_indexes"
        )

        connection = create_connection(
            self.database_path
        )

        initialize_database(connection)

        self._add_paper_with_chunk(
            connection=connection,
            paper_id="paper-001",
            title="混沌系统稳定性研究",
            section_title="2 稳定性分析",
            content=(
                "本文使用Lyapunov指数"
                "分析系统的稳定性。"
            ),
            page=3,
            file_hash="hash-001",
        )

        self._add_paper_with_chunk(
            connection=connection,
            paper_id="paper-002",
            title="非线性系统数值仿真研究",
            section_title="3 数值计算",
            content=(
                "本文采用四阶Runge-Kutta"
                "方法进行数值仿真。"
            ),
            page=5,
            file_hash="hash-002",
        )

        connection.close()

        self.encoder = FakeEncoder()

        self.tool = SearchPaperEvidenceTool(
            database_path=self.database_path,
            vector_cache_directory=(
                self.vector_cache_directory
            ),
            encoder=self.encoder,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _add_paper_with_chunk(
        *,
        connection,
        paper_id: str,
        title: str,
        section_title: str,
        content: str,
        page: int,
        file_hash: str,
    ) -> None:
        """向临时数据库添加论文、章节和Chunk。"""

        PaperRepository(
            connection
        ).add(
            Paper(
                id=paper_id,
                title=title,
                authors=("测试作者",),
                year=2026,
                file_name=f"{paper_id}.pdf",
                file_path=(
                    f"papers/{paper_id}.pdf"
                ),
                file_hash=file_hash,
            )
        )

        section_records = (
            SectionRepository(
                connection
            ).add_tree(
                paper_id=paper_id,
                root_sections=[
                    Section(
                        title=section_title,
                        level=1,
                    )
                ],
            )
        )

        chunk = Chunk(
            id=f"{paper_id}-chunk-001",
            paper_id=paper_id,
            chunk_index=0,
            section_title=section_title,
            section_path=(
                section_title,
            ),
            content=content,
            page_start=page,
            page_end=page,
            pages=(page,),
            source_blocks=(
                f"{page}:1",
            ),
            content_hash=(
                hash_chunk_content(
                    content
                )
            ),
        )

        ChunkRepository(
            connection
        ).add_many(
            chunks=[chunk],
            sections=section_records,
        )

    def test_generates_search_tool_schema(
        self,
    ) -> None:
        """工具Schema应描述查询、范围和Top-K。"""

        definition = (
            self.tool.get_definition()
        )

        function = definition["function"]
        parameters = function["parameters"]

        self.assertEqual(
            function["name"],
            "search_paper_evidence",
        )

        self.assertIn(
            "query",
            parameters["properties"],
        )

        self.assertIn(
            "paper_ids",
            parameters["properties"],
        )

        self.assertIn(
            "top_k",
            parameters["properties"],
        )

        self.assertIn(
            "query",
            parameters["required"],
        )

    def test_normalizes_input_arguments(
        self,
    ) -> None:
        """输入模型应清理问题和论文ID。"""

        arguments = (
            SearchPaperEvidenceInput
            .model_validate(
                {
                    "query": (
                        "  如何分析稳定性？  "
                    ),
                    "paper_ids": [
                        " paper-002 ",
                        "paper-001",
                        "paper-002",
                    ],
                    "top_k": 2,
                }
            )
        )

        self.assertEqual(
            arguments.query,
            "如何分析稳定性？",
        )

        self.assertEqual(
            arguments.paper_ids,
            [
                "paper-001",
                "paper-002",
            ],
        )

    def test_rejects_invalid_arguments(
        self,
    ) -> None:
        """空问题和非法Top-K应被Pydantic拒绝。"""

        with self.assertRaises(
            ValidationError
        ):
            self.tool.invoke(
                {
                    "query": "   ",
                }
            )

        with self.assertRaises(
            ValidationError
        ):
            self.tool.invoke(
                {
                    "query": "稳定性分析",
                    "top_k": 11,
                }
            )

    def test_returns_missing_database(
        self,
    ) -> None:
        """数据库不存在时应返回明确状态。"""

        tool = SearchPaperEvidenceTool(
            database_path=(
                self.directory
                / "missing.db"
            ),
            vector_cache_directory=(
                self.vector_cache_directory
            ),
            encoder=self.encoder,
        )

        result = tool.invoke(
            {
                "query": "如何分析稳定性？",
            }
        )

        self.assertFalse(
            result["database_available"]
        )

        self.assertEqual(
            result["indexed_count"],
            0,
        )

        self.assertEqual(
            result["evidences"],
            [],
        )

    def test_searches_all_papers(
        self,
    ) -> None:
        """全库检索应返回最相关的原文证据。"""

        result = self.tool.invoke(
            {
                "query": "如何分析系统稳定性？",
                "top_k": 2,
            }
        )

        self.assertTrue(
            result["database_available"]
        )

        self.assertIsNone(
            result["scope_paper_ids"]
        )

        self.assertEqual(
            result["indexed_count"],
            2,
        )

        self.assertEqual(
            result["result_count"],
            2,
        )

        first_evidence = (
            result["evidences"][0]
        )

        self.assertEqual(
            first_evidence["rank"],
            1,
        )

        self.assertEqual(
            first_evidence["paper_id"],
            "paper-001",
        )

        self.assertEqual(
            first_evidence["paper_title"],
            "混沌系统稳定性研究",
        )

        self.assertEqual(
            first_evidence["section_path"],
            [
                "2 稳定性分析",
            ],
        )

        self.assertEqual(
            first_evidence["page_start"],
            3,
        )

        self.assertIn(
            "Lyapunov",
            first_evidence["content"],
        )

    def test_limits_search_to_selected_paper(
        self,
    ) -> None:
        """paper_ids应限制实际索引和检索范围。"""

        result = self.tool.invoke(
            {
                "query": "如何分析系统稳定性？",
                "paper_ids": [
                    " paper-002 ",
                    "paper-002",
                ],
                "top_k": 5,
            }
        )

        self.assertEqual(
            result["scope_paper_ids"],
            [
                "paper-002",
            ],
        )

        self.assertEqual(
            result["indexed_count"],
            1,
        )

        self.assertEqual(
            result["result_count"],
            1,
        )

        self.assertEqual(
            result["evidences"][0][
                "paper_id"
            ],
            "paper-002",
        )

        self.assertIn(
            "Runge-Kutta",
            result["evidences"][0][
                "content"
            ],
        )

    def test_reuses_in_memory_index(
        self,
    ) -> None:
        """相同范围的重复检索不应重新编码Chunk。"""

        self.tool.invoke(
            {
                "query": "系统如何保持稳定？",
                "top_k": 1,
            }
        )

        self.tool.invoke(
            {
                "query": "使用了什么稳定性方法？",
                "top_k": 1,
            }
        )

        self.assertEqual(
            self.encoder.passage_call_count,
            1,
        )

        self.assertEqual(
            self.encoder.query_call_count,
            2,
        )


if __name__ == "__main__":
    unittest.main()
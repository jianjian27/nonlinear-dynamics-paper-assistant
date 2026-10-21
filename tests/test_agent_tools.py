import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from agent_tools.paper_tools import (
    ListPapersTool,
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
from domain.paper import Paper


class TestListPapersTool(
    unittest.TestCase
):
    """测试论文列表Agent工具。"""

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

    def tearDown(self) -> None:
            self.temporary_directory.cleanup()

    def test_generates_tool_definition(
        self,
    ) -> None:
        """工具应生成OpenAI兼容定义。"""

        tool = ListPapersTool(
            self.database_path
        )

        definition = (
            tool.get_definition()
        )

        self.assertEqual(
            definition["type"],
            "function",
        )

        function = definition[
            "function"
        ]

        self.assertEqual(
            function["name"],
            "list_papers",
        )

        self.assertIn(
            "parameters",
            function,
        )

        self.assertEqual(
            function["parameters"][
                "type"
            ],
            "object",
        )

        self.assertFalse(
            function["parameters"][
                "additionalProperties"
            ]
        )

    def test_returns_missing_database(
        self,
    ) -> None:
        """数据库不存在时应返回明确状态。"""

        tool = ListPapersTool(
            self.database_path
        )

        result = tool.invoke({})

        self.assertFalse(
            result["database_available"]
        )

        self.assertEqual(
            result["total"],
            0,
        )

        self.assertEqual(
            result["papers"],
            [],
        )

    def test_lists_papers(
        self,
    ) -> None:
        """工具应返回结构化论文数据。"""

        connection = create_connection(
            self.database_path
        )

        initialize_database(
            connection
        )

        repository = PaperRepository(
            connection
        )

        repository.add(
            Paper(
                id="paper-001",
                title="分数阶混沌系统研究",
                authors=(
                    "作者甲",
                    "作者乙",
                ),
                year=2026,
                file_name="paper.pdf",
                file_path="papers/paper.pdf",
                file_hash="hash-001",
                doi="10.0000/example",
            )
        )

        connection.close()

        tool = ListPapersTool(
            self.database_path
        )

        result = tool.invoke({})

        self.assertTrue(
            result["database_available"]
        )

        self.assertEqual(
            result["total"],
            1,
        )

        paper = result["papers"][0]

        self.assertEqual(
            paper["paper_id"],
            "paper-001",
        )

        self.assertEqual(
            paper["title"],
            "分数阶混沌系统研究",
        )

        self.assertEqual(
            paper["authors"],
            [
                "作者甲",
                "作者乙",
            ],
        )

        self.assertEqual(
            paper["section_count"],
            0,
        )

        self.assertEqual(
            paper["chunk_count"],
            0,
        )

    def test_rejects_unknown_arguments(
        self,
    ) -> None:
        """工具不应接受Schema外的参数。"""

        tool = ListPapersTool(
            self.database_path
        )

        with self.assertRaises(
            ValidationError
        ):
            tool.invoke(
                {
                    "unknown": "value",
                }
            )


if __name__ == "__main__":
    unittest.main()





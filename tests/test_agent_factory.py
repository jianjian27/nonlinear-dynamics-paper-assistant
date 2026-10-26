import unittest
from collections.abc import (
    Sequence,
)
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from agent_app import (
    create_paper_agent_runner,
    create_paper_tool_registry,
)
from agent_runtime import (
    AgentAssistantMessage,
    AgentMessage,
    ToolDefinition,
)
from agent_tools import (
    GetPaperOverviewTool,
    ListPapersTool,
    SearchPaperEvidenceTool,
)


class DirectAnswerModel:
    """直接返回最终回答的测试模型。"""

    def __init__(self) -> None:
        self.received_messages: list[
            dict[str, Any]
        ] = []

        self.received_tools: list[
            dict[str, Any]
        ] = []

    def complete(
        self,
        *,
        messages: Sequence[
            AgentMessage
        ],
        tools: Sequence[
            ToolDefinition
        ],
    ) -> AgentAssistantMessage:
        self.received_messages = [
            dict(message)
            for message in messages
        ]

        self.received_tools = [
            dict(tool)
            for tool in tools
        ]

        return AgentAssistantMessage(
            content="工厂测试回答",
            finish_reason="stop",
        )


class TestPaperAgentFactory(
    unittest.TestCase
):
    """测试科研论文Agent组合根。"""

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

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_registers_all_paper_tools(
        self,
    ) -> None:
        """工厂应注册三个论文只读工具。"""

        registry = (
            create_paper_tool_registry(
                database_path=(
                    self.database_path
                ),
                vector_cache_directory=(
                    self.vector_cache_directory
                ),
            )
        )

        self.assertEqual(
            registry.names,
            (
                "list_papers",
                "get_paper_overview",
                "search_paper_evidence",
            ),
        )

        self.assertIsInstance(
            registry.require(
                "list_papers"
            ),
            ListPapersTool,
        )

        self.assertIsInstance(
            registry.require(
                "get_paper_overview"
            ),
            GetPaperOverviewTool,
        )

        self.assertIsInstance(
            registry.require(
                "search_paper_evidence"
            ),
            SearchPaperEvidenceTool,
        )

    def test_passes_retrieval_configuration(
        self,
    ) -> None:
        """检索配置应传给证据检索工具。"""

        registry = (
            create_paper_tool_registry(
                database_path=(
                    self.database_path
                ),
                vector_cache_directory=(
                    self.vector_cache_directory
                ),
                embedding_model_name=(
                    "test-embedding-model"
                ),
                embedding_device="cpu",
            )
        )

        tool = registry.require(
            "search_paper_evidence"
        )

        self.assertIsInstance(
            tool,
            SearchPaperEvidenceTool,
        )

        self.assertEqual(
            tool.database_path,
            self.database_path.resolve(),
        )

        self.assertEqual(
            tool.vector_cache_directory,
            self.vector_cache_directory,
        )

        self.assertEqual(
            tool.embedding_model,
            "test-embedding-model",
        )

        self.assertEqual(
            tool.device,
            "cpu",
        )

    @patch(
        "agent_tools.retrieval_tools."
        "SentenceTransformerEncoder"
    )
    def test_does_not_load_embedding_during_creation(
        self,
        mock_encoder,
    ) -> None:
        """创建工具注册表时不应加载Embedding。"""

        create_paper_tool_registry(
            database_path=(
                self.database_path
            ),
            vector_cache_directory=(
                self.vector_cache_directory
            ),
        )

        mock_encoder.assert_not_called()

    def test_injects_chat_model(
        self,
    ) -> None:
        """工厂应使用传入的模型替身。"""

        model = DirectAnswerModel()

        runner = create_paper_agent_runner(
            database_path=(
                self.database_path
            ),
            vector_cache_directory=(
                self.vector_cache_directory
            ),
            chat_model=model,
        )

        self.assertIs(
            runner.model,
            model,
        )

        result = runner.run(
            "测试问题"
        )

        self.assertEqual(
            result.answer,
            "工厂测试回答",
        )

        self.assertEqual(
            [
                definition["function"][
                    "name"
                ]
                for definition
                in model.received_tools
            ],
            [
                "list_papers",
                "get_paper_overview",
                "search_paper_evidence",
            ],
        )

    def test_passes_runner_limits(
        self,
    ) -> None:
        """运行限制应传入AgentRunner。"""

        runner = create_paper_agent_runner(
            database_path=(
                self.database_path
            ),
            vector_cache_directory=(
                self.vector_cache_directory
            ),
            chat_model=(
                DirectAnswerModel()
            ),
            system_prompt="自定义提示词",
            max_model_rounds=3,
            max_tool_calls=4,
        )

        self.assertEqual(
            runner.system_prompt,
            "自定义提示词",
        )

        self.assertEqual(
            runner.max_model_rounds,
            3,
        )

        self.assertEqual(
            runner.max_tool_calls,
            4,
        )


if __name__ == "__main__":
    unittest.main()
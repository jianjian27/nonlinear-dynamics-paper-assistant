import unittest
from pathlib import Path
from tempfile import (
    TemporaryDirectory,
)
from unittest.mock import (
    Mock,
    patch,
)

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)

from agent_runtime.runner import (
    DEFAULT_AGENT_SYSTEM_PROMPT,
)
from framework_agent import (
    create_langchain_paper_agent,
)


class TestLangChainAgentFactory(
    unittest.TestCase
):
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

        self.chat_model = Mock(
            spec=BaseChatModel
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @patch(
        "framework_agent.agent_factory."
        "create_agent"
    )
    def test_assembles_agent(
        self,
        mock_create_agent,
    ) -> None:
        """应组装模型、工具和提示词。"""

        compiled_agent = object()

        mock_create_agent.return_value = (
            compiled_agent
        )

        result = (
            create_langchain_paper_agent(
                database_path=(
                    self.database_path
                ),
                vector_cache_directory=(
                    self
                    .vector_cache_directory
                ),
                chat_model=self.chat_model,
            )
        )

        self.assertIs(
            result,
            compiled_agent,
        )

        mock_create_agent.assert_called_once()

        arguments = (
            mock_create_agent
            .call_args
            .kwargs
        )

        self.assertIs(
            arguments["model"],
            self.chat_model,
        )

        self.assertEqual(
            arguments["system_prompt"],
            DEFAULT_AGENT_SYSTEM_PROMPT,
        )

        self.assertEqual(
            arguments["name"],
            "paper_research_agent",
        )

        self.assertFalse(
            arguments["debug"]
        )

        self.assertEqual(
            tuple(
                tool.name
                for tool
                in arguments["tools"]
            ),
            (
                "list_papers",
                "get_paper_overview",
                "search_paper_evidence",
            ),
        )

    @patch(
        "framework_agent.agent_factory."
        "create_agent"
    )
    def test_passes_custom_configuration(
        self,
        mock_create_agent,
    ) -> None:
        """应传递自定义提示词和调试开关。"""

        create_langchain_paper_agent(
            database_path=(
                self.database_path
            ),
            vector_cache_directory=(
                self.vector_cache_directory
            ),
            chat_model=self.chat_model,
            system_prompt="自定义科研提示词",
            debug=True,
        )

        arguments = (
            mock_create_agent
            .call_args
            .kwargs
        )

        self.assertEqual(
            arguments["system_prompt"],
            "自定义科研提示词",
        )

        self.assertTrue(
            arguments["debug"]
        )

    @patch(
        "framework_agent.agent_factory."
        "create_aliyun_langchain_model"
    )
    @patch(
        "framework_agent.agent_factory."
        "create_agent"
    )
    def test_creates_default_model(
        self,
        mock_create_agent,
        mock_model_factory,
    ) -> None:
        """未注入模型时应调用模型工厂。"""

        default_model = Mock(
            spec=BaseChatModel
        )

        mock_model_factory.return_value = (
            default_model
        )

        create_langchain_paper_agent(
            database_path=(
                self.database_path
            ),
            vector_cache_directory=(
                self.vector_cache_directory
            ),
        )

        (
            mock_model_factory
            .assert_called_once_with()
        )

        arguments = (
            mock_create_agent
            .call_args
            .kwargs
        )

        self.assertIs(
            arguments["model"],
            default_model,
        )

    @patch(
        "framework_agent.agent_factory."
        "create_aliyun_langchain_model"
    )
    @patch(
        "framework_agent.agent_factory."
        "create_agent"
    )
    def test_injected_model_skips_factory(
        self,
        mock_create_agent,
        mock_model_factory,
    ) -> None:
        """注入模型时不应创建默认模型。"""

        create_langchain_paper_agent(
            database_path=(
                self.database_path
            ),
            vector_cache_directory=(
                self.vector_cache_directory
            ),
            chat_model=self.chat_model,
        )

        mock_model_factory.assert_not_called()

    def test_rejects_blank_system_prompt(
        self,
    ) -> None:
        """系统提示词不能为空。"""

        with self.assertRaisesRegex(
            ValueError,
            "系统提示词",
        ):
            create_langchain_paper_agent(
                database_path=(
                    self.database_path
                ),
                vector_cache_directory=(
                    self
                    .vector_cache_directory
                ),
                chat_model=self.chat_model,
                system_prompt="   ",
            )


if __name__ == "__main__":
    unittest.main()
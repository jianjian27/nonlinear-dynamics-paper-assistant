import os
import unittest
from unittest.mock import (
    patch,
)

from langchain_openai import (
    ChatOpenAI,
)

from framework_agent import (
    LangChainModelConfigurationError,
    create_aliyun_langchain_model,
)


class TestLangChainModelFactory(
    unittest.TestCase
):
    def test_creates_chat_openai_model(
        self,
    ) -> None:
        """应使用显式配置创建模型。"""

        model = (
            create_aliyun_langchain_model(
                api_key="test-api-key",
                base_url=(
                    "https://example.com/v1"
                ),
                model_name="test-model",
                request_timeout=30,
                max_tokens=2000,
                enable_thinking=False,
                temperature=0.0,
                max_retries=3,
            )
        )

        self.assertIsInstance(
            model,
            ChatOpenAI,
        )

        self.assertEqual(
            model.model_name,
            "test-model",
        )

        self.assertEqual(
            model.openai_api_base,
            "https://example.com/v1",
        )

        self.assertEqual(
            (
                model
                .openai_api_key
                .get_secret_value()
            ),
            "test-api-key",
        )

        self.assertEqual(
            model.request_timeout,
            30,
        )

        self.assertEqual(
            model.temperature,
            0.0,
        )

        self.assertEqual(
            model.max_retries,
            3,
        )

        self.assertEqual(
            model.extra_body,
            {
                "enable_thinking": False,
                "max_tokens": 2000,
            },
        )

    def test_reads_environment_variables(
        self,
    ) -> None:
        """未显式传入时应读取环境变量。"""

        environment = {
            "DASHSCOPE_API_KEY": (
                "environment-key"
            ),
            "DASHSCOPE_BASE_URL": (
                "https://environment.test/v1"
            ),
        }

        with patch.dict(
            os.environ,
            environment,
            clear=True,
        ):
            model = (
                create_aliyun_langchain_model(
                    model_name="test-model",
                )
            )

        self.assertEqual(
            (
                model
                .openai_api_key
                .get_secret_value()
            ),
            "environment-key",
        )

        self.assertEqual(
            model.openai_api_base,
            "https://environment.test/v1",
        )

    def test_requires_api_key(
        self,
    ) -> None:
        """缺少API Key时应明确失败。"""

        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_BASE_URL": (
                    "https://example.com/v1"
                ),
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                LangChainModelConfigurationError,
                "DASHSCOPE_API_KEY",
            ):
                create_aliyun_langchain_model()

    def test_requires_base_url(
        self,
    ) -> None:
        """缺少Base URL时应明确失败。"""

        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": (
                    "test-key"
                ),
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                LangChainModelConfigurationError,
                "DASHSCOPE_BASE_URL",
            ):
                create_aliyun_langchain_model()

    def test_rejects_blank_model_name(
        self,
    ) -> None:
        """模型名称不能为空。"""

        with self.assertRaisesRegex(
            ValueError,
            "模型名称",
        ):
            create_aliyun_langchain_model(
                api_key="test-key",
                base_url=(
                    "https://example.com/v1"
                ),
                model_name="   ",
            )

    def test_rejects_invalid_limits(
        self,
    ) -> None:
        """超时和Token限制必须有效。"""

        with self.assertRaisesRegex(
            ValueError,
            "超时时间",
        ):
            create_aliyun_langchain_model(
                api_key="test-key",
                base_url=(
                    "https://example.com/v1"
                ),
                request_timeout=0,
            )

        with self.assertRaisesRegex(
            ValueError,
            "Token",
        ):
            create_aliyun_langchain_model(
                api_key="test-key",
                base_url=(
                    "https://example.com/v1"
                ),
                max_tokens=0,
            )

    def test_rejects_invalid_generation_options(
        self,
    ) -> None:
        """随机性和重试次数不能为负数。"""

        with self.assertRaisesRegex(
            ValueError,
            "temperature",
        ):
            create_aliyun_langchain_model(
                api_key="test-key",
                base_url=(
                    "https://example.com/v1"
                ),
                temperature=-0.1,
            )

        with self.assertRaisesRegex(
            ValueError,
            "重试次数",
        ):
            create_aliyun_langchain_model(
                api_key="test-key",
                base_url=(
                    "https://example.com/v1"
                ),
                max_retries=-1,
            )


if __name__ == "__main__":
    unittest.main()
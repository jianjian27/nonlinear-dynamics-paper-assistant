import unittest

from langchain_core.tools import (
    StructuredTool,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from agent_tools import (
    AgentTool,
)
from framework_agent import (
    adapt_agent_tool,
    adapt_agent_tools,
)


class ExampleInput(BaseModel):
    """测试工具输入。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    text: str = Field(
        min_length=1,
        description="需要处理的文本。",
    )


class ExampleOutput(BaseModel):
    """测试工具输出。"""

    original: str
    upper: str


class ExampleTool(AgentTool):
    """测试用原有工具。"""

    name = "example_tool"

    description = (
        "将输入文本转换成大写。"
    )

    input_model = ExampleInput

    def run(
        self,
        arguments: BaseModel,
    ) -> BaseModel:
        if not isinstance(
            arguments,
            ExampleInput,
        ):
            raise TypeError(
                "参数模型错误。"
            )

        return ExampleOutput(
            original=arguments.text,
            upper=arguments.text.upper(),
        )


class SecondExampleTool(ExampleTool):
    """另一个名称不同的测试工具。"""

    name = "second_example_tool"


class DuplicateExampleTool(ExampleTool):
    """名称重复的测试工具。"""

    name = "example_tool"


class TestLangChainToolAdapter(
    unittest.TestCase
):
    def test_adapts_metadata(
        self,
    ) -> None:
        """应保留名称、描述和输入模型。"""

        original_tool = ExampleTool()

        adapted_tool = adapt_agent_tool(
            original_tool
        )

        self.assertIsInstance(
            adapted_tool,
            StructuredTool,
        )

        self.assertEqual(
            adapted_tool.name,
            original_tool.name,
        )

        self.assertEqual(
            adapted_tool.description,
            original_tool.description,
        )

        self.assertIs(
            adapted_tool.args_schema,
            ExampleInput,
        )

    def test_invokes_existing_tool(
        self,
    ) -> None:
        """LangChain调用应进入原工具。"""

        adapted_tool = adapt_agent_tool(
            ExampleTool()
        )

        result = adapted_tool.invoke(
            {
                "text": "agent",
            }
        )

        self.assertEqual(
            result,
            {
                "original": "agent",
                "upper": "AGENT",
            },
        )

    def test_preserves_validation(
        self,
    ) -> None:
        """输入仍然受到Pydantic约束。"""

        adapted_tool = adapt_agent_tool(
            ExampleTool()
        )

        with self.assertRaises(
            ValidationError
        ):
            adapted_tool.invoke(
                {
                    "text": "",
                }
            )

    def test_rejects_non_agent_tool(
        self,
    ) -> None:
        """不能适配任意普通对象。"""

        with self.assertRaisesRegex(
            TypeError,
            "AgentTool",
        ):
            adapt_agent_tool(
                object()  # type: ignore[arg-type]
            )

    def test_adapts_multiple_tools(
        self,
    ) -> None:
        """应按照输入顺序批量适配。"""

        adapted_tools = adapt_agent_tools(
            (
                ExampleTool(),
                SecondExampleTool(),
            )
        )

        self.assertEqual(
            tuple(
                tool.name
                for tool in adapted_tools
            ),
            (
                "example_tool",
                "second_example_tool",
            ),
        )

    def test_rejects_duplicate_names(
        self,
    ) -> None:
        """LangChain工具名称不能重复。"""

        with self.assertRaisesRegex(
            ValueError,
            "名称重复",
        ):
            adapt_agent_tools(
                (
                    ExampleTool(),
                    DuplicateExampleTool(),
                )
            )


if __name__ == "__main__":
    unittest.main()
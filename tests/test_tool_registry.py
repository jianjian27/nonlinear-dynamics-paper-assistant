import unittest

from pydantic import BaseModel

from agent_runtime import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistry,
)
from agent_tools import (
    GetPaperOverviewTool,
    ListPapersTool,
    SearchPaperEvidenceTool,
)
from agent_tools.base import AgentTool


class EchoInput(BaseModel):
    """测试工具输入。"""

    text: str


class EchoOutput(BaseModel):
    """测试工具输出。"""

    text: str


class EchoTool(AgentTool):
    """测试使用的简单工具。"""

    name = "echo"
    description = "原样返回输入文字。"
    input_model = EchoInput

    def run(
        self,
        arguments: BaseModel,
    ) -> EchoOutput:
        if not isinstance(
            arguments,
            EchoInput,
        ):
            raise TypeError(
                "echo收到错误的参数模型。"
            )

        return EchoOutput(
            text=arguments.text
        )


class InvalidNameTool(EchoTool):
    """名称中带有空格的错误工具。"""

    name = "invalid tool"


class PaddedNameTool(EchoTool):
    """名称两侧带有空格的错误工具。"""

    name = " echo "


class TestToolRegistry(
    unittest.TestCase
):
    """测试Agent工具注册表。"""

    def test_registers_and_finds_tool(
        self,
    ) -> None:
        """注册后应能按照名称找到同一工具。"""

        tool = EchoTool()

        registry = ToolRegistry(
            [tool]
        )

        self.assertEqual(
            len(registry),
            1,
        )

        self.assertEqual(
            registry.names,
            ("echo",),
        )

        self.assertIn(
            "echo",
            registry,
        )

        self.assertIs(
            registry.get("echo"),
            tool,
        )

        self.assertIs(
            registry.require("echo"),
            tool,
        )

    def test_exports_tool_definitions(
        self,
    ) -> None:
        """注册表应导出模型需要的工具定义。"""

        registry = ToolRegistry(
            [
                EchoTool(),
            ]
        )

        definitions = (
            registry.definitions
        )

        self.assertEqual(
            len(definitions),
            1,
        )

        function = (
            definitions[0]["function"]
        )

        self.assertEqual(
            function["name"],
            "echo",
        )

        self.assertIn(
            "text",
            function["parameters"][
                "properties"
            ],
        )

    def test_registers_real_paper_tools(
        self,
    ) -> None:
        """三个真实论文工具应能同时注册。"""

        registry = ToolRegistry(
            [
                ListPapersTool(),
                GetPaperOverviewTool(),
                SearchPaperEvidenceTool(),
            ]
        )

        self.assertEqual(
            registry.names,
            (
                "list_papers",
                "get_paper_overview",
                "search_paper_evidence",
            ),
        )

        definition_names = [
            definition["function"]["name"]
            for definition
            in registry.definitions
        ]

        self.assertEqual(
            definition_names,
            [
                "list_papers",
                "get_paper_overview",
                "search_paper_evidence",
            ],
        )

    def test_rejects_duplicate_name(
        self,
    ) -> None:
        """不允许注册两个同名工具。"""

        registry = ToolRegistry(
            [
                EchoTool(),
            ]
        )

        with self.assertRaisesRegex(
            DuplicateToolError,
            "echo",
        ):
            registry.register(
                EchoTool()
            )

    def test_raises_for_unknown_tool(
        self,
    ) -> None:
        """require遇到未知名称应抛出异常。"""

        registry = ToolRegistry()

        self.assertIsNone(
            registry.get("missing_tool")
        )

        with self.assertRaisesRegex(
            ToolNotFoundError,
            "missing_tool",
        ):
            registry.require(
                "missing_tool"
            )

    def test_rejects_invalid_tool_names(
        self,
    ) -> None:
        """工具名称必须符合机器标识格式。"""

        registry = ToolRegistry()

        with self.assertRaises(
            ValueError
        ):
            registry.register(
                InvalidNameTool()
            )

        with self.assertRaises(
            ValueError
        ):
            registry.register(
                PaddedNameTool()
            )

    def test_rejects_non_tool_object(
        self,
    ) -> None:
        """注册表只接受AgentTool实例。"""

        registry = ToolRegistry()

        with self.assertRaises(
            TypeError
        ):
            registry.register(
                object()  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
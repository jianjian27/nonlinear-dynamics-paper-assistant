import json
import unittest

from pydantic import (
    BaseModel,
    ConfigDict,
)

from agent_runtime import (
    ToolDispatcher,
    ToolRegistry,
)
from agent_tools.base import AgentTool


class AddInput(BaseModel):
    """加法工具输入。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    left: int
    right: int


class AddOutput(BaseModel):
    """加法工具输出。"""

    total: int


class AddTool(AgentTool):
    """测试使用的加法工具。"""

    name = "add_numbers"
    description = "计算两个整数之和。"
    input_model = AddInput

    def run(
        self,
        arguments: BaseModel,
    ) -> AddOutput:
        if not isinstance(
            arguments,
            AddInput,
        ):
            raise TypeError(
                "add_numbers收到错误的参数模型。"
            )

        return AddOutput(
            total=(
                arguments.left
                + arguments.right
            )
        )


class ExplodingInput(BaseModel):
    """异常工具输入。"""

    model_config = ConfigDict(
        extra="forbid",
    )


class ExplodingTool(AgentTool):
    """执行时主动抛出异常的测试工具。"""

    name = "explode"
    description = "用于测试工具执行异常。"
    input_model = ExplodingInput

    def run(
        self,
        arguments: BaseModel,
    ) -> BaseModel:
        raise RuntimeError(
            "模拟工具内部故障"
        )


class TestToolDispatcher(
    unittest.TestCase
):
    """测试Agent工具调度器。"""

    def setUp(self) -> None:
        registry = ToolRegistry(
            [
                AddTool(),
                ExplodingTool(),
            ]
        )

        self.dispatcher = ToolDispatcher(
            registry
        )

    def test_dispatches_successful_call(
        self,
    ) -> None:
        """有效调用应执行目标工具。"""

        result = self.dispatcher.dispatch(
            tool_call_id="call-001",
            tool_name="add_numbers",
            arguments_json=(
                '{"left": 2, "right": 3}'
            ),
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.tool_call_id,
            "call-001",
        )

        self.assertEqual(
            result.tool_name,
            "add_numbers",
        )

        self.assertEqual(
            result.output,
            {
                "total": 5,
            },
        )

        self.assertIsNone(
            result.error
        )

    def test_converts_result_to_tool_message(
        self,
    ) -> None:
        """执行结果应能转换成模型需要的tool消息。"""

        result = self.dispatcher.dispatch(
            tool_call_id="call-002",
            tool_name="add_numbers",
            arguments_json=(
                '{"left": 10, "right": 7}'
            ),
        )

        message = (
            result.to_tool_message()
        )

        self.assertEqual(
            message["role"],
            "tool",
        )

        self.assertEqual(
            message["tool_call_id"],
            "call-002",
        )

        payload = json.loads(
            message["content"]
        )

        self.assertTrue(
            payload["success"]
        )

        self.assertEqual(
            payload["result"],
            {
                "total": 17,
            },
        )

    def test_rejects_blank_call_id(
        self,
    ) -> None:
        """工具调用ID不能为空。"""

        result = self.dispatcher.dispatch(
            tool_call_id="   ",
            tool_name="add_numbers",
            arguments_json=(
                '{"left": 1, "right": 2}'
            ),
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.error.code,
            "invalid_tool_call_id",
        )

    def test_rejects_blank_tool_name(
        self,
    ) -> None:
        """工具名称不能为空。"""

        result = self.dispatcher.dispatch(
            tool_call_id="call-003",
            tool_name="   ",
            arguments_json="{}",
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.error.code,
            "invalid_tool_name",
        )

    def test_rejects_invalid_json(
        self,
    ) -> None:
        """损坏的JSON参数应转换为错误结果。"""

        result = self.dispatcher.dispatch(
            tool_call_id="call-004",
            tool_name="add_numbers",
            arguments_json=(
                '{"left": 1,'
            ),
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.error.code,
            "invalid_json",
        )

    def test_rejects_non_object_json(
        self,
    ) -> None:
        """工具参数必须是JSON对象。"""

        result = self.dispatcher.dispatch(
            tool_call_id="call-005",
            tool_name="add_numbers",
            arguments_json=(
                '["left", "right"]'
            ),
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.error.code,
            "invalid_arguments",
        )

    def test_reports_unknown_tool(
        self,
    ) -> None:
        """未知工具名应返回tool_not_found。"""

        result = self.dispatcher.dispatch(
            tool_call_id="call-006",
            tool_name="missing_tool",
            arguments_json="{}",
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.error.code,
            "tool_not_found",
        )

        self.assertIn(
            "missing_tool",
            result.error.message,
        )

    def test_reports_validation_error(
        self,
    ) -> None:
        """参数缺失应返回Pydantic验证错误。"""

        result = self.dispatcher.dispatch(
            tool_call_id="call-007",
            tool_name="add_numbers",
            arguments_json=(
                '{"left": 1}'
            ),
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.error.code,
            "validation_error",
        )

        self.assertIn(
            "right",
            result.error.message,
        )

    def test_reports_tool_execution_error(
        self,
    ) -> None:
        """工具内部异常不应使调度器崩溃。"""

        result = self.dispatcher.dispatch(
            tool_call_id="call-008",
            tool_name="explode",
            arguments_json="{}",
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.error.code,
            "tool_execution_error",
        )

        self.assertIn(
            "RuntimeError",
            result.error.message,
        )

        self.assertIn(
            "模拟工具内部故障",
            result.error.message,
        )

    def test_serializes_failure_message(
        self,
    ) -> None:
        """失败结果也应转换成合法tool消息。"""

        result = self.dispatcher.dispatch(
            tool_call_id="call-009",
            tool_name="missing_tool",
            arguments_json="{}",
        )

        message = (
            result.to_tool_message()
        )

        payload = json.loads(
            message["content"]
        )

        self.assertFalse(
            payload["success"]
        )

        self.assertEqual(
            payload["error"]["code"],
            "tool_not_found",
        )


if __name__ == "__main__":
    unittest.main()
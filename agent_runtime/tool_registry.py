import re
from collections.abc import Iterable

from agent_tools.base import AgentTool


TOOL_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{1,64}$"
)


class DulicateToolError(ValueError):
    """注册了名称重复的工具。"""


class ToolNotFoundError(LookupError):
    """找不到模型请求调用的工具。"""


class ToolRegistry:
    """统一保存和查找Agent可以调用的工具。"""

    def __init__(
        self,
        tools: Iterable[AgentTool] = (), # Iterable 表示“可迭代对象”，也就是可以被 for 循环逐个读取的对象。
    ) -> None:
        self._tools: dict[
            str,
            AgentTool,
        ] = {}

        for tool in tools:
            self.register(tool)

    def register(
        self,
        tool: AgentTool,
    ) -> None:
        """注册一个工具。"""

        if not isinstance(tool, AgentTool):
            raise TypeError(
                "只能注册AgentTool实例。"
            )

        tool_name = tool.name.strip()

        if not tool_name:
            raise ValueError(
                "工具名称不能为空。"
            )

        if not TOOL_NAME_PATTERN.fullmatch(tool_name):
            raise ValueError(
                "工具名称只能包含英文字母、"
                "数字、下划线或连字符，"
                "且长度不能超过64个字符。"
            )

        if tool_name in self._tools:
            raise DulicateToolError(
                f"工具已经注册：{tool_name}"
            )

        self._tools[tool_name] = tool

    def get(
        self,
        tool_name: str,
    ) -> AgentTool | None:
        """查询工具；不存在时返回None。"""

        cleaned_name = tool_name.strip()

        if not cleaned_name:
            return None

        return  self._tools.get(cleaned_name)

    def require(
        self,
        tool_name: str,
    ) -> AgentTool:
        """查询工具；不存在时抛出异常。"""

        tool = self.get(tool_name)

        if tool is None:
            raise ToolNotFoundError(
                f"没有注册工具：{tool_name}"
            )

        return tool

    @property
    def names(
        self,
    ) -> tuple[str,...]:
        """按照注册顺序返回全部工具名称。"""

        return tuple(self._tools)

    @property
    def definitions(
            self,
    ) -> list[dict[str, object]]:
        """生成发送给大模型的工具定义列表。"""

        return [
            tool.get_definition()
            for tool in self._tools.values()
        ]

    def __len__(self) -> int:
        """返回已经注册的工具数量。"""

        return len(self._tools)

    def __contains__(
            self,
            tool_name: object,
    ) -> bool:
        """支持使用in判断工具是否存在。"""

        if not isinstance(
                tool_name,
                str,
        ):
            return False

        return tool_name in self._tools
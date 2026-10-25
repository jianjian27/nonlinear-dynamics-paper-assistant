from _collections_abc import (
    Mapping,
    Sequence,
)
from typing import (
    Any,
    Protocol,
    TypeAlias,
    runtime_checkable,
)
# Protocol 用来描述一个对象“应该提供哪些方法”，但不要求这个对象必须继承该类。
#
# 例如：
# class AgentChatModel(Protocol):
#     def complete(
#         self,
#         messages,
#         tools,
#     ):
#         ...
# 它表达的要求是：
# 只要某个对象有 complete() 方法，
# 就可以把它当成 AgentChatModel 使用。

from agent_runtime.messages import (
    AgentAssistantMessage,
)


AgentMessage: TypeAlias = Mapping[
    str,
    Any,
]

ToolDefinition: TypeAlias = Mapping[
    str,
    Any,
]


@runtime_checkable # 允许程序在运行时使用 isinstance() 检查某个对象是否符合这个 Protocol
class AgentChatModel(Protocol):
    """Agent运行时需要的聊天模型接口。"""

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
        """
        根据消息历史和工具定义生成下一条消息。

        模型可以返回最终文本，也可以请求调用工具。
        """

        ...


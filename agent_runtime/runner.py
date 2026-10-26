from collections.abc import (
    Sequence
)
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
)

from agent_runtime.model_protocol import (
    AgentChatModel,
    AgentMessage,
)
from agent_runtime.tool_dispatcher import (
    ToolDispatcher,
)
from agent_runtime.tool_registry import (
    ToolRegistry,
)


DEFAULT_AGENT_SYSTEM_PROMPT = """
你是一名非线性动力系统科研论文助手。

你可以使用工具查看论文库、查询论文结构，
以及检索带有章节和页码的论文原文证据。

工作要求：
1. 涉及论文库内容时，优先调用工具获取事实。
2. 不要根据记忆编造论文、公式、参数或实验结论。
3. 需要确定论文范围时，先使用list_papers。
4. 需要了解论文结构时，使用get_paper_overview。
5. 需要回答论文内容问题时，使用search_paper_evidence。
6. 工具返回失败时，可以修正参数后再次调用。
7. 最终回答应区分论文原文、合理推断和你的评价。
8. 引用证据时尽量注明论文、章节和页码。
""".strip()


class AgentRunLimitError(
    RuntimeError
):
    """Agent超过运行轮数或工具调用限制。"""

class AgentProtocolError(
    RuntimeError
):
    """模型返回了不符合Agent协议的结果。"""


class AgentRunResult(BaseModel):
    """一次完整Agent运行的结果。"""

    model_config = ConfigDict(
        frozen=True,
    )

    answer: str

    messages: tuple[
        dict[str, Any],
        ...
    ]

    model_rounds: int
    tool_call_count: int


class AgentRunner:
    """执行模型推理与工具调用的多轮循环。"""

    def __init__(
        self,
        *,
        model: AgentChatModel,
        registry: ToolRegistry,
        system_prompt: str = (
            DEFAULT_AGENT_SYSTEM_PROMPT
        ),
        max_model_rounds: int = 6,
        max_tool_calls: int = 12,
    ) -> None:
        cleaned_system_prompt = system_prompt.strip()

        if not cleaned_system_prompt:
            raise ValueError(
                "Agent系统提示词不能为空。"
            )

        if max_model_rounds <= 0:
            raise ValueError(
                "最大模型轮数必须大于0。"
            )

        if max_tool_calls <= 0:
            raise ValueError(
                "最大工具调用次数必须大于0。"
            )

        self.model = model
        self.registry = registry

        self.dispatcher = ToolDispatcher(registry)

        self.system_prompt = cleaned_system_prompt

        self.max_model_rounds = max_model_rounds
        self.max_tool_calls = max_tool_calls

    def run(
        self,
        question: str,
        *,
        history: Sequence[
            AgentMessage
        ] = (),
    ) -> AgentRunResult:
        """运行一次完整的Agent问答。"""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "用户问题不能为空。"
            )

        messages: list[
            dict[str, Any]
        ] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        messages.extend(
            dict(message)
            for message in history
        )

        messages.append(
            {
                "role": "user",
                "content": cleaned_question,
            }
        )

        tool_definitions = self.registry.definitions

        tool_call_count = 0

        used_tool_call_ids: set[
            str
        ] = set()

        for model_round in range(1, self.max_model_rounds + 1):
            assistant_message = (
                self.model.complete(
                    messages=messages,
                    tools=tool_definitions,
                )
            )

            messages.append(
                assistant_message.to_openai_dict()
            )

            if assistant_message.is_final_answer:
                if assistant_message.finish_reason == "length":
                    raise AgentProtocolError(
                        "模型回答达到长度上限，"
                        "不能视为完整最终答案。"
                    )

                return AgentRunResult(
                    answer=(
                        assistant_message.content
                    ),
                    messages=tuple(messages),
                    model_rounds=model_round,
                    tool_call_count=tool_call_count,
                )

            if not assistant_message.requests_tools:
                raise AgentProtocolError(
                    "模型既没有给出最终答案，"
                    "也没有请求调用工具。"
                )

            new_tool_call_count = (
                tool_call_count
                + len(assistant_message.tool_calls)
            )

            if new_tool_call_count > self.max_tool_calls:
                raise AgentRunLimitError(
                    "Agent工具调用次数超过限制："
                    f"{self.max_tool_calls}。"
                )

            for tool_call in assistant_message.tool_calls:
                if tool_call.id in used_tool_call_ids:
                    raise AgentProtocolError(
                        "模型重复使用了工具调用ID："
                        f"{tool_call.id}。"
                    )

                used_tool_call_ids.add(tool_call.id)

                execution_result = (
                    self.dispatcher.dispatch(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments_json=tool_call.arguments_json,
                    )
                )

                messages.append(execution_result.to_tool_message())

                tool_call_count += 1

        raise AgentRunLimitError(
            "Agent在达到最大模型轮数"
            f" {self.max_model_rounds} 后，"
            "仍未生成最终回答。"
        )
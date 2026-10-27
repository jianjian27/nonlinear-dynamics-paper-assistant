import json
from collections.abc import (
    Callable,
    Mapping,
)
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from agent_app.factory import (
    create_paper_agent_runner,
)
from agent_runtime import (
    AgentRunResult,
    AgentRunner,
)
from embedding.sentence_transformer_encoder import (
    DEFAULT_MODEL_NAME,
)



@dataclass(frozen=True)
class GradioAgentResult:
    """适合加给Gradio展示的Agent结果。"""

    answer_markdown: str
    trace_markdown: str
    status_text: str


class GradioAgentService:
    """为Gradio提供可复用的科研Agent服务。"""

    def __init__(
        self,
        *,
        database_path: str | Path = (
            "data/papers.db"
        ),
        vector_cache_directory: str | Path = (
            "data/vector_indexes"
        ),
        embedding_model_name: str = (
            DEFAULT_MODEL_NAME
        ),
        embedding_device: str | None = None,
        max_model_rounds: int = 6,
        max_tool_calls: int = 12,
        runner_factory: (
            Callable[[], AgentRunner] | None
        ) = None,
    ) -> None:
        self.database_path = Path(database_path).resolve()

        self.vector_cache_directory = Path(vector_cache_directory)

        self.embedding_model_name = embedding_model_name

        self.embedding_device = embedding_device

        self.max_model_rounds = max_model_rounds

        self.max_tool_calls = max_tool_calls

        if runner_factory is None:
            self._runner_factory = self._create_default_runner
        else:
            self._runner_factory = runner_factory

        self._runner: AgentRunner | None = None

        self._lock = Lock()

    def ask(
        self,
        question: str,
    ) ->GradioAgentResult :
        """运行一次科研Agent问答。"""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "请输入需要询问Agent的问题。"
            )

        if not self.database_path.is_file():
            raise FileNotFoundError(
                "找不到论文数据库："
                f"{self.database_path}"
            )

        with self._lock:
            runner = self._get_or_create_runner()

            result = runner.run(cleaned_question)

        return GradioAgentResult(
            answer_markdown=result.answer,
            trace_markdown=self._render_trace(result),
            status_text=self._render_status(result),
        )

    def reset(self) -> None:
        """
        清除当前Agent及其内存索引。

        论文入库或删除后调用，避免继续使用旧索引。
        """

        with self._lock:
            self._runner = None

    def _get_or_create_runner(self) -> AgentRunner:
        """延迟创建并复用AgentRunner。"""

        if self._runner is None:
            self._runner = self._runner_factory()

        return self._runner

    def _create_default_runner(self) -> AgentRunner:
        """按照网页配置创建真实Agent。"""

        return create_paper_agent_runner(
            database_path=self.database_path,
            vector_cache_directory=self.vector_cache_directory,
            embedding_model_name=self.embedding_model_name,
            embedding_device=self.embedding_device,
            max_model_rounds=self.max_model_rounds,
            max_tool_calls=self.max_tool_calls,
        )

    @staticmethod
    def _render_status(result: AgentRunResult) -> str:
        """生成简洁运行状态。"""

        return (
            "Agent运行完成；"
            f"模型轮数：{result.model_rounds}；"
            "工具调用数："
            f"{result.tool_call_count}。"
        )

    @staticmethod
    def _render_trace(
            result: AgentRunResult,
    ) -> str:
        """把工具调用过程转换为Markdown。"""

        tool_results: dict[
            str,
            dict[str, Any],
        ] = {}

        for message in result.messages:
            if message.get("role") != "tool":
                continue

            call_id = message.get(
                "tool_call_id"
            )

            content = message.get(
                "content"
            )

            if (
                    not isinstance(
                        call_id,
                        str,
                    )
                    or not isinstance(
                content,
                str,
            )
            ):
                continue

            try:
                payload = json.loads(
                    content
                )
            except json.JSONDecodeError:
                payload = {
                    "success": False,
                    "error": {
                        "code": (
                            "invalid_tool_result"
                        ),
                        "message": (
                            "无法解析工具返回值。"
                        ),
                    },
                }

            if isinstance(
                    payload,
                    dict,
            ):
                tool_results[
                    call_id
                ] = payload

        sections = [
            "## Agent 执行轨迹"
        ]

        call_number = 0

        for message in result.messages:
            if (
                    message.get("role")
                    != "assistant"
            ):
                continue

            tool_calls = message.get(
                "tool_calls"
            )

            if not isinstance(
                    tool_calls,
                    list,
            ):
                continue

            for tool_call in tool_calls:
                if not isinstance(
                        tool_call,
                        Mapping,
                ):
                    continue

                function = tool_call.get(
                    "function"
                )

                if not isinstance(
                        function,
                        Mapping,
                ):
                    continue

                call_number += 1

                call_id = str(
                    tool_call.get(
                        "id",
                        "unknown",
                    )
                )

                tool_name = str(
                    function.get(
                        "name",
                        "unknown",
                    )
                )

                raw_arguments = (
                    function.get(
                        "arguments",
                        "{}",
                    )
                )

                arguments_text = (
                    GradioAgentService
                    ._format_arguments(
                        raw_arguments
                    )
                )

                payload = (
                    tool_results.get(
                        call_id
                    )
                )

                status_text = (
                    GradioAgentService
                    ._format_tool_status(
                        payload
                    )
                )

                sections.append(
                    "\n".join(
                        [
                            (
                                f"### {call_number}. "
                                f"`{tool_name}`"
                            ),
                            "",
                            (
                                f"- 调用 ID："
                                f"`{call_id}`"
                            ),
                            (
                                f"- 执行状态："
                                f"{status_text}"
                            ),
                            "",
                            "**调用参数**",
                            "",
                            "```json",
                            arguments_text,
                            "```",
                        ]
                    )
                )

        if call_number == 0:
            sections.append(
                "模型没有调用工具，"
                "直接生成了回答。"
            )

        return "\n\n".join(
            sections
        )

    @staticmethod
    def _format_arguments(
            raw_arguments: Any,
    ) -> str:
        """格式化工具调用参数。"""

        if not isinstance(
                raw_arguments,
                str,
        ):
            return str(raw_arguments)

        try:
            parsed = json.loads(
                raw_arguments
            )
        except json.JSONDecodeError:
            return raw_arguments

        return json.dumps(
            parsed,
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _format_tool_status(
            payload: dict[
                         str,
                         Any,
                     ] | None,
    ) -> str:
        """格式化工具执行状态。"""

        if payload is None:
            return "⚠️ 未找到工具结果"

        if payload.get("success"):
            return "✅ 成功"

        error = payload.get(
            "error"
        )

        if not isinstance(
                error,
                Mapping,
        ):
            return "❌ 执行失败"

        code = error.get(
            "code",
            "unknown_error",
        )

        message = error.get(
            "message",
            "没有错误详情。",
        )

        return (
            f"❌ 失败（`{code}`）："
            f"{message}"
        )
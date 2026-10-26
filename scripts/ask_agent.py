import argparse
import json
import sys
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any

from agent_app import (
    create_paper_agent_runner,
)
from agent_runtime import (
    AgentProtocolError,
    AgentRunLimitError,
)
from embedding.sentence_transformer_encoder import (
    DEFAULT_MODEL_NAME,
)
from models.deepseek_api import (
    ModelServiceError,
)


def create_argument_parser(
) -> argparse.ArgumentParser:
    """创建科研Agent命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "运行科研论文Agent，"
            "由模型自主选择论文知识库工具。"
        )
    )

    parser.add_argument(
        "question",
        help="需要询问科研Agent的问题。",
    )

    parser.add_argument(
        "--database",
        default="data/papers.db",
        help=(
            "SQLite论文数据库路径，"
            "默认是data/papers.db。"
        ),
    )

    parser.add_argument(
        "--vector-cache-directory",
        default="data/vector_indexes",
        help=(
            "本地向量缓存目录，默认是"
            "data/vector_indexes。"
        ),
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL_NAME,
        help="本地Embedding模型名称。",
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Embedding运行设备，"
            "例如cuda或cpu。"
        ),
    )

    parser.add_argument(
        "--max-model-rounds",
        type=int,
        default=6,
        help="最大模型调用轮数，默认是6。",
    )

    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=12,
        help="最大工具调用次数，默认是12。",
    )

    parser.add_argument(
        "--show-trace",
        action="store_true",
        help=(
            "显示模型选择的工具及其"
            "执行成功或失败状态。"
        ),
    )

    return parser


def print_agent_trace(
    messages: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    """打印简洁的Agent工具调用轨迹。"""

    print()
    print("=== Agent执行轨迹 ===")

    found_tool_activity = False

    for message in messages:
        role = message.get("role")

        if role == "assistant":
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

                found_tool_activity = True

                call_id = tool_call.get(
                    "id",
                    "unknown",
                )

                name = function.get(
                    "name",
                    "unknown",
                )

                arguments = function.get(
                    "arguments",
                    "{}",
                )

                print(
                    f"模型请求：{name} "
                    f"[{call_id}]"
                )

                print(
                    f"参数：{arguments}"
                )

        elif role == "tool":
            found_tool_activity = True

            call_id = message.get(
                "tool_call_id",
                "unknown",
            )

            content = message.get(
                "content",
                "",
            )

            try:
                payload = json.loads(
                    content
                )
            except (
                json.JSONDecodeError,
                TypeError,
            ):
                print(
                    "工具结果："
                    f"[{call_id}] "
                    "无法解析"
                )
                continue

            if payload.get("success"):
                print(
                    "工具结果："
                    f"[{call_id}] 成功"
                )
            else:
                error = payload.get(
                    "error",
                    {},
                )

                if isinstance(
                    error,
                    Mapping,
                ):
                    error_code = error.get(
                        "code",
                        "unknown_error",
                    )
                else:
                    error_code = (
                        "unknown_error"
                    )

                print(
                    "工具结果："
                    f"[{call_id}] "
                    f"失败（{error_code}）"
                )

    if not found_tool_activity:
        print(
            "模型未调用工具，直接回答。"
        )


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """运行一次科研Agent问答。"""

    parser = create_argument_parser()
    options = parser.parse_args(
        arguments
    )

    question = options.question.strip()

    if not question:
        print(
            "错误：问题不能为空。",
            file=sys.stderr,
        )
        return 1

    if options.max_model_rounds <= 0:
        print(
            "错误：最大模型轮数必须大于0。",
            file=sys.stderr,
        )
        return 1

    if options.max_tool_calls <= 0:
        print(
            "错误：最大工具调用次数必须大于0。",
            file=sys.stderr,
        )
        return 1

    database_path = Path(
        options.database
    ).resolve()

    if not database_path.is_file():
        print(
            "错误：找不到论文数据库："
            f"{database_path}",
            file=sys.stderr,
        )
        return 1

    try:
        runner = create_paper_agent_runner(
            database_path=database_path,
            vector_cache_directory=(
                options
                .vector_cache_directory
            ),
            embedding_model_name=(
                options.embedding_model
            ),
            embedding_device=(
                options.device
            ),
            max_model_rounds=(
                options.max_model_rounds
            ),
            max_tool_calls=(
                options.max_tool_calls
            ),
        )

        print(
            "科研Agent正在分析问题……"
        )

        result = runner.run(
            question
        )

    except ModelServiceError as exc:
        print(
            f"模型服务错误：{exc}",
            file=sys.stderr,
        )
        return 1

    except AgentRunLimitError as exc:
        print(
            f"Agent达到运行限制：{exc}",
            file=sys.stderr,
        )
        return 1

    except AgentProtocolError as exc:
        print(
            f"Agent协议错误：{exc}",
            file=sys.stderr,
        )
        return 1

    except Exception as exc:
        print(
            "Agent运行失败："
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )
        return 1

    print()
    print("=== 科研Agent回答 ===")
    print(result.answer)

    print()
    print(
        "运行统计："
        f"模型轮数={result.model_rounds}，"
        f"工具调用数="
        f"{result.tool_call_count}"
    )

    if options.show_trace:
        print_agent_trace(
            result.messages
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
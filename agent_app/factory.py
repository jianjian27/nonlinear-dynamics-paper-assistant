from pathlib import Path

from agent_runtime.model_protocol import (
    AgentChatModel,
)
from agent_runtime.runner import (
    DEFAULT_AGENT_SYSTEM_PROMPT,
    AgentRunner,
)
from agent_runtime.tool_registry import (
    ToolRegistry,
)
from agent_tools import (
    GetPaperOverviewTool,
    ListPapersTool,
    SearchPaperEvidenceTool,
)
from embedding.sentence_transformer_encoder import (
    DEFAULT_MODEL_NAME,
)
from models.aliyun_agent_model import (
    AliyunAgentChatModel,
)


def create_paper_tool_registry(
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
) -> ToolRegistry:
    """创建包含全部论文只读工具的注册表。"""

    tools = [
        ListPapersTool(
            database_path=database_path,
        ),
        GetPaperOverviewTool(
            database_path=database_path,
        ),
        SearchPaperEvidenceTool(
            database_path=database_path,
            vector_cache_directory=vector_cache_directory,
            embedding_model=embedding_model_name,
            device=embedding_device,
        ),
    ]

    return ToolRegistry(tools)


def create_paper_agent_runner(
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
    chat_model: AgentChatModel | None = None,
    system_prompt: str = (
        DEFAULT_AGENT_SYSTEM_PROMPT
    ),
    max_model_rounds: int = 6,
    max_tool_calls: int = 12,
) -> AgentRunner:
    """组装可运行的科研论文Agent。"""

    registry = create_paper_tool_registry(
        database_path=database_path,
        vector_cache_directory=vector_cache_directory,
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
    )

    resolved_chat_model = (
        chat_model
        if chat_model is not None
        else AliyunAgentChatModel()
    )

    return AgentRunner(
        model=resolved_chat_model,
        registry=registry,
        system_prompt=system_prompt,
        max_model_rounds=max_model_rounds,
        max_tool_calls=max_tool_calls,
    )
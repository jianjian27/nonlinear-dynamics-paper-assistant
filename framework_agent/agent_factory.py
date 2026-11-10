# model_factory.py       创建模型
# tool_adapters.py       转换工具
# agent_factory.py       组装Agent

from pathlib import Path

from langchain.agents import (
    create_agent,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langgraph.graph.state import (
    CompiledStateGraph,
)

from agent_app.factory import (
    create_paper_tool_registry,
)
from agent_runtime.runner import (
    DEFAULT_AGENT_SYSTEM_PROMPT,
)
from embedding.sentence_transformer_encoder import (
    DEFAULT_MODEL_NAME,
)
from framework_agent.model_factory import (
    create_aliyun_langchain_model,
)
from framework_agent.tool_adapters import (
    adapt_agent_tools,
)


def create_langchain_paper_agent(
    *,
    database_path: str | Path =(
        "data/papers.db"
    ),
    vector_cache_directory: str | Path =(
        "data/vector_indexes"
    ),
    embedding_model_name: str =(
        DEFAULT_MODEL_NAME
    ),
    embedding_device: str | None = None,
    chat_model: BaseChatModel | None = None, # 依赖注入：由外部决定具体使用哪个实现。
    system_prompt: str =(
        DEFAULT_AGENT_SYSTEM_PROMPT
    ),
    debug: bool = False,
) -> CompiledStateGraph:
    """创建基于LangChain的科研论文Agent。"""

    cleaned_system_prompt = (
        system_prompt.strip()
    )

    if not cleaned_system_prompt:
        raise ValueError(
            "Agent系统提示词不能为空。"
        )

    registry = create_paper_tool_registry(
        database_path=database_path,
        vector_cache_directory=vector_cache_directory,
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
    )

    existing_tools = tuple(
        registry.require(tool_name)
        for tool_name in registry.names
    )

    langchain_tools = adapt_agent_tools(
        existing_tools
    )

    resolved_model = (
        chat_model
        if chat_model is not None
        else create_aliyun_langchain_model()
    )

    return create_agent(
        model=resolved_model,
        tools=langchain_tools,
        system_prompt=cleaned_system_prompt,
        debug=debug,
        name="paper_research_agent",
    )
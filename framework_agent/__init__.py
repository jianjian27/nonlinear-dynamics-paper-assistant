"""基于LangChain和LangGraph的科研Agent。"""

from framework_agent.agent_factory import (
    create_langchain_paper_agent,
)
from framework_agent.model_factory import (
    LangChainModelConfigurationError,
    create_aliyun_langchain_model,
)
from framework_agent.tool_adapters import (
    adapt_agent_tool,
    adapt_agent_tools,
)


__all__ = [
    "adapt_agent_tool",
    "adapt_agent_tools",
    "LangChainModelConfigurationError",
    "create_aliyun_langchain_model",
    "create_langchain_paper_agent",
]


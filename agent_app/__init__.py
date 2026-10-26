"""科研论文Agent的应用组装层。"""

from agent_app.factory import (
    create_paper_agent_runner,
    create_paper_tool_registry,
)


__all__ = [
    "create_paper_agent_runner",
    "create_paper_tool_registry",
]
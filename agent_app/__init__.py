"""科研论文Agent的应用组装层。"""

from agent_app.factory import (
    create_paper_agent_runner,
    create_paper_tool_registry,
)
from agent_app.gradio_service import (
    GradioAgentResult,
    GradioAgentService,
)


__all__ = [
    "GradioAgentResult",
    "GradioAgentService",
    "create_paper_agent_runner",
    "create_paper_tool_registry",
]
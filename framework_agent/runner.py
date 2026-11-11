from langchain_core.messages import AIMessage, convert_to_openai_messages
from langgraph.graph.state import CompiledStateGraph

from agent_runtime import AgentProtocolError, AgentRunResult


class LangChainPaperAgentRunner:
    """执行一次 LangChain Agent 任务并转换结果。"""

    def __init__(self, agent: CompiledStateGraph) -> None:
        self.agent = agent

    def run(self, question: str) -> AgentRunResult:
        question = question.strip()
        if not question:
            raise ValueError("用户问题不能为空。")

        state = self.agent.invoke({
            "messages": [{"role": "user", "content": question}],
        })

        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            raise AgentProtocolError("Agent 没有返回最终模型消息。")

        final_message = messages[-1]
        if final_message.tool_calls or final_message.invalid_tool_calls:
            raise AgentProtocolError("Agent 在工具请求阶段结束。")

        if final_message.response_metadata.get("finish_reason") == "length":
            raise AgentProtocolError("模型回答达到长度上限。")

        answer = final_message.text.strip()
        if not answer:
            raise AgentProtocolError("Agent 的最终回答为空。")

        model_messages = [
            message for message in messages if isinstance(message, AIMessage)
        ]
        if any(message.invalid_tool_calls for message in model_messages):
            raise AgentProtocolError("模型返回了无法解析的工具请求。")

        return AgentRunResult(
            answer=answer,
            messages=tuple(convert_to_openai_messages(messages)),
            model_rounds=len(model_messages),
            tool_call_count=sum(
                len(message.tool_calls) for message in model_messages
            ),
        )
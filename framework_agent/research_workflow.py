# 决定节点执行顺序和分支

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from framework_agent.research_evidence import ResearchEvidenceNode
from framework_agent.research_nodes import ResearchScopeNodes
from framework_agent.research_state import ResearchState


def route_after_confirmation(
    state: ResearchState,
) -> Literal["collect_evidence", "finish"]: # Literal 类型注解，表示函数预期只返回这两个字符串
    """根据确认结果选择下一步。"""

    if state.get("approved", False):
        return "collect_evidence"

    return "finish"


def create_research_workflow(
    scope_nodes: ResearchScopeNodes,
    evidence_node: ResearchEvidenceNode,
    *,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """组装多论文研究工作流。"""

    builder = StateGraph(ResearchState)

    builder.add_node(
        "load_candidates",
        scope_nodes.load_candidates,
    )
    builder.add_node(
        "confirm_scope",
        scope_nodes.confirm_scope,
    )
    builder.add_node(
        "collect_evidence",
        evidence_node.collect,
    )

    builder.add_edge(
        START,
        "load_candidates",
    )
    builder.add_edge(
        "load_candidates",
        "confirm_scope",
    )

    # LangGraph 的普通边连接固定的执行顺序
    # 条件边负责根据函数返回值选择后续节点

    builder.add_conditional_edges(
        "confirm_scope",
        route_after_confirmation, # confirm_scope 节点完成后，运行路由函数
        {
            "collect_evidence": "collect_evidence",
            "finish": END, # 路由函数读取更新后的状态，返回一个字符串，根据字典映射决定下一节点
        },
    )

    builder.add_edge(
        "collect_evidence",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer, # 表示由调用方传入检查点保存器。测试可以传 InMemorySaver()，后续命令行入口可以传 SQLite 保存器。
    )
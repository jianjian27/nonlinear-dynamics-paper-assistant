# LangGraph 开发流程复习笔记

本文结合当前论文助手中的多论文研究工作流，记录从定义到运行的开发过程。各代码块有独立用途，请不要把整份笔记直接当成一个 Python 脚本运行。第 4 节演示组装函数，第 6 节演示创建依赖、启动和恢复任务。

## 1. 总体开发顺序

设计任务流程 → 定义 State → 编写节点 → 连接节点 → 编译工作流 → 提交任务 → 检查结果。

当前流程：

```text
START
  ↓
读取候选论文（load_candidates）
  ↓
确认范围（confirm_scope）
  ├─ 论文不足：直接结束
  └─ 论文足够：interrupt 暂停，等待用户选择
       ├─ 确认：逐篇检索证据（collect_evidence）→ END
       └─ 取消：END
```

当前流程结束时得到的是证据结果。生成多论文对比回答是后续扩展步骤。

## 2. 先定义状态：流程需要记住什么

项目文件：research_state.py。

ResearchState 约定研究问题、候选论文、选定论文、确认结果和证据等字段。

```python
from typing import TypedDict


class ResearchState(TypedDict, total=False):
    question: str
    selected_paper_ids: list[str]
    approved: bool
    # 其他字段见实际源码
```

- State 定义数据结构，不会自行查询数据或执行数据传递。
- TypedDict 主要用于类型提示，不提供 Pydantic 那样的运行时输入验证。
- total=False 表示各字段可以暂时缺省，因此读取前需要考虑字段是否已经产生。

## 3. 再编写节点：每一步怎么处理状态

项目文件：research_nodes.py、research_evidence.py。

节点通常读取当前状态，执行工作，返回需要更新的字段：

```python
def normalize_question(state: dict) -> dict:
    question = state["question"].strip()
    return {"question": question}
```

LangGraph 负责将节点返回的更新合并进状态。当前状态没有配置 reducer，因此同名字段使用新值覆盖，未返回的字段保持原值。

节点可以是普通函数，也可以是对象的方法。项目使用类来保存工具对象、每篇证据数量等依赖和配置；LangGraph 并不要求节点必须写成类。

当前节点职责：

- ResearchScopeNodes.load_candidates：调用论文列表工具，准备候选论文。
- ResearchScopeNodes.confirm_scope：暂停等待选择，验证确认结果。
- ResearchEvidenceNode.collect：按论文分别检索，记录证据或错误。

## 4. 组装：登记节点，连接执行顺序

项目文件：research_workflow.py。

```python
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from framework_agent.research_evidence import ResearchEvidenceNode
from framework_agent.research_nodes import ResearchScopeNodes
from framework_agent.research_state import ResearchState
from framework_agent.research_workflow import route_after_confirmation


def assemble_example(
    scope_nodes: ResearchScopeNodes,
    evidence_node: ResearchEvidenceNode,
    *,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    builder = StateGraph(ResearchState)

    builder.add_node("load_candidates", scope_nodes.load_candidates)
    builder.add_node("confirm_scope", scope_nodes.confirm_scope)
    builder.add_node("collect_evidence", evidence_node.collect)

    builder.add_edge(START, "load_candidates")
    builder.add_edge("load_candidates", "confirm_scope")

    builder.add_conditional_edges(
        "confirm_scope",
        route_after_confirmation,
        {
            "collect_evidence": "collect_evidence",
            "finish": END,
        },
    )

    builder.add_edge("collect_evidence", END)

    return builder.compile(checkpointer=checkpointer)
```

- add_node：为节点命名，并登记要执行的函数或方法。
- 方法后面没有括号，因为这里传递方法本身，供框架稍后调用。
- add_edge：连接固定的下一步。
- add_conditional_edges：调用路由函数，根据返回值选择下一步。
- finish 是分支名称，通过字典映射到 END，不是另一个业务节点。
- START 和 END 是框架提供的入口、结束标记。

这个阶段只是在设计流程，还没有开始查询论文。

## 5. 编译：生成可执行工作流

上一节函数最后的 `return builder.compile(checkpointer=checkpointer)` 就是编译步骤。这里的两个变量分别来自函数内部创建的 builder 和调用方传入的 checkpointer。

builder 用来搭建流程；workflow 用来执行流程。compile 会检查图结构并绑定检查点等运行组件，不会开始执行研究任务。

保存器由外部传入：测试可以使用 InMemorySaver，后续持久化入口可以使用 SqliteSaver。

## 6. 运行：用 invoke 提交任务

下面是一个完整的调用示例，使用真实工具。它要求数据库中至少有两篇含 Chunk 的论文；恢复后会实际加载 Embedding 模型并检索。请从项目根目录运行，需要时将 device 调整为本机可用设备。这里只演示调用方式，无需为了复习反复运行。

```python
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_tools.paper_tools import ListPapersTool
from agent_tools.retrieval_tools import SearchPaperEvidenceTool
from framework_agent.research_evidence import ResearchEvidenceNode
from framework_agent.research_nodes import ResearchScopeNodes
from framework_agent.research_workflow import create_research_workflow


def main() -> None:
    list_tool = ListPapersTool(database_path="data/papers.db")
    search_tool = SearchPaperEvidenceTool(
        database_path="data/papers.db",
        device="cpu",
    )

    scope_nodes = ResearchScopeNodes(list_tool)
    evidence_node = ResearchEvidenceNode(search_tool)

    workflow = create_research_workflow(
        scope_nodes,
        evidence_node,
        checkpointer=InMemorySaver(),
    )

    config = {
        "configurable": {
            "thread_id": str(uuid4()),
        },
    }

    result = workflow.invoke(
        {"question": "比较两篇论文的分析方法"},
        config=config,
    )

    pending = result.get("__interrupt__", ())
    if not pending:
        print(result.get("message", "流程已结束。"))
        return

    # 只显示候选列表，不自动替用户选择。
    candidates = pending[0].value["candidate_papers"]
    for paper in candidates:
        print(paper["paper_id"], paper["title"])

    raw_ids = input("输入至少两个论文ID，用英文逗号分隔；留空取消：")
    selected_ids = [
        paper_id.strip()
        for paper_id in raw_ids.split(",")
        if paper_id.strip()
    ]

    result = workflow.invoke(
        Command(resume={
            "approved": bool(selected_ids),
            "selected_paper_ids": selected_ids,
        }),
        config=config,
    )

    print(result)


if __name__ == "__main__":
    main()

```

实际执行过程：接收输入 → 找到入口节点 → 把状态传给节点 → 合并节点更新 → 根据边确定下一步 → 继续执行，直到结束或中断。

ResearchState 保存业务数据；config 保存 thread_id 等运行配置。thread_id 标识检查点所属的任务线程，不是论文 ID，也不是操作系统线程。

## 7. 暂停与恢复

项目中的 confirm_scope 方法调用 `interrupt(...)` 暂停工作流；第 6 节的完整示例使用 `Command(resume=...)` 提交用户选择，并继续传入同一份 config 恢复任务。

这两处代码分别属于节点内部和工作流调用方，不能作为孤立的 Python 语句直接运行。

- interrupt 的载荷交给调用方展示；默认 invoke 返回结果中的 `__interrupt__` 可用于读取中断。
- 恢复值成为 interrupt() 的返回值。
- 恢复时，中断所在节点从头重新执行，interrupt 之前的代码会再次运行。
- 因此，数据库写入、外部请求等操作放在中断前时，需要考虑重复执行。
- 检查点保存状态和执行进度，不会自动保存所有 Python 对象、Embedding 模型和内存索引。
- InMemorySaver 只保存在当前进程内存中；SqliteSaver 可以将检查点持久化到文件。
- 论文数据库保存知识库内容，检查点数据库保存任务进度，两者用途不同。

## 8. 测试入口与正式入口的关系

测试不是唯一的工作流入口。测试、命令行和网页都可以复用 create_research_workflow。

```text
准备工具 → 创建节点对象 → 组装并编译 → invoke 启动 → 必要时恢复
```

区别在于依赖：

- 测试传入 Mock 工具，设置模拟返回值，检查调用过程。
- 正式入口传入真实工具，实际查询 SQLite 或运行检索。
- 节点测试验证单步逻辑；工作流测试验证节点之间的执行顺序、分支和恢复行为。
- Mock 测试通过不等于真实检索质量已经合格，还需要真实论文验收。

## 9. 如何判断项目得到改善

当前工作流应满足：

- 等待确认时不执行检索。
- 确认后只检索选中的论文，并逐篇收集证据。
- 取消或候选论文不足时结束。
- 恢复确认节点时不重新执行候选查询。
- 一篇论文检索失败时，记录错误并继续处理其他论文。

后续生成回答后，还需要检查回答是否有证据支持、引用是否正确，以及证据不足时是否明确说明。

## 10. 复习时定位代码

| 文件 | 关注的问题 |
| --- | --- |
| research_state.py | 流程需要传递哪些数据？ |
| research_nodes.py | 怎样查询论文、等待并验证用户确认？ |
| research_evidence.py | 怎样按论文检索并整理结果？ |
| research_workflow.py | 节点怎样连接，分支怎样选择？ |
| ../tests/test_research_workflow.py | 怎样注入模拟工具，启动、恢复并验证流程？ |

记住三个时刻：创建对象是在准备依赖；compile 是生成可执行流程；invoke 才是真正启动任务。

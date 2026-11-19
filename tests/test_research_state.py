import unittest

from langgraph.graph import END, START, StateGraph

from framework_agent.research_state import ResearchState


class TestResearchState(unittest.TestCase):
    def test_preserves_selected_scope_and_empty_evidence(self):
        def collect_evidence(state: ResearchState) -> ResearchState:
            return {
                "evidence_results": [
                    {
                        "paper_id": paper_id,
                        "evidences": [],
                        "error": None,
                    }
                    for paper_id in state["selected_paper_ids"]
                ]
            }

        # 把 collect_evidence 函数组装成一个可以运行的流程。
        builder = StateGraph(ResearchState) # 创建工作流的“设计图”，这个流程里的节点通过ResearchState所描述的状态传递数据。
        builder.add_node("collect_evidence", collect_evidence) # 添加一个执行任务的节点，节点名称为 "collect_evidence"，执行的函数为 collect_evidence。
        builder.add_edge(START, "collect_evidence") # 设置入口，工作流开始后，首先执行名为 "collect_evidence" 的节点。
        builder.add_edge("collect_evidence", END) # 设置结束位置，collect_evidence 执行完成、状态更新后，结束本次工作流。
        workflow = builder.compile() # 把设计图编译成可执行工作流

        result = workflow.invoke({
            "question": "比较两篇论文的动力学分析方法",
            "selected_paper_ids": ["paper-001", "paper-002"],
            "approved": True,
        })

        self.assertTrue(result["approved"])
        self.assertEqual(
            [item["paper_id"] for item in result["evidence_results"]],
            ["paper-001", "paper-002"],
        )
        self.assertEqual(
            result["evidence_results"][1]["evidences"],
            [],
        )
        self.assertIsNone(result["evidence_results"][1]["error"])


if __name__ == "__main__":
    unittest.main()

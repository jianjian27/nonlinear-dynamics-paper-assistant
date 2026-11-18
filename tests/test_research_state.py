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

        builder = StateGraph(ResearchState)
        builder.add_node("collect_evidence", collect_evidence)
        builder.add_edge(START, "collect_evidence")
        builder.add_edge("collect_evidence", END)
        workflow = builder.compile()

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

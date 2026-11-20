from pydantic import BaseModel, ConfigDict, Field
from langgraph.types import interrupt

from agent_tools.paper_tools import ListPapersTool
from framework_agent.research_state import ResearchState


class ResearchScopeDecision(BaseModel):
    """用户提交的研究范围确认结果。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    approved: bool
    selected_paper_ids: list[str] = Field(
        default_factory=list,
    )


class ResearchScopeNodes:
    """查询候选论文，并等待用户确认研究范围。"""

    def __init__(self, list_papers_tool: ListPapersTool,) -> None:
        self.list_papers_tool = list_papers_tool

    def load_candidates(self, state: ResearchState,) -> ResearchState:
        question = state.get("question", "").strip()

        if not question:
            raise ValueError("研究问题不能为空。")

        result = self.list_papers_tool.invoke({})

        if not result["database_available"]:
            message = "论文数据库不存在，请先导入论文。"
            candidates = []
        else:
            candidates = [
                {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                }
                for paper in result["papers"]
                if paper["chunk_count"] > 0
            ]

            if len(candidates) < 2:
                message = "可用论文不足两篇， 请先导入更多论文。"
            else:
                message = "请选择至少两篇论文作为研究范围。"

        return {
            "question": question,
            "candidate_papers": candidates,
            "selected_paper_ids": [],
            "approved": False,
            "evidence_results": [],
            "answer": "",
            "message": message,
        }

    def confirm_scope(self, state: ResearchState) -> ResearchState:
        candidates = state["candidate_papers"]

        # 没有足够的论文时，直接结束范围确认。
        if len(candidates) < 2:
            return {
                "approved": False,
                "selected_paper_ids": [],
            }

        response = interrupt({
            "question": state["question"],
            "message": "请选择至少两篇论文作为研究范围,或取消本次研究。",
            "candidate_papers": candidates,
        })

        decision = ResearchScopeDecision.model_validate(
            response,
        )

        if not decision.approved:
            return {
                "approved": False,
                "selected_paper_ids": [],
                "message": "已取消本次研究。",
            }

        # 去重，并保留用户选择的先后顺序。
        selected_ids = list(
            dict.fromkeys(decision.selected_paper_ids)
        )

        available_ids = {
            paper["paper_id"]
            for paper in candidates
        }

        if any(
            paper_id not in available_ids
            for paper_id in selected_ids
        ):
            raise ValueError("选择中包含不在候选列表里的论文。")

        if len(selected_ids) < 2:
            raise ValueError("至少选择两篇不同的论文。")

        return {
            "approved": True,
            "selected_paper_ids": selected_ids,
            "message": f"已确认 {len(selected_ids)} 篇论文。",
        }

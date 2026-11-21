from agent_tools.retrieval_tools import (
    SearchPaperEvidenceOutput,
    SearchPaperEvidenceTool,
)
from framework_agent.research_state import (
    PaperEvidenceResult,
    ResearchState,
)


class ResearchEvidenceNode:
    """为每篇选定论文分别检索证据。"""

    def __init__(
        self,
        search_tool: SearchPaperEvidenceTool,
        *,
        top_k_per_paper: int = 3,
    ) -> None:
        if not 1 <= top_k_per_paper <= 10:
            raise ValueError("每篇论文的证据数量必须为1到10。")

        self.search_tool = search_tool
        self.top_k_per_paper = top_k_per_paper

    def collect(
        self,
        state: ResearchState,
    ) -> ResearchState:
        if not state.get("approved", False):
            raise ValueError("请先确认研究范围。")

        question = state.get("question", "").strip()

        if not question:
            raise ValueError("研究问题不能为空。")

        paper_ids  = list(
            dict.fromkeys(
                state.get("selected_paper_ids", []),
            )
        )

        if len(paper_ids) < 2:
            raise ValueError("至少需要两篇不同的论文。")

        results: list[PaperEvidenceResult] = []

        for paper_id in paper_ids:
            result = self._search_one_paper(
                question,
                paper_id,
            )
            results.append(result)

        failed_count = sum(
            item["error"] is not None
            for item in results
        )

        return {
            "evidence_results": results,
            "message": (
                f"已处理 {len(results)} 篇论文，"
                f"其中 {failed_count} 篇检索失败。"
            ),
        }

    def _search_one_paper(self, question: str, paper_id: str,) -> PaperEvidenceResult:
        try:
            raw_result = self.search_tool.invoke({
                "query": question,
                "paper_ids": [paper_id],
                "top_k": self.top_k_per_paper,
            })

            result = SearchPaperEvidenceOutput.model_validate(
                raw_result,
            )

            if not result.database_available:
                raise RuntimeError("论文数据库不可用。")

            if any(
                evidence.paper_id != paper_id
                for evidence in result.evidences
            ):
                raise ValueError("检索结果包含研究范围外的论文。")

            return {
                "paper_id": paper_id,
                "evidences": [
                    evidence.model_dump(mode="json")
                    for evidence in result.evidences
                ],
                "error": None,
            }

        except Exception as exc:
            return {
                "paper_id": paper_id,
                "evidences": [],
                "error": f"{type(exc).__name__}: {exc}",
            }


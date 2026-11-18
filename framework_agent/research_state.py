from typing import Any, TypedDict


class ResearchPaperChoice(TypedDict):
    """展示给用户选择的一篇论文。"""

    paper_id: str
    title: str


class PaperEvidenceResult(TypedDict):
    """一篇选定论文的检索结果。"""

    paper_id: str

    # 保存已有检索工具返回的证据字典。
    evidences: list[dict[str, Any]]

    # 检索正常完成时为 None，失败时记录错误。
    error: str | None


class ResearchState(TypedDict, total=False):
    """多论文研究工作流在节点之间传递的状态。"""

    question: str

    candidate_papers: list[ResearchPaperChoice]
    selected_paper_ids: list[str]
    approved: bool

    evidence_results: list[PaperEvidenceResult]

    answer: str
    message: str
    
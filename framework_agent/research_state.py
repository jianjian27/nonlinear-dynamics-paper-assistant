# 定义节点之间传递哪些数据

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


# 规定流程需要记录什么
class ResearchState(TypedDict, total=False): # total=False 表示创建状态字典时，不要求一次提供全部字段
    """多论文研究工作流在节点之间传递的状态。"""

    question: str # 用户的研究问题

    candidate_papers: list[ResearchPaperChoice] # 提供的候选论文列表，每项遵循 ResearchPaperChoice 结构
    selected_paper_ids: list[str] # 用户选择的论文ID列表
    approved: bool # 用户是否确认检索对应证据

    evidence_results: list[PaperEvidenceResult] # 各篇选定论文的检索结果，每项遵循 PaperEvidenceResult 结构

    answer: str # 最终研究回答
    message: str # 流程提示，例如没有可选论文、用户取消等
    
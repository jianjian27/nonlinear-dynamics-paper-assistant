from dataclasses import dataclass

from database.paper_repository import (
    PaperRepository,
)
from models.deepseek_api import (
    AliyunChatClient,
)
from rag.context_builder import (
    ContextBuilder,
    RagContext,
)
from rag.prompt import (
    RagPromptBuilder,
)
from retrieval.semantic_retriever import (
    SemanticRetriever,
)


INSUFFICIENT_INFORMATION_ANSWER = (
    "当前论文库中没有足够信息回答这个问题。"
)


@dataclass(frozen=True)
class RagAnswer:
    """一次RAG问答的完整结果。"""

    question: str
    answer: str
    context: RagContext

    @property
    def retrieved_count(self) -> int:
        """返回本次使用的证据数量。"""

        return len(
            self.context.evidences
        )


class RagAnswerService:
    """协调检索、上下文、Prompt和模型调用。"""

    def __init__(
        self,
        *,
        retriever: SemanticRetriever,
        paper_repository: PaperRepository,
        context_builder: ContextBuilder,
        prompt_builder: RagPromptBuilder,
        chat_client: AliyunChatClient,
    ) -> None:
        self.retriever = retriever

        self.paper_repository = (
            paper_repository
        )

        self.context_builder = (
            context_builder
        )

        self.prompt_builder = (
            prompt_builder
        )

        self.chat_client = chat_client

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> RagAnswer:
        """根据论文库证据回答问题。"""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "用户问题不能为空。"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k必须大于0。"
            )

        results = self.retriever.search(
            cleaned_question,
            top_k=top_k,
        )

        paper_titles = (
            self._load_paper_titles(results)
        )

        context = self.context_builder.build(
            results,
            paper_titles=paper_titles,
        )

        if context.is_empty:
            return RagAnswer(
                question=cleaned_question,
                answer=(
                    INSUFFICIENT_INFORMATION_ANSWER
                ),
                context=context,
            )

        prompt = self.prompt_builder.build(
            cleaned_question,
            context,
        )

        answer = self.chat_client.ask(
            prompt.messages
        )

        return RagAnswer(
            question=cleaned_question,
            answer=answer,
            context=context,
        )

    def _load_paper_titles(
        self,
        results,
    ) -> dict[str, str]:
        """查询检索结果涉及的论文标题。"""

        paper_titles: dict[str, str] = {}

        for result in results:
            paper_id = (
                result.chunk.paper_id
            )

            if paper_id in paper_titles:
                continue

            paper = (
                self.paper_repository
                .get_by_id(paper_id)
            )

            paper_titles[paper_id] = (
                paper.title
                if paper is not None
                else paper_id
            )

        return paper_titles
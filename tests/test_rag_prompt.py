import unittest

from rag.context_builder import (
    ContextEvidence,
    RagContext,
)
from rag.prompt import (
    RagPromptBuilder,
    SYSTEM_PROMPT,
)


def make_context() -> RagContext:
    evidence = ContextEvidence(
        citation_number=1,
        paper_id="paper-1",
        paper_title="混沌系统稳定性研究",
        chunk_id="chunk-1",
        section_path=(
            "稳定性分析",
            "李雅普诺夫方法",
        ),
        page_start=3,
        page_end=4,
        score=0.91,
        content=(
            "作者构造李雅普诺夫函数，"
            "证明同步误差渐近收敛。"
        ),
    )

    return RagContext(
        evidences=(evidence,)
    )


class TestRagPromptBuilder(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.builder = RagPromptBuilder()
        self.context = make_context()

    def test_builds_prompt_from_question_and_context(
        self,
    ) -> None:
        prompt = self.builder.build(
            "论文如何证明系统稳定？",
            self.context,
        )

        self.assertEqual(
            prompt.system_prompt,
            SYSTEM_PROMPT,
        )

        self.assertIn(
            "论文如何证明系统稳定？",
            prompt.user_prompt,
        )

        self.assertIn(
            "[证据1]",
            prompt.user_prompt,
        )

        self.assertIn(
            "混沌系统稳定性研究",
            prompt.user_prompt,
        )

        self.assertIn(
            "李雅普诺夫函数",
            prompt.user_prompt,
        )

    def test_creates_openai_compatible_messages(
        self,
    ) -> None:
        prompt = self.builder.build(
            "论文如何证明系统稳定？",
            self.context,
        )

        messages = prompt.messages

        self.assertEqual(
            len(messages),
            2,
        )

        self.assertEqual(
            messages[0]["role"],
            "system",
        )

        self.assertEqual(
            messages[0]["content"],
            SYSTEM_PROMPT,
        )

        self.assertEqual(
            messages[1]["role"],
            "user",
        )

        self.assertEqual(
            messages[1]["content"],
            prompt.user_prompt,
        )

    def test_system_prompt_contains_grounding_rules(
        self,
    ) -> None:
        prompt = self.builder.build(
            "论文结论是什么？",
            self.context,
        )

        self.assertIn(
            "只能根据用户提供的检索证据",
            prompt.system_prompt,
        )

        self.assertIn(
            "当前论文库中没有足够信息",
            prompt.system_prompt,
        )

        self.assertIn(
            "[证据1]",
            prompt.system_prompt,
        )

        self.assertIn(
            "不得编造",
            prompt.system_prompt,
        )

    def test_strips_question_whitespace(
        self,
    ) -> None:
        prompt = self.builder.build(
            "  论文结论是什么？  ",
            self.context,
        )

        self.assertIn(
            "## 用户问题\n\n论文结论是什么？",
            prompt.user_prompt,
        )

    def test_rejects_blank_question(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "问题不能为空",
        ):
            self.builder.build(
                "   ",
                self.context,
            )

    def test_rejects_empty_context(
        self,
    ) -> None:
        empty_context = RagContext(
            evidences=()
        )

        with self.assertRaisesRegex(
            ValueError,
            "上下文不能为空",
        ):
            self.builder.build(
                "论文结论是什么？",
                empty_context,
            )


if __name__ == "__main__":
    unittest.main()
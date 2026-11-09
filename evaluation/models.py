from typing import  Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class EvaluationModel(BaseModel):
    """评估数据模型的共同配置。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class AgentEvalCase(EvaluationModel):
    """一个可以重复执行的Agent评估案例。"""

    case_id: str = Field(
        min_length=1,
        pattern=(
            r"^[A-Za-z0-9]"
            r"[A-Za-z0-9._-]*$"
        ),
    )

    question: str = Field(min_length=1)

    expected_tools: tuple[
        str,
        ...
    ] = ()

    forbidden_tools: tuple[
        str,
        ...
    ] = ()

    expected_answer_keywords: tuple[
        str,
        ...
    ] = ()

    @field_validator(
        "expected_tools",
        "forbidden_tools",
        "expected_answer_keywords",
    )
    @classmethod
    def normalize_string_items(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """清理列表内容并拒绝空值和重复值。"""

        cleaned_values = tuple(
            value.strip()
            for value in values
        )

        if any(
            not value
            for value in cleaned_values
        ):
            raise ValueError(
                "列表中不能包含空字符串。"
            )

        if (
            len(set(cleaned_values))
            != len(cleaned_values)
        ):
            raise ValueError(
                "列表中不能包含重复值。"
            )

        return cleaned_values

    @model_validator(mode="after")
    def reject_tool_conflicts(self,) -> Self:
        """同一工具不能既要求调用又禁止调用。"""

        conflicts = (
            set(self.expected_tools)
            & set(self.forbidden_tools)
        )

        if conflicts:
            conflict_text = ", ".join(sorted(conflicts))

            raise ValueError(
                "工具不能同时出现在"
                "expected_tools和"
                "forbidden_tools中: "
                f"{conflict_text}"
            )

        return self



from collections.abc import (
    Callable,
    Sequence,
)
from time import perf_counter

from agent_runtime import (
    AgentRunResult,
)
from evaluation.models import (
    AgentCaseEvaluation,
    AgentEvalCase,
    EvaluationSummary,
)
from evaluation.trace_evaluator import (
    evaluate_tool_trace,
)


AgentCallable = Callable[
    [str],
    AgentRunResult,
]

def find_missing_answer_keywords(
    *,
    answer: str,
    expected_keywords: Sequence[str],
) -> tuple[str, ...]:
    """找出回答中没有出现的必需关键词。"""

    normalized_answer = answer.casefold()

    return tuple(
        keyword
        for keyword in expected_keywords
        if keyword.casefold() not in normalized_answer
    )


def evaluate_agent_case(
    case: AgentEvalCase,
    run_agent: AgentCallable,
) -> AgentCaseEvaluation:
    """运行并评估一个Agent案例。"""

    started_at = perf_counter()

    try:
        result = run_agent(
            case.question
        )

        trace_evaluation = (
            evaluate_tool_trace(
                case,
                result,
            )
        )

        missing_keywords = find_missing_answer_keywords(
            answer=result.answer,
            expected_keywords=case.expected_answer_keywords,
        )

        passed = (
            trace_evaluation.passed
            and not missing_keywords
        )

        return AgentCaseEvaluation(
            case_id=case.case_id,
            question=case.question,
            answer=result.answer,
            trace=trace_evaluation,
            missing_answer_keywords=(
                missing_keywords
            ),
            duration_seconds=(
                    perf_counter()
                    - started_at
            ),
            error=None,
            passed=passed,
        )

    except Exception as error:
        return AgentCaseEvaluation(
            case_id=case.case_id,
            question=case.question,
            answer=None,
            trace=None,
            duration_seconds=(
                    perf_counter()
                    - started_at
            ),
            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
            passed=False,
        )


def run_evaluation_cases(
    cases: Sequence[AgentEvalCase],
    run_agent: AgentCallable,
) -> tuple[AgentCaseEvaluation, ...]:
    """依次执行一组Agent评估案例。"""

    return tuple(
        evaluate_agent_case(
            case,
            run_agent,
        )
        for case in cases
    )


def summarize_evaluations(
    evaluations: Sequence[AgentCaseEvaluation],
) -> EvaluationSummary:
    """汇总一组案例评估结果。"""

    total_cases = len(evaluations)

    passed_cases = sum(
        evaluation.passed
        for evaluation in evaluations
    )

    failed_cases = (
        total_cases - passed_cases
    )

    total_duration = sum(
        evaluation.duration_seconds
        for evaluation in evaluations
    )

    total_model_rounds = sum(
        evaluation.trace.model_rounds
        for evaluation in evaluations
        if evaluation.trace is not None
    )

    total_tool_calls = sum(
        evaluation.trace.tool_call_count
        for evaluation in evaluations
        if evaluation.trace is not None
    )

    failed_case_ids = tuple(
        evaluation.case_id
        for evaluation in evaluations
        if not evaluation.passed
    )

    pass_rate = (
        passed_cases / total_cases
        if total_cases
        else 0.0
    )

    average_duration = (
        total_duration / total_cases
        if total_cases
        else 0.0
    )

    return EvaluationSummary(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=pass_rate,
        average_duration_seconds=(
            average_duration
        ),
        total_model_rounds=(
            total_model_rounds
        ),
        total_tool_calls=(
            total_tool_calls
        ),
        failed_case_ids=failed_case_ids,
    )


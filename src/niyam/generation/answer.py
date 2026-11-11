from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from niyam.generation.llm import Effort, LLMClient, LLMError
from niyam.generation.pricing import Usage
from niyam.generation.prompts import (
    ANSWER_SYSTEM,
    GroundedAnswer,
    answer_user_message,
    passage_label,
    retry_user_message,
)
from niyam.models import AnswerStatus, RetrievedChunk

_MARKER_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
_LABEL_RE = re.compile(r"S\d+")


def cited_labels(answer: str) -> list[str]:
    """Passage labels referenced by ``[S#]`` markers, in order of first appearance."""
    seen: dict[str, None] = {}
    for group in _MARKER_RE.findall(answer):
        for label in _LABEL_RE.findall(group):
            seen.setdefault(label, None)
    return list(seen)


@dataclass(slots=True)
class CitationCheck:
    valid: bool
    problems: list[str]
    labels: list[str]


def check_citations(result: GroundedAnswer, allowed: list[str]) -> CitationCheck:
    allowed_set = set(allowed)
    in_text = cited_labels(result.answer)
    declared = [c.strip() for c in result.citations]
    labels = list(dict.fromkeys([*in_text, *declared]))
    problems: list[str] = []

    if not result.answerable:
        return CitationCheck(valid=True, problems=[], labels=[])
    if not result.answer.strip():
        problems.append("answerable was true but the answer is empty")
    if not in_text:
        problems.append("the answer contains no [S#] citation markers")
    unknown = [label for label in labels if label not in allowed_set]
    if unknown:
        problems.append(f"cited passages that were not provided: {', '.join(unknown)}")
    undeclared = [label for label in in_text if label not in declared]
    if undeclared and not unknown:
        problems.append(f"markers missing from the citations list: {', '.join(undeclared)}")
    return CitationCheck(valid=not problems, problems=problems, labels=in_text)


@dataclass(slots=True)
class GenerationOutcome:
    status: AnswerStatus
    answer: str
    cited: list[RetrievedChunk]
    labels: list[str]
    attempts: int
    usage: Usage
    latency_ms: float
    problems: list[str] = field(default_factory=list)


async def generate_answer(
    llm: LLMClient,
    question: str,
    chunks: list[RetrievedChunk],
    *,
    as_of: date,
    effort: Effort = "medium",
    max_tokens: int = 4000,
    max_attempts: int = 2,
) -> GenerationOutcome:
    labels = [passage_label(i) for i in range(len(chunks))]
    by_label = dict(zip(labels, chunks, strict=True))
    base_message = answer_user_message(question, chunks, as_of)
    message = base_message
    usage = Usage(provider=llm.provider, model=llm.model)
    latency = 0.0
    problems: list[str] = []

    for attempt in range(1, max_attempts + 1):
        try:
            result = await llm.structured(
                system=ANSWER_SYSTEM,
                user=message,
                schema=GroundedAnswer,
                effort=effort,
                max_tokens=max_tokens,
            )
        except LLMError as exc:
            problems = [str(exc)]
            break
        usage.add(result.usage)
        latency += result.latency_ms

        if result.parsed is None:
            problems = [
                "the model declined the request"
                if result.refusal
                else f"no structured output (stop_reason={result.stop_reason})"
            ]
            if result.refusal:
                break
            message = retry_user_message(base_message, problems, labels)
            continue

        check = check_citations(result.parsed, labels)
        if check.valid:
            if not result.parsed.answerable:
                return GenerationOutcome(
                    AnswerStatus.NOT_FOUND, "", [], [], attempt, usage, latency
                )
            return GenerationOutcome(
                status=AnswerStatus.ANSWERED,
                answer=result.parsed.answer.strip(),
                cited=[by_label[label] for label in check.labels],
                labels=check.labels,
                attempts=attempt,
                usage=usage,
                latency_ms=latency,
            )
        problems = check.problems
        message = retry_user_message(base_message, problems, labels)

    return GenerationOutcome(
        status=AnswerStatus.FAILED_VALIDATION,
        answer="",
        cited=[],
        labels=[],
        attempts=max_attempts,
        usage=usage,
        latency_ms=latency,
        problems=problems,
    )

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Literal

from pydantic import BaseModel, Field

from niyam.generation.llm import LLMClient, LLMError
from niyam.generation.pricing import Usage

JUDGE_SYSTEM = """\
You grade answers produced by a question-answering system over Reserve Bank of India regulations.

You receive the question, a reference answer written by a human, the system's answer, and the \
passages the system cited (with their ids).

Grade three things:

1. claims: split the system answer into atomic factual claims. For each claim, decide whether the \
cited passages fully support it ("supported"), support it only in part ("partial"), or do not \
support it ("unsupported"). Judge support only from the cited passages, never from your own \
knowledge.

2. correctness: compare the system answer with the reference answer.
   - "correct": states the same key facts (numbers, periods, conditions) as the reference, with no \
contradiction. Extra correct detail is fine.
   - "partial": gets some key facts right but misses or garbles at least one.
   - "incorrect": contradicts the reference or misses its main point.

3. passages: for each cited passage id, does it support at least one claim in the answer?

Be strict about numbers, time periods and conditions. Paraphrase is fine."""


class ClaimVerdict(BaseModel):
    claim: str
    verdict: Literal["supported", "partial", "unsupported"]


class PassageVerdict(BaseModel):
    passage_id: str
    supports_answer: bool


class JudgeOutput(BaseModel):
    claims: list[ClaimVerdict] = Field(default_factory=list)
    correctness: Literal["correct", "partial", "incorrect"]
    passages: list[PassageVerdict] = Field(default_factory=list)
    rationale: str = Field(description="One or two sentences explaining the correctness grade.")


@dataclass(slots=True)
class Judgement:
    faithfulness: float
    correctness: float
    correctness_label: str
    citation_precision: float
    raw: JudgeOutput
    usage: Usage


_CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
_CLAIM_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}


def judge_message(
    question: str, reference: str, answer: str, passages: list[tuple[str, str]]
) -> str:
    rendered = "\n".join(
        f'<passage id="{escape(pid)}">\n{escape(text, quote=False)}\n</passage>'
        for pid, text in passages
    )
    return (
        f"<question>{escape(question, quote=False)}</question>\n"
        f"<reference_answer>{escape(reference, quote=False)}</reference_answer>\n"
        f"<system_answer>{escape(answer, quote=False)}</system_answer>\n"
        f"<cited_passages>\n{rendered}\n</cited_passages>"
    )


async def judge_answer(
    llm: LLMClient,
    *,
    question: str,
    reference: str,
    answer: str,
    passages: list[tuple[str, str]],
) -> Judgement | None:
    try:
        result = await llm.structured(
            system=JUDGE_SYSTEM,
            user=judge_message(question, reference, answer, passages),
            schema=JudgeOutput,
            effort="medium",
            max_tokens=4000,
        )
    except LLMError:
        return None
    if result.parsed is None:
        return None
    out = result.parsed
    claims = out.claims
    faithfulness = sum(_CLAIM_SCORE[c.verdict] for c in claims) / len(claims) if claims else 0.0
    cited = {pid for pid, _ in passages}
    verdicts = [p for p in out.passages if p.passage_id in cited]
    precision = sum(1 for p in verdicts if p.supports_answer) / len(verdicts) if verdicts else 0.0
    return Judgement(
        faithfulness=faithfulness,
        correctness=_CORRECTNESS_SCORE[out.correctness],
        correctness_label=out.correctness,
        citation_precision=precision,
        raw=out,
        usage=result.usage,
    )

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from niyam.generation.llm import LLMClient, LLMError
from niyam.generation.pricing import Usage
from niyam.generation.prompts import REWRITE_SYSTEM, RewrittenQuery, rewrite_user_message

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Rewrite:
    question: str
    as_of: date | None
    usage: Usage
    latency_ms: float
    applied: bool


def _parse_date(value: str | None, today: date) -> date | None:
    if not value:
        return None
    try:
        parsed = date.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed < today else None


async def rewrite_query(
    llm: LLMClient,
    question: str,
    history: list[tuple[str, str]],
    *,
    today: date | None = None,
) -> Rewrite:
    today = today or date.today()
    usage = Usage(provider=llm.provider, model=llm.model)
    try:
        result = await llm.structured(
            system=REWRITE_SYSTEM,
            user=rewrite_user_message(question, history[-6:], today),
            schema=RewrittenQuery,
            effort="low",
            max_tokens=1000,
        )
    except LLMError:
        log.warning("query rewrite failed; using the original question", exc_info=True)
        return Rewrite(question, None, usage, 0.0, applied=False)

    usage.add(result.usage)
    if result.parsed is None or not result.parsed.standalone_question.strip():
        return Rewrite(question, None, usage, result.latency_ms, applied=False)
    rewritten = result.parsed.standalone_question.strip()
    return Rewrite(
        question=rewritten,
        as_of=_parse_date(result.parsed.as_of, today),
        usage=usage,
        latency_ms=result.latency_ms,
        applied=rewritten != question.strip(),
    )

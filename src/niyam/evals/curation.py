from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from niyam.db import Database
from niyam.evals.metrics import cohen_kappa, coverage, normalize_tokens
from niyam.generation.llm import LLMClient

DRAFT_SYSTEM = """\
You help build an evaluation set for a question-answering system over Reserve Bank of India \
regulations. You receive one passage from an RBI Master Direction.

Write one question a compliance analyst at a bank or NBFC might realistically ask that this passage \
answers. Prefer questions about obligations, limits, time periods, eligibility, documents and \
exceptions. Do not mention "the passage" or "the document" in the question.

Return:
- question: the question, phrased naturally.
- category: "single_fact" if one sentence answers it, "numeric_table" if the answer is a number, \
limit, percentage or date or comes from a table, otherwise "multi_clause".
- quote: a verbatim span copied character-for-character from the passage (one to three sentences) \
that contains the answer.
- expected_answer: a short, correct answer in your own words.
If the passage is boilerplate (titles, contents, signatures), set usable to false."""

UNANSWERABLE_SYSTEM = """\
You help build an evaluation set for a question-answering system over Reserve Bank of India \
Master Directions. Write plausible questions a compliance analyst might ask that sound like they \
belong to the listed directions, but that those directions are unlikely to answer: for example \
rules from other regulators (SEBI, IRDAI), tax law, internal bank policy, or details beyond what \
RBI prescribes. Keep them specific and realistic. A human will verify each one."""


class DraftQuestion(BaseModel):
    usable: bool
    question: str = ""
    category: Literal["single_fact", "multi_clause", "numeric_table"] = "single_fact"
    quote: str = ""
    expected_answer: str = ""


class UnanswerableBatch(BaseModel):
    questions: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class Candidate:
    data: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.data, ensure_ascii=False)


async def sample_chunks(db: Database, chunk_set: str, n: int, seed: int) -> list[dict[str, Any]]:
    async with db.connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT c.id, c.text, c.heading_text, d.source_key, d.title
                FROM chunks c
                JOIN chunk_sets s ON s.id = c.chunk_set_id
                JOIN documents d ON d.id = c.document_id
                WHERE s.name = %s AND c.token_count >= 80 AND d.withdrawn_on IS NULL
                ORDER BY c.id
                """,
                (chunk_set,),
            )
        ).fetchall()
    rng = random.Random(seed)
    rng.shuffle(rows)
    return list(rows[:n])


def quote_is_verbatim(quote: str, text: str) -> bool:
    return coverage(normalize_tokens(quote), normalize_tokens(text)) >= 0.95


async def draft_answerable(
    llm: LLMClient, db: Database, *, chunk_set: str, n: int, seed: int
) -> list[Candidate]:
    out: list[Candidate] = []
    for index, row in enumerate(await sample_chunks(db, chunk_set, n, seed)):
        passage = f"Title: {row['title']}\nSection: {row['heading_text']}\n\n{row['text']}"
        result = await llm.structured(
            system=DRAFT_SYSTEM, user=passage, schema=DraftQuestion, effort="low", max_tokens=1500
        )
        draft = result.parsed
        if draft is None or not draft.usable or not quote_is_verbatim(draft.quote, row["text"]):
            continue
        out.append(
            Candidate(
                {
                    "id": f"draft-{seed}-{index:04d}",
                    "question": draft.question,
                    "category": draft.category,
                    "expected_answer": draft.expected_answer,
                    "evidence": [{"source_key": row["source_key"], "quote": draft.quote}],
                    "notes": f"drafted from {row['id']}; needs human review",
                    "status": "draft",
                }
            )
        )
    return out


async def draft_unanswerable(llm: LLMClient, db: Database, *, n: int, seed: int) -> list[Candidate]:
    async with db.connection() as conn:
        titles = [
            r["title"]
            for r in await (
                await conn.execute(
                    "SELECT title FROM documents WHERE withdrawn_on IS NULL ORDER BY random() LIMIT 25"
                )
            ).fetchall()
        ]
    result = await llm.structured(
        system=UNANSWERABLE_SYSTEM,
        user="Directions:\n- " + "\n- ".join(titles) + f"\n\nWrite {n} questions.",
        schema=UnanswerableBatch,
        effort="low",
        max_tokens=3000,
    )
    questions = result.parsed.questions if result.parsed else []
    return [
        Candidate(
            {
                "id": f"draft-u-{seed}-{i:04d}",
                "question": q,
                "category": "unanswerable",
                "notes": "drafted; verify the corpus does not answer this",
                "status": "draft",
            }
        )
        for i, q in enumerate(questions[:n])
    ]


def write_candidates(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(candidate.to_json() + "\n")


def calibration_report(pairs: list[tuple[str, str]], field_name: str) -> dict[str, Any]:
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    agreement = sum(h == j for h, j in pairs) / len(pairs)
    labels = sorted(set(human) | set(judge))
    confusion = {h: {j: 0 for j in labels} for h in labels}
    for h, j in pairs:
        confusion[h][j] += 1
    return {
        "field": field_name,
        "n": len(pairs),
        "agreement": round(agreement, 4),
        "cohen_kappa": round(cohen_kappa(human, judge), 4),
        "confusion": confusion,
    }


def render_calibration(reports: list[dict[str, Any]], judge_model: str) -> str:
    lines = [
        "# Judge calibration",
        "",
        f"Judge model: `{judge_model}`. Human labels: `evals/labels.jsonl`.",
        "",
        "| Metric | n | Agreement | Cohen's kappa |",
        "|---|---|---|---|",
    ]
    for report in reports:
        lines.append(
            f"| {report['field']} | {report['n']} | {report['agreement'] * 100:.1f}% | "
            f"{report['cohen_kappa']:.3f} |"
        )
    for report in reports:
        labels = list(report["confusion"])
        lines += ["", f"### {report['field']}: rows = human, columns = judge", ""]
        lines.append("| | " + " | ".join(labels) + " |")
        lines.append("|---" * (len(labels) + 1) + "|")
        for human_label in labels:
            row = report["confusion"][human_label]
            lines.append(f"| **{human_label}** | " + " | ".join(str(row[j]) for j in labels) + " |")
    return "\n".join(lines) + "\n"

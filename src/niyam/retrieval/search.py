from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from psycopg import sql

from niyam.db.pool import Conn
from niyam.models import RetrievedChunk

_SELECT_COLUMNS = sql.SQL(
    """
    c.id AS chunk_id, c.document_id, c.text, c.heading_path, c.clause_ref,
    c.page_from, c.page_to, c.anchor, c.token_count,
    d.title, d.source_key, d.source_url, d.pdf_url, d.circular_no, d.issued_on,
    d.withdrawn_on, d.last_amended_on,
    EXISTS (SELECT 1 FROM supersessions s WHERE s.old_document_id = d.id) AS superseded
    """
)


@dataclass(slots=True, frozen=True)
class TimeFilter:
    """Restrict retrieval to documents in force on ``as_of`` (today when omitted)."""

    as_of: date | None = None
    include_withdrawn: bool = False

    def effective_date(self) -> date:
        return self.as_of or date.today()

    def clause(self) -> tuple[sql.Composable, list[Any]]:
        parts: list[sql.Composable] = []
        params: list[Any] = []
        if not self.include_withdrawn:
            parts.append(sql.SQL("(d.withdrawn_on IS NULL OR d.withdrawn_on > %s)"))
            params.append(self.effective_date())
        if self.as_of is not None:
            parts.append(sql.SQL("(d.issued_on IS NULL OR d.issued_on <= %s)"))
            params.append(self.as_of)
        if not parts:
            return sql.SQL("TRUE"), params
        return sql.SQL(" AND ").join(parts), params


def _to_chunk(row: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        text=row["text"],
        heading_path=list(row["heading_path"] or []),
        clause_ref=row["clause_ref"],
        page_from=row["page_from"],
        page_to=row["page_to"],
        anchor=row["anchor"],
        title=row["title"],
        source_key=row["source_key"],
        source_url=row["source_url"],
        pdf_url=row["pdf_url"],
        circular_no=row["circular_no"],
        issued_on=row["issued_on"],
        withdrawn_on=row["withdrawn_on"],
        last_amended_on=row["last_amended_on"],
        superseded=bool(row["superseded"]),
        token_count=row["token_count"],
    )


async def dense_search(
    conn: Conn,
    *,
    query_vector: np.ndarray,
    model_id: int,
    chunk_set_id: int,
    dim: int,
    k: int,
    time_filter: TimeFilter,
    ef_search: int = 100,
) -> list[RetrievedChunk]:
    where, params = time_filter.clause()
    await conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(int(ef_search))))
    vector_cast = sql.SQL("::vector({})").format(sql.Literal(dim))
    query = sql.SQL(
        """
        SELECT * FROM (
            SELECT {columns}, 1 - (e.embedding{cast} <=> %s{cast}) AS score
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE e.model_id = %s AND e.chunk_set_id = %s AND {where}
            ORDER BY e.embedding{cast} <=> %s{cast}
            LIMIT %s
        ) ranked
        ORDER BY score DESC
        """
    ).format(columns=_SELECT_COLUMNS, cast=vector_cast, where=where)
    rows = await (
        await conn.execute(
            query,
            [query_vector, model_id, chunk_set_id, *params, query_vector, k],
        )
    ).fetchall()
    out: list[RetrievedChunk] = []
    for rank, row in enumerate(rows, start=1):
        chunk = _to_chunk(row)
        chunk.dense_rank = rank
        chunk.dense_score = float(row["score"])
        out.append(chunk)
    return out


async def lexical_search(
    conn: Conn,
    *,
    question: str,
    chunk_set_id: int,
    k: int,
    time_filter: TimeFilter,
) -> list[RetrievedChunk]:
    """Postgres full-text search with OR semantics, ranked by ``ts_rank_cd`` (cover density).

    This is not BM25: there is no IDF term saturation. See docs/decisions.md.
    """
    where, params = time_filter.clause()
    query = sql.SQL(
        """
        WITH q AS (
            SELECT NULLIF(replace(plainto_tsquery('english', %s)::text, ' & ', ' | '), '')::tsquery
                AS query
        )
        SELECT {columns}, ts_rank_cd(c.tsv, q.query, 32) AS score
        FROM q
        JOIN chunks c ON c.tsv @@ q.query
        JOIN documents d ON d.id = c.document_id
        WHERE c.chunk_set_id = %s AND {where}
        ORDER BY score DESC, c.id
        LIMIT %s
        """
    ).format(columns=_SELECT_COLUMNS, where=where)
    rows = await (await conn.execute(query, [question, chunk_set_id, *params, k])).fetchall()
    out: list[RetrievedChunk] = []
    for rank, row in enumerate(rows, start=1):
        chunk = _to_chunk(row)
        chunk.lexical_rank = rank
        chunk.lexical_score = float(row["score"])
        out.append(chunk)
    return out


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk], k: int = 60, weights: list[float] | None = None
) -> list[RetrievedChunk]:
    """Fuse ranked lists by sum of ``weight / (k + rank)``. Scores need no normalisation."""
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("one weight per ranked list")
    merged: dict[str, RetrievedChunk] = {}
    for weight, ranked in zip(weights, ranked_lists, strict=True):
        for rank, chunk in enumerate(ranked, start=1):
            current = merged.get(chunk.chunk_id)
            if current is None:
                current = chunk
                current.fused_score = 0.0
                merged[chunk.chunk_id] = current
            else:
                if chunk.dense_rank is not None:
                    current.dense_rank, current.dense_score = chunk.dense_rank, chunk.dense_score
                if chunk.lexical_rank is not None:
                    current.lexical_rank = chunk.lexical_rank
                    current.lexical_score = chunk.lexical_score
            current.fused_score += weight / (k + rank)
    return sorted(merged.values(), key=lambda c: (-c.fused_score, c.chunk_id))

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from psycopg.types.json import Jsonb

from niyam.db.pool import Conn
from niyam.models import Chunk, ParsedDocument


def chunk_ids(chunk_set: str, source_key: str, chunks: list[Chunk]) -> list[str]:
    """Content-addressed IDs: unchanged chunk text keeps its ID (and its embedding) across re-ingests."""
    seen: Counter[str] = Counter()
    ids: list[str] = []
    for chunk in chunks:
        basis = f"{chunk_set}|{source_key}|{chunk.heading_text}|{chunk.text}"
        occurrence = seen[basis]
        seen[basis] += 1
        digest = hashlib.blake2b(f"{basis}|{occurrence}".encode(), digest_size=10).hexdigest()
        ids.append(f"c_{digest}")
    return ids


async def get_or_create_chunk_set(
    conn: Conn, name: str, strategy: str, params: dict[str, Any]
) -> int:
    row = await (
        await conn.execute(
            """
            INSERT INTO chunk_sets (name, strategy, params) VALUES (%s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET params = EXCLUDED.params
            RETURNING id
            """,
            (name, strategy, Jsonb(params)),
        )
    ).fetchone()
    assert row is not None
    return int(row["id"])


async def get_or_create_embedding_model(conn: Conn, name: str, dim: int) -> int:
    row = await (
        await conn.execute(
            """
            INSERT INTO embedding_models (name, dim) VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE SET dim = embedding_models.dim
            RETURNING id, dim
            """,
            (name, dim),
        )
    ).fetchone()
    assert row is not None
    if int(row["dim"]) != dim:
        raise ValueError(f"embedding model {name} registered with dim {row['dim']}, got {dim}")
    return int(row["id"])


async def lookup_ids(conn: Conn, chunk_set: str, embedding_model: str) -> tuple[int, int, int]:
    row = await (
        await conn.execute(
            """
            SELECT cs.id AS chunk_set_id, em.id AS model_id, em.dim
            FROM chunk_sets cs, embedding_models em
            WHERE cs.name = %s AND em.name = %s
            """,
            (chunk_set, embedding_model),
        )
    ).fetchone()
    if row is None:
        raise LookupError(
            f"no index for chunk set {chunk_set!r} and embedding model {embedding_model!r}; "
            "run `niyam ingest` and `niyam embed` first"
        )
    return int(row["chunk_set_id"]), int(row["model_id"]), int(row["dim"])


@dataclass(slots=True)
class UpsertResult:
    document_id: int
    changed: bool


async def upsert_document(conn: Conn, doc: ParsedDocument) -> UpsertResult:
    meta = doc.meta
    existing = await (
        await conn.execute(
            "SELECT id, content_hash FROM documents WHERE source_key = %s", (meta.source_key,)
        )
    ).fetchone()
    params = (
        meta.source_url,
        meta.pdf_url,
        meta.title,
        meta.doc_type,
        meta.circular_no,
        meta.reference_nos,
        meta.issued_on,
        meta.last_amended_on,
        meta.amended_on,
        meta.withdrawn_on,
        meta.department,
        doc.page_count,
        doc.parser,
        doc.content_hash,
    )
    if existing is None:
        row = await (
            await conn.execute(
                """
                INSERT INTO documents (
                    source_key, source_url, pdf_url, title, doc_type, circular_no, reference_nos,
                    issued_on, last_amended_on, amended_on, withdrawn_on, department, page_count,
                    parser, content_hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (meta.source_key, *params),
            )
        ).fetchone()
        assert row is not None
        return UpsertResult(document_id=int(row["id"]), changed=True)

    await conn.execute(
        """
        UPDATE documents SET
            source_url = %s, pdf_url = %s, title = %s, doc_type = %s, circular_no = %s,
            reference_nos = %s, issued_on = %s, last_amended_on = %s, amended_on = %s,
            withdrawn_on = %s, department = %s, page_count = %s, parser = %s, content_hash = %s,
            fetched_at = now(), updated_at = now()
        WHERE id = %s
        """,
        (*params, existing["id"]),
    )
    return UpsertResult(
        document_id=int(existing["id"]), changed=existing["content_hash"] != doc.content_hash
    )


async def has_chunks(conn: Conn, document_id: int, chunk_set_id: int) -> bool:
    row = await (
        await conn.execute(
            "SELECT 1 FROM chunks WHERE document_id = %s AND chunk_set_id = %s LIMIT 1",
            (document_id, chunk_set_id),
        )
    ).fetchone()
    return row is not None


async def replace_chunks(
    conn: Conn,
    document_id: int,
    chunk_set_id: int,
    ids: list[str],
    chunks: list[Chunk],
) -> tuple[int, int]:
    """Replace a document's chunks, keeping rows (and embeddings) whose content ID is unchanged.

    Returns (inserted, removed).
    """
    rows = await (
        await conn.execute(
            "SELECT id FROM chunks WHERE document_id = %s AND chunk_set_id = %s",
            (document_id, chunk_set_id),
        )
    ).fetchall()
    existing = {r["id"] for r in rows}
    wanted = set(ids)
    stale = list(existing - wanted)
    if stale:
        await conn.execute("DELETE FROM chunks WHERE id = ANY(%s)", (stale,))

    inserted = 0
    async with conn.cursor() as cur:
        for chunk_id, chunk in zip(ids, chunks, strict=True):
            if chunk_id in existing:
                await cur.execute(
                    "UPDATE chunks SET ordinal = %s WHERE id = %s", (chunk.ordinal, chunk_id)
                )
                continue
            await cur.execute(
                """
                INSERT INTO chunks (
                    id, chunk_set_id, document_id, ordinal, heading_path, heading_text, clause_ref,
                    text, token_count, page_from, page_to, anchor, has_table
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chunk_id,
                    chunk_set_id,
                    document_id,
                    chunk.ordinal,
                    chunk.heading_path,
                    chunk.heading_text,
                    chunk.clause_ref,
                    chunk.text,
                    chunk.token_count,
                    chunk.page_from,
                    chunk.page_to,
                    chunk.anchor,
                    chunk.has_table,
                ),
            )
            inserted += 1
    return inserted, len(stale)


async def chunks_missing_embeddings(
    conn: Conn, chunk_set_id: int, model_id: int, limit: int
) -> list[dict[str, Any]]:
    rows = await (
        await conn.execute(
            """
            SELECT c.id, c.text, c.heading_text, d.title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model_id = %s
            WHERE c.chunk_set_id = %s AND e.chunk_id IS NULL
            ORDER BY c.document_id, c.ordinal
            LIMIT %s
            """,
            (model_id, chunk_set_id, limit),
        )
    ).fetchall()
    return list(rows)


async def insert_embeddings(
    conn: Conn,
    model_id: int,
    chunk_set_id: int,
    items: list[tuple[str, np.ndarray]],
) -> None:
    async with conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO embeddings (chunk_id, model_id, chunk_set_id, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (model_id, chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            [(chunk_id, model_id, chunk_set_id, vector) for chunk_id, vector in items],
        )


async def detect_supersessions(conn: Conn, chunk_set_id: int) -> int:
    """Link documents whose reference numbers are cited in another document's repeal provisions."""
    await conn.execute("DELETE FROM supersessions")
    result = await conn.execute(
        """
        INSERT INTO supersessions (old_document_id, new_document_id, evidence_chunk_id)
        SELECT DISTINCT ON (old.id, newer.id) old.id, newer.id, c.id
        FROM documents old
        JOIN chunks c ON c.chunk_set_id = %s AND c.document_id <> old.id
        JOIN documents newer ON newer.id = c.document_id
        WHERE cardinality(old.reference_nos) > 0
          AND (c.heading_text ILIKE '%%repeal%%' OR c.text ILIKE '%%repeal%%'
               OR c.text ILIKE '%%supersed%%')
          AND (newer.issued_on IS NULL OR old.issued_on IS NULL OR newer.issued_on >= old.issued_on)
          AND EXISTS (
              SELECT 1 FROM unnest(old.reference_nos) AS ref
              WHERE length(ref) >= 12 AND position(ref IN replace(c.text, ' ', '')) > 0
          )
        ORDER BY old.id, newer.id, c.ordinal
        ON CONFLICT DO NOTHING
        """,
        (chunk_set_id,),
    )
    return result.rowcount or 0


async def record_query(conn: Conn, record: dict[str, Any]) -> None:
    await conn.execute(
        """
        INSERT INTO queries (
            trace_id, question, rewritten, as_of, status, answer, config, timings_ms, usage, cost_usd
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (trace_id) DO NOTHING
        """,
        (
            record["trace_id"],
            record["question"],
            record.get("rewritten"),
            record.get("as_of"),
            record["status"],
            Jsonb(record["answer"]),
            Jsonb(record["config"]),
            Jsonb(record["timings_ms"]),
            Jsonb(record["usage"]),
            record.get("cost_usd", 0),
        ),
    )


async def record_feedback(conn: Conn, trace_id: str, rating: int, comment: str | None) -> bool:
    exists = await (
        await conn.execute("SELECT 1 FROM queries WHERE trace_id = %s", (trace_id,))
    ).fetchone()
    if exists is None:
        return False
    await conn.execute(
        "INSERT INTO feedback (trace_id, rating, comment) VALUES (%s, %s, %s)",
        (trace_id, rating, comment),
    )
    return True

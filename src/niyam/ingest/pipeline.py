from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from niyam.config import Settings
from niyam.db import Database, ensure_vector_index
from niyam.db import repository as repo
from niyam.embeddings import Embedder, embedding_input
from niyam.ingest.chunking import chunker_from_name
from niyam.ingest.fetcher import FetchError, PoliteFetcher
from niyam.ingest.html import PARSER_VERSION as HTML_PARSER_VERSION
from niyam.ingest.html import parse_detail_body
from niyam.ingest.local import load_local_document
from niyam.ingest.rbi import MASTER_DIRECTIONS_INDEX, IndexEntry, RBIMasterDirections
from niyam.models import DocumentMeta, Element, ParsedDocument

log = logging.getLogger(__name__)

MIN_ELEMENTS = 3


def content_hash(parser: str, elements: list[Element]) -> str:
    digest = hashlib.sha256(parser.encode())
    for element in elements:
        digest.update(b"\x00")
        digest.update(f"{element.kind}|{element.level}|{element.text}".encode())
    return digest.hexdigest()


@dataclass(slots=True)
class IngestReport:
    seen: int = 0
    changed: int = 0
    unchanged: int = 0
    failed: list[str] = field(default_factory=list)
    chunks_inserted: int = 0
    chunks_removed: int = 0
    supersessions: int = 0
    seconds: float = 0.0


class Ingestor:
    def __init__(self, db: Database, chunk_sets: Iterable[str]) -> None:
        self._db = db
        self._chunkers = [chunker_from_name(name) for name in chunk_sets]

    async def store(self, doc: ParsedDocument, report: IngestReport) -> None:
        report.seen += 1
        async with self._db.transaction() as conn:
            upsert = await repo.upsert_document(conn, doc)
            any_written = False
            for chunker in self._chunkers:
                set_id = await repo.get_or_create_chunk_set(
                    conn, chunker.name, chunker.strategy, chunker.params()
                )
                if not upsert.changed and await repo.has_chunks(conn, upsert.document_id, set_id):
                    continue
                chunks = chunker.chunk(doc.elements)
                ids = repo.chunk_ids(chunker.name, doc.meta.source_key, chunks)
                inserted, removed = await repo.replace_chunks(
                    conn, upsert.document_id, set_id, ids, chunks
                )
                report.chunks_inserted += inserted
                report.chunks_removed += removed
                any_written = True
        if upsert.changed or any_written:
            report.changed += 1
        else:
            report.unchanged += 1

    async def link_supersessions(self, report: IngestReport) -> None:
        async with self._db.transaction() as conn:
            for chunker in self._chunkers:
                set_id = await repo.get_or_create_chunk_set(
                    conn, chunker.name, chunker.strategy, chunker.params()
                )
                report.supersessions = await repo.detect_supersessions(conn, set_id)
                break


def _matches(entry: IndexEntry, pattern: re.Pattern[str] | None) -> bool:
    if pattern is None:
        return True
    return bool(pattern.search(entry.title) or pattern.search(entry.department or ""))


async def ingest_rbi(
    settings: Settings,
    db: Database,
    *,
    chunk_sets: list[str],
    match: str | None = None,
    limit: int | None = None,
    ids: list[str] | None = None,
    refresh: bool = False,
) -> IngestReport:
    started = time.perf_counter()
    report = IngestReport()
    ingestor = Ingestor(db, chunk_sets)
    pattern = re.compile(match, re.I) if match else None

    async with PoliteFetcher(
        cache_dir=settings.cache_dir / "http",
        user_agent=settings.fetch_user_agent,
        min_delay_s=settings.fetch_min_delay_s,
        timeout_s=settings.fetch_timeout_s,
        max_cache_age_s=settings.fetch_cache_max_age_s,
    ) as fetcher:
        source = RBIMasterDirections(fetcher)
        entries = await source.list_entries(refresh=refresh)
        if ids:
            wanted = [i.removeprefix("rbi-md-") for i in ids]
            listed = {e.source_key: e for e in entries}
            # Withdrawn directions drop off the index but their pages stay online.
            entries = [
                listed.get(f"rbi-md-{i}")
                or IndexEntry(
                    source_key=f"rbi-md-{i}",
                    url=f"{MASTER_DIRECTIONS_INDEX}?id={i}",
                    title="",
                    department=None,
                    listed_on=None,
                    pdf_url=None,
                )
                for i in wanted
            ]
        entries = [e for e in entries if _matches(e, pattern)]
        if limit is not None:
            entries = entries[:limit]
        log.info("ingesting %d master directions", len(entries))

        for entry in entries:
            try:
                page = await source.fetch_detail(entry, refresh=refresh)
            except FetchError:
                log.exception("fetch failed for %s", entry.url)
                report.failed.append(entry.source_key)
                continue
            if not page.body_html:
                report.failed.append(entry.source_key)
                continue
            elements = parse_detail_body(page.body_html, page.meta.title)
            if len(elements) < MIN_ELEMENTS:
                log.warning("skipping %s: only %d elements parsed", entry.source_key, len(elements))
                report.failed.append(entry.source_key)
                continue
            doc = ParsedDocument(
                meta=page.meta,
                elements=elements,
                content_hash=content_hash(HTML_PARSER_VERSION, elements),
                parser=HTML_PARSER_VERSION,
            )
            await ingestor.store(doc, report)

    await ingestor.link_supersessions(report)
    report.seconds = time.perf_counter() - started
    return report


async def ingest_local(db: Database, paths: list[Path], *, chunk_sets: list[str]) -> IngestReport:
    started = time.perf_counter()
    report = IngestReport()
    ingestor = Ingestor(db, chunk_sets)
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in {".md", ".pdf"}))
        else:
            files.append(path)

    for file in files:
        try:
            meta, elements, page_count, parser = load_local_document(file)
        except Exception:
            log.exception("failed to load %s", file)
            report.failed.append(str(file))
            continue
        await ingestor.store(
            ParsedDocument(
                meta=meta,
                elements=elements,
                content_hash=content_hash(parser, elements),
                parser=parser,
                page_count=page_count,
            ),
            report,
        )
    await ingestor.link_supersessions(report)
    report.seconds = time.perf_counter() - started
    return report


@dataclass(slots=True)
class EmbedReport:
    embedded: int = 0
    seconds: float = 0.0


async def embed_missing(
    db: Database, embedder: Embedder, chunk_set: str, batch_size: int = 256
) -> EmbedReport:
    started = time.perf_counter()
    report = EmbedReport()
    async with db.transaction() as conn:
        row = await (
            await conn.execute("SELECT id FROM chunk_sets WHERE name = %s", (chunk_set,))
        ).fetchone()
        if row is None:
            raise LookupError(f"chunk set {chunk_set!r} does not exist; ingest documents first")
        set_id = int(row["id"])
        model_id = await repo.get_or_create_embedding_model(conn, embedder.name, embedder.dim)
        await ensure_vector_index(conn, model_id, set_id, embedder.dim)

    while True:
        async with db.connection() as conn:
            pending = await repo.chunks_missing_embeddings(conn, set_id, model_id, batch_size)
        if not pending:
            break
        texts = [embedding_input(r["title"], r["heading_text"], r["text"]) for r in pending]
        vectors = await embedder.embed_documents(texts)
        async with db.transaction() as conn:
            await repo.insert_embeddings(
                conn,
                model_id,
                set_id,
                [(r["id"], v) for r, v in zip(pending, vectors, strict=True)],
            )
        report.embedded += len(pending)
        log.info("embedded %d chunks (%s, %s)", report.embedded, chunk_set, embedder.name)

    report.seconds = time.perf_counter() - started
    return report


__all__ = [
    "DocumentMeta",
    "EmbedReport",
    "IngestReport",
    "embed_missing",
    "ingest_local",
    "ingest_rbi",
]

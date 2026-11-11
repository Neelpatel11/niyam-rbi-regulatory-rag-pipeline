from __future__ import annotations

import asyncio
import json
import logging
import platform
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from psycopg.types.json import Jsonb

from niyam import __version__
from niyam.config import Settings
from niyam.db import Database
from niyam.embeddings import Embedder, build_embedder
from niyam.evals.dataset import GoldenItem
from niyam.evals.judge import judge_answer
from niyam.evals.metrics import (
    BinaryCounts,
    ChunkView,
    evidence_hits,
    hit_at_k,
    ndcg_at_k,
    normalize_tokens,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from niyam.generation.llm import LLMClient
from niyam.generation.pricing import PriceBook, Usage
from niyam.models import AnswerStatus, RetrievedChunk
from niyam.qa import AskRequest, PipelineConfig, QAService, refusal_signal
from niyam.retrieval import HybridRetriever, RetrievalConfig, TimeFilter
from niyam.retrieval.rerank import CrossEncoderReranker, Reranker

log = logging.getLogger(__name__)

Mode = Literal["retrieval", "full"]
RETRIEVAL_FIELDS = set(RetrievalConfig.__dataclass_fields__)


@dataclass(slots=True)
class EvalConfig:
    name: str
    description: str = ""
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str | None = None
    retrieval: dict[str, Any] = field(default_factory=dict)
    query_rewrite: bool = True
    refusal_threshold: float | None = None
    refusal_threshold_dense: float | None = None

    def pipeline(self, settings: Settings) -> PipelineConfig:
        base = PipelineConfig.from_settings(settings)
        retrieval = base.retrieval.with_overrides(**self.retrieval)
        return PipelineConfig(
            retrieval=retrieval,
            query_rewrite=self.query_rewrite,
            refusal_threshold=self.refusal_threshold
            if self.refusal_threshold is not None
            else base.refusal_threshold,
            refusal_threshold_dense=self.refusal_threshold_dense
            if self.refusal_threshold_dense is not None
            else base.refusal_threshold_dense,
            effort=base.effort,
            max_output_tokens=base.max_output_tokens,
        )


def load_configs(path: Path, settings: Settings) -> list[EvalConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults", {})
    out: list[EvalConfig] = []
    for entry in raw.get("configs", []):
        merged = {**defaults, **entry}
        retrieval = {k: merged.pop(k) for k in list(merged) if k in RETRIEVAL_FIELDS}
        merged.setdefault("embedding_provider", settings.embedding_provider)
        merged.setdefault("embedding_model", settings.embedding_model)
        if retrieval.get("use_rerank", True):
            merged.setdefault("rerank_model", settings.rerank_model)
        out.append(EvalConfig(retrieval=retrieval, **merged))
    return out


@dataclass(slots=True)
class ItemResult:
    id: str
    category: str
    split: str
    answerable: bool
    retrieved: list[str]
    hits_at_rank: list[int]
    recall_at_5: float
    recall_at_20: float
    hit_at_5: float
    mrr: float
    ndcg_at_10: float
    refusal_score: float | None
    refused: bool
    latency_ms: float
    status: str | None = None
    answer: str | None = None
    rewritten: str | None = None
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    faithfulness: float | None = None
    correctness: float | None = None
    correctness_label: str | None = None
    citation_precision: float | None = None
    judge_rationale: str | None = None


def _views(chunks: list[RetrievedChunk]) -> list[ChunkView]:
    return [
        ChunkView(
            c.chunk_id,
            c.source_key,
            tuple(normalize_tokens(f"{' '.join(c.heading_path)} {c.text}")),
        )
        for c in chunks
    ]


def _retrieval_scores(item: GoldenItem, ranked: list[RetrievedChunk]) -> dict[str, Any]:
    if not item.answerable:
        return {
            "hits_at_rank": [],
            "recall_at_5": 0.0,
            "recall_at_20": 0.0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
            "ndcg_at_10": 0.0,
        }
    hits = evidence_hits(item.evidence, _views(ranked[:20]))
    width = len(hits[0]) if hits else 0
    return {
        "hits_at_rank": [r + 1 for r in range(width) if any(row[r] for row in hits)],
        "recall_at_5": recall_at_k(hits, 5),
        "recall_at_20": recall_at_k(hits, 20),
        "hit_at_5": hit_at_k(hits, 5),
        "mrr": reciprocal_rank(hits),
        "ndcg_at_10": ndcg_at_k(hits, 10),
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize(items: list[ItemResult], mode: Mode) -> dict[str, Any]:
    answerable = [i for i in items if i.answerable]
    refusal = BinaryCounts()
    for item in items:
        refusal.add(item.refused, not item.answerable)
    latencies = [i.latency_ms for i in items]
    summary: dict[str, Any] = {
        "n": len(items),
        "n_answerable": len(answerable),
        "recall@5": _mean([i.recall_at_5 for i in answerable]),
        "recall@20": _mean([i.recall_at_20 for i in answerable]) if mode == "retrieval" else None,
        "hit@5": _mean([i.hit_at_5 for i in answerable]),
        "mrr": _mean([i.mrr for i in answerable]),
        "ndcg@10": _mean([i.ndcg_at_10 for i in answerable]),
        "refusal_precision": round(refusal.precision, 4),
        "refusal_recall": round(refusal.recall, 4),
        "refusal_f1": round(refusal.f1, 4),
        "false_refusal_rate": round(refusal.fp / len(answerable), 4) if answerable else None,
        "latency_p50_ms": round(percentile(latencies, 0.5), 1),
        "latency_p95_ms": round(percentile(latencies, 0.95), 1),
    }
    if mode == "full":
        answered = [i for i in answerable if i.status == AnswerStatus.ANSWERED.value]
        costs = [i.cost_usd for i in items if i.cost_usd is not None]
        summary.update(
            {
                "answered_rate": round(len(answered) / len(answerable), 4) if answerable else None,
                "faithfulness": _mean(
                    [i.faithfulness for i in answered if i.faithfulness is not None]
                ),
                "correctness": _mean(
                    [i.correctness if i.correctness is not None else 0.0 for i in answerable]
                ),
                "citation_precision": _mean(
                    [i.citation_precision for i in answered if i.citation_precision is not None]
                ),
                "cost_per_query_usd": round(statistics.fmean(costs), 6) if costs else None,
                "cost_known_for": f"{len(costs)}/{len(items)}",
                "tokens_per_query": round(
                    statistics.fmean([i.input_tokens + i.output_tokens for i in items]), 1
                )
                if items
                else None,
                "validation_failures": sum(
                    1 for i in items if i.status == AnswerStatus.FAILED_VALIDATION.value
                ),
            }
        )
    return summary


class EvalRunner:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        *,
        llm: LLMClient | None = None,
        judge: LLMClient | None = None,
        prices: PriceBook | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._llm = llm
        self._judge = judge
        self._prices = prices or PriceBook.load(settings.prices_file)
        self._embedders: dict[str, Embedder] = {}
        self._rerankers: dict[str, Reranker] = {}

    def _embedder(self, cfg: EvalConfig) -> Embedder:
        key = f"{cfg.embedding_provider}:{cfg.embedding_model}"
        if key not in self._embedders:
            self._embedders[key] = build_embedder(
                self._settings, model=cfg.embedding_model, provider=cfg.embedding_provider
            )
        return self._embedders[key]

    def _reranker(self, cfg: EvalConfig) -> Reranker | None:
        if not cfg.rerank_model:
            return None
        if cfg.rerank_model not in self._rerankers:
            self._rerankers[cfg.rerank_model] = CrossEncoderReranker(
                cfg.rerank_model, cache_dir=str(self._settings.data_dir / "models")
            )
        return self._rerankers[cfg.rerank_model]

    def register(self, embedder: Embedder | None = None, reranker: Reranker | None = None) -> None:
        """Inject pre-built components (used by tests to avoid model downloads)."""
        if embedder is not None:
            self._embedders[embedder.name] = embedder
        if reranker is not None:
            self._rerankers[reranker.name] = reranker

    async def run(
        self,
        cfg: EvalConfig,
        items: list[GoldenItem],
        *,
        mode: Mode = "retrieval",
        concurrency: int = 1,
        judge: bool = True,
    ) -> dict[str, Any]:
        pipeline = cfg.pipeline(self._settings)
        embedder = self._embedder(cfg)
        reranker = self._reranker(cfg)
        retriever = HybridRetriever(self._db, embedder, reranker, pipeline.retrieval)
        started = time.perf_counter()

        if mode == "retrieval":
            results = [await self._retrieval_item(retriever, pipeline, item) for item in items]
        else:
            if self._llm is None:
                raise ValueError("full mode needs an LLM client")
            qa = QAService(
                db=None,
                retriever=retriever,
                llm=self._llm,
                rewrite_llm=None,
                config=pipeline,
                prices=self._prices,
                persist=False,
            )
            semaphore = asyncio.Semaphore(max(1, concurrency))

            async def bounded(item: GoldenItem) -> ItemResult:
                async with semaphore:
                    return await self._full_item(qa, pipeline, item, judge=judge)

            results = list(await asyncio.gather(*(bounded(i) for i in items)))

        by_category: dict[str, list[ItemResult]] = defaultdict(list)
        by_split: dict[str, list[ItemResult]] = defaultdict(list)
        for result in results:
            by_category[result.category].append(result)
            by_split[result.split].append(result)

        return {
            "run_id": str(uuid.uuid4()),
            "name": cfg.name,
            "description": cfg.description,
            "mode": mode,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "niyam_version": __version__,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "config": {
                "pipeline": pipeline.to_dict(),
                "embedding": embedder.name,
                "reranker": reranker.name if reranker else None,
                "llm": f"{self._llm.provider}:{self._llm.model}"
                if self._llm and mode == "full"
                else None,
                "judge": f"{self._judge.provider}:{self._judge.model}"
                if self._judge and mode == "full" and judge
                else None,
            },
            "seconds": round(time.perf_counter() - started, 1),
            "metrics": summarize(results, mode),
            "by_category": {k: summarize(v, mode) for k, v in sorted(by_category.items())},
            "by_split": {k: summarize(v, mode) for k, v in sorted(by_split.items())},
            "items": [_as_dict(r) for r in results],
        }

    async def _retrieval_item(
        self, retriever: HybridRetriever, pipeline: PipelineConfig, item: GoldenItem
    ) -> ItemResult:
        start = time.perf_counter()
        result = await retriever.retrieve(
            item.question,
            time_filter=TimeFilter(as_of=item.as_of, include_withdrawn=item.include_withdrawn),
            config=pipeline.retrieval,
        )
        latency = (time.perf_counter() - start) * 1000
        score, threshold = refusal_signal(result, pipeline)
        scores = _retrieval_scores(item, result.candidates)
        return ItemResult(
            id=item.id,
            category=item.category.value,
            split=item.split(),
            answerable=item.answerable,
            retrieved=[c.chunk_id for c in result.candidates[:20]],
            refusal_score=score,
            refused=score is None or score < threshold,
            latency_ms=round(latency, 1),
            **scores,
        )

    async def _full_item(
        self, qa: QAService, pipeline: PipelineConfig, item: GoldenItem, *, judge: bool
    ) -> ItemResult:
        response = await qa.ask(
            AskRequest(
                question=item.question,
                history=list(item.history),
                as_of=item.as_of,
                include_withdrawn=item.include_withdrawn,
            ),
            pipeline,
        )
        scores = _retrieval_scores(item, response.retrieved)
        usage = Usage(provider="all", model="all")
        for u in response.usage:
            usage.add(u)
        result = ItemResult(
            id=item.id,
            category=item.category.value,
            split=item.split(),
            answerable=item.answerable,
            retrieved=[c.chunk_id for c in response.retrieved],
            refusal_score=response.retrieval_score,
            refused=response.status is not AnswerStatus.ANSWERED,
            latency_ms=response.timings_ms.get("total", 0.0),
            status=response.status.value,
            answer=response.answer,
            rewritten=response.rewritten_question,
            cost_usd=response.cost_usd,
            input_tokens=usage.input_tokens + usage.cache_read_tokens,
            output_tokens=usage.output_tokens,
            **scores,
        )
        if (
            judge
            and self._judge is not None
            and item.answerable
            and response.status is AnswerStatus.ANSWERED
        ):
            by_id = {c.chunk_id: c for c in response.retrieved}
            passages = [
                (c.label, by_id[c.chunk_id].text) for c in response.citations if c.chunk_id in by_id
            ]
            verdict = await judge_answer(
                self._judge,
                question=item.question,
                reference=item.expected_answer,
                answer=response.answer,
                passages=passages,
            )
            if verdict is not None:
                result.faithfulness = verdict.faithfulness
                result.correctness = verdict.correctness
                result.correctness_label = verdict.correctness_label
                result.citation_precision = verdict.citation_precision
                result.judge_rationale = verdict.raw.rationale
        return result


def _as_dict(obj: Any) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in obj.__dataclass_fields__}


def write_results(run: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = run["created_at"].replace(":", "").replace("-", "")[:15]
    path = out_dir / f"{stamp}_{run['mode']}_{run['name']}.json"
    path.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    return path


async def save_run(db: Database, run: dict[str, Any]) -> None:
    async with db.connection() as conn:
        await conn.execute(
            "INSERT INTO eval_runs (id, name, config, metrics) VALUES (%s, %s, %s, %s)",
            (
                run["run_id"],
                f"{run['mode']}:{run['name']}",
                Jsonb(run["config"]),
                Jsonb(run["metrics"]),
            ),
        )


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "–"
    if pct:
        return f"{value * 100:.1f}%"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(runs: list[dict[str, Any]]) -> str:
    full = any(r["mode"] == "full" for r in runs)
    if full:
        header = (
            "| Config | Recall@5 | MRR | Faithfulness | Correctness | Refusal F1 | p95 latency | Cost/query |\n"
            "|---|---|---|---|---|---|---|---|"
        )
    else:
        header = (
            "| Config | Recall@5 | Recall@20 | MRR | nDCG@10 | Refusal F1 | p95 latency |\n"
            "|---|---|---|---|---|---|---|"
        )
    lines = [header]
    for run in runs:
        m = run["metrics"]
        if full:
            cost = m.get("cost_per_query_usd")
            lines.append(
                f"| {run['name']} | {_fmt(m['recall@5'], True)} | {_fmt(m['mrr'])} | "
                f"{_fmt(m.get('faithfulness'))} | {_fmt(m.get('correctness'))} | "
                f"{_fmt(m['refusal_f1'])} | {m['latency_p95_ms']:.0f} ms | "
                f"{'$' + format(cost, '.4f') if cost is not None else '–'} |"
            )
        else:
            lines.append(
                f"| {run['name']} | {_fmt(m['recall@5'], True)} | {_fmt(m['recall@20'], True)} | "
                f"{_fmt(m['mrr'])} | {_fmt(m['ndcg@10'])} | {_fmt(m['refusal_f1'])} | "
                f"{m['latency_p95_ms']:.0f} ms |"
            )
    n = runs[0]["metrics"]["n"] if runs else 0
    lines.append("")
    lines.append(f"_n = {n} questions; generated {datetime.now(UTC).date().isoformat()}._")
    return "\n".join(lines)

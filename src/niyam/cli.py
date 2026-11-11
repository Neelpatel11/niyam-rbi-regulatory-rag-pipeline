from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Coroutine
from datetime import date
from pathlib import Path
from typing import Annotated, Any, TypeVar, cast

import typer
from rich.console import Console
from rich.table import Table

from niyam.config import Settings, get_settings
from niyam.db import Database, apply_schema
from niyam.log_config import configure_logging

T = TypeVar("T")

app = typer.Typer(
    help="Niyam: citation-checked answers over RBI Master Directions.", no_args_is_help=True
)
ingest_app = typer.Typer(help="Fetch, parse and chunk documents.", no_args_is_help=True)
eval_app = typer.Typer(
    help="Evaluate retrieval and answers against a golden set.", no_args_is_help=True
)
app.add_typer(ingest_app, name="ingest")
app.add_typer(eval_app, name="eval")

console = Console()


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on a selector loop (psycopg's async driver does not support Proactor)."""
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(coro)


def _settings() -> Settings:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    return settings


async def _db(settings: Settings, migrate: bool = True) -> Database:
    db = Database(settings.database_url, settings.db_pool_min, settings.db_pool_max)
    await db.open()
    if migrate:
        await apply_schema(db)
    return db


ChunkSets = Annotated[
    list[str] | None,
    typer.Option("--chunk-set", "-c", help="Chunk set(s) to build, e.g. structure-v1, fixed-512."),
]


@app.command("init-db")
def init_db() -> None:
    """Create the extension, tables and indexes."""
    settings = _settings()

    async def main() -> None:
        db = await _db(settings)
        await db.close()

    run(main())
    console.print("[green]schema applied[/green]")


def _print_ingest(report: Any) -> None:
    console.print(
        f"documents seen={report.seen} changed={report.changed} unchanged={report.unchanged} "
        f"failed={len(report.failed)} | chunks +{report.chunks_inserted} -{report.chunks_removed} "
        f"| supersession links={report.supersessions} | {report.seconds:.1f}s"
    )
    if report.failed:
        console.print(f"[yellow]failed: {', '.join(report.failed[:20])}[/yellow]")


@ingest_app.command("rbi")
def ingest_rbi_cmd(
    match: Annotated[str | None, typer.Option(help="Regex on title or department.")] = None,
    limit: Annotated[int | None, typer.Option(help="Maximum documents to ingest.")] = None,
    ids: Annotated[list[str] | None, typer.Option("--id", help="RBI direction id(s).")] = None,
    refresh: Annotated[bool, typer.Option(help="Ignore the HTTP cache.")] = False,
    chunk_set: ChunkSets = None,
) -> None:
    """Crawl RBI Master Directions (politely, with caching) and chunk them."""
    from niyam.ingest.pipeline import ingest_rbi

    settings = _settings()

    async def main() -> Any:
        db = await _db(settings)
        try:
            return await ingest_rbi(
                settings,
                db,
                chunk_sets=chunk_set or [settings.chunk_set],
                match=match,
                limit=limit,
                ids=ids,
                refresh=refresh,
            )
        finally:
            await db.close()

    _print_ingest(run(main()))


@ingest_app.command("local")
def ingest_local_cmd(
    paths: Annotated[
        list[Path], typer.Argument(exists=True, help="Markdown/PDF files or folders.")
    ],
    chunk_set: ChunkSets = None,
) -> None:
    """Ingest local Markdown (YAML front matter) or PDF files (with a sibling .yaml)."""
    from niyam.ingest.pipeline import ingest_local

    settings = _settings()

    async def main() -> Any:
        db = await _db(settings)
        try:
            return await ingest_local(db, paths, chunk_sets=chunk_set or [settings.chunk_set])
        finally:
            await db.close()

    _print_ingest(run(main()))


@app.command()
def embed(
    chunk_set: Annotated[str | None, typer.Option("--chunk-set", "-c")] = None,
    provider: Annotated[
        str | None, typer.Option(help="fastembed | openai | gemini | hashing")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Embedding model name.")] = None,
) -> None:
    """Embed every chunk that has no vector for the chosen model yet."""
    from niyam.embeddings import build_embedder
    from niyam.ingest.pipeline import embed_missing

    settings = _settings()

    async def main() -> Any:
        db = await _db(settings)
        try:
            embedder = build_embedder(settings, model=model, provider=provider)
            return await embed_missing(db, embedder, chunk_set or settings.chunk_set)
        finally:
            await db.close()

    report = run(main())
    console.print(f"embedded {report.embedded} chunks in {report.seconds:.1f}s")


@app.command()
def ask(
    question: str,
    as_of: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    include_withdrawn: bool = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Ask a question from the terminal."""
    from niyam.api.schemas import AskOut
    from niyam.qa import AskRequest
    from niyam.services import build_services

    settings = _settings()

    async def main() -> Any:
        services = await build_services(settings)
        try:
            return await services.qa.ask(
                AskRequest(
                    question=question,
                    as_of=date.fromisoformat(as_of) if as_of else None,
                    include_withdrawn=include_withdrawn,
                )
            )
        finally:
            await services.close()

    response = run(main())
    if as_json:
        console.print_json(AskOut.from_response(response).model_dump_json())
        return
    console.print(f"\n[bold]{response.answer}[/bold]\n")
    for citation in response.citations:
        where = (
            f"para {citation.clause_ref}"
            if citation.clause_ref
            else " > ".join(citation.heading_path)
        )
        console.print(f"  [{citation.label}] {citation.title} — {where}\n       {citation.url}")
    cost = f"${response.cost_usd:.4f}" if response.cost_usd is not None else "unknown"
    console.print(
        f"\n[dim]status={response.status.value} score={response.retrieval_score} "
        f"total={response.timings_ms.get('total')}ms cost={cost}[/dim]"
    )


@app.command()
def models(
    reranker: Annotated[list[str] | None, typer.Option(help="Extra reranker model(s).")] = None,
) -> None:
    """Download the configured embedding and reranker models (used at image build time)."""
    from niyam.embeddings import build_embedder
    from niyam.services import build_reranker

    settings = _settings()
    embedder = build_embedder(settings)
    console.print(f"embedding model ready: {embedder.name} (dim {embedder.dim})")
    for name in [settings.rerank_model, *(reranker or [])]:
        model = build_reranker(settings, model=name)
        console.print(f"reranker ready: {model.name if model else name}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API."""
    import uvicorn

    from niyam.api.app import create_app

    settings = _settings()
    config = uvicorn.Config(create_app(settings), host=host, port=port, log_config=None)
    run(uvicorn.Server(config).serve())


GoldenOpt = Annotated[Path, typer.Option("--golden", "-g", exists=True)]
ConfigsOpt = Annotated[Path, typer.Option("--configs", exists=True)]


def _parse_gate(gate: str) -> tuple[str | None, str, str, float]:
    match = re.fullmatch(r"\s*(?:([\w.-]+):)?([\w@]+)\s*(>=|<=)\s*([0-9.]+)\s*", gate)
    if not match:
        raise typer.BadParameter(
            f"gate must look like 'recall@5>=0.8' or 'hybrid-rerank:refusal_f1>=0.8', got {gate!r}"
        )
    return match.group(1), match.group(2), match.group(3), float(match.group(4))


@eval_app.command("run")
def eval_run(
    golden: GoldenOpt = Path("evals/golden.jsonl"),
    configs: ConfigsOpt = Path("evals/configs/ablations.yaml"),
    name: Annotated[list[str] | None, typer.Option("--name", "-n", help="Config name(s).")] = None,
    mode: Annotated[str, typer.Option(help="retrieval | full")] = "retrieval",
    split: Annotated[str | None, typer.Option(help="dev | test")] = None,
    category: Annotated[str | None, typer.Option(help="Only this category.")] = None,
    limit: Annotated[int | None, typer.Option()] = None,
    concurrency: int = 1,
    judge: Annotated[bool, typer.Option(help="Run the LLM judge in full mode.")] = True,
    out: Annotated[Path, typer.Option()] = Path("evals/results"),
    gate: Annotated[
        list[str] | None,
        typer.Option(help="e.g. 'recall@5>=0.8' or 'hybrid-rerank:refusal_f1>=0.8'"),
    ] = None,
) -> None:
    """Run one or more configurations and write results JSON plus a markdown table."""
    from niyam.evals.dataset import load_golden
    from niyam.evals.runner import (
        EvalRunner,
        Mode,
        load_configs,
        markdown_table,
        save_run,
        write_results,
    )
    from niyam.generation.llm import build_llm

    if mode not in {"retrieval", "full"}:
        raise typer.BadParameter("mode must be retrieval or full")
    settings = _settings()
    items = load_golden(golden)
    if split:
        items = [i for i in items if i.split() == split]
    if category:
        items = [i for i in items if i.category.value == category]
    if limit:
        items = items[:limit]
    selected = load_configs(configs, settings)
    if name:
        selected = [c for c in selected if c.name in set(name)]
        if not selected:
            raise typer.BadParameter(f"no config named {name}")

    async def main() -> list[dict[str, Any]]:
        db = await _db(settings)
        try:
            llm = build_llm(settings) if mode == "full" else None
            judge_llm = (
                build_llm(settings, model=settings.judge_model)
                if mode == "full" and judge
                else None
            )
            runner = EvalRunner(settings, db, llm=llm, judge=judge_llm)
            runs = []
            for cfg in selected:
                console.print(f"[cyan]running {cfg.name} ({mode}, {len(items)} questions)[/cyan]")
                result = await runner.run(
                    cfg, items, mode=cast(Mode, mode), concurrency=concurrency, judge=judge
                )
                path = write_results(result, out)
                await save_run(db, result)
                console.print(f"  wrote {path}")
                runs.append(result)
            return runs
        finally:
            await db.close()

    runs = run(main())
    table = markdown_table(runs)
    (out / f"latest_{mode}.md").write_text(table + "\n", encoding="utf-8")
    console.print(table)

    failures: list[str] = []
    for spec in gate or []:
        target, metric, op, bound = _parse_gate(spec)
        for result in runs:
            if target is not None and result["name"] != target:
                continue
            value = result["metrics"].get(metric)
            if value is None:
                failures.append(f"{result['name']}: metric {metric} is missing")
            elif (op == ">=" and value < bound) or (op == "<=" and value > bound):
                failures.append(f"{result['name']}: {metric}={value} violates {op} {bound}")
    if failures:
        for failure in failures:
            console.print(f"[red]gate failed: {failure}[/red]")
        raise typer.Exit(code=1)


@eval_app.command("tune")
def eval_tune(
    golden: GoldenOpt = Path("evals/golden.jsonl"),
    configs: ConfigsOpt = Path("evals/configs/ablations.yaml"),
    name: Annotated[str, typer.Option("--name", "-n")] = "hybrid-rerank",
) -> None:
    """Pick the refusal threshold that maximises refusal F1 on the dev split."""
    from niyam.evals.dataset import load_golden
    from niyam.evals.metrics import best_refusal_threshold
    from niyam.evals.runner import EvalRunner, load_configs

    settings = _settings()
    items = [i for i in load_golden(golden) if i.split() == "dev"]
    cfg = next((c for c in load_configs(configs, settings) if c.name == name), None)
    if cfg is None:
        raise typer.BadParameter(f"no config named {name}")

    async def main() -> dict[str, Any]:
        db = await _db(settings)
        try:
            return await EvalRunner(settings, db).run(cfg, items, mode="retrieval")
        finally:
            await db.close()

    result = run(main())
    scores = [i["refusal_score"] for i in result["items"]]
    answerable = [i["answerable"] for i in result["items"]]
    threshold, counts = best_refusal_threshold(scores, answerable)
    console.print(
        f"dev n={len(items)} best threshold={threshold:.4f} refusal P={counts.precision:.3f} "
        f"R={counts.recall:.3f} F1={counts.f1:.3f}\n"
        f"Set it in {configs} (refusal_threshold) or NIYAM_REFUSAL_THRESHOLD, then report on --split test."
    )


@eval_app.command("draft")
def eval_draft(
    n: Annotated[int, typer.Option(help="Answerable candidates to draft.")] = 40,
    unanswerable: Annotated[int, typer.Option(help="Unanswerable candidates to draft.")] = 10,
    seed: int = 7,
    out: Path = Path("evals/candidates.jsonl"),
) -> None:
    """Draft candidate questions from sampled chunks. Every candidate needs human review."""
    from niyam.evals.curation import draft_answerable, draft_unanswerable, write_candidates
    from niyam.generation.llm import build_llm

    settings = _settings()

    async def main() -> int:
        db = await _db(settings)
        try:
            llm = build_llm(settings)
            candidates = await draft_answerable(
                llm, db, chunk_set=settings.chunk_set, n=n, seed=seed
            )
            if unanswerable:
                candidates += await draft_unanswerable(llm, db, n=unanswerable, seed=seed)
            write_candidates(out, candidates)
            return len(candidates)
        finally:
            await db.close()

    count = run(main())
    console.print(f"wrote {count} candidates to {out}; review them with `niyam eval review`")


@eval_app.command("review")
def eval_review(
    candidates: Annotated[Path, typer.Option(exists=True)] = Path("evals/candidates.jsonl"),
    golden: Annotated[Path, typer.Option()] = Path("evals/golden.jsonl"),
) -> None:
    """Accept, edit or reject drafted candidates one by one."""
    from niyam.evals.dataset import GoldenItem

    rows = [
        json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines() if line
    ]
    pending = [r for r in rows if r.get("status") == "draft"]
    existing = golden.read_text(encoding="utf-8").splitlines() if golden.exists() else []
    next_id = len([line for line in existing if line.strip()]) + 1

    for row in pending:
        console.rule(row["id"])
        console.print(
            f"[bold]Q:[/bold] {row['question']}\n[bold]category:[/bold] {row['category']}"
        )
        if row.get("expected_answer"):
            console.print(f"[bold]A:[/bold] {row['expected_answer']}")
        for ev in row.get("evidence", []):
            console.print(f"[dim]{ev['source_key']}: {ev['quote']}[/dim]")
        choice = typer.prompt("[a]ccept / [e]dit / [r]eject / [q]uit", default="a").lower()
        if choice.startswith("q"):
            break
        if choice.startswith("r"):
            row["status"] = "rejected"
            continue
        if choice.startswith("e"):
            row["question"] = typer.prompt("question", default=row["question"])
            if row["category"] != "unanswerable":
                row["expected_answer"] = typer.prompt(
                    "expected answer", default=row["expected_answer"]
                )
            row["category"] = typer.prompt("category", default=row["category"])
        item = GoldenItem.model_validate(
            {k: v for k, v in row.items() if k not in {"status"}} | {"id": f"q{next_id:04d}"}
        )
        with golden.open("a", encoding="utf-8") as handle:
            handle.write(item.model_dump_json(exclude_defaults=True) + "\n")
        row["status"] = "accepted"
        next_id += 1

    candidates.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


@eval_app.command("label")
def eval_label(
    results: Annotated[Path, typer.Option(exists=True, help="A full-mode results JSON.")],
    golden: GoldenOpt = Path("evals/golden.jsonl"),
    out: Path = Path("evals/labels.jsonl"),
    n: int = 50,
) -> None:
    """Hand-label answers from a full run, blind to the judge's verdict."""
    from niyam.evals.dataset import load_golden

    run_data = json.loads(results.read_text(encoding="utf-8"))
    by_id = {i.id: i for i in load_golden(golden)}
    done = set()
    if out.exists():
        done = {
            json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line
        }
    todo = [
        i
        for i in run_data["items"]
        if i["status"] == "answered" and i["answerable"] and i["id"] not in done
    ][:n]
    for entry in todo:
        item = by_id[entry["id"]]
        console.rule(item.id)
        console.print(
            f"[bold]Q:[/bold] {item.question}\n[bold]Reference:[/bold] {item.expected_answer}"
        )
        console.print(f"[bold]System:[/bold] {entry['answer']}")
        correctness = typer.prompt("correctness [c]orrect/[p]artial/[i]ncorrect", default="c")
        faithful = typer.prompt("every claim supported by its citations? [y/n]", default="y")
        label = {
            "id": item.id,
            "run_id": run_data["run_id"],
            "correctness": {"c": "correct", "p": "partial", "i": "incorrect"}[
                correctness[0].lower()
            ],
            "faithful": faithful.lower().startswith("y"),
        }
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(label) + "\n")


@eval_app.command("calibrate")
def eval_calibrate(
    results: Annotated[Path, typer.Option(exists=True)],
    labels: Annotated[Path, typer.Option(exists=True)] = Path("evals/labels.jsonl"),
    out: Path = Path("evals/judge_calibration.md"),
) -> None:
    """Measure agreement between the LLM judge and human labels."""
    from niyam.evals.curation import calibration_report, render_calibration

    run_data = json.loads(results.read_text(encoding="utf-8"))
    judged = {i["id"]: i for i in run_data["items"] if i.get("correctness_label")}
    human = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines() if line]
    correctness = [
        (h["correctness"], judged[h["id"]]["correctness_label"]) for h in human if h["id"] in judged
    ]
    faithful = [
        (
            "yes" if h["faithful"] else "no",
            "yes" if (judged[h["id"]]["faithfulness"] or 0) >= 0.999 else "no",
        )
        for h in human
        if h["id"] in judged and judged[h["id"]].get("faithfulness") is not None
    ]
    if not correctness:
        raise typer.BadParameter("no overlap between labels and judged items")
    reports = [calibration_report(correctness, "correctness")]
    if faithful:
        reports.append(calibration_report(faithful, "faithfulness (all claims supported)"))
    out.write_text(
        render_calibration(reports, run_data["config"].get("judge") or "unknown"), encoding="utf-8"
    )
    console.print(out.read_text(encoding="utf-8"))


@eval_app.command("table")
def eval_table(files: Annotated[list[Path], typer.Argument(exists=True)]) -> None:
    """Render a markdown table from saved result files."""
    from niyam.evals.runner import markdown_table

    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    console.print(markdown_table(runs))


@eval_app.command("validate")
def eval_validate(golden: GoldenOpt = Path("evals/golden.jsonl")) -> None:
    """Check a golden file and print its composition."""
    from collections import Counter

    from niyam.evals.dataset import load_golden

    items = load_golden(golden)
    table = Table("category", "dev", "test", "total")
    counts = Counter((i.category.value, i.split()) for i in items)
    for cat in sorted({i.category.value for i in items}):
        dev, test = counts[(cat, "dev")], counts[(cat, "test")]
        table.add_row(cat, str(dev), str(test), str(dev + test))
    table.add_row(
        "all",
        str(sum(v for (c, s), v in counts.items() if s == "dev")),
        str(sum(v for (c, s), v in counts.items() if s == "test")),
        str(len(items)),
    )
    console.print(table)


if __name__ == "__main__":
    app()

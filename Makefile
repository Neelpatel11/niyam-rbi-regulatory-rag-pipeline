.PHONY: install lint format typecheck test smoke up down ingest embed serve web

install:
	uv sync --all-extras --group dev
	cd web && npm install

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:
	uv run mypy
	cd web && npm run typecheck

test:
	uv run pytest

smoke:
	uv run niyam ingest local tests/fixtures/corpus -c structure-v1 -c fixed-512
	uv run niyam embed -c structure-v1
	uv run niyam embed -c fixed-512
	uv run niyam eval run --golden evals/smoke.jsonl -n dense-fixed -n dense-structure -n hybrid -n hybrid-rerank \
		--gate "recall@5>=0.95" --gate "hybrid-rerank:refusal_f1>=0.8"

up:
	docker compose up --build -d

down:
	docker compose down

ingest:
	uv run niyam ingest rbi

embed:
	uv run niyam embed

serve:
	uv run niyam serve

web:
	cd web && npm run dev

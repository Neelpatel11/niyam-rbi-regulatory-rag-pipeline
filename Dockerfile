# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    NIYAM_DATA_DIR=/app/data \
    HF_HUB_DISABLE_PROGRESS_BARS=1
WORKDIR /app

FROM base AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install ".[tracing]"

FROM base AS runtime
RUN useradd --create-home --uid 10001 niyam
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN mkdir -p /app/data && chown -R niyam:niyam /app
USER niyam
# Bake the local embedding and reranker models into the image so cold starts do not download them.
ARG PRELOAD_MODELS=true
RUN if [ "$PRELOAD_MODELS" = "true" ]; then niyam models; fi
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz',timeout=3)"
CMD ["sh", "-c", "uvicorn niyam.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --no-access-log"]

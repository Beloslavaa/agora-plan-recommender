# syntax=docker/dockerfile:1

# ── Stage 1: builder ──────────────────────────────────────────────
# Installs dependencies into a venv, isolated from the runtime image so uv
# itself and any transient build artifacts never ship to production.
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.4 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Only the manifest + lockfile, so this layer is cached across code changes
# and rebuilds only when dependencies actually change.
# --frozen: install exactly what uv.lock pins, and fail rather than silently
# re-resolving if it's out of date with pyproject.toml.
# --no-dev: skip the offline-training group (torch/jupyter) — see pyproject.toml.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime ──────────────────────────────────────────────
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# Non-root user: the container never needs root once dependencies are installed.
RUN groupadd --system app && useradd --system --gid app --home-dir /app app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# This image serves both roles (see README's Deployment section): the web
# process below, and the ingestion worker — `python main.py --mode fixed`,
# run by a Dokploy Schedule Job that execs into this same container, hence
# main.py being here and not just the API's own files.
COPY --chown=app:app agora ./agora
COPY --chown=app:app data ./data
COPY --chown=app:app index.html ./index.html
COPY --chown=app:app main.py ./main.py

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/')" || exit 1

# sh -c so $PORT (set by the host platform, e.g. Render) is expanded at runtime;
# `exec` makes uvicorn replace the shell as PID 1's child so it receives
# SIGTERM directly instead of the shell swallowing it on shutdown.
CMD ["sh", "-c", "exec uvicorn agora.backend.infrastructure.web.api:app --host 0.0.0.0 --port $PORT"]

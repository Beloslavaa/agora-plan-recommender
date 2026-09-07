# syntax=docker/dockerfile:1

# ── Stage 1: builder ──────────────────────────────────────────────
# Installs dependencies into a venv, isolated from the runtime image so
# pip/setuptools and any transient build artifacts never ship to production.
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the dependency manifest first so this layer is cached across
# code changes and only rebuilds when requirements-web.txt changes.
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt

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

# Only what agora.backend.infrastructure.web.api needs at request time:
# index.html (served at "/"), data/ (cities + fixed sources), agora/ (app code).
COPY --chown=app:app agora ./agora
COPY --chown=app:app data ./data
COPY --chown=app:app index.html ./index.html

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/')" || exit 1

# sh -c so $PORT (set by the host platform, e.g. Render) is expanded at runtime;
# `exec` makes uvicorn replace the shell as PID 1's child so it receives
# SIGTERM directly instead of the shell swallowing it on shutdown.
CMD ["sh", "-c", "exec uvicorn agora.backend.infrastructure.web.api:app --host 0.0.0.0 --port $PORT"]

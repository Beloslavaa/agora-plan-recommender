# Agora — Plan Recommender

A curated gallery of things to do — concerts, art exhibitions, cinema,
fashion events, and more — scraped from a mix of fixed and discovered
sources, deduplicated, and ranked per user. Each city gets its own separate,
switchable feed rather than one combined gallery; currently available for
Madrid and Barcelona (`data/cities.json`). **This is a gallery, not a
marketplace**: no inventory, no purchase flow. Every plan links out to its
original source.

## How it's put together

Structured hexagonal-ish: `domain/` (entities + pure business rules, no I/O),
`application/` (use-cases that orchestrate domain + infrastructure), and
`infrastructure/` (DB, HTTP, LLM/search/embedding providers, the FastAPI and
CLI adapters).

| Layer | Where | What it does |
|---|---|---|
| Domain | `agora/backend/domain/` | Entities (`schemas.py`), dedup rules (`plan_matching.py`), URL/content safety (`url_safety.py`), JSON-LD/dice.fm parsing (`event_parsing.py`), plan validation (`validation.py`), recommender scoring math (`ranking.py`). No I/O. |
| Application | `agora/backend/application/` | Use-cases: `extraction.py`/`explorer.py`/`enrichment.py`/`sources_admin.py`/`ingestion.py` (the scraping pipeline) and `recommendation.py` (ranking for a user). Orchestrates domain rules + infrastructure ports. |
| Infrastructure | `agora/backend/infrastructure/` | `persistence/postgres_repository.py` (Supabase — `plans`, `interactions`, `users`, schema owned by `init_db()`), `persistence/json_files.py` (fixed sources/cities), `llm/`, `search/`, `embeddings/` (provider adapters), `http/fetcher.py` (SSRF-guarded fetching), `web/api.py` (FastAPI), `cli/ingest_cli.py` (ingestion CLI). |
| Frontend | `index.html` | Single-file vanilla HTML/JS UI, served directly by the API — no build step. |

## Recommender

**Tier 1 (live):** content-based ranking. Every plan is embedded (Gemini
`gemini-embedding-001`, from its title/category/tags/description) and cached
in `plans.embedding`. A user's taste profile is the weighted average of the
embeddings of what they've interacted with (`saved` > `view_link` >
`click`), and `/recommendations/{user_id}` ranks candidates by cosine
similarity to that profile — falling back to popularity for brand-new users
with no usable signal yet. Cinema hub cards are scored by their single
best-matching movie rather than automatically leading the feed regardless of
relevance.

**Planned:** a LightGCN-based graph recommender (co-consumption — "the same
crowd attends plan A and B" — rather than text similarity), once there's
enough real multi-user interaction volume for it to learn anything
meaningful. See `AGENTS.md` for the fuller design rationale.

## Setup

Requires Python 3.13 (pinned in `.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

- `LLM_PROVIDER` (`openai` | `gemini` | `anthropic`) + its matching API key — used for scraping/extraction.
- `GEMINI_API_KEY` — **required regardless of `LLM_PROVIDER`**, since the recommender always embeds through Gemini's API specifically.
- `SEARCH_PROVIDER` + its key — used by the exploratory ingestion pipeline.
- `DATABASE_URL` — a Supabase Postgres connection string (pooled "Transaction mode", port 6543).

## Running it

```bash
uvicorn agora.backend.infrastructure.web.api:app --reload
```

Open `http://localhost:8000`.

## Running ingestion

```bash
python main.py --mode explorer   # discover new sources for each category
python main.py --mode fixed      # scrape the known/promoted sources
python main.py --mode full       # both
python main.py --city Madrid Barcelona   # omit to run every city in data/cities.json
```

Newly-scraped plans are embedded automatically at the end of the run.

## Deployment

`render.yaml` deploys this as a single Render web service (`DATABASE_URL` set
via Render's dashboard, not synced from the repo).

The ingestion schedule runs separately via a GitHub Actions workflow
(`.github/workflows/ingestion.yml`), twice a month — the 1st and 15th —
rather than a Render Cron Job, since Render has no free tier for cron
(billed per run) while GitHub Actions' scheduled workflows are free. It runs
`python main.py --mode full` for every city in `data/cities.json`, which
does three things back to back: scrapes, embeds any new plans, and
soft-deletes stale ones (`mark_stale_plans()` — anything whose end date, or
start date if it has no end date, is in the past; undated "DATES TBA" plans
are left alone). Soft delete just flips `plans.is_stale`, so interaction
history is preserved for the recommender; every browse/recommendation query
filters `NOT is_stale` to keep expired plans out of the feed.

The workflow needs these set as **GitHub repo secrets** (Settings → Secrets
and variables → Actions): `DATABASE_URL`, `LLM_PROVIDER`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `SEARCH_PROVIDER`, `SERPAPI_KEY` — same values as `.env`.
It can also be triggered manually from the repo's Actions tab
(`workflow_dispatch`).

## Project state

Demo/research repo — favors clarity and runnability over production
hardening. See `AGENTS.md` for the architecture rationale and conventions.

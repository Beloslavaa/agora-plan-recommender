# Agora — Plan Recommender

A curated gallery of things to do — concerts, art exhibitions, cinema,
fashion events, and more — scraped from a mix of fixed and discovered
sources, deduplicated, and ranked per user. Each city gets its own separate,
switchable feed rather than one combined gallery; currently available for
Madrid and Barcelona (`data/cities.json`). **This is a gallery, not a
marketplace**: no inventory, no purchase flow. Every plan links out to its
original source.

## How it's put together

| Layer | Where | What it does |
|---|---|---|
| Ingestion | `agora/backend/ingestion/` | LLM-driven scraping — a fixed list of known sources (`data/fixed_sources.json`) plus an exploratory search pipeline that discovers and promotes new ones. Extracts structured plans from raw HTML via an LLM (OpenAI / Gemini / Anthropic, configurable). |
| Storage | Supabase (Postgres) | `plans`, `interactions` (click / saved / view_link), `users` — schema owned entirely by `store.py`'s `init_db()`. |
| API | `agora/backend/api.py` | FastAPI: plans, auth, interactions, recommendations, cinema groupings. |
| Recommender | `agora/backend/recommender/` | See below. |
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
uvicorn agora.backend.api:app --reload
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

## Utility scripts

Run from the repo root with `PYTHONPATH=.` (they're standalone, not part of
the package's normal import path):

```bash
PYTHONPATH=. python scripts/backfill_embeddings.py           # embed any plan missing one
PYTHONPATH=. python scripts/smoke_test_recommender.py --title "..."  # save a plan, see what ranks close to it, clean up after itself
```

## Deployment

`render.yaml` deploys this as a Render web service (`DATABASE_URL` set via
Render's dashboard, not synced from the repo) plus a cron job
(`agora-ingestion`, twice a month — the 1st and 15th) that runs `python
main.py --mode full` for every city in `data/cities.json`. That single run
does three things back to back: scrapes, embeds any new plans, and
soft-deletes stale ones (`mark_stale_plans()` — anything whose end date, or
start date if it has no end date, is in the past; undated "DATES TBA" plans
are left alone). Soft delete just flips `plans.is_stale`, so interaction
history is preserved for the recommender; every browse/recommendation query
filters `NOT is_stale` to keep expired plans out of the feed. No separate
cleanup cron is needed.

## Project state

Demo/research repo — favors clarity and runnability over production
hardening. See `AGENTS.md` for the architecture rationale and conventions.

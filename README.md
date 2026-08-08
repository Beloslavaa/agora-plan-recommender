# Agora — Plan Recommender

**Live: [agora-plan-recommender.onrender.com](https://agora-plan-recommender.onrender.com/)**

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

**Tier 2 (live):** a LightGCN graph recommender on the bipartite user-plan
interaction graph — co-consumption ("the same crowd attends plan A and B")
rather than text similarity. Trained offline
(`notebooks/train_lightgcn.ipynb`), initialized from and regularized toward
Tier 1's semantic embeddings, then exported to `plans.graph_embedding` /
`user_embeddings`. `/recommendations/{user_id}` blends the graph score with
Tier 1's semantic score, folding in a live embedding for users not yet in
the trained graph, and falling back through Tier 1 then popularity when
there's no graph signal at all. Real interaction volume is still low, so the
graph is currently bootstrapped mostly with synthetic archetype users
(`scripts/generate_synthetic_interactions.py`) — useful for validating the
pipeline end to end, not yet evidence the graph model beats Tier 1 on real
taste. See `AGENTS.md` for the fuller design rationale.

**Cold-start plans:** a plan with zero interactions never becomes a node in
the trained graph, so it has no learned embedding of its own. The training
notebook's export step proxies one instead: embed the plan's text, find its
5 nearest neighbors in semantic space among plans that *were* trained, and
take the similarity-weighted average of those neighbors' trained graph
embeddings. It inherits a hint of whatever crowd its closest analogues
attracted, not just what its own text says about it.

**New users get tuned live, no retraining required:** a user who joined (or
started interacting) after the last training run has no row in
`user_embeddings` yet. Rather than wait for the next retrain,
`/recommendations/{user_id}` folds one in on the spot — a weighted average
of the *trained* graph embeddings of whatever plans that user has already
interacted with (`ranking.py`'s `fold_in_user_embedding`), recomputed fresh
on every request. Every new interaction sharpens it immediately. It's still
a shallower proxy than a properly trained embedding, though — it never gets
the reciprocal, multi-hop refinement real training provides, and the plan
embeddings themselves never move in response to it. Only the next full run
of `notebooks/train_lightgcn.ipynb` does that.

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

`render.yaml` deploys this as a single Render web service. Ingestion runs
separately on a schedule via GitHub Actions
(`.github/workflows/ingestion.yml`), twice a month, and can also be
triggered manually from the repo's Actions tab.

## Project state

Demo/research repo — favors clarity and runnability over production
hardening. See `AGENTS.md` for the architecture rationale and conventions.

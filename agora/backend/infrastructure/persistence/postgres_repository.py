import json

import bcrypt
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from agora.backend.domain.cinemas import CINEMA_SOURCES
from agora.backend.domain.plan_matching import (
    compute_dedup_merge_updates,
    compute_merge_updates,
    fuzzy_same_event_by_title,
    same_event_by_title,
)
from agora.backend.domain.schemas import PlanData
from agora.backend.infrastructure.config import settings

# Movies from cinema sources get grouped into one "cinema" card each (see
# list_cinemas/list_cinema_plans) rather than appearing individually in the
# main feed, so the general listing/ranking queries exclude them. Plain
# substring ILIKE matching (rather than parsing source_url's hostname) is good
# enough for this fixed, known set of domains and keeps it consistent with
# the ILIKE-based matching list_cinema_plans uses to select them back out.
_CINEMA_EXCLUDE_WHERE = " AND ".join("source_url NOT ILIKE %s" for _ in CINEMA_SOURCES)
_CINEMA_EXCLUDE_PARAMS = [f"%{domain}%" for domain in CINEMA_SOURCES]

# Columns update_plan_fields() is allowed to touch — keeps its dynamically-built
# SET clause from ever interpolating an arbitrary column name.
PATCHABLE_PLAN_COLUMNS = {"url", "image_url", "ticket_url", "price", "start_date", "end_date", "location"}

# Lazily-opened (open=False) so importing this module never makes a network call.
# prepare_threshold=None disables server-side prepared statements, which don't
# survive Supabase's transaction-mode pooler handing a query to a different
# backend connection.
pool = ConnectionPool(
    settings.database_url,
    min_size=1,
    max_size=5,
    kwargs={"row_factory": dict_row, "prepare_threshold": None},
    open=False,
)


def _conn() -> psycopg.connection.Connection:
    pool.open()  # idempotent no-op if already open
    return pool.connection()


# Bump this whenever a statement is added to (or changed in) the migration
# block below. init_db() is called on every process boot AND on nearly every
# write/list call in this module (upsert_plans, list_cinemas, ...) — before
# this gate existed it re-ran all ~25 migration statements as ~25 sequential
# round-trips to Supabase every single time, which is most of what made cold
# starts (and every /cinemas request) slow. The gate below turns the
# steady-state cost into 2 round-trips (create-if-missing + version check)
# once a deploy has run the migrations once, at the cost of remembering to
# bump this number when the block changes — forgetting only means a change
# silently doesn't apply until the next manual bump, which get_plan_count()-
# style smoke testing after a schema change will catch immediately.
_SCHEMA_VERSION = 1


def init_db() -> None:
    """Create every table this app owns. This is the ONE place the schema lives."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id      BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
                version INTEGER NOT NULL
            )
        """)
        row = conn.execute("SELECT version FROM schema_migrations").fetchone()
        if row and row["version"] >= _SCHEMA_VERSION:
            return

        conn.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                id          SERIAL  PRIMARY KEY,
                title       TEXT    NOT NULL,
                short_title TEXT    NOT NULL DEFAULT '',
                description TEXT    NOT NULL DEFAULT '',
                start_date  TEXT,
                end_date    TEXT,
                url         TEXT,
                ticket_url  TEXT,
                location    TEXT,
                image_url   TEXT,
                price       REAL,
                tags        TEXT    NOT NULL DEFAULT '[]',
                category    TEXT,
                source_url  TEXT    NOT NULL,
                source_type TEXT    NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        # Migration: add columns that may be missing in existing databases
        conn.execute("ALTER TABLE plans ADD COLUMN IF NOT EXISTS short_title TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE plans ADD COLUMN IF NOT EXISTS url TEXT")
        # Migration: multi-city support. Backfill existing (pre-city) rows to
        # 'Madrid' — the only city this app ever ingested before — then make
        # the column NOT NULL so every future insert must supply one.
        conn.execute("ALTER TABLE plans ADD COLUMN IF NOT EXISTS city TEXT")
        conn.execute("UPDATE plans SET city = 'Madrid' WHERE city IS NULL")
        conn.execute("ALTER TABLE plans ALTER COLUMN city SET NOT NULL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plans_city ON plans(city)")
        # Migration: semantic recommender (agora/backend/recommender/semantic.py).
        # JSON-encoded float list rather than pgvector — plan counts are small
        # enough (hundreds, not millions) that scoring in Python is plenty fast,
        # and it avoids requiring the pgvector extension on Supabase.
        conn.execute("ALTER TABLE plans ADD COLUMN IF NOT EXISTS embedding TEXT")
        # Migration: soft delete for the stale-plan cleanup cron. A flag rather
        # than a hard DELETE — keeps interaction history intact for the
        # recommender (and avoids the interactions.plan_id FK entirely) while
        # still hiding expired plans from browsing/recommendations (see the
        # "NOT is_stale" filters below and mark_stale_plans()).
        conn.execute("ALTER TABLE plans ADD COLUMN IF NOT EXISTS is_stale BOOLEAN NOT NULL DEFAULT FALSE")
        # Migration: LightGCN graph recommender (notebooks/train_lightgcn.ipynb).
        # Trained offline; the backend only ever reads this column. Plans with
        # too few interactions to train a real embedding get one synthesized
        # at export time (semantic-neighbor average — see the notebook's
        # cold-start section); NULL means neither exists yet, so callers fall
        # back to the semantic-only score for that plan.
        conn.execute("ALTER TABLE plans ADD COLUMN IF NOT EXISTS graph_embedding TEXT")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id          SERIAL  PRIMARY KEY,
                user_id     TEXT    NOT NULL,
                plan_id     INTEGER NOT NULL REFERENCES plans(id),
                interaction_type TEXT NOT NULL CHECK(interaction_type IN ('click', 'saved', 'view_link')),
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(user_id, plan_id, interaction_type)
            )
        """)
        # Migration: drop the CHECK before renaming data (Postgres validates ALL
        # existing rows when a CHECK is (re)added, so 'attendance' rows would
        # violate a constraint that no longer lists 'attendance' if renamed after).
        conn.execute("ALTER TABLE interactions DROP CONSTRAINT IF EXISTS interactions_interaction_type_check")
        conn.execute("UPDATE interactions SET interaction_type = 'saved' WHERE interaction_type = 'attendance'")
        conn.execute(
            "ALTER TABLE interactions ADD CONSTRAINT interactions_interaction_type_check "
            "CHECK (interaction_type IN ('click', 'saved', 'view_link'))"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                username      TEXT   NOT NULL UNIQUE,
                password_hash TEXT   NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        # Trained LightGCN user embeddings (notebooks/train_lightgcn.ipynb).
        # Keyed by interactions.user_id (a free-text id — anon/synthetic users
        # included, not just rows in `users`) and city, since a graph is
        # trained per-city and a user embedding is only meaningful within the
        # graph it came from. No row here means this user wasn't in the graph
        # at last training time — callers fall back to folding the user in
        # live from their interacted plans' graph_embedding, or to the
        # semantic-only score if even that's unavailable.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_embeddings (
                user_id     TEXT NOT NULL,
                city        TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (user_id, city)
            )
        """)
        # Migration: lock down Supabase's auto-generated PostgREST API, which
        # exposes every public-schema table to anyone with the project's anon
        # key regardless of whether this app uses that API (it doesn't — we
        # only ever connect directly via DATABASE_URL as the `postgres` role,
        # which has BYPASSRLS, so this changes nothing for store.py itself).
        # No policies added: nothing needs anon/authenticated REST access, so
        # RLS-enabled-with-zero-policies is a correct default-deny.
        for table in ("plans", "interactions", "users", "user_embeddings"):
            conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        # Migration: dedup key moved from (title, source_url) to
        # (title, city, start_date) — the same event scraped from many
        # different listing pages (a chain's category pages, a search
        # discovering it twice) has a different source_url every time, so the
        # old key let every one of those in as a "new" row. NULL start_date
        # (an undated plan) is normalised to a sentinel so two undated plans
        # with the same title/city still collide instead of NULL != NULL
        # silently letting them both through.
        conn.execute("ALTER TABLE plans DROP CONSTRAINT IF EXISTS plans_title_source_url_key")
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS plans_title_city_date_uniq
            ON plans (lower(btrim(title)), city, COALESCE(start_date, '0001-01-01'))
        """)

        conn.execute(
            "INSERT INTO schema_migrations (id, version) VALUES (TRUE, %s) "
            "ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version",
            (_SCHEMA_VERSION,),
        )


# ── Ingestion-side writes ────────────────────────────────

# On a (title, city, start_date) conflict we BACKFILL: keep any value the
# stored row already has, and only fill columns that are currently empty/NULL
# from the new scrape. This is what lets a re-run pick up URLs/images/short
# titles that a previous run missed, without clobbering good existing data.
# title/city/start_date themselves are never touched here — a conflict on
# this key means those three are already identical between the two rows.
_UPSERT_SQL = """
INSERT INTO plans
    (title, short_title, description, start_date, end_date,
     url, ticket_url, location, image_url, price, tags, category,
     source_url, source_type, city)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (lower(btrim(title)), city, COALESCE(start_date, '0001-01-01')) DO UPDATE SET
    short_title = CASE WHEN plans.short_title IS NULL OR plans.short_title = ''
                       THEN excluded.short_title ELSE plans.short_title END,
    description = CASE WHEN plans.description IS NULL OR plans.description = ''
                       THEN excluded.description ELSE plans.description END,
    end_date    = COALESCE(plans.end_date, excluded.end_date),
    url         = COALESCE(plans.url, excluded.url),
    ticket_url  = COALESCE(plans.ticket_url, excluded.ticket_url),
    location    = COALESCE(plans.location, excluded.location),
    image_url   = COALESCE(plans.image_url, excluded.image_url),
    price       = COALESCE(plans.price, excluded.price),
    tags        = CASE WHEN plans.tags IS NULL OR plans.tags = '' OR plans.tags = '[]'
                       THEN excluded.tags ELSE plans.tags END,
    category    = COALESCE(plans.category, excluded.category)
"""


_CANDIDATE_MATCH_COLUMNS = (
    "id, title, short_title, description, ticket_url, "
    "location, image_url, price, category, start_date, end_date, is_stale"
)


def _merge_into_existing(conn: psycopg.connection.Connection, existing: dict, p: PlanData) -> None:
    """Persist the merge decision (see plan_matching.compute_merge_updates)
    for two plans matched as the same event."""
    updates = compute_merge_updates(existing, p)
    if updates:
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        conn.execute(f"UPDATE plans SET {set_clause} WHERE id = %s", (*updates.values(), existing["id"]))


def upsert_plans(plans: list[PlanData]) -> int:
    """Insert new plans; backfill empty fields on existing ones.

    Returns the number of brand-new rows inserted (existing rows that were only
    backfilled are not counted as new).
    """
    init_db()
    inserted = 0
    with _conn() as conn:
        for p in plans:
            try:
                # Each row gets its own savepoint (conn.transaction() nested inside
                # the outer transaction): unlike SQLite, Postgres aborts the WHOLE
                # transaction on any statement error, so without this a single bad
                # row would silently kill every row after it in the batch.
                with conn.transaction():
                    # A url match catches same-event-different-title-text
                    # duplicates the title/city/date key below can't.
                    # Scheme/www-insensitive; gated on word overlap (see
                    # same_event_by_title) so a shared listing page doesn't
                    # get treated as one event.
                    existing_match = None
                    if p.url:
                        existing_match = conn.execute(
                            f"SELECT {_CANDIDATE_MATCH_COLUMNS} FROM plans "
                            r"WHERE lower(regexp_replace(url, '^https?://(www\.)?', '')) "
                            r"= lower(regexp_replace(%s, '^https?://(www\.)?', '')) AND city = %s",
                            (p.url, p.city),
                        ).fetchone()
                        if existing_match and not same_event_by_title(existing_match["title"], p.title):
                            existing_match = None

                    if not existing_match:
                        # Fuzzy/exact title match with a compatible date
                        # (equal, or one side TBA — see dates_compatible)
                        # catches "scraped once before the date was known,
                        # again after" — a NULL start_date never matches a
                        # real one via the (title, city, date) key alone.
                        p_date = p.start_date.isoformat() if p.start_date else None
                        candidates = conn.execute(
                            f"SELECT {_CANDIDATE_MATCH_COLUMNS} FROM plans "
                            "WHERE city = %s AND (start_date IS NULL OR %s IS NULL OR start_date = %s)",
                            (p.city, p_date, p_date),
                        ).fetchall()
                        for cand in candidates:
                            if fuzzy_same_event_by_title(cand["title"], p.title):
                                existing_match = cand
                                break

                    if existing_match:
                        _merge_into_existing(conn, existing_match, p)
                        existed = True
                    else:
                        existed = conn.execute(
                            "SELECT 1 FROM plans WHERE lower(btrim(title)) = lower(btrim(%s)) "
                            "AND city = %s AND COALESCE(start_date, '0001-01-01') = COALESCE(%s, '0001-01-01')",
                            (p.title, p.city, p.start_date.isoformat() if p.start_date else None),
                        ).fetchone()
                        conn.execute(
                            _UPSERT_SQL,
                            (
                                p.title,
                                p.short_title or "",
                                p.description or "",
                                p.start_date.isoformat() if p.start_date else None,
                                p.end_date.isoformat() if p.end_date else None,
                                p.url,
                                p.ticket_url,
                                p.location,
                                p.image_url,
                                p.price,
                                json.dumps(p.tags),
                                p.category,
                                p.source_url,
                                p.source_type,
                                p.city,
                            ),
                        )
                if not existed:
                    inserted += 1
            except Exception:
                continue
    return inserted


# ── Read API (used by the web layer; no SQL leaks upward) ─

def get_all_plans() -> list[dict]:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM plans ORDER BY start_date ASC NULLS LAST"
        ).fetchall()
        return [dict(r) for r in rows]


def get_plan_count() -> int:
    init_db()
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM plans").fetchone()["n"]


def list_plans(
    city: str | None = None,
    category: str | None = None,
    location: str | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    where: list[str] = [_CINEMA_EXCLUDE_WHERE, "NOT is_stale"]
    params: list = list(_CINEMA_EXCLUDE_PARAMS)

    if city:
        # Exact match — this is a clean canonical value now, unlike the
        # messy free-text `location` column below.
        where.append("city = %s")
        params.append(city)
    if category:
        placeholders = ",".join("%s" for _ in category.split(","))
        where.append(f"category IN ({placeholders})")
        params.extend(c.strip() for c in category.split(","))
    if location:
        where.append("location ILIKE %s")
        params.append(f"%{location}%")
    if search:
        where.append("(title ILIKE %s OR description ILIKE %s OR tags ILIKE %s)")
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

    sql = "SELECT * FROM plans"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY start_date ASC NULLS LAST LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with _conn() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_plan(plan_id: int) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM plans WHERE id = %s", (plan_id,)).fetchone()
    return dict(row) if row else None


def record_interaction(user_id: str, plan_id: int, interaction_type: str) -> None:
    """Insert an interaction (idempotent via UNIQUE constraint). May raise on bad input."""
    with _conn() as conn:
        conn.execute(
            """INSERT INTO interactions (user_id, plan_id, interaction_type)
               VALUES (%s, %s, %s)
               ON CONFLICT (user_id, plan_id, interaction_type) DO NOTHING""",
            (user_id, plan_id, interaction_type),
        )


def remove_interaction(user_id: str, plan_id: int, interaction_type: str) -> None:
    """Undo a 'saved' interaction so the plan drops out of get_saved_plans()."""
    with _conn() as conn:
        conn.execute(
            "DELETE FROM interactions WHERE user_id = %s AND plan_id = %s AND interaction_type = %s",
            (user_id, plan_id, interaction_type),
        )


def get_user_interactions(user_id: str) -> list[dict]:
    """Every interaction this user has recorded (any type), most recent first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT plan_id, interaction_type, created_at FROM interactions "
            "WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_interactions_with_embeddings(user_id: str) -> list[dict]:
    """Same as get_user_interactions, but with each plan's semantic AND
    graph embedding joined in (NULL if not set) — one query instead of
    separate calls, since Tier 1 needs the former and Tier 2
    (graph_recommendation.py) needs the latter on every single request (not
    cacheable, since it must reflect interactions the instant they're
    recorded)."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT i.plan_id, i.interaction_type, i.created_at, p.embedding, p.graph_embedding
               FROM interactions i JOIN plans p ON p.id = i.plan_id
               WHERE i.user_id = %s
               ORDER BY i.created_at DESC""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_saved_plans(user_id: str) -> list[dict]:
    """Plans this user has saved, most-recently-saved first — lets the Saved
    list follow the same user_id across browsers/devices instead of living
    only in that browser's localStorage."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT p.* FROM plans p
               JOIN interactions i ON i.plan_id = p.id
               WHERE i.user_id = %s AND i.interaction_type = 'saved'
               ORDER BY i.created_at DESC""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recommendations(user_id: str, city: str, limit: int = 10) -> list[dict]:
    """Popularity-ranked plans (for the given city) the user hasn't
    interacted with. Each dict carries a `score`."""
    with _conn() as conn:
        interacted = {
            r["plan_id"]
            for r in conn.execute(
                "SELECT plan_id FROM interactions WHERE user_id = %s", (user_id,)
            ).fetchall()
        }
        popular = conn.execute(
            f"""SELECT p.*, COUNT(i.id) as score
               FROM plans p
               LEFT JOIN interactions i ON i.plan_id = p.id
               WHERE {_CINEMA_EXCLUDE_WHERE} AND p.city = %s AND NOT p.is_stale
               GROUP BY p.id
               ORDER BY score DESC, p.start_date ASC NULLS LAST
               LIMIT %s""",
            (*_CINEMA_EXCLUDE_PARAMS, city, limit + len(interacted)),
        ).fetchall()

    out: list[dict] = []
    for row in popular:
        d = dict(row)
        if d["id"] in interacted:
            continue
        out.append(d)
        if len(out) >= limit:
            break
    return out


# ── Semantic recommender support (agora/backend/recommender/semantic.py) ─

def get_plans_missing_embedding() -> list[dict]:
    """Plans not yet embedded — a fresh scrape, or anything ingested before
    the embedding column existed. Safe to call repeatedly/incrementally."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, description, category, tags FROM plans WHERE embedding IS NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def set_plan_embedding(plan_id: int, embedding: list[float]) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE plans SET embedding = %s WHERE id = %s",
            (json.dumps(embedding), plan_id),
        )


# ── Category backfill support (scripts/backfill_categories.py) ──────────

def get_plans_missing_category() -> list[dict]:
    """Plans with no category — mainly older fixed-source scrapes from
    before run_fixed_pipeline threaded the source's category through (see
    cli.py's _category_from_promoted_by). Safe to call repeatedly."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, description, tags FROM plans WHERE category IS NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def set_plan_category(plan_id: int, category: str) -> None:
    """Also nulls the semantic embedding: it's computed from title +
    CATEGORY + tags + description (gemini_embeddings.plan_text), so an
    embedding computed under the old category is stale the moment this
    changes it. get_plans_missing_embedding()/backfill_embeddings() picks
    up the null and re-embeds on the next run — this is the only writer of
    `category` for an existing row, so there's no case where nulling it
    here is wrong."""
    with _conn() as conn:
        conn.execute(
            "UPDATE plans SET category = %s, embedding = NULL WHERE id = %s",
            (category, plan_id),
        )


def get_embeddings_for_plans(plan_ids: list[int]) -> dict[int, list[float]]:
    """id -> embedding, skipping any plan that hasn't been embedded yet."""
    if not plan_ids:
        return {}
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM plans WHERE id = ANY(%s) AND embedding IS NOT NULL",
            (plan_ids,),
        ).fetchall()
    return {r["id"]: json.loads(r["embedding"]) for r in rows}


def get_all_interactions_for_city(city: str) -> list[dict]:
    """Every (user_id, plan_id, interaction_type) edge for a city's plans —
    the full bipartite graph LightGCN would train on. Used by graph-structure
    diagnostics (scripts/inspect_interaction_graph.py) and the training
    notebook itself."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT i.user_id, i.plan_id, i.interaction_type
               FROM interactions i JOIN plans p ON p.id = i.plan_id
               WHERE p.city = %s""",
            (city,),
        ).fetchall()
    return [dict(r) for r in rows]


def set_plan_graph_embeddings_bulk(rows: list[tuple[int, list[float]]]) -> None:
    """Write every plan's trained (or cold-start-synthesized) graph embedding
    in ONE round trip — one UPDATE per row would mean hundreds of individual
    network round trips to Supabase (this is what made
    generate_synthetic_interactions.py's plain per-row record_interaction
    calls slow; not repeating that here)."""
    if not rows:
        return
    with _conn() as conn:
        conn.execute(
            """UPDATE plans SET graph_embedding = data.embedding
               FROM (SELECT * FROM unnest(%s::int[], %s::text[]) AS t(id, embedding)) AS data
               WHERE plans.id = data.id""",
            ([r[0] for r in rows], [json.dumps(r[1]) for r in rows]),
        )


def upsert_user_embeddings_bulk(city: str, rows: list[tuple[str, list[float]]]) -> None:
    """Write every trained user embedding for a city in ONE round trip (see
    set_plan_graph_embeddings_bulk for why bulk matters here)."""
    if not rows:
        return
    with _conn() as conn:
        conn.execute(
            """INSERT INTO user_embeddings (user_id, city, embedding, updated_at)
               SELECT * FROM unnest(%s::text[], %s::text[], %s::text[], array_fill(now(), ARRAY[%s]))
               ON CONFLICT (user_id, city) DO UPDATE
               SET embedding = EXCLUDED.embedding, updated_at = EXCLUDED.updated_at""",
            (
                [r[0] for r in rows],
                [city] * len(rows),
                [json.dumps(r[1]) for r in rows],
                len(rows),
            ),
        )


def get_plan_graph_embeddings(plan_ids: list[int]) -> dict[int, list[float]]:
    """id -> graph_embedding, skipping any plan that doesn't have one
    (never trained, or too new for the last export's cold-start pass)."""
    if not plan_ids:
        return {}
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, graph_embedding FROM plans WHERE id = ANY(%s) AND graph_embedding IS NOT NULL",
            (plan_ids,),
        ).fetchall()
    return {r["id"]: json.loads(r["graph_embedding"]) for r in rows}


def get_user_embedding(user_id: str, city: str) -> list[float] | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT embedding FROM user_embeddings WHERE user_id = %s AND city = %s",
            (user_id, city),
        ).fetchone()
    return json.loads(row["embedding"]) if row else None


def get_interaction_counts(plan_ids: list[int]) -> dict[int, int]:
    """Total interactions (any type) per plan id — used to score a cinema
    hub by its most-interacted-with movie when there's no taste profile to
    compare against (see semantic.py's popularity-fallback branch)."""
    if not plan_ids:
        return {}
    with _conn() as conn:
        rows = conn.execute(
            "SELECT plan_id, COUNT(*) AS n FROM interactions WHERE plan_id = ANY(%s) GROUP BY plan_id",
            (plan_ids,),
        ).fetchall()
    return {r["plan_id"]: r["n"] for r in rows}


def get_plans_for_scoring(city: str) -> list[dict]:
    """Every non-cinema plan in a city, embedding included (may be NULL for
    plans not backfilled yet — the recommender skips those)."""
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM plans WHERE {_CINEMA_EXCLUDE_WHERE} AND city = %s AND NOT is_stale",
            (*_CINEMA_EXCLUDE_PARAMS, city),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_city_plans(city: str) -> list[dict]:
    """Every non-stale plan row for a city in ONE query — cinema movies
    included, unlike get_plans_for_scoring. semantic.py splits the result into
    scoring candidates vs. cinema-hub buckets in Python from this single
    fetch, instead of one query for the main list plus one ILIKE query per
    cinema chain (5 extra network round-trips to Supabase — the actual source
    of a slow /recommendations call, not the similarity math)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM plans WHERE city = %s AND NOT is_stale", (city,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Cinemas (grouped movie sources) ──────────────────────
# Movies from a cinema source are excluded from list_plans/get_recommendations
# above and surfaced instead as one card per cinema (this section), which
# opens out to the cinema's own movie list.

def list_cinemas(city: str) -> list[dict]:
    """One entry per cinema chain (for the given city) with at least one plan:
    name, a representative image (soonest upcoming movie that has one), and
    how many movies it has."""
    init_db()
    out: list[dict] = []
    with _conn() as conn:
        for domain, info in CINEMA_SOURCES.items():
            if info["city"] != city:
                continue
            rows = conn.execute(
                "SELECT image_url FROM plans WHERE source_url ILIKE %s AND NOT is_stale "
                "ORDER BY start_date ASC NULLS LAST",
                (f"%{domain}%",),
            ).fetchall()
            if not rows:
                continue
            image_url = next((r["image_url"] for r in rows if r["image_url"]), None)
            out.append({"key": domain, "name": info["name"], "image_url": image_url, "movie_count": len(rows)})
    return out


def list_cinema_plans(key: str) -> list[dict]:
    """All plans for one cinema (identified by its CINEMA_SOURCES domain key),
    soonest showing first.

    Scoped to the hub's own city — some chains (Cines Renoir) have branches
    in more than one city under the same domain, tagged by their real city
    via correct_city_from_location(), so a hub reachable from one city's
    view must not also pull in another city's showings on that domain.
    """
    if key not in CINEMA_SOURCES:
        return []
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM plans WHERE source_url ILIKE %s AND city = %s AND NOT is_stale "
            "ORDER BY start_date ASC NULLS LAST",
            (f"%{key}%", CINEMA_SOURCES[key]["city"]),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_stale_plans() -> int:
    """Soft-delete plans whose run is fully over: COALESCE(end_date,
    start_date) is before today. Plans with no date at all ("DATES TBA") are
    left alone — an unknown date isn't a stale one. Flips is_stale rather than
    deleting the row, so interaction history survives for the recommender;
    every browse/recommendation-facing query filters on "NOT is_stale" to
    keep these out of the feed. Returns the number of plans newly marked."""
    init_db()
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE plans SET is_stale = TRUE "
            "WHERE NOT is_stale "
            "AND COALESCE(end_date, start_date) IS NOT NULL "
            "AND COALESCE(end_date, start_date)::date < CURRENT_DATE"
        )
        return cur.rowcount


# ── Auth (UI-only gate: username identifies the user consistently across
# browsers, no session token — see AGENTS.md discussion) ──

def authenticate_user(username: str, password: str) -> None:
    """Signup-or-login in one step: unknown usernames are created on the spot,
    known ones must match. Raises ValueError (caller maps to 401) on a wrong
    password for an existing username."""
    with _conn() as conn:
        with conn.transaction():
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = %s", (username,)
            ).fetchone()
            if row is None:
                password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                    (username, password_hash),
                )
            elif not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
                raise ValueError("Incorrect password")


# ── Maintenance (used by scripts/backfill_jsonld.py) ─────

def get_plans_missing_url() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, source_url, url, image_url, ticket_url, price, start_date, end_date, location "
            "FROM plans WHERE url IS NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def update_plan_fields(plan_id: int, updates: dict) -> None:
    """Patch a subset of PATCHABLE_PLAN_COLUMNS on one row by id."""
    if not updates:
        return
    bad = set(updates) - PATCHABLE_PLAN_COLUMNS
    if bad:
        raise ValueError(f"update_plan_fields: not patchable: {bad}")
    set_clause = ", ".join(f"{k} = %s" for k in updates)
    with _conn() as conn:
        conn.execute(
            f"UPDATE plans SET {set_clause} WHERE id = %s",
            (*updates.values(), plan_id),
        )


# ── Dedup (used by scripts/dedupe_plans.py) ──────────────

def get_plans_for_dedup(city: str) -> list[dict]:
    """Every plan row for a city (stale included), with just the columns a
    fuzzy-title dedup pass needs — no embedding columns."""
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT {_CANDIDATE_MATCH_COLUMNS} FROM plans WHERE city = %s", (city,)
        ).fetchall()
    return [dict(r) for r in rows]


def merge_duplicate_plan(keep_id: int, drop_id: int) -> None:
    """Collapse `drop_id` into `keep_id`: backfill missing fields onto
    `keep_id`, re-point `drop_id`'s interactions (skipping any that'd
    collide with the UNIQUE(user_id, plan_id, interaction_type) constraint),
    then delete `drop_id`."""
    with _conn() as conn:
        with conn.transaction():
            keep = conn.execute(
                f"SELECT {_CANDIDATE_MATCH_COLUMNS} FROM plans WHERE id = %s", (keep_id,)
            ).fetchone()
            drop = conn.execute(
                f"SELECT {_CANDIDATE_MATCH_COLUMNS} FROM plans WHERE id = %s", (drop_id,)
            ).fetchone()
            if keep is None or drop is None:
                raise ValueError(f"merge_duplicate_plan: missing row(s) {keep_id}/{drop_id}")

            updates = compute_dedup_merge_updates(dict(keep), dict(drop))

            # drop_id must be gone before the keep_id update runs — it can
            # prefer drop's title, which would collide with drop's own
            # still-existing row on the unique index.
            conn.execute(
                "UPDATE interactions SET plan_id = %s "
                "WHERE plan_id = %s AND NOT EXISTS ("
                "  SELECT 1 FROM interactions i2 WHERE i2.plan_id = %s "
                "  AND i2.user_id = interactions.user_id AND i2.interaction_type = interactions.interaction_type"
                ")",
                (keep_id, drop_id, keep_id),
            )
            # Any left on drop_id now collide with one already on keep_id — drop those.
            conn.execute("DELETE FROM interactions WHERE plan_id = %s", (drop_id,))
            conn.execute("DELETE FROM plans WHERE id = %s", (drop_id,))

            if updates:
                set_clause = ", ".join(f"{k} = %s" for k in updates)
                conn.execute(f"UPDATE plans SET {set_clause} WHERE id = %s", (*updates.values(), keep_id))


def delete_plan(plan_id: int) -> None:
    """Hard-delete one plan and its interactions — for content that doesn't
    belong at all (wrong city/region), unlike mark_stale_plans()'s soft
    delete for events whose run is just over."""
    with _conn() as conn:
        with conn.transaction():
            conn.execute("DELETE FROM interactions WHERE plan_id = %s", (plan_id,))
            conn.execute("DELETE FROM plans WHERE id = %s", (plan_id,))

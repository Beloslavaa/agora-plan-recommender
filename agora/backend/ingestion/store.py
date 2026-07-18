import json

import bcrypt
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from agora.backend.cinemas import CINEMA_SOURCES
from agora.backend.config import settings
from agora.backend.ingestion.schemas import PlanData

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


def init_db() -> None:
    """Create every table this app owns. This is the ONE place the schema lives."""
    with _conn() as conn:
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
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(title, source_url)
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


# ── Ingestion-side writes ────────────────────────────────

# On a (title, source_url) conflict we BACKFILL: keep any value the stored row
# already has, and only fill columns that are currently empty/NULL from the new
# scrape. This is what lets a re-run pick up URLs/images/short titles that a
# previous run missed, without clobbering good existing data.
_UPSERT_SQL = """
INSERT INTO plans
    (title, short_title, description, start_date, end_date,
     url, ticket_url, location, image_url, price, tags, category,
     source_url, source_type, city)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT(title, source_url) DO UPDATE SET
    short_title = CASE WHEN plans.short_title IS NULL OR plans.short_title = ''
                       THEN excluded.short_title ELSE plans.short_title END,
    description = CASE WHEN plans.description IS NULL OR plans.description = ''
                       THEN excluded.description ELSE plans.description END,
    start_date  = COALESCE(plans.start_date, excluded.start_date),
    end_date    = COALESCE(plans.end_date, excluded.end_date),
    url         = COALESCE(plans.url, excluded.url),
    ticket_url  = COALESCE(plans.ticket_url, excluded.ticket_url),
    location    = COALESCE(plans.location, excluded.location),
    image_url   = COALESCE(plans.image_url, excluded.image_url),
    price       = COALESCE(plans.price, excluded.price),
    tags        = CASE WHEN plans.tags IS NULL OR plans.tags = '' OR plans.tags = '[]'
                       THEN excluded.tags ELSE plans.tags END,
    category    = COALESCE(plans.category, excluded.category),
    city        = COALESCE(plans.city, excluded.city)
"""


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
                    existed = conn.execute(
                        "SELECT 1 FROM plans WHERE title = %s AND source_url = %s",
                        (p.title, p.source_url),
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
                if existed is None:
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
    where: list[str] = [_CINEMA_EXCLUDE_WHERE]
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
               WHERE {_CINEMA_EXCLUDE_WHERE} AND p.city = %s
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
                "SELECT image_url FROM plans WHERE source_url ILIKE %s "
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
    soonest showing first."""
    if key not in CINEMA_SOURCES:
        return []
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM plans WHERE source_url ILIKE %s ORDER BY start_date ASC NULLS LAST",
            (f"%{key}%",),
        ).fetchall()
    return [dict(r) for r in rows]


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

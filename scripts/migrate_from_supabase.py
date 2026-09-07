"""One-off (and re-runnable) migration: copy app data out of the old Supabase
database into whatever DATABASE_URL currently points at.

Written to run INSIDE the deployed container (Dokploy dashboard → the app
service → Terminal), which is the only place both endpoints are reachable
when the self-hosted Postgres has no published port and the host has no SSH:
the container can reach *out* to Supabase over the public internet, and *in*
to Postgres over Dokploy's internal network. Nothing has to be exposed.

Unlike pg_dump, this only issues plain SELECTs against the source, so
Supabase's POOLED (transaction-mode, port 6543) URI works fine — no need for
the direct connection.

Copies plans, interactions, users and user_embeddings. Deliberately NOT
schema_migrations: the target's own init_db() owns its schema version.

    # in the container's terminal, from the project root (/app)
    export SUPABASE_URL='postgresql://postgres.<ref>:<pw>@<host>:6543/postgres'
    PYTHONPATH=. python scripts/migrate_from_supabase.py --dry-run  # count both sides
    PYTHONPATH=. python scripts/migrate_from_supabase.py            # copy

Re-runnable: every table is inserted with ON CONFLICT DO NOTHING, so a second
run tops up rather than duplicating. Existing rows are never overwritten.
"""

import argparse
import logging
import os

import psycopg
from psycopg.rows import dict_row, tuple_row

from agora.backend.infrastructure.persistence.postgres_repository import init_db, pool

logger = logging.getLogger(__name__)

# Order matters: interactions.plan_id references plans(id), so plans must
# land first. user_embeddings/users are independent of both.
#
# Each entry is (table, columns, conflict_target). Columns are listed
# explicitly rather than SELECT * so a column added on one side but not the
# other fails loudly here instead of silently shifting values.
TABLES = [
    (
        "plans",
        [
            "id", "title", "short_title", "description", "start_date", "end_date",
            "url", "ticket_url", "location", "image_url", "price", "tags",
            "category", "source_url", "source_type", "city", "created_at",
            "embedding", "is_stale", "graph_embedding",
        ],
        "(id)",
    ),
    (
        "users",
        ["id", "username", "password_hash", "created_at"],
        "(id)",
    ),
    (
        "interactions",
        ["id", "user_id", "plan_id", "interaction_type", "created_at"],
        "(id)",
    ),
    (
        "user_embeddings",
        ["user_id", "city", "embedding", "updated_at"],
        "(user_id, city)",
    ),
]

BATCH = 500

# Sequences to fast-forward after copying explicit ids, so the next INSERT
# doesn't collide on a primary key already in use.
SEQUENCES = [("plans", "id"), ("users", "id"), ("interactions", "id")]


def _copy_table(src: psycopg.Connection, table: str, columns: list[str], conflict: str) -> int:
    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    copied = 0

    # A plain client-side cursor, deliberately: a server-side (named) cursor
    # spans round trips, which is exactly what Supabase's transaction-mode
    # pooler is free to reassign to a different backend. Volumes here are
    # thousands of rows, so buffering the result set is the safer trade.
    with src.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {col_list} FROM {table}")
        with pool.connection() as dst, dst.cursor() as dcur:
            while rows := cur.fetchmany(BATCH):
                dcur.executemany(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                    f"ON CONFLICT {conflict} DO NOTHING",
                    [tuple(r[c] for c in columns) for r in rows],
                )
                copied += len(rows)
                logger.info("  %s: %d rows sent", table, copied)
    return copied


def _sync_sequences() -> None:
    with pool.connection() as conn:
        for table, col in SEQUENCES:
            # setval to max(id) so the next nextval() clears every copied row.
            # Guarded with COALESCE for an empty table (setval rejects 0).
            conn.execute(
                f"SELECT setval(pg_get_serial_sequence('{table}', '{col}'), "
                f"COALESCE((SELECT MAX({col}) FROM {table}), 1))"
            )
            logger.info("  sequence for %s.%s fast-forwarded", table, col)


def _counts(conn: psycopg.Connection) -> dict[str, int]:
    out = {}
    # tuple_row explicitly: the app's pool sets dict_row as its default, while
    # a plain psycopg.connect() yields tuples — so indexing [0] is only safe
    # if the row factory is pinned here rather than inherited.
    for table, _, _ in TABLES:
        with conn.cursor(row_factory=tuple_row) as cur:
            out[table] = cur.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy app data from Supabase into DATABASE_URL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report row counts on both sides and exit without writing",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source_url = os.environ.get("SUPABASE_URL")
    if not source_url:
        raise SystemExit("SUPABASE_URL is not set — export the old Supabase connection string first.")

    init_db()  # make sure the target schema exists before copying into it

    # prepare_threshold=None: Supabase's transaction-mode pooler hands queries
    # to different backends, which breaks server-side prepared statements —
    # same reason postgres_repository's pool sets it.
    with psycopg.connect(source_url, prepare_threshold=None, connect_timeout=15) as src:
        before_src = _counts(src)
        with pool.connection() as dst:
            before_dst = _counts(dst)

        print("\nsource (Supabase)     :", before_src)
        print("target (DATABASE_URL) :", before_dst)

        if args.dry_run:
            print("\n--dry-run: nothing written.")
            pool.close()
            return

        print()
        for table, columns, conflict in TABLES:
            logger.info("copying %s ...", table)
            _copy_table(src, table, columns, conflict)

    logger.info("syncing sequences ...")
    _sync_sequences()

    with pool.connection() as dst:
        after = _counts(dst)
    print("\ntarget after migration:", after)
    print("source for comparison :", before_src)

    pool.close()


if __name__ == "__main__":
    main()

"""One-off (and re-runnable) backfill: embed every plan missing an `embedding`.

Needed once for plans ingested before the semantic recommender existed;
after that, ingestion runs (agora/backend/ingestion/cli.py) call this
automatically so newly-scraped plans stay embedded.

Run from the project root (same folder as main.py / data/):

    python scripts/backfill_embeddings.py
"""

import logging

from agora.backend.ingestion.store import init_db, pool
from agora.backend.recommender.semantic import backfill_embeddings

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()  # runs the `embedding` column migration if this DB predates it
    n = backfill_embeddings()
    print(f"Embedded {n} plan(s)")
    pool.close()


if __name__ == "__main__":
    main()

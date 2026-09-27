"""Retrain LightGCN for every city in data/cities.json — one run of
notebooks/train_lightgcn.ipynb per city (each city is its own graph: its
own users, plans and vector space), selected via the AGORA_CITY env var the
notebook reads.

One city failing (e.g. no interactions yet) never stops the others. Each
executed copy, outputs included, lands in notebooks/runs/ (gitignored) so
the recall numbers and loss curve for that city stay inspectable.

Needs the dev dependency group (torch, jupyter) — `uv sync`. Run from the
project root (same folder as main.py / data/):

    python scripts/train_all_cities.py
    python scripts/train_all_cities.py --city Barcelona
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from agora.backend.infrastructure.persistence.json_files import load_cities

NOTEBOOK = Path("notebooks/train_lightgcn.ipynb")
RUNS_DIR = Path("notebooks/runs")
ANSI = re.compile(r"\x1b\[[0-9;]*m")  # IPython colours its tracebacks


def train(city: str) -> bool:
    out_name = f"train_lightgcn_{city.lower().replace(' ', '_')}"
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=-1",
            "--output-dir", str(RUNS_DIR), "--output", out_name, str(NOTEBOOK),
        ],
        env={**os.environ, "AGORA_CITY": city},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # nbconvert's stderr ends with the failing cell's traceback — the last
        # lines carry the actual reason (e.g. the notebook's "No interactions" exit).
        print(f"[{city}] FAILED:\n" + "\n".join(ANSI.sub("", result.stderr).strip().splitlines()[-3:]))
        return False
    print(f"[{city}] done — {RUNS_DIR / out_name}.ipynb")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrain LightGCN for every configured city")
    ap.add_argument("--city", help="Train only this city instead of all of data/cities.json")
    args = ap.parse_args()

    cities = [args.city] if args.city else load_cities()
    results = {city: train(city) for city in cities}
    failed = [c for c, ok in results.items() if not ok]
    print(f"\nTrained {len(cities) - len(failed)}/{len(cities)} cities" + (f" — failed: {', '.join(failed)}" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

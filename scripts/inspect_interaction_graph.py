"""Diagnostic: inspect the structure of the bipartite user-plan interaction
graph LightGCN would actually train on for a city.

Checks the things that matter for a graph model specifically (not just
"how many rows exist"):

- degree distribution on both sides (users, plans) — LightGCN needs enough
  edges per node to propagate anything; isolated or near-isolated nodes are
  dead weight
- connected components — a graph that's secretly N disjoint islands (e.g.
  one per synthetic archetype) means the model can't propagate signal
  *across* those islands, which defeats the point of a graph over pure
  content similarity
- whether components correlate with synthetic archetype, as a sanity check
  on scripts/generate_synthetic_interactions.py's NOISE_FLOOR actually doing
  its job of creating cross-archetype ties

Run from the project root:

    PYTHONPATH=. python scripts/inspect_interaction_graph.py --city Madrid
"""

import argparse
import re
from collections import Counter, defaultdict

from agora.backend.infrastructure.persistence.postgres_repository import get_all_interactions_for_city, init_db, pool

_SYNTHETIC_USER_RE = re.compile(r"^synthetic_([a-z_]+)_\d+$")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _describe(values: list[int], label: str) -> None:
    if not values:
        print(f"  {label}: (none)")
        return
    values_sorted = sorted(values)
    n = len(values_sorted)
    mean = sum(values_sorted) / n
    median = values_sorted[n // 2]
    print(
        f"  {label}: n={n}  min={values_sorted[0]}  median={median}  "
        f"mean={mean:.1f}  max={values_sorted[-1]}"
    )


def inspect(city: str) -> None:
    init_db()
    edges = get_all_interactions_for_city(city)
    if not edges:
        print(f"No interactions found for {city}. Run the generator first.")
        pool.close()
        return

    def user_node(u: str) -> str:
        return f"u:{u}"

    def plan_node(p: int) -> str:
        return f"p:{p}"

    uf = UnionFind()
    user_degree: Counter = Counter()
    plan_degree: Counter = Counter()
    edge_types: Counter = Counter()
    seen_edges: set[tuple[str, int]] = set()

    for e in edges:
        pair = (e["user_id"], e["plan_id"])
        edge_types[e["interaction_type"]] += 1
        if pair in seen_edges:
            continue  # count each user-plan tie once for structure, even if multiple interaction types link them
        seen_edges.add(pair)
        user_degree[e["user_id"]] += 1
        plan_degree[e["plan_id"]] += 1
        uf.union(user_node(e["user_id"]), plan_node(e["plan_id"]))

    n_users = len(user_degree)
    n_plans = len(plan_degree)
    n_edges = len(seen_edges)

    print(f"\n=== Interaction graph for {city} ===")
    print(f"users: {n_users}   plans touched: {n_plans}   distinct user-plan ties: {n_edges}")
    print(f"raw interaction rows: {len(edges)}  by type: {dict(edge_types)}")
    print()
    _describe(list(user_degree.values()), "user degree (plans per user)")
    _describe(list(plan_degree.values()), "plan degree (users per plan)")

    # ── Connected components ──────────────────────────────
    components: dict[str, set[str]] = defaultdict(set)
    for node in user_degree:
        components[uf.find(user_node(node))].add(user_node(node))
    for node in plan_degree:
        components[uf.find(plan_node(node))].add(plan_node(node))

    sizes = sorted((len(members) for members in components.values()), reverse=True)
    print(f"\nconnected components: {len(components)}")
    print(f"  component sizes (top 10): {sizes[:10]}")
    if len(sizes) > 1:
        largest_share = sizes[0] / (n_users + n_plans)
        print(f"  largest component covers {largest_share:.1%} of all nodes")

    # ── Does component membership track synthetic archetype? ─────────
    archetype_by_component: dict[str, Counter] = defaultdict(Counter)
    for user_id in user_degree:
        m = _SYNTHETIC_USER_RE.match(user_id)
        archetype = m.group(1) if m else "real_or_unlabeled"
        archetype_by_component[uf.find(user_node(user_id))][archetype] += 1

    mixed = sum(1 for c in archetype_by_component.values() if len(c) > 1)
    print(
        f"\ncomponents containing more than one archetype: {mixed}/{len(archetype_by_component)} "
        "(low is bad — it means archetypes never actually connect)"
    )

    pool.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="Madrid")
    args = ap.parse_args()
    inspect(args.city)


if __name__ == "__main__":
    main()

"""Route and Evidence scorers (paper_spec §C3, solutions_v0 §2.2).

Route — under distractor pressure (all surfaces always loaded), score the set
of surfaces the agent actually used against the gold required_surfaces:

    precision = |chosen ∩ needed| / |chosen|
    recall    = |chosen ∩ needed| / |needed|
    f1        = 2PR / (P+R)          # Route score in aggregate

Route is over {rag, graph, table} only — Skill is metadata in v0.1, never a
routable surface. Leaving extra tools on the shelf is penalized via Efficiency,
not Route (solutions §2.2).

Evidence — did the agent ground its answer in the right artifacts? Scored
per surface then averaged (weighted by number of gold evidence items on that
surface), so a task with 2 table + 1 rag gold items weights table 2:1.

  rag    a gold doc/span is hit if the agent read/cited the source file
  table  a gold table is hit if the agent queried that view
  graph  a gold graph_path node is hit if the agent traversed/returned it

The scorer takes the agent's tool trace (list of {tool, args, surface, ...})
and the gold_evidence; it is trace-format-agnostic via small accessor lambdas
the runner supplies, but ships sensible defaults for the runner's own trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ROUTABLE = ("rag", "graph", "table")


@dataclass
class RouteResult:
    precision: float
    recall: float
    f1: float
    chosen: list = field(default_factory=list)
    needed: list = field(default_factory=list)


def score_route(needed_surfaces, chosen_surfaces) -> RouteResult:
    needed = {s for s in needed_surfaces if s in ROUTABLE}
    chosen = {s for s in chosen_surfaces if s in ROUTABLE}
    if not needed:
        # abstain / no-surface tasks: perfect route iff nothing was chosen
        p = r = f = 1.0 if not chosen else 0.0
        return RouteResult(p, r, f, sorted(chosen), sorted(needed))
    tp = len(needed & chosen)
    p = tp / len(chosen) if chosen else 0.0
    r = tp / len(needed)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return RouteResult(round(p, 4), round(r, 4), round(f, 4),
                       sorted(chosen), sorted(needed))


@dataclass
class EvidenceResult:
    score: float
    per_surface: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


def _rag_hit(ev, trace_files) -> bool:
    f = ev.get("file")
    if f:
        return any(f in tf or tf in f for tf in trace_files)
    # span-only evidence: credit if any doc was read (weak) — the runner can
    # tighten this by populating file.
    return bool(trace_files)


def _table_hit(ev, trace_tables) -> bool:
    t = ev.get("table")
    return t in trace_tables if t else bool(trace_tables)


def _graph_hit(ev, trace_nodes) -> bool:
    path = ev.get("graph_path") or []
    nodes = [p for p in path]
    return any(n in trace_nodes for n in nodes) if nodes else bool(trace_nodes)


def score_evidence(gold_evidence, trace) -> EvidenceResult:
    """trace: {'rag_files': set, 'tables': set, 'graph_nodes': set}."""
    trace_files = set(trace.get("rag_files", set()))
    trace_tables = set(trace.get("tables", set()))
    trace_nodes = set(trace.get("graph_nodes", set()))

    by_surface: dict[str, list[bool]] = {}
    for ev in gold_evidence:
        s = ev.get("surface")
        if s == "rag":
            hit = _rag_hit(ev, trace_files)
        elif s == "table":
            hit = _table_hit(ev, trace_tables)
        elif s == "graph":
            hit = _graph_hit(ev, trace_nodes)
        else:
            continue
        by_surface.setdefault(s, []).append(hit)

    if not by_surface:
        return EvidenceResult(0.0, {}, {"reason": "no scorable evidence"})

    per_surface = {s: sum(h) / len(h) for s, h in by_surface.items()}
    # weight each surface by its gold-item count
    total_items = sum(len(h) for h in by_surface.values())
    weighted = sum(sum(h) for h in by_surface.values()) / total_items
    return EvidenceResult(round(weighted, 4),
                          {s: round(v, 4) for s, v in per_surface.items()},
                          {"n_gold_items": total_items})

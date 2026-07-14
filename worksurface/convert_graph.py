"""Graph surface: surface_graph.json with Phase-A edge enrichment.

Starts from Workspace-Bench's per-task ``file_dep_graph`` (from/to file
edges) and enriches it with cheap, high-precision programmatic edges
(solutions_v0 §1.2 Phase A):

  mentions          basename(B) appears verbatim in text(A)
  schema_overlap    two tabular files share >= 50% normalized columns
  version_of        same stem after stripping version markers + overlap
  shared_artifact   identical content hash across >= 2 tasks (cross-profile)

Node types: file, task, output. Edges carry a ``source`` tag (wsb | phaseA)
so the runner and scorer can distinguish gold dependency edges from enriched
ones. LLM adjudication (Phase B) is deliberately out of scope for v0.1 core;
the interface leaves room for it.

The *edge-density gate* (Phase C): if the median count of non-trivial edges
per task is < 5, graph_only tasks are disabled for that profile and Graph is
kept as a distractor surface only. This number lands in Appendix A2.
"""

from __future__ import annotations

import os
import re
import statistics
from collections import defaultdict

from .common import Task, strip_hash_prefix

# Explicit version markers only. A bare trailing integer is NOT a version
# marker — "event_plan_33" and "event_plan_34" are different events, not
# versions of each other — so we require v-prefixed numbers, dates, ordinal
# words, or parenthesized copies.
VERSION_MARKER_RE = re.compile(
    r"[ _\-]*("
    r"v\d+"
    r"|final|draft|copy|revised|updated|old|new"
    r"|\(\d+\)"
    r"|\d{4}[-_]\d{2}[-_]\d{2}"
    r")$",
    re.IGNORECASE,
)


def _strip_version(stem: str) -> str:
    prev = None
    s = stem
    while prev != s:
        prev = s
        s = VERSION_MARKER_RE.sub("", s).strip()
    return s.lower()


def _read_text_safe(abspath: str, limit: int = 200_000) -> str:
    try:
        with open(abspath, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""


def build_graph(
    profile_tasks: list[Task],
    profile_dir: str,
    table_registry: dict | None = None,
    global_hash_index: dict | None = None,
) -> tuple[dict, dict]:
    """Build ``surface_graph.json`` for a profile.

    ``table_registry`` (from build_tables) supplies normalized columns for
    schema_overlap edges. ``global_hash_index`` maps content-hash ->
    [(persona, task, filename)] across ALL profiles for shared_artifact edges;
    pass the same dict across profiles to accumulate cross-profile links.

    Returns (graph, density_stats).
    """
    graph_dir = os.path.join(profile_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)

    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(nid: str, ntype: str, **attrs):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "type": ntype, **attrs}

    # columns per file node, for schema_overlap
    file_cols: dict[str, set[str]] = {}
    if table_registry:
        for meta in table_registry.values():
            fid = f"t{meta['task']}::{meta['source_file']}"
            cols = {c["name"] for c in meta["columns"]
                    if not c["name"].startswith("_source_")}
            file_cols.setdefault(fid, set()).update(cols)

    # per-task text cache for mentions edges
    for task in profile_tasks:
        task_node = f"task_{task.task_id}"
        add_node(task_node, "task", persona=task.persona,
                 difficulty=task.difficulty)

        present = [e for e in task.manifest() if e["exists"]]
        clean_names = {e["filename"]: strip_hash_prefix(e["filename"])
                       for e in present}
        file_ids = {}
        texts = {}
        for e in present:
            clean = clean_names[e["filename"]]
            fid = f"t{task.task_id}::{clean}"
            file_ids[e["filename"]] = fid
            add_node(fid, "file", filename=clean, ext=e["ext"],
                     task=task.task_id)
            edges.append({"from": task_node, "to": fid,
                          "rel": "task_requires_file", "source": "wsb"})
            if e["ext"] in {".md", ".txt", ".html", ".py", ".java", ".json"}:
                texts[fid] = _read_text_safe(e["abspath"])

        # output nodes
        for out in task.output_files:
            oid = f"t{task.task_id}::out::{out}"
            add_node(oid, "output", filename=out, task=task.task_id)
            edges.append({"from": task_node, "to": oid,
                          "rel": "task_produces_output", "source": "wsb"})

        # WSB file_dep_graph edges (from/to are clean basenames within task)
        for dep in task.dep_edges:
            f_from = file_ids.get(_match(dep["from"], clean_names))
            f_to = file_ids.get(_match(dep["to"], clean_names))
            if f_from and f_to:
                edges.append({"from": f_from, "to": f_to,
                              "rel": "depends_on", "source": "wsb"})

        # ---- Phase A enrichment, within this task ----
        items = list(file_ids.items())
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                (na, fa), (nb, fb) = items[i], items[j]
                ca, cb = clean_names[na], clean_names[nb]

                # mentions: basename(B) verbatim in text(A)
                ta = texts.get(fa, "")
                if ta and cb and cb in ta:
                    edges.append({"from": fa, "to": fb,
                                  "rel": "mentions", "source": "phaseA"})

                # schema_overlap (tabular, both directions -> add once i<j)
                if i < j:
                    sa, sb = file_cols.get(fa, set()), file_cols.get(fb, set())
                    if sa and sb:
                        jac = len(sa & sb) / len(sa | sb)
                        if jac >= 0.5:
                            edges.append({"from": fa, "to": fb,
                                          "rel": "schema_overlap",
                                          "source": "phaseA",
                                          "jaccard": round(jac, 3)})

                    # version_of: same stripped stem, different name
                    stem_a = _strip_version(os.path.splitext(ca)[0])
                    stem_b = _strip_version(os.path.splitext(cb)[0])
                    if stem_a and stem_a == stem_b and ca != cb:
                        edges.append({"from": fa, "to": fb,
                                      "rel": "version_of", "source": "phaseA"})

    # ---- cross-task / cross-profile shared_artifact edges ----
    if global_hash_index is not None:
        _register_shared_artifacts(profile_tasks, nodes, edges,
                                   global_hash_index, add_node)

    graph = {"nodes": list(nodes.values()), "edges": edges}

    # ---- Phase C: edge-density gate ----
    trivial = {"task_requires_file", "task_produces_output"}
    per_task_nontrivial = defaultdict(int)
    for e in edges:
        if e["rel"] in trivial:
            continue
        # attribute an edge to a task via its endpoint prefix
        m = re.match(r"t(\d+)::", e["from"])
        if m:
            per_task_nontrivial[m.group(1)] += 1
    counts = [per_task_nontrivial.get(t.task_id, 0) for t in profile_tasks]
    median = statistics.median(counts) if counts else 0

    density = {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_wsb_edges": sum(1 for e in edges if e["source"] == "wsb"),
        "n_phaseA_edges": sum(1 for e in edges if e["source"] == "phaseA"),
        "median_nontrivial_edges_per_task": median,
        "graph_only_eligible": median >= 5,
        "per_task_nontrivial_counts": dict(per_task_nontrivial),
    }
    return graph, density


def _match(name: str, clean_names: dict[str, str]) -> str | None:
    """Resolve a dep-graph endpoint (a clean basename) to a manifest key."""
    target = strip_hash_prefix(name)
    for k, v in clean_names.items():
        if v == target or strip_hash_prefix(k) == target:
            return k
    return None


def _register_shared_artifacts(profile_tasks, nodes, edges, index, add_node):
    import hashlib

    for task in profile_tasks:
        for e in task.manifest():
            if not e["exists"]:
                continue
            try:
                h = hashlib.sha256(open(e["abspath"], "rb").read()).hexdigest()[:16]
            except OSError:
                continue
            clean = strip_hash_prefix(e["filename"])
            index.setdefault(h, []).append((task.persona, task.task_id, clean))
    # edges added in a second pass by convert_profiles after all profiles seen

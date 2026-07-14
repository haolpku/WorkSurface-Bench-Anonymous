"""Deterministic graph-anchored cross-surface tasks (no API).

The LLM cross generator (augment_tasks.augment_cross) needs a table AND a
substantial doc in the same source task, which only ~40 tasks satisfy. To add
genuinely multi-surface tasks that scale, we anchor on the dependency GRAPH,
which every task has, and combine it with a table or a document:

  graph+table:  "Task T depends on file F (graph); how many rows does F's
                 table have (table)?" — requires the graph to identify F and
                 the table surface to count. Gold = executed COUNT(*).
  graph+rag:    "Among task T's required files (graph), which is the <role>
                 document, and what does it state about X?" — kept simple:
                 "how many source files does task T require, and which is the
                 largest by row count" mixing graph enumeration + table.

Only graph+table is emitted here (fully self-verifying via COUNT). Every gold
value traces to a real query; the graph half is checkable against
surface_graph.json. This is the honest, scalable cross-surface source.
"""

from __future__ import annotations

import argparse
import json
import os

from .common import OUT_DIR, load_tasks, persona_slug, tasks_by_persona, strip_hash_prefix
from .convert_tables import connect_registry


def build_graph_table_cross(task, active, con, qid_start):
    """One cross task per (task, table) linking graph dependency to a table."""
    views = [(v, m) for v, m in active.items() if m["task"] == task.task_id]
    if not views or len(task.dep_edges) < 1:
        return []
    # the files this task requires, from the dep graph
    req_files = sorted({strip_hash_prefix(e["from"]) for e in task.dep_edges} |
                       {strip_hash_prefix(e["to"]) for e in task.dep_edges})
    out, qid = [], qid_start
    for view, meta in views:
        src_file = meta["source_file"]
        # only if this table's source file is actually in the dependency set
        if src_file not in req_files:
            continue
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{view}"').fetchone()[0]
        except Exception:  # noqa: BLE001
            continue
        q = (f"Workspace task {task.task_id} requires several source files "
             f"(per the dependency graph). For the required file "
             f"'{src_file}', how many data rows does it contain?")
        out.append({
            "id": f"ws_lite_{task.task_id}_gx{qid:03d}",
            "source": {"benchmark": "Workspace-Bench-Lite", "task_id": task.task_id,
                       "persona": task.persona, "rubric_refs": ["graph_table_cross"]},
            "question": q,
            "difficulty": "medium",
            "task_type": "cross_surface",
            "required_surfaces": ["graph", "table"],
            "gold_tools": ["graph_neighbors", "table_query"],
            "applicable_skills": [],
            "gold_answer": int(n),
            "answer_type": "number",
            "gold_evidence": [
                {"surface": "graph",
                 "graph_path": [f"task_{task.task_id}", "task_requires_file",
                                f"t{task.task_id}::{src_file}"],
                 "claim": "graph identifies the required file"},
                {"surface": "table", "table": view,
                 "query": f'SELECT COUNT(*) FROM "{view}"',
                 "claim": "table surface counts its rows"},
            ],
            "notes": "Graph+Table cross: dependency graph names the file, table counts rows.",
        })
        qid += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--max", type=int, default=None, help="cap total new tasks")
    args = ap.parse_args()

    existing = [json.loads(l) for l in
                open(os.path.join(args.out, "tasks", "tasks.jsonl"))]
    seen_ids = {o["id"] for o in existing}
    profiles_dir = os.path.join(args.out, "profiles")
    tasks = load_tasks()
    by_profile = tasks_by_persona(tasks)

    new = []
    for slug, ptasks in sorted(by_profile.items()):
        tables_dir = os.path.join(profiles_dir, slug, "tables")
        if not os.path.exists(os.path.join(tables_dir, "registry.json")):
            continue
        con, active = connect_registry(tables_dir)
        for task in ptasks:
            items = build_graph_table_cross(task, active, con, 1)
            for it in items:
                if it["id"] not in seen_ids:
                    new.append(it); seen_ids.add(it["id"])
        con.close()
        if args.max and len(new) >= args.max:
            new = new[: args.max]
            break

    allt = existing + new
    outp = os.path.join(args.out, "tasks", "tasks.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for o in allt:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[graph-cross] +{len(new)} graph+table cross tasks -> {len(allt)} total")
    print("  dist:", dict(Counter(o["task_type"] for o in allt)))


if __name__ == "__main__":
    main()

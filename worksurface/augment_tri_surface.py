"""Experimental 3-surface task generator (disabled for the v0.1 release).

The deterministic and graph-anchored augmenters produce 2-surface combos
naturally; a 3-surface task requires deliberate construction. Given a source
task that has both dep-graph docs and DuckDB tables, we ask an LLM to compose
one question that forces all three surfaces: it must reference (a) an entity
resolvable via graph, (b) a fact in a specific document, and (c) a table
value computed via DuckDB. The LLM emits question + doc_span + SQL; we
verify:
  - doc_span occurs verbatim in a real doc under the task
  - SQL executes over the task's table registry
  - gold = executed SQL result
The pilot did not establish that all three surfaces were causally necessary,
and its graph evidence used placeholder targets. It is retained only as a
record of the rejected experiment and must not write release tasks.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

from .augment_tasks import _exec_gold, _extract_json_array, _sample_rows
from .common import (OUT_DIR, load_tasks, persona_slug, strip_hash_prefix,
                     tasks_by_persona)
from .convert_tables import connect_registry
from .llm_client import OpenAIClient

TRI_SYS = (
    "Compose ONE question that requires THREE surfaces:\n"
    "  1. RAG: a doc_fact copied verbatim from the DOCUMENT below.\n"
    "  2. GRAPH: the question must name/reference the workspace task or one "
    "of its dependency files (use the given task id).\n"
    "  3. TABLE: the answer must be computable by ONE read-only DuckDB SQL "
    "query over the TABLE below.\n"
    "Return a JSON object: "
    '{"question": "...", "doc_fact": "<verbatim span>", "sql": "SELECT ..."}. '
    "JSON only. Use the exact table name and column names given."
)


def _augment(client, active, con, kb_dir, task, qid_start, max_items=1):
    # pick a doc + a table both belonging to this task
    task_views = [(v, m) for v, m in active.items() if m["task"] == task.task_id]
    docs = sorted(glob.glob(os.path.join(kb_dir, f"t{task.task_id}__*.md")))
    if not task_views or not docs:
        return []
    doc_text, doc_name = "", None
    for d in docs:
        t = open(d, encoding="utf-8").read()
        if len(t) > 300:
            doc_text, doc_name = t[:3500], os.path.basename(d)
            break
    if not doc_name:
        return []

    out, qid = [], qid_start
    for view, _ in task_views[:2]:
        if len(out) >= max_items:
            break
        cols, rows = _sample_rows(con, view)
        if not cols:
            continue
        prompt = (f"WORKSPACE TASK ID: {task.task_id}\n\n"
                  f"TABLE {view}\ncolumns={cols}\nsample_rows={rows}\n\n"
                  f"DOCUMENT ({doc_name}):\n{doc_text}")
        try:
            raw = client.complete(TRI_SYS, prompt, max_tokens=600)
        except Exception:                        # noqa: BLE001
            continue
        # llm returns a single object or an array
        obj = None
        arr = _extract_json_array("[" + raw + "]" if raw.strip().startswith("{") else raw)
        if arr:
            obj = arr[0]
        if not obj:
            continue
        sql = obj.get("sql", "")
        fact = str(obj.get("doc_fact", "")).strip()
        q = obj.get("question", "").strip()
        if not (sql and fact and q):
            continue
        if view not in sql or fact not in doc_text:
            continue
        gold = _exec_gold(con, sql)
        if gold is None:
            continue
        answer, atype = gold
        out.append({
            "id": f"ws_lite_{task.task_id}_tri{qid:03d}",
            "source": {"benchmark": "Workspace-Bench-Lite",
                       "task_id": task.task_id, "persona": task.persona,
                       "rubric_refs": ["llm_tri_aug"]},
            "question": q,
            "difficulty": "hard",
            "task_type": "cross_surface",
            "required_surfaces": ["rag", "table", "graph"],
            "gold_tools": ["kb_search", "graph_neighbors", "table_query"],
            "applicable_skills": [],
            "gold_answer": answer,
            "answer_type": atype,
            "gold_evidence": [
                {"surface": "graph",
                 "graph_path": [f"task_{task.task_id}", "task_requires_file", ""],
                 "claim": "task/file identity resolved from graph"},
                {"surface": "rag", "file": doc_name, "span": fact,
                 "claim": "doc_fact verified verbatim in document"},
                {"surface": "table", "table": view, "query": sql,
                 "claim": "gold = executed query result"},
            ],
            "notes": "LLM-augmented tri-surface: 3-surface required, all evidence auto-verified.",
        })
        qid += 1
    return out


def main():
    raise SystemExit(
        "disabled: the tri-surface pilot failed graph-path and surface-necessity "
        "validation; rebuild it before enabling this generator"
    )


def _disabled_legacy_main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--target", type=int, default=30, help="max new tasks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    existing = [json.loads(l) for l in
                open(os.path.join(args.out, "tasks", "tasks.jsonl"))]
    seen_ids = {o["id"] for o in existing}
    profiles_dir = os.path.join(args.out, "profiles")
    src = load_tasks()
    by = tasks_by_persona(src)

    # walk personas fairly (round-robin) so distribution stays balanced
    order = []
    personas = sorted(by.keys())
    per_iters = [iter(by[p]) for p in personas]
    from itertools import zip_longest
    for row in zip_longest(*per_iters):
        for t in row:
            if t is not None:
                order.append(t)

    client = OpenAIClient()
    new = []
    for st in order:
        if len(new) >= args.target:
            break
        slug = persona_slug(st.persona)
        tables_dir = os.path.join(profiles_dir, slug, "tables")
        kb_dir = os.path.join(profiles_dir, slug, "kb_docs")
        if not os.path.exists(os.path.join(tables_dir, "registry.json")):
            continue
        con, active = connect_registry(tables_dir)
        items = _augment(client, active, con, kb_dir, st, 1, max_items=1)
        con.close()
        for it in items:
            if it["id"] in seen_ids:
                continue
            new.append(it); seen_ids.add(it["id"])
        if len(new) % 5 == 0 and new:
            print(f"  progress: {len(new)}/{args.target}  ({client.report()})")

    allt = existing + new
    outp = os.path.join(args.out, "tasks", "tasks.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for o in allt:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[tri] +{len(new)} tri-surface tasks -> {len(allt)} total")
    print(f"  {client.report()}")
    from collections import Counter as C
    print("  by persona:",
          dict(C(o["source"]["persona"] for o in new)))


if __name__ == "__main__":
    main()

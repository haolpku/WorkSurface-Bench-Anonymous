"""Quality-Control v2: strict cleanup + uniqueness assertions.

Addresses the audit findings on the 534-task set:
  1. 23 duplicate task IDs (55 tasks). Runner keyed on id, so duplicates
     silently overwrote each other's traces -- must fix or discard.
  2. 10-19 tasks with empty-string gold (LLM path emitted them from queries
     over blank/unnamed columns).
  3. 36 questions containing CJK characters in the English split.
  4. 10 questions that literally say ``unnamed_N`` (leaked internal header).
  5. LLM-generated items had no semantic sanity check.

Policy: DROP rather than repair anything ambiguous. A smaller, clean set
beats a bigger, tainted one. Also enforce a schema-level uniqueness
invariant so the same class of bugs cannot regress silently.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, OrderedDict

from .common import OUT_DIR


CJK = re.compile(r"[一-鿿]")
UNNAMED = re.compile(r"\bunnamed_\d+\b", re.I)


def _empty_gold(g) -> bool:
    """Return True if gold answer is empty / all-blank."""
    if isinstance(g, list):
        return not g or all(not str(x).strip() for x in g)
    if isinstance(g, str):
        return not g.strip()
    return False


def _clean_list_gold(g):
    """For list gold: drop empty entries; return None if the result is empty."""
    if not isinstance(g, list):
        return g
    cleaned = [x for x in g if str(x).strip()]
    return cleaned if cleaned else None


def _has_degenerate_graph_path(task) -> bool:
    """Reject graph evidence whose path is missing or contains blank nodes."""
    for ev in task.get("gold_evidence", []):
        if ev.get("surface") != "graph":
            continue
        path = ev.get("graph_path")
        if not isinstance(path, list) or len(path) < 2:
            return True
        if any(not str(node).strip() for node in path):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--schema", default=os.path.join(
        os.path.dirname(__file__), "..", "schemas", "task.schema.json"))
    args = ap.parse_args()

    path = os.path.join(args.out, "tasks", "tasks.jsonl")
    tasks = [json.loads(l) for l in open(path)]
    n0 = len(tasks)
    stats = Counter()

    # ---- 1. drop duplicate IDs (keep first) ----
    seen_ids = set()
    unique = []
    for t in tasks:
        if t["id"] in seen_ids:
            stats["dropped_duplicate_id"] += 1
            continue
        seen_ids.add(t["id"])
        unique.append(t)

    # ---- 2-4. filter tainted items ----
    kept = []
    for t in unique:
        q = t.get("question", "")
        g = t.get("gold_answer")

        if CJK.search(q):
            stats["dropped_cjk_question"] += 1
            continue
        if UNNAMED.search(q):
            stats["dropped_unnamed_column"] += 1
            continue
        if _has_degenerate_graph_path(t):
            stats["dropped_degenerate_graph_path"] += 1
            continue

        # try to salvage list gold by stripping empty entries; else drop
        if isinstance(g, list):
            cleaned = _clean_list_gold(g)
            if cleaned is None:
                stats["dropped_empty_gold"] += 1
                continue
            if cleaned != g:
                stats["salvaged_list_gold"] += 1
                t["gold_answer"] = cleaned
        elif _empty_gold(g):
            stats["dropped_empty_gold"] += 1
            continue

        kept.append(t)

    # ---- 5. schema validation ----
    import jsonschema
    schema = json.load(open(args.schema))
    v = jsonschema.Draft202012Validator(schema)
    validated = []
    for t in kept:
        errs = list(v.iter_errors(t))
        if errs:
            stats["dropped_schema"] += 1
            continue
        validated.append(t)

    # ---- 6. assertion: unique IDs (crash if we ever regress) ----
    ids = [t["id"] for t in validated]
    assert len(ids) == len(set(ids)), (
        f"REGRESSION: {len(ids)-len(set(ids))} duplicate ids after QC"
    )
    assert not any(_has_degenerate_graph_path(t) for t in validated), (
        "REGRESSION: degenerate graph_path survived QC"
    )

    # ---- 7. dedup by normalized question (from qc_tasks.py) ----
    seen_q, dedup = set(), []
    for t in validated:
        k = (str(t["source"]["task_id"]), t["task_type"],
             re.sub(r"\s+", " ", t["question"].lower().strip()))
        if k in seen_q:
            stats["dropped_duplicate_question"] += 1
            continue
        seen_q.add(k)
        dedup.append(t)

    # ---- write ----
    shutil.copy(path, path + ".preqc2")
    with open(path, "w", encoding="utf-8") as f:
        for t in dedup:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    tt = Counter(t["task_type"] for t in dedup)
    print(f"[qcv2] {n0} -> {len(dedup)} ({n0-len(dedup)} removed)")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  final task_type: {dict(tt)}")
    print(f"  backup: {path}.preqc2")


if __name__ == "__main__":
    main()

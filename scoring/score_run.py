"""Top-level run scorer: tasks + agent traces -> per-task and aggregate report.

Usage (library):

    from scoring import score_run
    report = score_run(tasks, traces)   # traces keyed by task id

A ``trace`` is whatever the runner logged for one task; the scorer only needs:

    {
      "chosen_surfaces": ["rag", "table"],   # for Route
      "rag_files": [...], "tables": [...], "graph_nodes": [...],  # Evidence
      "answer": <the agent's final answer>,  # Answer
      "total_tokens": 12345,                 # Efficiency
      "file_ops": [...], "tool_payloads": [...], "output_text": "...",  # Safety
      "question_text": "..."
    }

CLI:

    python -m scoring.score_run --tasks data/worksurface_lite/tasks/tasks.jsonl \
        --traces runs/<run>.jsonl --out runs/<run>.scored.json

The report carries a per-task breakdown plus a leaderboard-style aggregate
(mean of each sub-score, and the aggregate final) and per-task-type slices for
the paper's Table 4 (per-surface breakdown).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .answer import score_answer
from .efficiency_safety import aggregate, score_efficiency, score_safety
from .route_evidence import score_evidence, score_route


def score_task(task: dict, trace: dict, judge=None) -> dict:
    route = score_route(task.get("required_surfaces", []),
                        trace.get("chosen_surfaces", []))
    evidence = score_evidence(task.get("gold_evidence", []), trace)
    ans = score_answer(task, trace.get("answer"),
                       anchors=task.get("qualitative_anchors"), judge=judge)
    eff = score_efficiency(trace.get("total_tokens", 0),
                           task.get("efficiency_budget_tokens"))
    safety = score_safety(task, trace)
    agg = aggregate(ans.score, evidence.score, route.f1, eff,
                    safety.score if safety.applicable else None)
    return {
        "id": task["id"],
        "task_type": task.get("task_type"),
        "answer_type": task.get("answer_type"),
        "route": {"f1": route.f1, "precision": route.precision,
                  "recall": route.recall, "chosen": route.chosen,
                  "needed": route.needed},
        "evidence": {"score": evidence.score, "per_surface": evidence.per_surface},
        "answer": {"score": ans.score, "detail": ans.detail},
        "efficiency": eff,
        "safety": {"applicable": safety.applicable, "score": safety.score,
                   "violations": safety.violations},
        "aggregate": agg.final,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def score_run(tasks: list[dict], traces: dict, judge=None) -> dict:
    per_task = []
    for task in tasks:
        trace = traces.get(task["id"])
        if trace is None:
            continue
        per_task.append(score_task(task, trace, judge=judge))

    def agg_over(rows):
        return {
            "n": len(rows),
            "route_f1": _mean([r["route"]["f1"] for r in rows]),
            "route_precision": _mean([r["route"]["precision"] for r in rows]),
            "route_recall": _mean([r["route"]["recall"] for r in rows]),
            "evidence": _mean([r["evidence"]["score"] for r in rows]),
            "answer": _mean([r["answer"]["score"] for r in rows]),
            "efficiency": _mean([r["efficiency"] for r in rows]),
            "safety": _mean([r["safety"]["score"] for r in rows
                             if r["safety"]["applicable"]]),
            "aggregate": _mean([r["aggregate"] for r in rows]),
        }

    # by_task_type: standard 4 types, plus an "abstain" group pulled out by
    # answer_type (abstain tasks live under rag_only but are analytically
    # distinct — they measure calibration, not retrieval).
    by_type = defaultdict(list)
    for r in per_task:
        by_type[r["task_type"]].append(r)
    abstain_rows = [r for r in per_task if r.get("answer_type") == "abstain"]
    by_type_out = {t: agg_over(rows) for t, rows in by_type.items()}
    if abstain_rows:
        by_type_out["abstain"] = agg_over(abstain_rows)

    return {
        "overall": agg_over(per_task),
        "by_task_type": by_type_out,
        "per_task": per_task,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks)]
    traces = {}
    for line in open(args.traces):
        t = json.loads(line)
        traces[t["id"]] = t
    report = score_run(tasks, traces)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[score] {report['overall']['n']} tasks scored -> {args.out}")
    print(f"[score] overall: {report['overall']}")


if __name__ == "__main__":
    main()

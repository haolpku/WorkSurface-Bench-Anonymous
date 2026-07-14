#!/usr/bin/env python3
"""Offline robustness diagnostics for the released model traces.

Produces source-task-clustered bootstrap intervals, exact/Jaccard routing
diagnostics, unnecessary tool-call rates, and aggregate-weight sensitivity.
No model or judge API is used.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


MODELS = ["gpt-4o-mini", "deepseek-v4-pro",
          "gemini-3.1-pro-preview", "gpt-5.5"]
WEIGHT_SETS = {
    "default": {"answer": .35, "evidence": .30, "route_f1": .25,
                "efficiency": .10},
    "equal": {"answer": .25, "evidence": .25, "route_f1": .25,
              "efficiency": .25},
    "answer_heavy": {"answer": .50, "evidence": .25, "route_f1": .20,
                     "efficiency": .05},
    "route_heavy": {"answer": .30, "evidence": .25, "route_f1": .35,
                    "efficiency": .10},
}


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def clustered_ci(rows: list[dict], source_by_id: dict[str, str], metric: str,
                 n_boot: int, seed: int) -> list[float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[source_by_id[row["id"]]].append(
            row[metric] if metric != "answer" else row["answer"]["score"])
    keys = sorted(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sampled = [rng.choice(keys) for _ in keys]
        vals = [value for key in sampled for value in groups[key]]
        draws.append(sum(vals) / len(vals))
    return [round(percentile(draws, .025), 4),
            round(percentile(draws, .975), 4)]


def metric_value(row: dict, metric: str) -> float:
    if metric == "answer":
        return row["answer"]["score"]
    if metric == "evidence":
        return row["evidence"]["score"]
    if metric == "route_f1":
        return row["route"]["f1"]
    return row[metric]


def clustered_difference(rows_a: list[dict], rows_b: list[dict],
                         source_by_id: dict[str, str], metric: str,
                         n_boot: int, seed: int) -> dict:
    a = {row["id"]: row for row in rows_a}
    b = {row["id"]: row for row in rows_b}
    groups: dict[str, list[float]] = defaultdict(list)
    for task_id in sorted(a.keys() & b.keys()):
        groups[source_by_id[task_id]].append(
            metric_value(b[task_id], metric) - metric_value(a[task_id], metric))
    keys = sorted(groups)
    all_diffs = [value for values in groups.values() for value in values]
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sampled = [rng.choice(keys) for _ in keys]
        vals = [value for key in sampled for value in groups[key]]
        draws.append(sum(vals) / len(vals))
    return {"difference": round(sum(all_diffs) / len(all_diffs), 4),
            "clustered_95ci": [round(percentile(draws, .025), 4),
                               round(percentile(draws, .975), 4)]}


def route_diagnostics(rows: list[dict]) -> dict:
    exact = []
    jaccard = []
    for row in rows:
        chosen = set(row["route"]["chosen"])
        needed = set(row["route"]["needed"])
        exact.append(chosen == needed)
        union = chosen | needed
        jaccard.append(len(chosen & needed) / len(union) if union else 1.0)
    return {"exact_set_accuracy": round(sum(exact) / len(exact), 4),
            "mean_jaccard": round(sum(jaccard) / len(jaccard), 4)}


def tool_diagnostics(tasks: dict[str, dict], traces: dict[str, dict]) -> dict:
    extra_counts = []
    any_extra = []
    for task_id, trace in traces.items():
        needed = set(tasks[task_id].get("required_surfaces", []))
        extras = sum(1 for call in trace.get("tool_trace", [])
                     if call.get("surface") not in needed)
        extra_counts.append(extras)
        any_extra.append(extras > 0)
    return {
        "mean_unnecessary_calls": round(sum(extra_counts) / len(extra_counts), 4),
        "tasks_with_unnecessary_call": round(sum(any_extra) / len(any_extra), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/worksurface_lite/tasks/tasks_final_1151.jsonl")
    ap.add_argument("--runs", default="runs_final1151")
    ap.add_argument("--out", default="results/scoring_robustness.json")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    task_list = [json.loads(line) for line in open(args.tasks)]
    tasks = {task["id"]: task for task in task_list}
    source_by_id = {task["id"]: str(task["source"]["task_id"])
                    for task in task_list}
    root = Path(args.runs)
    output = {"bootstrap_unit": "source.task_id", "n_bootstrap": args.bootstrap,
              "weight_sets": WEIGHT_SETS, "runs": {}, "contrasts": {},
              "rankings": {}}

    means_by_setting: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    rows_by_run: dict[tuple[str, str], list[dict]] = {}
    for model in MODELS:
        for scored_path in sorted((root / model).glob("S*.scored.json")):
            setting = scored_path.name.split("_", 1)[0]
            trace_path = scored_path.with_name(scored_path.name.replace(".scored.json", ".jsonl"))
            report = json.load(open(scored_path))
            traces = {x["id"]: x for x in map(json.loads, open(trace_path))}
            key = f"{setting}/{model}"
            run = {
                "n": report["overall"]["n"],
                "answer": report["overall"]["answer"],
                "answer_clustered_95ci": clustered_ci(
                    report["per_task"], source_by_id, "answer", args.bootstrap,
                    seed=17),
                "aggregate": report["overall"]["aggregate"],
                "aggregate_clustered_95ci": clustered_ci(
                    report["per_task"], source_by_id, "aggregate", args.bootstrap,
                    seed=29),
                **route_diagnostics(report["per_task"]),
                **tool_diagnostics(tasks, traces),
            }
            output["runs"][key] = run
            rows_by_run[(setting, model)] = report["per_task"]
            means_by_setting[setting][model] = {
                metric: report["overall"][metric]
                for metric in ("answer", "evidence", "route_f1", "efficiency")
            }

    for model in MODELS:
        for start, end in (("S4", "S6"), ("S6", "S5")):
            if (start, model) not in rows_by_run or (end, model) not in rows_by_run:
                continue
            key = f"{start}->{end}/{model}"
            output["contrasts"][key] = {
                metric: clustered_difference(
                    rows_by_run[(start, model)], rows_by_run[(end, model)],
                    source_by_id, metric, args.bootstrap,
                    seed=101 + index,
                )
                for index, metric in enumerate(
                    ("route_f1", "evidence", "answer", "efficiency", "aggregate"))
            }

    for setting, by_model in sorted(means_by_setting.items()):
        output["rankings"][setting] = {}
        for weight_name, weights in WEIGHT_SETS.items():
            scores = {
                model: round(sum(values[k] * w for k, w in weights.items()), 4)
                for model, values in by_model.items()
            }
            output["rankings"][setting][weight_name] = sorted(
                scores.items(), key=lambda item: (-item[1], item[0]))

    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}: {len(output['runs'])} runs")


if __name__ == "__main__":
    main()

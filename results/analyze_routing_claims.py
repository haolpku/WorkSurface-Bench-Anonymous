"""Reproducible uncertainty and route-conditioned analyses for the paper.

Outputs:
  results/routing_claims_analysis.json
  results/routing_claims_analysis.md
  results/manual_audit_sample.json

The bootstrap is paired at the task level: each replicate resamples task IDs
and applies the same indices to S4 and S5.  This quantifies uncertainty across
the released task set; it is not a substitute for repeated stochastic runs.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TASKS = ROOT / "data" / "worksurface_lite" / "tasks" / "tasks.jsonl"
OUT_JSON = ROOT / "results" / "routing_claims_analysis.json"
OUT_MD = ROOT / "results" / "routing_claims_analysis.md"
AUDIT_JSON = ROOT / "results" / "manual_audit_sample.json"

SEED = 20260712
BOOTSTRAP_REPLICATES = 20_000

MODELS = [
    ("GPT-4o-mini", "gpt-4o-mini", True),
    ("DeepSeek-v3.2", "deepseek-v3.2", False),
    ("Grok-4-fast", "grok-4-fast-non-reasoning", False),
    ("Gemini-3.1-Pro", "gemini-3.1-pro-preview", True),
    ("GPT-5.5", "gpt-5.5", True),
]


def load_scored(setting: str, slug: str) -> dict[str, dict]:
    path = RUNS / f"{setting}_{slug}.scored.json"
    rows = json.loads(path.read_text())["per_task"]
    return {row["id"]: row for row in rows}


def paired_ci(differences: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = len(differences)
    # Chunk the replicates to keep memory bounded and deterministic.
    means = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        k = min(1_000, remaining)
        indices = rng.integers(0, n, size=(k, n))
        means.append(differences[indices].mean(axis=1))
        remaining -= k
    samples = np.concatenate(means)
    low, high = np.quantile(samples, [0.025, 0.975])
    return [round(float(low * 100), 2), round(float(high * 100), 2)]


def conditioned(rows: dict[str, dict]) -> dict:
    exact = [r for r in rows.values() if abs(r["route"]["f1"] - 1.0) < 1e-12]
    inexact = [r for r in rows.values() if r not in exact]

    def block(group: list[dict]) -> dict:
        return {
            "n": len(group),
            "answer_pct": round(100 * np.mean([r["answer"]["score"] for r in group]), 1)
            if group else None,
        }

    result = {"exact_route": block(exact), "inexact_route": block(inexact)}
    nonperfect = [r for r in exact if r["answer"]["score"] < 1.0]
    complete_evidence = [r for r in nonperfect if r["evidence"]["score"] == 1.0]
    result["exact_route"]["nonperfect_answer_n"] = len(nonperfect)
    result["exact_route"]["complete_evidence_nonperfect_n"] = len(complete_evidence)
    result["exact_route"]["complete_evidence_nonperfect_pct"] = (
        round(100 * len(complete_evidence) / len(nonperfect), 1) if nonperfect else None
    )
    return result


def analyze() -> dict:
    rng = np.random.default_rng(SEED)
    output = {
        "bootstrap": {
            "seed": SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "unit": "task",
            "interval": "percentile 95% paired bootstrap",
        },
        "models": [],
    }

    for display, slug, primary in MODELS:
        s4 = load_scored("S4", slug)
        s5 = load_scored("S5", slug)
        ids = sorted(set(s4) & set(s5))
        if len(ids) != 517:
            raise ValueError(f"{display}: expected 517 paired tasks, found {len(ids)}")

        route_diff = np.asarray([s5[i]["route"]["f1"] - s4[i]["route"]["f1"] for i in ids])
        answer_diff = np.asarray([s5[i]["answer"]["score"] - s4[i]["answer"]["score"] for i in ids])
        output["models"].append({
            "model": display,
            "slug": slug,
            "primary_high_coverage": primary,
            "n_paired": len(ids),
            "s4_to_s5": {
                "route_f1_delta_points": round(float(route_diff.mean() * 100), 1),
                "route_f1_ci95_points": paired_ci(route_diff, rng),
                "answer_delta_points": round(float(answer_diff.mean() * 100), 1),
                "answer_ci95_points": paired_ci(answer_diff, rng),
            },
            "route_conditioned_answer": {
                "S4": conditioned(s4),
                "S5": conditioned(s5),
            },
        })
    return output


def format_md(data: dict) -> str:
    lines = [
        "# Routing-claim uncertainty and conditioning analysis",
        "",
        f"Paired task bootstrap with {data['bootstrap']['replicates']:,} replicates "
        f"and seed {data['bootstrap']['seed']}. Intervals quantify task-sample "
        "uncertainty, not run-to-run model variance.",
        "",
        "## S4 to S5 paired differences",
        "",
        "| Model | Route F1 Δ (95% CI) | Answer Δ (95% CI) |",
        "|---|---:|---:|",
    ]
    for row in data["models"]:
        d = row["s4_to_s5"]
        mark = "*" if row["primary_high_coverage"] else ""
        lines.append(
            f"| {row['model']}{mark} | {d['route_f1_delta_points']:.1f} "
            f"[{d['route_f1_ci95_points'][0]:.2f}, {d['route_f1_ci95_points'][1]:.2f}] | "
            f"{d['answer_delta_points']:.1f} "
            f"[{d['answer_ci95_points'][0]:.2f}, {d['answer_ci95_points'][1]:.2f}] |"
        )
    lines += ["", "*Primary high-coverage backbone.", "", "## Route-conditioned Answer", "",
              "| Model | Setting | Exact route n | Answer if exact | Inexact route n | Answer if inexact |",
              "|---|---|---:|---:|---:|---:|"]
    for row in data["models"]:
        for setting in ("S4", "S5"):
            d = row["route_conditioned_answer"][setting]
            lines.append(
                f"| {row['model']} | {setting} | {d['exact_route']['n']} | "
                f"{d['exact_route']['answer_pct']:.1f} | {d['inexact_route']['n']} | "
                f"{d['inexact_route']['answer_pct']:.1f} |"
            )
    lines += ["", "## Residual errors after exact S4 routing", "",
              "| Model | Non-perfect Answer n | Complete evidence n | Complete-evidence share |",
              "|---|---:|---:|---:|"]
    for row in data["models"]:
        d = row["route_conditioned_answer"]["S4"]["exact_route"]
        lines.append(
            f"| {row['model']} | {d['nonperfect_answer_n']} | "
            f"{d['complete_evidence_nonperfect_n']} | "
            f"{d['complete_evidence_nonperfect_pct']:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def audit_sample() -> list[dict]:
    tasks = [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for task in tasks:
        combo = "+".join(sorted(task.get("required_surfaces", [])))
        groups[(task["task_type"], combo)].append(task)

    rng = random.Random(SEED)
    selected = []
    targets = {
        ("rag_only", "rag"): 20,
        ("table_only", "table"): 20,
        ("graph_only", "graph"): 20,
        ("cross_surface", "graph+table"): 17,
        ("cross_surface", "graph+rag"): 16,
        ("cross_surface", "rag+table"): 7,  # audit the entire rare subgroup
    }
    for key, count in targets.items():
        pool = sorted(groups[key], key=lambda x: x["id"])
        picked = pool if count == len(pool) else rng.sample(pool, count)
        selected.extend(picked)

    selected.sort(key=lambda x: (x["task_type"], x["id"]))
    if len(selected) != 100 or len({x["id"] for x in selected}) != 100:
        raise AssertionError("audit sample must contain 100 unique tasks")
    return selected


def main() -> None:
    data = analyze()
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    OUT_MD.write_text(format_md(data))
    AUDIT_JSON.write_text(json.dumps(audit_sample(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"wrote {AUDIT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Paired analysis of explicit-graph-cue removal on 61 RAG+Graph tasks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring.score_run import score_task


TASKS = ROOT / "data/worksurface_lite/tasks/tasks_rag_graph_no_cue.jsonl"
MODELS = ["gpt-4o-mini", "gemini-3.1-pro-preview", "gpt-5.5"]
SETTINGS = ["S3", "S4"]
METRICS = ["route_f1", "evidence", "answer"]
SEED = 20260712
N_BOOT = 20_000


def load_jsonl(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def values(task: dict, trace: dict) -> dict[str, float]:
    scored = score_task(task, trace)
    return {
        "route_f1": scored["route"]["f1"],
        "evidence": scored["evidence"]["score"],
        "answer": scored["answer"]["score"],
    }


def main() -> None:
    tasks = list(map(json.loads, TASKS.read_text().splitlines()))
    rng = np.random.default_rng(SEED)
    output = {"n_tasks": len(tasks), "seed": SEED, "bootstrap_replicates": N_BOOT, "rows": []}
    for model in MODELS:
        for setting in SETTINGS:
            original = load_jsonl(
                ROOT / "runs_ablations/graph_cue_control" / model / f"{setting}_{model}.jsonl"
            )
            ablated = load_jsonl(ROOT / "runs_ablations/no_graph_cue" / model / f"{setting}_{model}.jsonl")
            paired = {metric: [] for metric in METRICS}
            original_means = {metric: [] for metric in METRICS}
            ablated_means = {metric: [] for metric in METRICS}
            for task in tasks:
                before = values(task, original[task["id"]])
                after = values(task, ablated[task["id"]])
                for metric in METRICS:
                    original_means[metric].append(before[metric])
                    ablated_means[metric].append(after[metric])
                    paired[metric].append(after[metric] - before[metric])
            row = {"model": model, "setting": setting, "n": len(tasks), "metrics": {}}
            indices = rng.integers(0, len(tasks), size=(N_BOOT, len(tasks)))
            for metric in METRICS:
                delta = np.asarray(paired[metric])
                boot = delta[indices].mean(axis=1)
                row["metrics"][metric] = {
                    "original": float(np.mean(original_means[metric])),
                    "no_cue": float(np.mean(ablated_means[metric])),
                    "delta": float(np.mean(delta)),
                    "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                }
            output["rows"].append(row)

    json_path = ROOT / "results/cue_ablation_analysis.json"
    md_path = ROOT / "results/cue_ablation_analysis.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = ["# Explicit Graph-Cue Ablation", "", "No-cue minus original; 95% paired task-bootstrap CI.", ""]
    lines.append("| Model | Setting | Route F1 | Evidence | Answer |")
    lines.append("|---|---|---:|---:|---:|")
    for row in output["rows"]:
        cells = []
        for metric in METRICS:
            item = row["metrics"][metric]
            cells.append(
                f"{100*item['original']:.1f}→{100*item['no_cue']:.1f} "
                f"(Δ {100*item['delta']:+.1f} [{100*item['ci95'][0]:+.1f}, {100*item['ci95'][1]:+.1f}])"
            )
        lines.append(f"| {row['model']} | {row['setting']} | " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()

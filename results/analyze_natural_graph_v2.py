#!/usr/bin/env python3
"""Compare completed natural-graph v2 runs with the current 517-task runs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "runs_natural_graph_v2"
OLD = ROOT / "runs"
OUT = ROOT / "results/natural_graph_v2_baseline_comparison.json"
MODELS = ["gpt-4o-mini", "gpt-5.5", "gemini-3.1-pro-preview"]
SETTINGS = ["S1", "S2", "S3", "S4", "S5"]


def load(path: Path):
    return json.loads(path.read_text())["overall"] if path.exists() else None


def main() -> None:
    rows = []
    for model in MODELS:
        safe = model.replace("/", "-").replace(":", "-")
        for setting in SETTINGS:
            new = load(NEW / model / f"{setting}_{safe}.scored.json")
            old = load(OLD / f"{setting}_{safe}.scored.json")
            row = {"model": model, "setting": setting, "old": old, "new": new}
            if old and new:
                row["delta"] = {
                    metric: new.get(metric, 0) - old.get(metric, 0)
                    for metric in ["route_f1", "evidence", "answer", "aggregate"]
                }
            rows.append(row)
    payload = {
        "complete": all(row["new"] is not None for row in rows),
        "rows": rows,
        "note": "Do not update manuscript headline results until complete=true and the human re-audit is returned.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

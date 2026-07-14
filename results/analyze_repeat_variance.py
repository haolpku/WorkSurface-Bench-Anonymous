"""Summarize three GPT-4o-mini benchmark runs as mean and sample SD."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("canonical", ROOT / "runs"),
    ("repeat2", ROOT / "runs_repeats/gpt4o_repeat2"),
    ("repeat3", ROOT / "runs_repeats/gpt4o_repeat3"),
]
SETTINGS = ["S1", "S2", "S3", "S4", "S5"]
METRICS = ["route_f1", "evidence", "answer", "efficiency", "aggregate"]


def main() -> None:
    output: dict = {"model": "gpt-4o-mini", "n_runs": 3, "settings": {}}
    for setting in SETTINGS:
        rows = []
        for name, directory in RUNS:
            path = directory / f"{setting}_gpt-4o-mini.scored.json"
            overall = json.loads(path.read_text())["overall"]
            rows.append({"run": name, **{metric: overall[metric] for metric in METRICS}})
        summary = {}
        for metric in METRICS:
            values = [row[metric] for row in rows]
            summary[metric] = {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
                "values": values,
            }
        output["settings"][setting] = {"runs": rows, "summary": summary}

    json_path = ROOT / "results/repeat_variance_analysis.json"
    md_path = ROOT / "results/repeat_variance_analysis.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = ["# GPT-4o-mini Three-Run Variance", "", "Values are percentages; SD is the sample SD across three runs.", ""]
    lines.append("| Setting | Route F1 | Evidence | Answer | Efficiency | Aggregate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for setting in SETTINGS:
        cells = []
        for metric in METRICS:
            item = output["settings"][setting]["summary"][metric]
            cells.append(f"{100*item['mean']:.2f} ± {100*item['sample_sd']:.2f}")
        lines.append(f"| {setting} | " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()

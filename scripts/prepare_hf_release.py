#!/usr/bin/env python3
"""Build the public Hugging Face release for WorkSurface-Bench.

The release contains the frozen benchmark, canonical surface resources,
anonymized human-audit votes, normalized per-trajectory scores, and the raw
model trajectories used in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd


MODELS = (
    "gpt-4o-mini",
    "deepseek-v4-pro",
    "gemini-3.1-pro-preview",
    "gpt-5.5",
)
SETTINGS = ("S1", "S2", "S3", "S4", "S5", "S6")
AUDIT_COLUMNS = (
    "Question natural?",
    "Answerable from evidence?",
    "Required surfaces necessary?",
    "Gold answer correct?",
    "Atomic and unambiguous?",
    "Leakage cue",
)
PASS_VALUE = {
    "Question natural?": "Yes",
    "Answerable from evidence?": "Yes",
    "Required surfaces necessary?": "Yes",
    "Gold answer correct?": "Yes",
    "Atomic and unambiguous?": "Yes",
    "Leakage cue": "None",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def prepare_tasks(project: Path, release: Path) -> tuple[list[dict], dict]:
    source = project / "data/worksurface_lite/tasks/tasks_final_1151.jsonl"
    tasks = read_jsonl(source)
    if len(tasks) != 1151:
        raise ValueError(f"Expected 1,151 tasks, found {len(tasks)}")
    ids = [row["id"] for row in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Task IDs are not unique")

    task_dir = release / "data"
    task_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, task_dir / "tasks.jsonl")

    flat_rows = []
    for task in tasks:
        source_meta = task.get("source", {})
        quality = task.get("quality_screen", {})
        if not isinstance(quality, dict):
            quality = {}
        flat_rows.append(
            {
                "id": task["id"],
                "source_benchmark": source_meta.get("benchmark"),
                "source_task_id": str(source_meta.get("task_id", "")),
                "persona": source_meta.get("persona"),
                "question": task["question"],
                "difficulty": task["difficulty"],
                "task_type": task["task_type"],
                "surface_combo": "+".join(task["required_surfaces"]),
                "required_surfaces": json_text(task["required_surfaces"]),
                "gold_tools": json_text(task.get("gold_tools", [])),
                "gold_answer": json_text(task.get("gold_answer")),
                "answer_type": task["answer_type"],
                "gold_evidence": json_text(task.get("gold_evidence", [])),
                "applicable_skills": json_text(task.get("applicable_skills", [])),
                "rubric_refs": json_text(source_meta.get("rubric_refs", [])),
                "quality_screen_votes": quality.get("strict_pass_votes"),
                "quality_screen_models": json_text(quality.get("models", [])),
                "efficiency_budget_tokens": task.get("efficiency_budget_tokens"),
            }
        )
    pd.DataFrame(flat_rows).to_parquet(task_dir / "tasks.parquet", index=False)

    stats = {
        "n_tasks": len(tasks),
        "n_source_tasks": len({str(t.get("source", {}).get("task_id")) for t in tasks}),
        "task_type": dict(sorted(Counter(t["task_type"] for t in tasks).items())),
        "surface_combo": dict(
            sorted(Counter("+".join(t["required_surfaces"]) for t in tasks).items())
        ),
        "difficulty": dict(sorted(Counter(t["difficulty"] for t in tasks).items())),
        "persona": dict(
            sorted(Counter(t["source"]["persona"] for t in tasks).items())
        ),
        "answer_type": dict(sorted(Counter(t["answer_type"] for t in tasks).items())),
    }
    return tasks, stats


def prepare_resources(project: Path, release: Path) -> None:
    source = project / "data/worksurface_lite"
    target = release / "resources"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "profiles", target / "profiles")
    shutil.copytree(source / "skills", target / "skills")
    for filename in ("manifest.json", "gates.json", "skills_meta.json", "task_skill_map.json"):
        shutil.copy2(source / filename, target / filename)
    shutil.copy2(project / "data/wsb_lock.json", target / "wsb_lock.json")
    (release / "schema").mkdir(parents=True, exist_ok=True)
    shutil.copy2(project / "schemas/task.schema.json", release / "schema/task.schema.json")


def prepare_runs(project: Path, release: Path, task_ids: set[str]) -> dict:
    score_rows = []
    run_summaries = []
    for model in MODELS:
        for setting in SETTINGS:
            raw_path = project / "runs_final1151" / model / f"{setting}_{model}.jsonl"
            scored_path = (
                project / "runs_final1151" / model / f"{setting}_{model}.scored.json"
            )
            raw_rows = read_jsonl(raw_path)
            scored = json.loads(scored_path.read_text(encoding="utf-8"))
            per_task = scored["per_task"]
            if len(raw_rows) != 1151 or len(per_task) != 1151:
                raise ValueError(f"Incomplete run: {model}/{setting}")
            raw_by_id = {row["id"]: row for row in raw_rows}
            scored_by_id = {row["id"]: row for row in per_task}
            if set(raw_by_id) != task_ids or set(scored_by_id) != task_ids:
                raise ValueError(f"Task IDs differ in {model}/{setting}")
            error_count = sum(bool(row.get("error")) for row in raw_rows)
            if error_count:
                raise ValueError(f"Run {model}/{setting} contains {error_count} errors")

            raw_target = release / "trajectories" / model
            scored_target = release / "results/scored_reports" / model
            raw_target.mkdir(parents=True, exist_ok=True)
            scored_target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_path, raw_target / f"{setting}.jsonl")
            shutil.copy2(scored_path, scored_target / f"{setting}.json")

            for task_id in sorted(task_ids):
                raw = raw_by_id[task_id]
                score = scored_by_id[task_id]
                score_rows.append(
                    {
                        "task_id": task_id,
                        "model": model,
                        "setting": setting,
                        "task_type": score["task_type"],
                        "answer_type": score["answer_type"],
                        "route_f1": score["route"]["f1"],
                        "route_precision": score["route"]["precision"],
                        "route_recall": score["route"]["recall"],
                        "evidence": score["evidence"]["score"],
                        "answer": score["answer"]["score"],
                        "efficiency": score["efficiency"],
                        "aggregate": score["aggregate"],
                        "total_tokens": raw.get("total_tokens"),
                        "tool_calls": len(raw.get("tool_trace", [])),
                        "chosen_surfaces": json_text(score["route"].get("chosen", [])),
                        "needed_surfaces": json_text(score["route"].get("needed", [])),
                    }
                )
            run_summaries.append(
                {
                    "model": model,
                    "setting": setting,
                    "n": scored["overall"]["n"],
                    "errors": error_count,
                    **{key: scored["overall"].get(key) for key in (
                        "route_f1", "route_precision", "route_recall", "evidence",
                        "answer", "efficiency", "safety", "aggregate"
                    )},
                }
            )

    if len(score_rows) != 27624:
        raise ValueError(f"Expected 27,624 trajectory scores, found {len(score_rows)}")
    result_dir = release / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).to_parquet(result_dir / "trajectory_scores.parquet", index=False)
    with (result_dir / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_summaries[0]))
        writer.writeheader()
        writer.writerows(run_summaries)
    return {
        "n_models": len(MODELS),
        "n_settings": len(SETTINGS),
        "n_runs": len(run_summaries),
        "n_trajectories": len(score_rows),
        "n_errors": 0,
        "runs": run_summaries,
    }


def majority(values: list[str]) -> str:
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    if count < 2:
        raise ValueError(f"No majority among {values}")
    return value


def prepare_audits(project: Path, annotations: Path, release: Path) -> dict:
    frames = []
    for annotator_id in (1, 2, 3):
        path = annotations / (
            f"WorkSurface-Bench_final_1151_human_audit_200_annotator_{annotator_id}.xlsx"
        )
        frame = pd.read_excel(path, sheet_name="Audit", keep_default_na=False)
        if len(frame) != 200:
            raise ValueError(f"Expected 200 rows in {path}, found {len(frame)}")
        required = {"Sample #", "Task ID", "Task type", "Surface combo", "Persona", *AUDIT_COLUMNS}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing audit columns in {path}")
        public = frame[[
            "Sample #", "Task ID", "Task type", "Surface combo", "Persona", *AUDIT_COLUMNS
        ]].copy()
        public.insert(2, "Annotator ID", f"A{annotator_id}")
        frames.append(public)

    votes = pd.concat(frames, ignore_index=True)
    if len(votes) != 600:
        raise ValueError("Expected 600 individual audit rows")
    audit_dir = release / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    votes.to_csv(audit_dir / "human_audit_votes.csv", index=False)

    majority_rows = []
    disagreement_rows = []
    grouped = votes.groupby(["Sample #", "Task ID", "Task type", "Surface combo", "Persona"], sort=True)
    for group_key, group in grouped:
        row = dict(zip(("Sample #", "Task ID", "Task type", "Surface combo", "Persona"), group_key))
        strict_pass = True
        for dimension in AUDIT_COLUMNS:
            values = [str(value) for value in group[dimension].tolist()]
            result = majority(values)
            row[dimension] = result
            strict_pass &= result == PASS_VALUE[dimension]
            if len(set(values)) > 1:
                disagreement_rows.append(
                    {
                        "Sample #": row["Sample #"],
                        "Task ID": row["Task ID"],
                        "Surface combo": row["Surface combo"],
                        "Dimension": dimension,
                        "Annotator 1": values[0],
                        "Annotator 2": values[1],
                        "Annotator 3": values[2],
                        "Majority": result,
                    }
                )
        row["Strict pass"] = "Yes" if strict_pass else "No"
        majority_rows.append(row)

    majority_frame = pd.DataFrame(majority_rows)
    disagreement_frame = pd.DataFrame(disagreement_rows)
    majority_frame.to_csv(audit_dir / "human_audit_majority.csv", index=False)
    majority_frame.to_parquet(audit_dir / "human_audit_majority.parquet", index=False)
    disagreement_frame.to_csv(audit_dir / "human_audit_disagreements.csv", index=False)

    unanimous_all = 0
    dimension_stats = {}
    for dimension in AUDIT_COLUMNS:
        unanimous = 0
        majority_pass = 0
        pairwise_agree = 0
        pairwise_total = 0
        for _, group in votes.groupby("Task ID", sort=True):
            values = [str(value) for value in group[dimension].tolist()]
            unanimous += len(set(values)) == 1
            majority_pass += majority(values) == PASS_VALUE[dimension]
            pairwise_agree += sum(values[i] == values[j] for i, j in ((0, 1), (0, 2), (1, 2)))
            pairwise_total += 3
        dimension_stats[dimension] = {
            "majority_pass": majority_pass,
            "unanimous": unanimous,
            "pairwise_agreement": round(pairwise_agree / pairwise_total, 6),
        }
    for _, group in votes.groupby("Task ID", sort=True):
        unanimous_all += all(len(set(group[dimension].astype(str))) == 1 for dimension in AUDIT_COLUMNS)

    summary = {
        "n_annotators": 3,
        "n_sampled_tasks": len(majority_frame),
        "n_individual_ratings": len(votes),
        "strict_majority_pass": int((majority_frame["Strict pass"] == "Yes").sum()),
        "fully_unanimous_across_all_dimensions": unanimous_all,
        "dimensions": dimension_stats,
        "surface_sample": dict(sorted(Counter(majority_frame["Surface combo"]).items())),
    }
    write_json(audit_dir / "human_audit_summary.json", summary)

    expected = pd.read_excel(
        project / "output/audits/WorkSurface-Bench_final_1151_human_audit_results.xlsx",
        sheet_name="Majority",
        keep_default_na=False,
    )
    check_columns = ["Task ID", *AUDIT_COLUMNS, "Strict pass"]
    expected_check = expected[check_columns].sort_values("Task ID").reset_index(drop=True)
    actual_check = majority_frame[check_columns].sort_values("Task ID").reset_index(drop=True)
    if not expected_check.equals(actual_check):
        raise ValueError("Recomputed human-audit majorities differ from the paper workbook")
    return summary


def write_release_docs(release: Path, task_stats: dict, run_stats: dict, audit_stats: dict) -> None:
    write_json(
        release / "release_manifest.json",
        {"version": "1.0.0", "tasks": task_stats, "experiments": run_stats, "human_audit": audit_stats},
    )
    (release / "LICENSE_DATA").write_text(
        "WorkSurface-Bench derived data are released under CC BY 4.0.\n"
        "See https://creativecommons.org/licenses/by/4.0/\n\n"
        "The canonical resources are derived from Workspace-Bench-Lite; users must also\n"
        "follow the upstream dataset's license and terms.\n",
        encoding="utf-8",
    )
    (release / "audits/README.md").write_text(
        "# Human audit\n\n"
        "Three annotators independently reviewed the same stratified sample of 200 tasks.\n"
        "`human_audit_votes.csv` contains anonymized item-level votes; free-text notes and\n"
        "reviewer-identifying fields are intentionally excluded. `human_audit_majority.csv`\n"
        "contains 2-of-3 decisions, and `human_audit_disagreements.csv` contains every\n"
        "non-unanimous dimension.\n",
        encoding="utf-8",
    )
    readme = f"""---
license: cc-by-4.0
language:
- en
task_categories:
- question-answering
tags:
- agents
- benchmark
- rag
- tables
- knowledge-graphs
- tool-use
pretty_name: WorkSurface-Bench
size_categories:
- 1K<n<10K
configs:
- config_name: tasks
  default: true
  data_files:
  - split: test
    path: data/tasks.parquet
- config_name: scores
  data_files:
  - split: test
    path: results/trajectory_scores.parquet
- config_name: human_audit
  data_files:
  - split: test
    path: audits/human_audit_majority.parquet
---

# WorkSurface-Bench

WorkSurface-Bench evaluates whether enterprise agents can route questions across
document retrieval (RAG), structured tables, and dependency graphs, acquire the
right evidence, and produce correct answers.

## Release at a glance

- **1,151** atomic tasks derived from **100** Workspace-Bench-Lite source tasks
- **5** persona-scoped workspaces
- **{task_stats['task_type']['cross_surface']}** cross-surface, **{task_stats['task_type']['table_only']}** table-only, **{task_stats['task_type']['rag_only']}** RAG-only, and **{task_stats['task_type']['graph_only']}** graph-only tasks
- Cross-surface composition: **{task_stats['surface_combo']['rag+graph']} RAG+Graph**, **{task_stats['surface_combo']['graph+table']} Graph+Table**, **{task_stats['surface_combo']['rag+table']} RAG+Table**, and **{task_stats['surface_combo']['rag+graph+table']} RAG+Graph+Table**
- **27,624** retained trajectories: 4 models × 6 settings × 1,151 tasks, with zero protocol errors
- Human audit: 3 annotators on a stratified 200-task sample; all 200 pass all six criteria by majority vote, and 192 are unanimous across all criteria

## Repository structure

```text
data/                  Viewer-friendly Parquet and complete task JSONL
resources/profiles/    Canonical KB documents, table Parquet files, and graphs
resources/skills/      Skill metadata used by the benchmark
schema/                Task JSON schema
trajectories/          Raw model trajectories for all 24 official runs
results/               Normalized trajectory scores and full scored reports
audits/                Anonymized human votes, majorities, and disagreements
release_manifest.json  Machine-readable counts and run summaries
```

The Parquet task view stores nested annotations as JSON strings for stable
cross-tool loading. `data/tasks.jsonl` preserves the complete native objects.

## Loading

```python
from datasets import load_dataset

tasks = load_dataset("parquet", data_files="data/tasks.parquet", split="train")
scores = load_dataset("parquet", data_files="results/trajectory_scores.parquet", split="train")
audit = load_dataset("parquet", data_files="audits/human_audit_majority.parquet", split="train")
```

## Evaluation settings

- **S1 — Closed book:** no knowledge-surface tools.
- **S2 — Always RAG:** document retrieval is the only exposed surface.
- **S3 — Single-surface routing:** the model selects one surface before answering.
- **S4 — All tools:** RAG, table, and graph tools are exposed to a ReAct agent.
- **S5 — Gold-constrained:** gold surface labels are supplied and only the required surface tools are exposed.
- **S6 — Gold-hint/all:** the same labels are supplied while all surface tools remain exposed, isolating the informational intervention from tool removal.

The aggregate score is `0.35 Answer + 0.30 Evidence + 0.25 Route + 0.10 Efficiency`.

## Human evaluation

The 200-task sample is stratified by surface combination: 30 Graph, 40 Table,
30 RAG, 25 Graph+Table, 35 RAG+Graph, 25 RAG+Table, and all 15 three-surface
tasks. Annotators independently judged question naturalness, evidence
answerability, surface necessity, gold-answer correctness, atomicity, and
surface leakage. The release includes every anonymized vote and disagreement.

## Data provenance and limitations

WorkSurface-Bench projects the English Workspace-Bench-Lite release into
canonical document, table, and graph surfaces. Workspace-Bench uses a hybrid
construction process: task scenarios and dependency annotations are human
curated, while workspace files combine public resources with grounded generated
artifacts. The 1,151 benchmark items are atomic derivatives rather than 1,151
independent source workspaces. Distributional asymmetries are documented in the
paper and `release_manifest.json`; in particular, only 15 tasks require all
three surfaces.

The benchmark is intended for evaluation, not for training or safety-critical
deployment decisions.

## Code and citation

Evaluation code is included in the accompanying anonymous repository.

Please cite the WorkSurface-Bench paper when available. WorkSurface-Bench is
derived from [Workspace-Bench 1.0](https://arxiv.org/abs/2605.03596).

## License

Derived benchmark data are released under CC BY 4.0. Code is released separately
under Apache 2.0. Canonical resources inherit applicable Workspace-Bench-Lite
licensing and attribution requirements; see `LICENSE_DATA` and the upstream
dataset.
"""
    (release / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()

    release = args.release.resolve()
    release.mkdir(parents=True, exist_ok=True)
    for dirname in ("data", "resources", "schema", "trajectories", "results", "audits"):
        target = release / dirname
        if target.exists():
            shutil.rmtree(target)
    for filename in ("README.md", "LICENSE_DATA", "release_manifest.json"):
        target = release / filename
        if target.exists():
            target.unlink()

    tasks, task_stats = prepare_tasks(args.project.resolve(), release)
    prepare_resources(args.project.resolve(), release)
    run_stats = prepare_runs(args.project.resolve(), release, {task["id"] for task in tasks})
    audit_stats = prepare_audits(args.project.resolve(), args.annotations.resolve(), release)
    write_release_docs(release, task_stats, run_stats, audit_stats)
    print(json.dumps({"tasks": task_stats, "experiments": {
        key: run_stats[key] for key in ("n_models", "n_settings", "n_runs", "n_trajectories", "n_errors")
    }, "human_audit": audit_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

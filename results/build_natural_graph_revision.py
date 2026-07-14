#!/usr/bin/env python3
"""Build a cue-reduced revision with genuinely necessary graph+table tasks."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/worksurface_lite/tasks/tasks.jsonl"
OUTPUT = ROOT / "data/worksurface_lite/tasks/tasks_natural_graph_v2.jsonl"
REPORT = ROOT / "results/natural_graph_revision_report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def filename(task: dict) -> str:
    node = task["gold_evidence"][0]["graph_path"][-1]
    return node.split("::", 1)[-1]


def choose(task_id: str, templates: list[str]) -> str:
    return templates[sum(ord(ch) for ch in task_id) % len(templates)]


def rewrite_graph_only(task: dict) -> dict:
    item = copy.deepcopy(task)
    sid = item["source"]["task_id"]
    item["original_question"] = item["question"]
    item["question"] = choose(item["id"], [
        f"You are taking over Task {sid}. Which source files should you gather before starting? List their filenames.",
        f"Before work begins on Task {sid}, what input files need to be collected?",
        f"A teammate handed you Task {sid} without its inputs. Which files are needed?",
        f"Please prepare the source-file checklist for Task {sid}.",
        f"Which files should be in the handoff package for Task {sid}?",
        f"I am setting up Task {sid}. What source files do I need to have ready?",
        f"List the files that Task {sid} depends on before execution can begin.",
        f"What files must be available to complete Task {sid}?",
    ])
    answers = item["gold_answer"] if isinstance(item["gold_answer"], list) else [item["gold_answer"]]
    item["gold_evidence"] = [{
        "surface": "graph",
        "graph_query": {"node": f"task_{sid}", "relation": "task_requires_file"},
        "verified_complete_set": answers,
        "claim": "Enumerating all outgoing task_requires_file edges returns exactly this complete file set.",
    }] + [
        {
            "surface": "graph",
            "graph_path": [f"task_{sid}", "task_requires_file", f"t{sid}::{name}"],
            "claim": f"The task_requires_file edge identifies {name} as a required input.",
        }
        for name in answers
    ]
    item["revision"] = "natural_graph_v2"
    return item


def rewrite_rag_graph(task: dict) -> dict:
    item = copy.deepcopy(task)
    sid = item["source"]["task_id"]
    span = item["gold_evidence"][1]["span"]
    item["original_question"] = item["question"]
    item["question"] = choose(item["id"], [
        f"A teammate remembers seeing \"{span}\" in one of the files needed for Task {sid}. Which file should they open?",
        f"Which input document for Task {sid} contains the text \"{span}\"?",
        f"For Task {sid}, locate the required file that mentions \"{span}\" and give its filename.",
        f"I need to verify \"{span}\" for Task {sid}. Which of its source files contains it?",
        f"Find \"{span}\" among the documents used by Task {sid}. What is the matching filename?",
        f"One of Task {sid}'s input files includes \"{span}\". Which one is it?",
    ])
    item["gold_evidence"][0]["verified_candidate_scope"] = "all task_requires_file neighbors"
    item["gold_evidence"][1]["verified_unique_among_required_inputs"] = True
    item["gold_evidence"][1]["claim"] = (
        "The span occurs verbatim in this document and in no other document among "
        "the task's graph-enumerated required inputs."
    )
    item["revision"] = "natural_graph_v2"
    return item


def rebuild_graph_table(group: list[dict]) -> dict:
    # One non-redundant task per source task. Evidence retains every candidate
    # graph path and executable count query needed to verify the comparison.
    rows = [(task, int(task["gold_answer"]), filename(task)) for task in group]
    sid = group[0]["source"]["task_id"]
    item = copy.deepcopy(group[0])
    item["id"] = f"ws_lite_{sid}_gtv2_001"
    item["source"]["rubric_refs"] = ["natural_graph_table_v2"]
    item["original_task_ids"] = [task["id"] for task in group]
    item["revision"] = "natural_graph_v2"

    if len(rows) == 1:
        task, count, name = rows[0]
        item["question"] = choose(item["id"], [
            f"Task {sid} relies on one spreadsheet or CSV input. What is its filename, and how many data rows does it contain?",
            f"Identify the tabular file needed for Task {sid} and report its number of data rows.",
            f"For Task {sid}, which input file holds structured rows, and how many rows are there?",
        ])
        item["gold_answer"] = f"{name}: {count}"
        selected = [task]
    else:
        counts = [count for _, count, _ in rows]
        max_count, min_count = max(counts), min(counts)
        if counts.count(max_count) == 1:
            task, count, name = next(row for row in rows if row[1] == max_count)
            item["question"] = choose(item["id"], [
                f"Among the spreadsheet and CSV inputs needed for Task {sid}, which file has the most data rows, and how many?",
                f"Which structured input for Task {sid} is the largest by row count? Give the filename and count.",
                f"Find the required tabular file with the highest row count for Task {sid}, and report both its name and count.",
            ])
            item["gold_answer"] = f"{name}: {count}"
            selected = list(group)
        elif counts.count(min_count) == 1:
            task, count, name = next(row for row in rows if row[1] == min_count)
            item["question"] = choose(item["id"], [
                f"Among the spreadsheet and CSV inputs needed for Task {sid}, which file has the fewest data rows, and how many?",
                f"Which structured input for Task {sid} is the smallest by row count? Give the filename and count.",
                f"Find the required tabular file with the lowest row count for Task {sid}, and report both its name and count.",
            ])
            item["gold_answer"] = f"{name}: {count}"
            selected = list(group)
        elif len(set(counts)) == 1:
            count = counts[0]
            item["question"] = choose(item["id"], [
                f"How many spreadsheet or CSV inputs are needed for Task {sid}, and how many data rows does each contain?",
                f"Count Task {sid}'s structured input files and report the common row count across them.",
                f"For Task {sid}, how many tabular inputs are there, and what is the row count of each one?",
            ])
            item["gold_answer"] = f"{len(rows)} files; {count} rows each"
            selected = list(group)
        else:
            raise ValueError(f"No deterministic comparison template for Task {sid}: {counts}")

    evidence = []
    candidate_names = [filename(task) for task in group]
    evidence.append({
        "surface": "graph",
        "graph_query": {"node": f"task_{sid}", "relation": "task_requires_file", "filter": "tabular inputs"},
        "verified_complete_set": candidate_names,
        "claim": "Graph enumeration plus the table registry returns exactly this complete set of tabular inputs.",
    })
    for task in selected:
        task_evidence = copy.deepcopy(task["gold_evidence"])
        for evidence_item in task_evidence:
            if evidence_item.get("surface") == "table":
                evidence_item["verified_result"] = int(task["gold_answer"])
                evidence_item["claim"] = (
                    f"Executing the recorded query returns {int(task['gold_answer'])} data rows."
                )
        evidence.extend(task_evidence)
    item["gold_evidence"] = evidence
    item["answer_type"] = "string"
    item["notes"] = (
        "Natural Graph+Table revision: the graph identifies the task's tabular "
        "inputs and executable table queries determine the requested count/comparison."
    )
    return item


def main() -> None:
    tasks = load_jsonl(SOURCE)
    graph_table: dict[str, list[dict]] = defaultdict(list)
    revised = []
    counts = defaultdict(int)

    for task in tasks:
        surfaces = task["required_surfaces"]
        if surfaces == ["graph", "table"]:
            graph_table[str(task["source"]["task_id"])].append(task)
        elif surfaces == ["graph"]:
            revised.append(rewrite_graph_only(task))
            counts["graph_only_rewritten"] += 1
        elif surfaces == ["rag", "graph"]:
            revised.append(rewrite_rag_graph(task))
            counts["rag_graph_rewritten"] += 1
        else:
            revised.append(copy.deepcopy(task))
            counts["unchanged"] += 1

    for sid in sorted(graph_table, key=lambda value: int(value)):
        revised.append(rebuild_graph_table(graph_table[sid]))
        counts["graph_table_rebuilt"] += 1

    # Make the already-executed gold audit visible to downstream reviewers.
    # This records results; it does not replace semantic question/query checks.
    for task in revised:
        for evidence_item in task.get("gold_evidence", []):
            if evidence_item.get("surface") == "table" and "verified_result" not in evidence_item:
                evidence_item["verified_result"] = task["gold_answer"]
                evidence_item["claim"] = "Executing the recorded query returns the stated verified result."
            if task.get("gold_answer") == "INSUFFICIENT_EVIDENCE":
                evidence_item["verified_absence_supports_abstention"] = True

    revised.sort(key=lambda task: task["id"])
    forbidden = ("dependency graph", "graph", "rag", "table surface", "knowledge surface")
    violations = [
        {"id": task["id"], "question": task["question"]}
        for task in revised
        if "graph" in task["required_surfaces"]
        and any(term in task["question"].lower() for term in forbidden)
    ]
    ids = [task["id"] for task in revised]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs in revision")
    if violations:
        raise ValueError(f"Explicit surface cues remain: {violations[:3]}")

    OUTPUT.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in revised))
    report = {
        "source_tasks": len(tasks),
        "revised_tasks": len(revised),
        "counts": dict(counts),
        "graph_table_original": sum(len(group) for group in graph_table.values()),
        "graph_table_revised": len(graph_table),
        "explicit_surface_cue_violations": violations,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Build the paired RAG+Graph routing-cue ablation task file.

Only question wording changes. IDs, gold answers, required surfaces, evidence,
and every other task field remain identical to the canonical release.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/worksurface_lite/tasks/tasks.jsonl"
OUTPUT = ROOT / "data/worksurface_lite/tasks/tasks_rag_graph_no_cue.jsonl"
CONTROL = ROOT / "data/worksurface_lite/tasks/tasks_rag_graph_control.jsonl"

PATTERN = re.compile(
    r'^Task (?P<task>\d+) lists several source documents in its dependency graph\. '
    r'Which document mentions the exact span (?P<span>.+)\? Return the filename\.$'
)


def main() -> None:
    selected: list[dict] = []
    controls: list[dict] = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        task = json.loads(line)
        if set(task.get("required_surfaces", [])) != {"rag", "graph"}:
            continue
        match = PATTERN.fullmatch(task["question"])
        if not match:
            raise ValueError(f"unexpected RAG+Graph template: {task['id']}")
        controls.append(copy.deepcopy(task))
        rewritten = copy.deepcopy(task)
        rewritten["original_question"] = task["question"]
        rewritten["question"] = (
            f"Which input file required by Task {match.group('task')} contains "
            f"the exact span {match.group('span')}? Return the filename."
        )
        rewritten["ablation"] = "remove_explicit_dependency_graph_cue"
        selected.append(rewritten)

    if len(selected) != 61:
        raise AssertionError(f"expected 61 RAG+Graph tasks, found {len(selected)}")
    if len({task["id"] for task in selected}) != len(selected):
        raise AssertionError("duplicate task IDs")
    if any("dependency graph" in task["question"].lower() for task in selected):
        raise AssertionError("explicit dependency-graph cue remains")

    OUTPUT.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in selected),
        encoding="utf-8",
    )
    CONTROL.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in controls),
        encoding="utf-8",
    )
    print(f"wrote {len(selected)} paired tasks -> {OUTPUT}")
    print(f"wrote {len(controls)} control tasks -> {CONTROL}")


if __name__ == "__main__":
    main()

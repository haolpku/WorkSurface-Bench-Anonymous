#!/usr/bin/env python3
"""Fill final1151 token budgets from the fixed GPT-4o-mini S5 reference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default="data/worksurface_lite/tasks/tasks_final_1151.jsonl",
    )
    parser.add_argument(
        "--reference",
        default="runs_final1151_flat/S5_gpt-4o-mini.jsonl",
    )
    args = parser.parse_args()

    task_path = Path(args.tasks)
    tasks = [json.loads(line) for line in task_path.open()]
    traces = {
        row["id"]: row
        for row in (json.loads(line) for line in Path(args.reference).open())
    }
    task_ids = {task["id"] for task in tasks}
    if task_ids != set(traces):
        missing = sorted(task_ids - set(traces))[:5]
        extra = sorted(set(traces) - task_ids)[:5]
        raise ValueError(f"reference/task ID mismatch: missing={missing}, extra={extra}")

    checked_existing = 0
    changed_existing = 0
    for task in tasks:
        tokens = int(traces[task["id"]].get("total_tokens", 0))
        if tokens <= 0:
            raise ValueError(f"non-positive reference token count: {task['id']}")
        budget = 2 * tokens
        old = task.get("efficiency_budget_tokens")
        if old:
            checked_existing += 1
            if int(old) != budget:
                changed_existing += 1
        task["efficiency_budget_tokens"] = budget

    tmp = task_path.with_suffix(task_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for task in tasks:
            out.write(json.dumps(task, ensure_ascii=False) + "\n")
    tmp.replace(task_path)
    print(
        f"filled {len(tasks)} budgets from {args.reference}; "
        f"checked {checked_existing} existing values and refreshed "
        f"{changed_existing} against the canonical trace"
    )


if __name__ == "__main__":
    main()

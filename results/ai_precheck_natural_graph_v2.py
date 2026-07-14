#!/usr/bin/env python3
"""LLM-assisted six-dimension precheck for benchmark tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "data/worksurface_lite/tasks/tasks_natural_graph_v2.jsonl"
OUT = ROOT / "results/ai_precheck_natural_graph_v2"


def stable_hash(text: str) -> int:
    h = 2166136261
    for ch in text:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def sample_tasks(tasks: list[dict]) -> list[dict]:
    def combo(task: dict) -> str:
        return "+".join(task["required_surfaces"])

    def take(label: str, n: int) -> list[dict]:
        rows = [task for task in tasks if combo(task) == label]
        return sorted(rows, key=lambda task: stable_hash(task["id"]))[:n]

    return sorted(take("graph", 40) + take("rag+graph", 30) + take("graph+table", 30), key=lambda task: task["id"])


def parse_json(text: str) -> dict:
    text = text.strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON object: {text[:200]}")


def call(task: dict, model: str, base: str, key: str, audit_full: bool) -> dict:
    system = (
        "You are auditing a benchmark item before human annotation. Judge these properties independently: "
        "(1) whether the supplied gold evidence is sufficient to answer the question, and "
        "(2) whether the stated gold answer follows exactly from that evidence. Be skeptical. "
        "A graph_path proves a task-file dependency. A table evidence item contains an executable "
        "SQL query and may include a verified_result field. Do not penalize surface terminology or style. "
        "For an abstention task whose gold is INSUFFICIENT_EVIDENCE, evidence proving the requested fact "
        "is absent makes the task answerable with that abstention and makes the gold correct. "
        "Question natural=Yes only for fluent, plausible workplace wording, not benchmark/meta wording or a visibly awkward template. "
        "Atomic and unambiguous=Yes only when the requested output and interpretation are uniquely scoped; a tightly coupled multi-field answer is allowed. "
        "Leakage cue=None when the question does not reveal an internal retrieval surface. Ordinary words such as file, document, sheet, or rows are workplace nouns, not leakage. "
        "Leakage cue=Surface named only when it explicitly says graph, RAG, retrieval surface, DuckDB, table tool, graph_neighbors, kb_search, or equivalent internal mechanism; "
        "Surface implied is reserved for an unnatural instruction that effectively tells the agent which internal surface/tool to select. Return one JSON object only."
    )
    payload = {
        "task_id": task["id"],
        "question": task["question"],
        "required_surfaces": task["required_surfaces"],
        "gold_answer": task["gold_answer"],
        "gold_evidence": task["gold_evidence"],
    }
    schema = ("{\"answerable\":\"Yes|No|Unsure\",\"gold_correct\":\"Yes|No|Unsure\","
              "\"question_natural\":\"Yes|No|Unsure\",\"atomic_unambiguous\":\"Yes|No|Unsure\","
              "\"leakage_cue\":\"None|Surface implied|Surface named|Unsure\","
              "\"reason\":\"concise evidence-based reason\",\"repair\":\"none or a concrete correction\"}") if audit_full else (
              "{\"answerable\":\"Yes|No|Unsure\",\"gold_correct\":\"Yes|No|Unsure\","
              "\"reason\":\"concise evidence-based reason\",\"repair\":\"none or a concrete correction\"}")
    user = json.dumps(payload, ensure_ascii=False, indent=2) + (
        f"\nReturn exactly: {schema}. If a query result needed for the gold "
        "is not present in verified_result or another evidence claim, mark gold_correct Unsure rather than guessing."
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 4096,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        data = json.load(response)
    verdict = parse_json(data["choices"][0]["message"]["content"])
    return {"id": task["id"], "model": model, **verdict}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scope", choices=["sample", "all"], default="sample")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--id-regex", default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--audit-full", action="store_true")
    args = parser.parse_args()
    base = os.getenv("WSB_API_BASE")
    key = os.getenv("WSB_API_KEY")
    if not base or not key:
        raise SystemExit("WSB_API_BASE and WSB_API_KEY are required")

    task_path = Path(args.tasks)
    all_tasks = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
    if args.id_regex:
        pattern = re.compile(args.id_regex)
        all_tasks = [task for task in all_tasks if pattern.search(task["id"])]
    tasks = sample_tasks(all_tasks) if args.scope == "sample" else all_tasks
    OUT.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "-").replace(":", "-")
    suffix = "" if args.scope == "sample" else "_all"
    if args.output_tag:
        suffix += f"_{args.output_tag}"
    path = OUT / f"{safe}{suffix}.jsonl"
    completed = {}
    if args.resume and path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row.get("error"):
                completed[row["id"]] = row
    todo = [task for task in tasks if task["id"] not in completed]

    with path.open("a", encoding="utf-8") as handle, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(call, task, args.model, base, key, args.audit_full): task for task in todo}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {"id": task["id"], "model": args.model, "error": repr(exc)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 20 == 0:
                print(f"[{args.model}] {index}/{len(todo)}", flush=True)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    latest = {row["id"]: row for row in rows}
    ordered = [latest[task["id"]] for task in tasks if task["id"] in latest]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))
    counts = {}
    for field in (("answerable", "gold_correct", "question_natural", "atomic_unambiguous") if args.audit_full else ("answerable", "gold_correct")):
        counts[field] = {label: sum(row.get(field) == label for row in ordered) for label in ("Yes", "No", "Unsure")}
    if args.audit_full:
        counts["leakage_cue"] = {label: sum(row.get("leakage_cue") == label for row in ordered) for label in ("None", "Surface implied", "Surface named", "Unsure")}
    counts["errors"] = sum(bool(row.get("error")) for row in ordered)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

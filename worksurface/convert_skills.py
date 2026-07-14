"""Skill surface: rubric-cluster SOP derivation (solutions_v0 §1.4).

Skills are derived from *patterns that repeat across rubrics*, not from any
single rubric — a rubric promoted verbatim would leak that task's answer.
In v0.1 they are attached to tasks as ``applicable_skills`` metadata and are
NOT a routable surface (README v0.1 note).

Pipeline:
  1. Cluster rubrics by verb+object bigram (a coarse intent signature).
  2. Retain clusters with >= MIN_TASKS distinct source tasks AND
     >= MIN_OUTPUT_TYPES distinct output types (single-task clusters leak).
  3. Emit an abstract SOP markdown per retained cluster — no task-specific
     values.

The leak-check (§1.4 Step 4) — running a naive S-only agent and confirming it
does NOT pass the source rubrics — needs the runner, so it is executed later
by scripts and its per-skill pass rates recorded in Appendix. Here we emit the
candidate skills + the task->skill applicability map.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

from .common import Task

MIN_TASKS = 3
MIN_OUTPUT_TYPES = 2

# Coarse intent signatures. Each maps a regex over the rubric text to a
# cluster key + a human-authored abstract SOP body.
INTENT_PATTERNS = [
    (
        "output_creation_check",
        re.compile(r"\b(generat|creat|produc|sav|writ).{0,40}\b(file|report|document|\.md|\.txt|\.csv)", re.I),
        "# Output creation check\n\n"
        "1. Read the task description and identify the target output filename(s).\n"
        "2. After producing the deliverable, list files in the output directory.\n"
        "3. Assert each target filename exists and is non-empty.\n",
    ),
    (
        "list_completeness_check",
        re.compile(r"\b(includ|contain|list|cover).{0,40}\b(all|each|every|following)", re.I),
        "# List completeness check\n\n"
        "1. Extract the required set of items from the task specification.\n"
        "2. Collect the items your answer actually reports.\n"
        "3. Confirm every required item is present; report any omissions.\n",
    ),
    (
        "numeric_accuracy_check",
        re.compile(r"\b(total|sum|count|average|margin|percentage|rate)\b.{0,40}(report|calculat|correct|\$|\d)", re.I),
        "# Numeric accuracy check\n\n"
        "1. Identify each quantity the task asks you to compute.\n"
        "2. Compute it from the authoritative source (table query preferred).\n"
        "3. Report the value with the units/precision the task expects.\n",
    ),
    (
        "format_conformance_check",
        re.compile(r"\b(present|format|structur|organiz).{0,30}\b(table|markdown|section|header|chart)", re.I),
        "# Format conformance check\n\n"
        "1. Note the required output format (table, sections, headers, ...).\n"
        "2. Render the deliverable in that exact structure.\n"
        "3. Verify the structural constraints before finishing.\n",
    ),
    (
        "cross_file_integration_check",
        re.compile(r"\b(integrat|combin|merg|traverse|across|each region|all data).{0,40}", re.I),
        "# Cross-file integration check\n\n"
        "1. Enumerate every source file the task depends on.\n"
        "2. Join / aggregate them on the shared key rather than reading one.\n"
        "3. Ensure the final result reflects all sources, not a subset.\n",
    ),
]


def _output_type(task: Task) -> str:
    outs = task.output_files
    if not outs:
        return "none"
    return os.path.splitext(outs[0])[1].lower() or "none"


def build_skills(all_tasks: list[Task], out_root: str) -> tuple[dict, dict]:
    """Derive shared skills across ALL profiles (skills are enterprise-wide).

    Writes ``{out_root}/skills/{cluster}/SKILL.md`` for retained clusters.
    Returns (skills_meta, task_to_skills).
    """
    skills_dir = os.path.join(out_root, "skills")
    os.makedirs(skills_dir, exist_ok=True)

    # cluster -> {tasks:set, output_types:set, rubric_hits:int}
    clusters: dict[str, dict] = defaultdict(
        lambda: {"tasks": set(), "output_types": set(), "rubric_hits": 0}
    )
    task_hits: dict[str, set[str]] = defaultdict(set)

    for task in all_tasks:
        otype = _output_type(task)
        for rubric in task.rubrics:
            for key, pat, _body in INTENT_PATTERNS:
                if pat.search(rubric):
                    c = clusters[key]
                    c["tasks"].add(task.task_id)
                    c["output_types"].add(otype)
                    c["rubric_hits"] += 1
                    task_hits[task.task_id].add(key)

    skills_meta: dict[str, dict] = {}
    retained: set[str] = set()
    bodies = {k: b for k, _p, b in INTENT_PATTERNS}
    for key, c in clusters.items():
        keep = len(c["tasks"]) >= MIN_TASKS and len(c["output_types"]) >= MIN_OUTPUT_TYPES
        skills_meta[key] = {
            "skill_type": "procedure",
            "n_source_tasks": len(c["tasks"]),
            "n_output_types": len(c["output_types"]),
            "rubric_hits": c["rubric_hits"],
            "retained": keep,
            "leak_check": "pending",  # filled by runner-based leak-check later
        }
        if keep:
            retained.add(key)
            cdir = os.path.join(skills_dir, key)
            os.makedirs(cdir, exist_ok=True)
            with open(os.path.join(cdir, "SKILL.md"), "w", encoding="utf-8") as f:
                f.write(bodies[key])

    # applicability map, restricted to retained skills
    task_to_skills = {
        tid: sorted(s & retained) for tid, s in task_hits.items() if s & retained
    }
    return skills_meta, task_to_skills

"""Deterministic rag+graph cross tasks (no API).

Complements augment_graph_cross (graph+table). The pattern here is:
the graph identifies which documents are dependencies of a source task, and
the RAG surface reads one of them for a specific fact. Structure of every
emitted task:

    Q: "Task T requires several source documents. Which of them contains the
       fact '<span>'? Return the filename."

Gold = a specific document filename, discovered by scanning the task's
    dep-graph doc dependencies for a distinctive numeric or string span that
    appears in exactly one document. Both surfaces are genuinely required:
    the graph enumerates the candidate files, and RAG reads them for the fact.
Auto-verified: the gold doc must actually contain the span, and no other
    dep-doc for the same task may contain it (uniqueness).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

from .common import (OUT_DIR, load_tasks, persona_slug, strip_hash_prefix,
                     tasks_by_persona)

# Match distinctive-looking spans: currency, percentage, or 4+ digit numbers.
_SPAN_RE = re.compile(r"\$\s*[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?\s*%|\b\d{4,}\b")


def _load_doc(path):
    try:
        return open(path, encoding="utf-8").read()
    except OSError:
        return ""


def build_rag_graph_cross(task, kb_dir, qid_start):
    """Emit rag+graph cross tasks for this source task."""
    # dep-graph derived docs of this task, mapped to on-disk canonical files
    from_ = {strip_hash_prefix(e["from"]) for e in task.dep_edges}
    to_ = {strip_hash_prefix(e["to"]) for e in task.dep_edges}
    dep_files = sorted(from_ | to_)

    # keep only files that exist as canonical KB docs
    doc_paths = {}
    for f in dep_files:
        stem = os.path.splitext(f)[0]
        cand = os.path.join(kb_dir, f"t{task.task_id}__{stem}.md")
        if os.path.exists(cand):
            doc_paths[f] = cand
    if len(doc_paths) < 2:
        return []                    # need >=2 docs to make routing non-trivial

    # extract candidate spans, keep only those appearing in exactly ONE doc
    span_locations = {}
    for fname, path in doc_paths.items():
        text = _load_doc(path)
        for m in _SPAN_RE.finditer(text):
            s = m.group(0).strip()
            if len(s) < 4:
                continue
            span_locations.setdefault(s, set()).add(fname)

    unique_spans = [(s, list(locs)[0]) for s, locs in span_locations.items()
                    if len(locs) == 1]
    unique_spans = unique_spans[:2]   # cap per task so distribution stays even
    if not unique_spans:
        return []

    out, qid = [], qid_start
    for span, gold_file in unique_spans:
        out.append({
            "id": f"ws_lite_{task.task_id}_rg{qid:03d}",
            "source": {"benchmark": "Workspace-Bench-Lite",
                       "task_id": task.task_id, "persona": task.persona,
                       "rubric_refs": ["rag_graph_cross"]},
            "question": (f"Task {task.task_id} lists several source documents "
                         f"in its dependency graph. Which document mentions "
                         f"the exact span \"{span}\"? Return the filename."),
            "difficulty": "medium",
            "task_type": "cross_surface",
            "required_surfaces": ["rag", "graph"],
            "gold_tools": ["graph_neighbors", "kb_search"],
            "applicable_skills": [],
            "gold_answer": gold_file,
            "answer_type": "string",
            "gold_evidence": [
                {"surface": "graph",
                 "graph_path": [f"task_{task.task_id}", "task_requires_file",
                                f"t{task.task_id}::{gold_file}"],
                 "claim": "graph enumerates the candidate documents"},
                {"surface": "rag", "file": f"t{task.task_id}__{os.path.splitext(gold_file)[0]}.md",
                 "span": span,
                 "claim": "the target span occurs verbatim in only this doc"},
            ],
            "notes": "Deterministic rag+graph cross: span verified unique to one doc.",
        })
        qid += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--max-per-persona", type=int, default=None,
                    help="cap new tasks per persona (default: no cap)")
    args = ap.parse_args()

    existing = [json.loads(l) for l in
                open(os.path.join(args.out, "tasks", "tasks.jsonl"))]
    seen_ids = {o["id"] for o in existing}
    profiles_dir = os.path.join(args.out, "profiles")
    all_tasks = load_tasks()

    by = tasks_by_persona(all_tasks)
    new_all = []
    for slug, ptasks in sorted(by.items()):
        kb_dir = os.path.join(profiles_dir, slug, "kb_docs")
        added = 0
        for task in ptasks:
            if args.max_per_persona and added >= args.max_per_persona:
                break
            items = build_rag_graph_cross(task, kb_dir, 1)
            for it in items:
                if it["id"] in seen_ids:
                    continue
                new_all.append(it); seen_ids.add(it["id"]); added += 1
                if args.max_per_persona and added >= args.max_per_persona:
                    break

    allt = existing + new_all
    outp = os.path.join(args.out, "tasks", "tasks.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for o in allt:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[rag-graph] +{len(new_all)} rag+graph cross tasks -> {len(allt)} total")
    print("  by persona:", dict(Counter(o["source"]["persona"] for o in new_all)))


if __name__ == "__main__":
    main()

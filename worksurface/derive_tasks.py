"""Derive atomic WorkSurface tasks from Workspace-Bench-Lite rubrics + graph.

Implements the deterministic core of solutions_v0 §1.3 (rubric 3-way
classification + extractive rewriter) and §1.2 graph-task derivation. Produces
JSONL conforming to schemas/task.schema.json.

Three deterministic derivation paths (no API needed — high precision, the
"gold_answer for free" backbone of the Lite pilot):

  graph_only   From each task's file_dep_graph / task_requires_file edges:
               "which source files are required for task T?" -> list answer,
               gold_evidence is a graph_path. required_surfaces=[graph].

  table_only   From numeric rubrics that name an aggregate over a table that
               EXISTS in the registry, with an executable DuckDB gold query
               whose result matches the rubric's stated number. Only emitted
               when the query reproduces the number (self-verifying).

  rag_only /   From extractive numeric / percentage / currency / list rubrics.
  cross_surface  answer_type in {number, list, boolean}. required_surfaces are
               inferred from evidence: if the number is reproducible from a
               table -> cross_surface [rag, table]; else rag_only.

Qualitative / process rubrics without an extractable answer are recorded as
``freeform`` candidates (answer_type=freeform, needs judge+anchors) but are
NOT emitted into the deterministic pilot set unless --include-freeform.

The LLM-based rewriter / verifier / cue-stripper (solutions §1.3 Step2,
§2.1) is a documented, model-agnostic hook (worksurface.llm_hooks) — the
pilot picks a concrete model; the deterministic path runs standalone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

from .common import OUT_DIR, load_tasks, persona_slug, tasks_by_persona
from .convert_tables import connect_registry

# ---- number / list extraction --------------------------------------------

CURRENCY_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# "a total of 1000 order records", "1000 orders", "101 orders", "300"
COUNT_CTX_RE = re.compile(
    r"(?:total of|a total of|is|are|with|of|contains?|includes?|exactly)?\s*"
    r"(\d{1,3}(?:,\d{3})*|\d+)\s+"
    r"(order records?|orders?|records?|rows?|items?|files?|roles?|entries?|"
    r"tasks?|categories|segments?|libraries|dependencies|bugs?)",
    re.I,
)

# a boolean rubric on file/output existence
EXISTENCE_RE = re.compile(
    r"\b(generat|creat|produc|sav|writ|output)\w*\b.*?\b([\w\-. ]+\.(md|txt|csv|xlsx|docx?|pptx?|json|pdf))\b",
    re.I,
)


def _num(s: str) -> float:
    v = float(s.replace(",", ""))
    return int(v) if v.is_integer() else v


def extract_numeric_answers(rubric: str) -> list[dict]:
    """Return candidate (value, unit, kind, span) tuples from one rubric."""
    out = []
    for m in CURRENCY_RE.finditer(rubric):
        out.append({"value": _num(m.group(1)), "unit": "currency",
                    "kind": "number", "span": m.group(0)})
    for m in PERCENT_RE.finditer(rubric):
        out.append({"value": _num(m.group(1)), "unit": "percent",
                    "kind": "number", "span": m.group(0)})
    for m in COUNT_CTX_RE.finditer(rubric):
        out.append({"value": _num(m.group(1)), "unit": m.group(2).lower(),
                    "kind": "count", "span": m.group(0)})
    return out


# ---- rubric classification (§1.3 Step 1) ----------------------------------

def classify_rubric(rubric: str) -> str:
    r = rubric.lower()
    if CURRENCY_RE.search(rubric) or PERCENT_RE.search(rubric) or COUNT_CTX_RE.search(rubric):
        return "extractive_numeric"
    if re.search(r"\b(include|contain|list|cover)\b.*\b(all|each|every|following|:)\b", r):
        return "extractive_string_set"
    if EXISTENCE_RE.search(rubric):
        return "extractive_boolean"
    if re.search(r"\b(present|format|table|markdown|section|header|structur)\b", r):
        return "structural"
    return "qualitative"


# ---- graph_only derivation ------------------------------------------------

def derive_graph_tasks(task, qid_start: int) -> list[dict]:
    """One 'required source files' routing task per source task."""
    required = sorted({e["filename"] for e in
                       [{"filename": _basename(d["from"])} for d in task.dep_edges] +
                       [{"filename": _basename(d["to"])} for d in task.dep_edges]})
    # fall back to manifest if the dep graph is empty
    if not required:
        required = sorted({os.path.basename(e["filename"])
                           for e in task.manifest() if e["exists"]})
    if len(required) < 2:
        return []
    q = (f"According to the file dependency graph, which source files are "
         f"required inputs for workspace task {task.task_id}?")
    return [{
        "id": f"ws_lite_{task.task_id}_q{qid_start:03d}",
        "source": {"benchmark": "Workspace-Bench-Lite", "task_id": task.task_id,
                   "persona": task.persona, "rubric_refs": ["file_dep_graph"]},
        "question": q,
        "difficulty": _diff(task.difficulty),
        "task_type": "graph_only",
        "required_surfaces": ["graph"],
        "gold_tools": ["graph_neighbors", "graph_traverse"],
        "applicable_skills": [],
        "gold_answer": [_clean(f) for f in required],
        "answer_type": "list",
        "gold_evidence": [{
            "surface": "graph",
            "graph_path": [f"task_{task.task_id}", "task_requires_file",
                           _clean(required[0])],
            "claim": "task_requires_file edges enumerate the required inputs",
        }],
        "notes": "Derived from file_dep_graph.",
    }]


# ---- table_only derivation (self-verifying) -------------------------------

def derive_table_tasks(task, rc_index: dict, qid_start: int) -> list[dict]:
    """Numeric rubrics whose value equals a table's row count (self-verifying).

    ``rc_index`` is {view: rowcount} for this task's views, precomputed by the
    caller. Conservative: only COUNT(*) rowcount questions are auto-derived
    here; richer aggregates go through derive_aggregate_tasks.
    """
    if not rc_index:
        return []
    rowcounts = {}
    for view, n in rc_index.items():
        rowcounts.setdefault(n, []).append(view)
    out, qid = [], qid_start
    used_counts = set()
    for rubric in task.rubrics:
        for cand in extract_numeric_answers(rubric):
            if cand["kind"] != "count":
                continue
            v = cand["value"]
            if v in rowcounts and v not in used_counts and isinstance(v, int):
                view = rowcounts[v][0]
                used_counts.add(v)
                out.append({
                    "id": f"ws_lite_{task.task_id}_q{qid:03d}",
                    "source": {"benchmark": "Workspace-Bench-Lite",
                               "task_id": task.task_id, "persona": task.persona,
                               "rubric_refs": [rubric[:80]]},
                    "question": (f"How many {cand['unit']} are in the "
                                 f"{_view_label(view)} table?"),
                    "difficulty": _diff(task.difficulty),
                    "task_type": "table_only",
                    "required_surfaces": ["table"],
                    "gold_tools": ["table_describe", "table_query"],
                    "applicable_skills": [],
                    "gold_answer": v,
                    "answer_type": "number",
                    "gold_evidence": [{
                        "surface": "table", "table": view,
                        "query": f'SELECT COUNT(*) FROM "{view}"',
                        "claim": f"row count reproduces the stated {cand['unit']} count",
                    }],
                    "notes": "Self-verified: gold query result equals gold_answer.",
                })
                qid += 1
    return out


# ---- rag / cross-surface extractive numeric -------------------------------

def _load_kb_texts(profile_dir: str, task_id: str) -> dict:
    """Return {doc_name: text} for a source task's canonical KB docs."""
    import glob
    kb_dir = os.path.join(profile_dir, "kb_docs")
    out = {}
    for path in glob.glob(os.path.join(kb_dir, f"t{task_id}__*.md")):
        try:
            out[os.path.basename(path)] = open(path, encoding="utf-8").read()
        except OSError:
            continue
    return out


def _find_number_in_docs(value, unit, kb_texts: dict):
    """Return the doc that contains this figure, or None.

    Matches formatting variants: 1710971.47 may appear as "1,710,971.47",
    "1710971.47", "$1,710,971", "16.89%", "16.89", etc. Requires the digit
    core to appear as a token, so unrelated substrings don't false-match.
    """
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    variants = set()
    sval = f"{value}"
    variants.add(sval)
    if isinstance(value, int):
        variants.add(f"{value:,}")               # thousands separators
    else:
        variants.add(f"{value:,.2f}")
        variants.add(f"{value:.2f}")
        variants.add(f"{value:.1f}")
    # numbers with trailing .0 stripped, and rounded integer form
    variants.add(str(value).rstrip("0").rstrip(".") if "." in str(value) else str(value))
    for doc, text in kb_texts.items():
        for v in variants:
            if not v:
                continue
            # token-boundary-ish match: the digit string with optional $/% around
            pat = r"(?<![\d.,])" + re.escape(v) + r"(?![\d])"
            if re.search(pat, text):
                return doc
    return None


def derive_rag_numeric_tasks(task, kb_texts: dict, qid_start: int) -> list[dict]:
    """Extractive numeric rubrics -> rag_only questions, ONLY when the gold
    figure is actually present in some input KB document.

    Fixes the data-quality bug where gold numbers came from the OUTPUT report
    (which the agent must produce) rather than a retrievable input doc, making
    the task unanswerable. We now verify presence and attach the containing
    doc as gold_evidence.file.
    """
    out, qid = [], qid_start
    seen = set()
    for rubric in task.rubrics:
        if classify_rubric(rubric) != "extractive_numeric":
            continue
        cands = extract_numeric_answers(rubric)
        currency = [c for c in cands if c["unit"] == "currency"]
        percent = [c for c in cands if c["unit"] == "percent"]
        chosen = None
        if len(currency) == 1 and not percent:
            chosen = currency[0]
        elif len(percent) == 1 and not currency:
            chosen = percent[0]
        if not chosen:
            continue
        key = (chosen["value"], chosen["unit"])
        if key in seen:
            continue
        # VERIFY the figure exists in an input document; else discard.
        doc = _find_number_in_docs(chosen["value"], chosen["unit"], kb_texts)
        if not doc:
            continue
        seen.add(key)
        subject = _rubric_subject(rubric)
        unit_word = "total amount (USD)" if chosen["unit"] == "currency" else "percentage"
        out.append({
            "id": f"ws_lite_{task.task_id}_q{qid:03d}",
            "source": {"benchmark": "Workspace-Bench-Lite", "task_id": task.task_id,
                       "persona": task.persona, "rubric_refs": [rubric[:80]]},
            "question": f"For workspace task {task.task_id}: {subject} "
                        f"What is the {unit_word}?",
            "difficulty": _diff(task.difficulty),
            "task_type": "rag_only",
            "required_surfaces": ["rag"],
            "gold_tools": ["kb_search"],
            "applicable_skills": [],
            "gold_answer": chosen["value"],
            "answer_type": "number",
            "gold_evidence": [{
                "surface": "rag",
                "file": chosen and doc,
                "span": chosen["span"],
                "claim": rubric[:160],
            }],
            "notes": "Extractive-numeric, verified present in input KB doc.",
        })
        qid += 1
    return out


def derive_abstain_tasks(task, kb_texts: dict, qid_start: int,
                         max_per_task: int = 1) -> list[dict]:
    """Abstain tasks (solutions §4.1): questions whose answer is genuinely not
    in the retrievable data, so the correct response is INSUFFICIENT_EVIDENCE.

    We reuse the extractive-numeric rubrics whose figure is NOT present in any
    input document (the counterfactual of derive_rag_numeric_tasks: those the
    grounding check discards because the number lives only in the to-be-written
    report). This gives a natural, verifiable abstain: a well-formed question
    with no supporting evidence. Answering with any specific value scores 0.
    """
    out, qid, made = [], qid_start, 0
    seen = set()
    for rubric in task.rubrics:
        if made >= max_per_task:
            break
        if classify_rubric(rubric) != "extractive_numeric":
            continue
        cands = extract_numeric_answers(rubric)
        currency = [c for c in cands if c["unit"] == "currency"]
        percent = [c for c in cands if c["unit"] == "percent"]
        chosen = None
        if len(currency) == 1 and not percent:
            chosen = currency[0]
        elif len(percent) == 1 and not currency:
            chosen = percent[0]
        if not chosen:
            continue
        key = (chosen["value"], chosen["unit"])
        if key in seen:
            continue
        # keep ONLY figures absent from every input document
        if _find_number_in_docs(chosen["value"], chosen["unit"], kb_texts):
            continue
        seen.add(key)
        subject = _rubric_subject(rubric)
        unit_word = "total amount (USD)" if chosen["unit"] == "currency" else "percentage"
        out.append({
            "id": f"ws_lite_{task.task_id}_q{qid:03d}",
            "source": {"benchmark": "Workspace-Bench-Lite", "task_id": task.task_id,
                       "persona": task.persona, "rubric_refs": [rubric[:80]]},
            "question": f"Based only on the available source documents for "
                        f"workspace task {task.task_id}: {subject} "
                        f"What is the {unit_word}?",
            "difficulty": _diff(task.difficulty),
            "task_type": "rag_only",
            "required_surfaces": ["rag"],
            "gold_tools": ["kb_search"],
            "applicable_skills": [],
            "gold_answer": "INSUFFICIENT_EVIDENCE",
            "answer_type": "abstain",
            "gold_evidence": [{
                "surface": "rag",
                "claim": "figure is not present in any input document; "
                         "the value would only exist in the to-be-produced report",
            }],
            "notes": "Abstain: gold figure verified ABSENT from all input KB docs (§4.1).",
        })
        qid += 1
        made += 1
    return out

def _basename(x: str) -> str:
    return os.path.basename(x)


def _clean(name: str) -> str:
    from .common import strip_hash_prefix
    return strip_hash_prefix(name)


def _diff(d: str) -> str:
    return d if d in ("easy", "medium", "hard") else "unknown"


def _view_label(view: str) -> str:
    # t107__usca_orders -> "usca_orders"
    return view.split("__", 1)[1] if "__" in view else view


# ---- self-verifying aggregate search (table_only / cross_surface) ---------

def _precompute_aggregates(con, view: str) -> list[dict]:
    """One pass over a view: every numeric column's SUM/AVG/MAX/MIN/COUNT_DISTINCT.

    Returns [{col, agg, value, query}]. Non-numeric columns yield nothing.
    Casting strips $ , % so currency/percent string columns become numbers.
    """
    try:
        desc = con.execute(f'DESCRIBE "{view}"').fetchall()
    except Exception:  # noqa: BLE001
        return []
    # Skip provenance columns AND headerless "unnamed_*" columns: a numeric
    # coincidence over an unlabeled spreadsheet cell is not a real derivation,
    # and matching on it produces false-positive gold queries (a benchmark
    # correctness hazard — see buglog aggregate-precision).
    cols = [r[0] for r in desc
            if not r[0].startswith("_source_") and not r[0].startswith("unnamed")]
    if not cols:
        return []

    def cast(c):
        return (f"TRY_CAST(REPLACE(REPLACE(REPLACE(\"{c}\", '$',''), ',',''), "
                f"'%','') AS DOUBLE)")

    # Build one wide SELECT computing all aggregates for all columns at once.
    selects = []
    plan = []  # (idx, col, agg, query_for_evidence)
    idx = 0
    for c in cols:
        cst = cast(c)
        for agg, expr in (
            ("SUM", f"ROUND(SUM({cst}),2)"),
            ("AVG", f"ROUND(AVG({cst}),2)"),
            ("MAX", f"ROUND(MAX({cst}),2)"),
            ("MIN", f"ROUND(MIN({cst}),2)"),
            ("COUNT_DISTINCT", f'COUNT(DISTINCT "{c}")'),
        ):
            selects.append(f"{expr} AS a{idx}")
            ev_expr = expr.replace("ROUND(", "").split(",2)")[0] if agg != "COUNT_DISTINCT" else f'COUNT(DISTINCT "{c}")'
            q = f'SELECT {expr} FROM "{view}"'
            plan.append((idx, c, agg, q))
            idx += 1
    if not selects:
        return []
    try:
        row = con.execute(f'SELECT {", ".join(selects)} FROM "{view}"').fetchone()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for i, col, agg, q in plan:
        v = row[i]
        if v is None:
            continue
        out.append({"col": col, "agg": agg, "value": float(v), "query": q})
    return out


def _find_match(agg_index: dict, target: float, unit: str):
    """Look up an aggregate reproducing target across all views. agg_index:
    view -> [agg dicts]. Returns (view, entry) or None.

    Precision guards (avoid coincidental / semantically-invalid matches):
      - COUNT_DISTINCT never backs a currency/percent figure (a distinct-value
        count equalling a dollar amount is meaningless).
      - percentages require a near-exact match (no rounding slack that would
        let 21.05 match a SUM of 21.0).
      - currency allows 0.5% relative slack (rounding of large sums is real).
    """
    for view, entries in agg_index.items():
        for e in entries:
            got = e["value"]
            if unit in ("currency", "percent") and e["agg"] == "COUNT_DISTINCT":
                continue
            if unit == "percent":
                if abs(got - target) <= 0.02:
                    return view, e
            elif unit == "currency":
                if abs(got - target) / max(abs(target), 1e-9) <= 0.005:
                    return view, e
            else:
                if abs(got - target) < 0.5:
                    return view, e
    return None


def derive_aggregate_tasks(task, agg_index: dict, qid_start: int) -> list[dict]:
    """Currency/percent rubric values reproduced by a precomputed table aggregate.

    ``agg_index`` is {view: [aggregate dicts]} for this task's views, computed
    once by the caller. Self-verifying: emitted gold query matches gold_answer.
    """
    if not agg_index:
        return []
    out, qid = [], qid_start
    seen_vals = set()
    for rubric in task.rubrics:
        for cand in extract_numeric_answers(rubric):
            if cand["unit"] not in ("currency", "percent"):
                continue
            key = (cand["value"], cand["unit"])
            if key in seen_vals:
                continue
            hit = _find_match(agg_index, float(cand["value"]), cand["unit"])
            if not hit:
                continue
            seen_vals.add(key)
            view, e = hit
            subject = _rubric_subject(rubric)
            unit_word = "value (USD)" if cand["unit"] == "currency" else "percentage"
            out.append({
                "id": f"ws_lite_{task.task_id}_q{qid:03d}",
                "source": {"benchmark": "Workspace-Bench-Lite",
                           "task_id": task.task_id, "persona": task.persona,
                           "rubric_refs": [rubric[:80]]},
                "question": f"For workspace task {task.task_id}: {subject} "
                            f"Compute the {unit_word} from the source data.",
                "difficulty": _diff(task.difficulty),
                "task_type": "cross_surface",
                "required_surfaces": ["rag", "table"],
                "gold_tools": ["kb_search", "table_query"],
                "applicable_skills": [],
                "gold_answer": cand["value"],
                "answer_type": "number",
                "gold_evidence": [
                    {"surface": "rag", "span": cand["span"], "claim": rubric[:160]},
                    {"surface": "table", "table": view, "columns": [e["col"]],
                     "query": e["query"],
                     "claim": f"{e['agg']} over {e['col']} reproduces the stated figure"},
                ],
                "notes": "Self-verified cross_surface: gold query matches gold_answer.",
            })
            qid += 1
    return out


def _rubric_subject(rubric: str) -> str:
    """Trim a yes/no rubric into a declarative subject clue (no gold leak of #)."""
    s = re.sub(r"^\s*(is|are|does|do|did|was|were|has|have|can)\b\s*", "", rubric,
               flags=re.I).strip()
    s = CURRENCY_RE.sub("___", s)
    s = PERCENT_RE.sub("___", s)
    return (s[:140] + "…") if len(s) > 140 else s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    tasks = load_tasks(only=args.tasks)
    profiles_dir = os.path.join(args.out, "profiles")
    task_skill_map = {}
    tsm_path = os.path.join(args.out, "task_skill_map.json")
    if os.path.exists(tsm_path):
        task_skill_map = json.load(open(tsm_path))

    derived = []
    stats = defaultdict(int)
    # Process per profile so we open one DuckDB connection and precompute
    # aggregates per view exactly once (rather than per source task).
    by_profile = tasks_by_persona(tasks)
    for slug, ptasks in sorted(by_profile.items()):
        tables_dir = os.path.join(profiles_dir, slug, "tables")
        con = active = None
        view_aggs: dict = {}
        rowcounts_by_view: dict = {}
        if os.path.exists(os.path.join(tables_dir, "registry.json")):
            con, active = connect_registry(tables_dir)
            for view in active:
                view_aggs[view] = _precompute_aggregates(con, view)
                try:
                    rowcounts_by_view[view] = con.execute(
                        f'SELECT COUNT(*) FROM "{view}"').fetchone()[0]
                except Exception:  # noqa: BLE001
                    pass

        for task in ptasks:
            # views owned by this source task
            task_views = [v for v, m in (active or {}).items()
                          if m["task"] == task.task_id]
            agg_index = {v: view_aggs.get(v, []) for v in task_views}
            rc_index = {v: rowcounts_by_view[v] for v in task_views
                        if v in rowcounts_by_view}

            qid = 1
            block = []
            g = derive_graph_tasks(task, qid); block += g; qid += len(g)
            t = derive_table_tasks(task, rc_index, qid); block += t; qid += len(t)
            # self-verifying cross_surface aggregates first; they claim values
            # so the rag-numeric path won't re-emit the same figure.
            agg = derive_aggregate_tasks(task, agg_index, qid); block += agg
            qid += len(agg)
            claimed = {(o["gold_answer"],) for o in agg}
            kb_texts = _load_kb_texts(os.path.join(profiles_dir, slug), task.task_id)
            r = [o for o in derive_rag_numeric_tasks(task, kb_texts, qid)
                 if (o["gold_answer"],) not in claimed]
            for o in r:
                o["id"] = f"ws_lite_{task.task_id}_q{qid:03d}"; qid += 1
            block += r
            # abstain tasks (§4.1): only if this task has an unanswerable figure
            ab = derive_abstain_tasks(task, kb_texts, qid, max_per_task=1)
            for o in ab:
                o["id"] = f"ws_lite_{task.task_id}_q{qid:03d}"; qid += 1
            block += ab
            for item in block:
                item["applicable_skills"] = task_skill_map.get(task.task_id, [])
                stats[item["task_type"]] += 1
            derived += block

        if con is not None:
            con.close()

    # deterministic order by source task then id
    derived.sort(key=lambda o: (int(o["source"]["task_id"]), o["id"]))

    out_path = os.path.join(args.out, "tasks", "tasks.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in derived:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[derive] {len(derived)} atomic tasks -> {out_path}")
    for k in ("rag_only", "table_only", "graph_only", "cross_surface"):
        print(f"    {k}: {stats[k]}")


if __name__ == "__main__":
    main()

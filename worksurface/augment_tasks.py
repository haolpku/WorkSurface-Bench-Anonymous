"""LLM-assisted task augmentation (solutions §1.3, §1.4) — verifiable by design.

Scales the deterministic core toward the paper's target distribution. The
guiding rule is that gold answers are NEVER taken from the LLM's free text:

  table_only / cross_surface:  the LLM proposes a question AND a DuckDB query;
      we EXECUTE the query and use its result as gold. If the query errors or
      returns null, the item is dropped. Gold is therefore always reproducible
      (same guarantee as the deterministic aggregate path, just richer SQL:
      sum / avg / top-k / group-by / filter).

  rag_only:  the LLM reads one canonical doc and proposes a question whose
      answer is a short string/number copied verbatim from the doc; we VERIFY
      the answer substring occurs in the doc, else drop.

This keeps the "no synthesized facts" property: every gold value traces to
either a real query result over real tables or a verbatim doc span.

    python -m worksurface.augment_tasks --target-table 125 --target-cross 300 \
        --target-rag 175 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re

from .common import OUT_DIR, load_tasks, persona_slug, tasks_by_persona
from .convert_tables import connect_registry
from .llm_client import OpenAIClient

# ---------- schema-guided table question generation ----------

TABLE_SYS = (
    "You are given one SQL table's schema and sample rows. Propose "
    "{n} DISTINCT analytical questions a business analyst would ask, each "
    "answerable by ONE read-only DuckDB SQL query over this exact table. "
    "Vary the operation: SUM, AVG, MAX/MIN, COUNT with WHERE filter, "
    "GROUP BY + ORDER BY (top-k). Return a JSON array; each element is "
    '{"question": "...", "sql": "SELECT ...", "answer_type": "number|string|list"}. '
    "Use the exact table name and column names given. No prose, JSON only."
)


def _sample_rows(con, view, k=5):
    try:
        cols = [c[0] for c in con.execute(f'DESCRIBE "{view}"').fetchall()
                if not c[0].startswith("_source_")]
        rows = con.execute(
            f'SELECT {", ".join(chr(34)+c+chr(34) for c in cols)} '
            f'FROM "{view}" LIMIT {k}').fetchall()
        return cols, rows
    except Exception:  # noqa: BLE001
        return [], []


def _exec_gold(con, sql):
    """Run the proposed query; return (answer, answer_type) or None."""
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        return None
    try:
        rows = con.execute(sql).fetchall()
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    # scalar
    if len(rows) == 1 and len(rows[0]) == 1:
        v = rows[0][0]
        if v is None:
            return None
        if isinstance(v, (int, float)):
            fv = float(v)
            return (int(fv) if fv.is_integer() else round(fv, 2)), "number"
        return str(v), "string"
    # single column, multiple rows -> list (top-k)
    if rows and len(rows[0]) == 1:
        vals = [str(r[0]) for r in rows[:10] if r[0] is not None]
        return vals, "list"
    # multi-col: take first column as list
    vals = [str(r[0]) for r in rows[:10] if r[0] is not None]
    return vals, "list"


def _extract_json_array(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if m:
        text = m.group(1)
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        return json.loads(text[s:e + 1])
    except ValueError:
        return []


def augment_tables(client, active, con, task_id, persona, qid_start,
                   n_per_view=3, max_items=None):
    out, qid = [], qid_start
    for view, meta in active.items():
        if max_items is not None and len(out) >= max_items:
            break
        if meta["task"] != task_id:
            continue
        cols, rows = _sample_rows(con, view)
        if not cols:
            continue
        prompt = (f"Table name: {view}\nColumns: {cols}\n"
                  f"Sample rows: {rows}")
        try:
            raw = client.complete(TABLE_SYS.replace("{n}", str(n_per_view)),
                                  prompt, max_tokens=700)
        except Exception:  # noqa: BLE001
            continue
        for item in _extract_json_array(raw):
            sql = item.get("sql", "")
            if view not in sql:
                continue
            gold = _exec_gold(con, sql)
            if gold is None:
                continue
            answer, atype = gold
            out.append({
                "id": f"ws_lite_{task_id}_aq{qid:03d}",
                "source": {"benchmark": "Workspace-Bench-Lite", "task_id": task_id,
                           "persona": persona, "rubric_refs": ["llm_table_aug"]},
                "question": item.get("question", "").strip(),
                "difficulty": "medium",
                "task_type": "table_only",
                "required_surfaces": ["table"],
                "gold_tools": ["table_describe", "table_query"],
                "applicable_skills": [],
                "gold_answer": answer,
                "answer_type": atype,
                "gold_evidence": [{"surface": "table", "table": view,
                                   "query": sql, "columns": [],
                                   "claim": "LLM-proposed query; gold = executed result"}],
                "notes": "LLM-augmented table_only; gold self-verified by query execution.",
            })
            qid += 1
    return out


# ---------- doc-grounded rag question generation ----------

RAG_SYS = (
    "You are given the text of ONE workspace document. Propose {n} DISTINCT "
    "questions whose answer is a SHORT fact (a number, name, or short phrase) "
    "copied VERBATIM from the document. Return a JSON array of "
    '{"question": "...", "answer": "<verbatim span>"}. The answer MUST appear '
    "exactly in the document. JSON only, no prose."
)


def augment_rag(client, kb_dir, task_id, persona, qid_start, docs,
                n_per_doc=2, max_items=None):
    import glob
    out, qid = [], qid_start
    for path in sorted(glob.glob(os.path.join(kb_dir, f"t{task_id}__*.md"))):
        if max_items is not None and len(out) >= max_items:
            break
        text = open(path, encoding="utf-8").read()
        if len(text) < 200:
            continue
        body = text[:6000]
        try:
            raw = client.complete(RAG_SYS.replace("{n}", str(n_per_doc)),
                                  body, max_tokens=500)
        except Exception:  # noqa: BLE001
            continue
        doc_name = os.path.basename(path)
        for item in _extract_json_array(raw):
            ans = str(item.get("answer", "")).strip()
            q = item.get("question", "").strip()
            if not ans or not q or ans not in text:
                continue
            atype = "number" if re.fullmatch(r"[\d,.$%]+", ans) else "string"
            gold = ans
            if atype == "number":
                try:
                    fv = float(ans.replace(",", "").replace("$", "").replace("%", ""))
                    gold = int(fv) if fv.is_integer() else fv
                except ValueError:
                    atype, gold = "string", ans
            out.append({
                "id": f"ws_lite_{task_id}_aq{qid:03d}",
                "source": {"benchmark": "Workspace-Bench-Lite", "task_id": task_id,
                           "persona": persona, "rubric_refs": ["llm_rag_aug"]},
                "question": q,
                "difficulty": "easy",
                "task_type": "rag_only",
                "required_surfaces": ["rag"],
                "gold_tools": ["kb_search"],
                "applicable_skills": [],
                "gold_answer": gold,
                "answer_type": atype,
                "gold_evidence": [{"surface": "rag", "file": doc_name,
                                   "span": ans, "claim": "verbatim doc span"}],
                "notes": "LLM-augmented rag_only; answer verified verbatim in doc.",
            })
            qid += 1
    return out


# ---------- cross-surface generation ----------
# Strategy: a cross_surface item pairs a table computation with a doc lookup
# from the SAME source task. The question asks for the table figure but
# references an entity/condition that must be found in a document first, so
# both surfaces are genuinely required. Gold = executed query result.

CROSS_SYS = (
    "You are given a workspace task with (a) a data TABLE (schema + samples) "
    "and (b) an excerpt from a related DOCUMENT. Propose {n} questions that "
    "require BOTH: the reader must find a fact/name/threshold in the DOCUMENT "
    "and then run a DuckDB query over the TABLE using it. Return a JSON array "
    'of {"question": "...", "sql": "SELECT ...", "doc_fact": "<span from doc>"}. '
    "The SQL must be valid over the given table; doc_fact must appear verbatim "
    "in the document. JSON only."
)


def augment_cross(client, active, con, kb_dir, task_id, persona, qid_start,
                  max_items=None):
    import glob
    # pick this task's views + first sizeable doc
    views = [(v, m) for v, m in active.items() if m["task"] == task_id]
    docs = sorted(glob.glob(os.path.join(kb_dir, f"t{task_id}__*.md")))
    if not views or not docs:
        return []
    doc_text = ""
    doc_name = None
    for d in docs:
        t = open(d, encoding="utf-8").read()
        if len(t) > 300:
            doc_text, doc_name = t[:4000], os.path.basename(d)
            break
    if not doc_name:
        return []

    out, qid = [], qid_start
    for view, meta in views:
        if max_items is not None and len(out) >= max_items:
            break
        cols, rows = _sample_rows(con, view)
        if not cols:
            continue
        prompt = (f"TABLE name: {view}\nColumns: {cols}\nSample rows: {rows}\n\n"
                  f"DOCUMENT ({doc_name}):\n{doc_text}")
        try:
            raw = client.complete(CROSS_SYS.replace("{n}", "3"), prompt,
                                  max_tokens=800)
        except Exception:  # noqa: BLE001
            continue
        for item in _extract_json_array(raw):
            sql = item.get("sql", "")
            fact = str(item.get("doc_fact", "")).strip()
            if view not in sql or not fact or fact not in doc_text:
                continue
            gold = _exec_gold(con, sql)
            if gold is None:
                continue
            answer, atype = gold
            out.append({
                "id": f"ws_lite_{task_id}_aq{qid:03d}",
                "source": {"benchmark": "Workspace-Bench-Lite", "task_id": task_id,
                           "persona": persona, "rubric_refs": ["llm_cross_aug"]},
                "question": item.get("question", "").strip(),
                "difficulty": "hard",
                "task_type": "cross_surface",
                "required_surfaces": ["rag", "table"],
                "gold_tools": ["kb_search", "table_query"],
                "applicable_skills": [],
                "gold_answer": answer,
                "answer_type": atype,
                "gold_evidence": [
                    {"surface": "rag", "file": doc_name, "span": fact,
                     "claim": "condition/entity resolved from document"},
                    {"surface": "table", "table": view, "query": sql,
                     "claim": "gold = executed query result"},
                ],
                "notes": "LLM-augmented cross_surface; doc_fact verbatim + query self-verified.",
            })
            qid += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--target-table", type=int, default=125)
    ap.add_argument("--target-cross", type=int, default=300)
    ap.add_argument("--target-rag", type=int, default=175)
    ap.add_argument("--limit-tasks", type=int, default=None,
                    help="only process first N source tasks (for a cheap trial)")
    args = ap.parse_args()

    from collections import Counter
    existing = [json.loads(l) for l in
                open(os.path.join(args.out, "tasks", "tasks.jsonl"))]
    have = Counter(o["task_type"] for o in existing)
    print(f"[augment] existing {len(existing)}: {dict(have)}")

    client = OpenAIClient()
    profiles_dir = os.path.join(args.out, "profiles")
    tasks = load_tasks()
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
    by_profile = tasks_by_persona(tasks)

    new_items = []
    need_table = max(0, args.target_table - have["table_only"])
    need_cross = max(0, args.target_cross - have["cross_surface"])
    need_rag = max(0, args.target_rag - have["rag_only"])

    for slug, ptasks in sorted(by_profile.items()):
        tables_dir = os.path.join(profiles_dir, slug, "tables")
        kb_dir = os.path.join(profiles_dir, slug, "kb_docs")
        con = active = None
        if os.path.exists(os.path.join(tables_dir, "registry.json")):
            con, active = connect_registry(tables_dir)
        for task in ptasks:
            tid, per = task.task_id, task.persona
            # ONE running qid per source task, shared across the three augment
            # paths so their aq-ids never collide. (bug-015 fix.)
            qid = 1
            if con is not None and active:
                if need_table > 0:
                    it = augment_tables(client, active, con, tid, per, qid,
                                        max_items=3)
                    qid += len(it)
                    new_items += it; need_table -= len(it)
                if need_cross > 0:
                    it = augment_cross(client, active, con, kb_dir, tid, per, qid,
                                       max_items=4)
                    qid += len(it)
                    new_items += it; need_cross -= len(it)
            if need_rag > 0:
                it = augment_rag(client, kb_dir, tid, per, qid, docs=None,
                                 max_items=3)
                qid += len(it)
                new_items += it; need_rag -= len(it)
        if con is not None:
            con.close()
        print(f"[augment] after {slug}: +{len(new_items)} new "
              f"(need table={need_table} cross={need_cross} rag={need_rag})")
        print("  " + client.report())

    # merge + write
    allt = existing + new_items
    outp = os.path.join(args.out, "tasks", "tasks.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for o in allt:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[augment] wrote {len(allt)} tasks (+{len(new_items)}) -> {outp}")
    print("  " + client.report())
    from collections import Counter as C
    print("  dist:", dict(C(o["task_type"] for o in allt)))


if __name__ == "__main__":
    main()

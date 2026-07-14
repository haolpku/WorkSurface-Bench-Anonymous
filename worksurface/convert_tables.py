"""Table surface: per-workbook DuckDB view registry (solutions_v0 §1.1).

For every tabular input file (csv/xlsx/xls) we register a DuckDB view over a
normalized copy, add provenance columns (_source_file, _source_sheet,
_source_row_id), and record enough schema metadata for the runner's
``table_list`` / ``table_describe`` / ``table_query`` tools.

The Table *coverage gate* decides whether a sheet is table-worthy. A sheet is
kept as a real table only if it is non-trivial:

    row_count >= 30  OR  distinct groups >= 5  OR  it is wide (>= 4 columns)

Trivial sheets are demoted to RAG (rendered as markdown) — the (task_id,
filename) pairs are returned so convert_rag can pick them up. This is the
gate whose aggregate number (table_track_coverage) is reported in the paper's
Appendix A2 and drives the fallback in paper_spec §3.10.
"""

from __future__ import annotations

import os

import duckdb
import pandas as pd

from .common import Task, normalize_col, safe_ident, strip_hash_prefix

MIN_ROWS = 30
MIN_GROUPS = 5
MIN_COLS = 4


def _load_sheets(abspath: str, ext: str) -> dict[str, pd.DataFrame]:
    """Return {sheet_name: dataframe}. CSV -> single sheet named 'sheet'."""
    try:
        if ext == ".csv":
            return {"sheet": pd.read_csv(abspath, dtype=str, keep_default_na=False)}
        if ext in (".xlsx", ".xls"):
            engine = "openpyxl" if ext == ".xlsx" else None
            xls = pd.read_excel(
                abspath, sheet_name=None, dtype=str, engine=engine
            )
            return {str(k): v.fillna("") for k, v in xls.items()}
    except Exception as e:  # noqa: BLE001
        print(f"    [table] failed to read {abspath}: {type(e).__name__}: {e}")
    return {}


def _is_table_worthy(df: pd.DataFrame) -> bool:
    if df.shape[0] >= MIN_ROWS:
        return True
    if df.shape[1] >= MIN_COLS and df.shape[0] >= 3:
        return True
    # distinct groups: any column with >= MIN_GROUPS distinct non-empty values
    for col in df.columns:
        if df[col].replace("", pd.NA).nunique(dropna=True) >= MIN_GROUPS:
            return True
    return False


def build_tables(
    profile_tasks: list[Task], profile_dir: str
) -> tuple[dict, set[tuple[str, str]], dict]:
    """Build the DuckDB registry for a profile.

    Returns (registry, demoted_to_rag, coverage_stats).
      registry: view_name -> {task, source_file, sheet, rows, columns[...]}
      demoted_to_rag: {(task_id, filename)} trivial sheets sent to RAG
      coverage_stats: numbers for Appendix A2
    """
    tables_dir = os.path.join(profile_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    registry: dict[str, dict] = {}
    demoted: set[tuple[str, str]] = set()
    n_tabular_files = 0
    n_tasks_with_tabular = set()
    n_tasks_with_table = set()

    for task in profile_tasks:
        for entry in task.manifest():
            if not entry["exists"] or entry["ext"] not in {".csv", ".xlsx", ".xls"}:
                continue
            n_tabular_files += 1
            n_tasks_with_tabular.add(task.task_id)
            clean = strip_hash_prefix(entry["filename"])
            sheets = _load_sheets(entry["abspath"], entry["ext"])

            for sheet_name, df in sheets.items():
                if df.empty or df.shape[1] == 0:
                    continue
                if not _is_table_worthy(df):
                    demoted.add((task.task_id, entry["filename"]))
                    continue

                # Normalize headers; keep a rename map for provenance.
                orig_cols = list(df.columns)
                norm_cols, seen = [], {}
                for c in orig_cols:
                    nc = normalize_col(c)
                    if nc in seen:
                        seen[nc] += 1
                        nc = f"{nc}_{seen[nc]}"
                    else:
                        seen[nc] = 0
                    norm_cols.append(nc)
                df = df.copy()
                df.columns = norm_cols
                df["_source_file"] = clean
                df["_source_sheet"] = sheet_name
                df["_source_row_id"] = range(len(df))

                stem = safe_ident(entry["filename"])
                sheet_id = normalize_col(sheet_name) if sheet_name != "sheet" else ""
                view = f"t{task.task_id}__{stem}" + (f"__{sheet_id}" if sheet_id else "")
                # Source of truth is the parquet + registry.json; the runner
                # rebuilds an in-memory DuckDB via connect_registry(). This
                # avoids persisting views with cwd-relative paths.
                pq = os.path.join(tables_dir, f"{view}.parquet")
                df.to_parquet(pq, index=False)

                registry[view] = {
                    "task": task.task_id,
                    "source_file": clean,
                    "sheet": sheet_name,
                    "rows": int(df.shape[0]),
                    "parquet": os.path.basename(pq),
                    "columns": [
                        {"name": n, "orig": o}
                        for n, o in zip(norm_cols, orig_cols)
                    ],
                }
                n_tasks_with_table.add(task.task_id)

    coverage = {
        "n_tabular_files": n_tabular_files,
        "n_views": len(registry),
        "n_tasks_with_tabular": len(n_tasks_with_tabular),
        "n_tasks_with_table": len(n_tasks_with_table),
        "n_demoted_to_rag": len(demoted),
        "table_track_coverage": (
            len(n_tasks_with_table) / len(n_tasks_with_tabular)
            if n_tasks_with_tabular
            else 0.0
        ),
    }
    return registry, demoted, coverage


def connect_registry(tables_dir: str, tasks: list[str] | None = None):
    """Open an in-memory DuckDB with every view in a profile's registry.json.

    If ``tasks`` is given, only views whose source task is in that set are
    registered (used to scope a single task's tables in the runner). Returns
    (connection, {view_name: registry_entry}).
    """
    import json

    reg_path = os.path.join(tables_dir, "registry.json")
    registry = json.load(open(reg_path)) if os.path.exists(reg_path) else {}
    con = duckdb.connect(":memory:")
    keep = set(tasks) if tasks else None
    active = {}
    for view, meta in registry.items():
        if keep is not None and meta["task"] not in keep:
            continue
        pq = os.path.join(tables_dir, meta["parquet"]).replace("'", "''")
        con.execute(
            f'CREATE OR REPLACE VIEW "{view}" AS '
            f"SELECT * FROM read_parquet('{pq}')"
        )
        active[view] = meta
    return con, active

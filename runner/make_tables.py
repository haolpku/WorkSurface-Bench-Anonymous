"""Assemble paper tables from scored runs (paper_spec §3.2 / §3.4).

Given a directory of ``*.scored.json`` files (one per setting x model), emit:

  table3_main_results.{csv,md}   rows = setting x model, cols = Route F1 /
                                 Evidence / Answer / Efficiency / Aggregate
  table4_per_surface.{csv,md}    rows = setting x model, cols = Answer per
                                 task_type (rag_only / table_only /
                                 graph_only / cross_surface)

Run naming convention: ``<setting>_<model>.scored.json`` (e.g. S4_mock,
S3_opus-4-7). The setting/model are parsed from the filename.

    python -m runner.make_tables --runs runs/ --out runs/tables
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

SUBSCORES = [("route_precision", "Route-P"), ("route_recall", "Route-R"),
             ("route_f1", "Route-F1"), ("evidence", "Evid."),
             ("answer", "Answer"), ("efficiency", "Eff."),
             ("aggregate", "Agg.")]
TASK_TYPES = ["rag_only", "table_only", "graph_only", "cross_surface"]
# per-surface Answer table columns (adds an abstain calibration column)
SURFACE_COLS = ["rag_only", "table_only", "graph_only", "cross_surface", "abstain"]


def _parse_name(path: str):
    base = os.path.basename(path).replace(".scored.json", "")
    setting, _, model = base.partition("_")
    return setting, (model or "?")


def _fmt(v):
    # report scores as percentages (0.052 -> 5.2) — cleaner in tables
    return "--" if v is None else f"{v * 100:.1f}"


def load_runs(runs_dir: str):
    rows = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*.scored.json"))):
        rep = json.load(open(p))
        setting, model = _parse_name(p)
        rows.append({"setting": setting, "model": model, "report": rep})
    return rows


def build_table3(rows):
    header = ["Setting", "Model"] + [lbl for _, lbl in SUBSCORES]
    out = [header]
    for r in sorted(rows, key=lambda x: (x["model"], x["setting"])):
        o = r["report"]["overall"]
        out.append([r["setting"], r["model"]] +
                   [_fmt(o.get(k)) for k, _ in SUBSCORES])
    return out


def build_table4(rows):
    header = ["Setting", "Model"] + TASK_TYPES
    out = [header]
    for r in sorted(rows, key=lambda x: (x["model"], x["setting"])):
        bt = r["report"]["by_task_type"]
        cells = []
        for t in TASK_TYPES:
            cells.append(_fmt(bt.get(t, {}).get("answer")))
        out.append([r["setting"], r["model"]] + cells)
    return out


def write_csv(table, path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(table)


def write_md(table, path):
    lines = ["| " + " | ".join(table[0]) + " |",
             "| " + " | ".join("---" for _ in table[0]) + " |"]
    for row in table[1:]:
        lines.append("| " + " | ".join(row) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


# canonical model + setting display order for the paper leaderboard
MODEL_ORDER = ["gpt-4o-mini", "deepseek-v4-pro",
               "gemini-3.1-pro-preview", "gpt-5.5"]
MODEL_LABEL = {
    "gpt-4o-mini": "GPT-4o-mini", "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "gpt-5.5": "GPT-5.5",
    "gemini-3.1-pro-preview": "Gemini-3.1-Pro",
}
SETTING_LABEL = {"S1": "No-tool", "S2": "Always-RAG", "S3": "Naive-router",
                 "S4": "ReAct-all", "S5": "Gold-guided"}


def _rows_by_model(rows):
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["setting"]] = r["report"]
    return by


def _emit_grouped_latex(rows, col_keys, col_labels, cell_fn, caption, label,
                        placeholder_models=None, highlight_guidance=False):
    """model-grouped booktabs table: each model is a \\multirow block over its
    five settings. ``placeholder_models`` are listed with em-dash rows even if
    they have no runs yet. Per (setting, column) the best value across models
    is bolded so the leaderboard is scannable."""
    by = _rows_by_model(rows)
    models = [m for m in MODEL_ORDER if m in by or (placeholder_models and m in placeholder_models)]
    for m in by:
        if m not in models:
            models.append(m)

    settings = ["S1", "S2", "S3", "S4", "S5"]

    # First pass: collect raw cells so we can identify per-(setting, col) maxima.
    raw = {}   # raw[m][s] = list[str] of cells for that row
    for m in models:
        raw[m] = {}
        for s in settings:
            raw[m][s] = cell_fn(by.get(m, {}).get(s), s)

    def _numeric(cell):
        try:
            return float(cell)
        except (TypeError, ValueError):
            return None

    # Within each model, bold the best setting for each metric. This makes the
    # table answer the natural ablation question: which agent configuration
    # works best for this backbone?
    best = {}
    for m in models:
        for ci in range(len(col_keys)):
            vals = [(s, _numeric(raw[m][s][ci])) for s in settings]
            vals = [(s, v) for s, v in vals if v is not None]
            if not vals:
                continue
            top = max(v for _, v in vals)
            for s, v in vals:
                if v == top:
                    best[(s, ci, m)] = True

    out = []
    out.append("\\begin{table*}[t]")
    out.append("\\centering\\small")
    out.append("\\renewcommand{\\arraystretch}{1.08}")
    out.append("\\setlength{\\tabcolsep}{4.2pt}")
    out.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll"
               + "c" * len(col_keys) + "}")
    out.append("\\toprule")
    if label == "tab:main":
        out.append("\\multicolumn{2}{c}{\\textbf{Configuration}} & "
                   "\\multicolumn{3}{c}{\\textbf{Routing}} & "
                   "\\multicolumn{2}{c}{\\textbf{Agent performance}} & "
                   "\\textbf{Resource use} & \\textbf{Overall} \\\\")
        out.append("\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}")
        out.append("\\textbf{Model} & \\textbf{Setting} & "
                   "\\textbf{P} & \\textbf{R} & \\textbf{F1} & "
                   "\\textbf{Evidence} & \\textbf{Answer} & "
                   "\\textbf{Eff.} & \\textbf{Agg.} \\\\")
    else:
        out.append("\\multicolumn{2}{c}{\\textbf{Configuration}} & "
                   "\\multicolumn{4}{c}{\\textbf{Answer score by task type}} \\\\")
        out.append("\\cmidrule(lr){3-6}")
        out.append("\\textbf{Model} & \\textbf{Setting} & "
                   "\\textbf{RAG} & \\textbf{Table} & \\textbf{Graph} & "
                   "\\textbf{Cross} \\\\")
    out.append("\\midrule")
    for mi, m in enumerate(models):
        for si, s in enumerate(settings):
            cells = raw[m][s]
            # bold the winning cells in this (setting, col)
            bold_cells = []
            for ci, c in enumerate(cells):
                if best.get((s, ci, m)) and c != "--":
                    bold_cells.append(f"\\textbf{{{c}}}")
                else:
                    bold_cells.append(c)
            # Avoid multirow: colortbl paints highlighted cells after spanning
            # text and can occlude model labels. Group rules preserve the same
            # visual hierarchy without relying on overlapping TeX boxes.
            label_cell = MODEL_LABEL.get(m, m) if si == 0 else ""
            setting_cell = SETTING_LABEL[s]
            if s == "S5" and highlight_guidance:
                box = "gbox" if label == "tab:persurface" else "gcell"
                setting_cell = f"\\{box}{{{setting_cell}}}"
                bold_cells = [f"\\{box}{{{cell}}}" for cell in bold_cells]
            out.append(f"{label_cell} & {setting_cell} & " +
                       " & ".join(bold_cells) + " \\\\")
        if mi < len(models) - 1:
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular*}")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{table*}")
    return "\n".join(out) + "\n"


def _main_cell(rep, setting):
    if rep is None:
        return ["--"] * len(SUBSCORES)
    o = rep["overall"]
    cells = []
    for k, _ in SUBSCORES:
        # S1 (no-tool) has no routing at all
        if setting == "S1" and k in ("route_precision", "route_recall", "route_f1"):
            cells.append("--")
        else:
            cells.append(_fmt(o.get(k)))
    return cells


def _persurface_cell(rep, setting):
    if rep is None:
        return ["--"] * len(SURFACE_COLS)
    bt = rep["by_task_type"]
    return [_fmt(bt.get(t, {}).get("answer")) for t in SURFACE_COLS]


def build_table3_latex(rows, placeholder_models=None):
    return _emit_grouped_latex(
        rows, [k for k, _ in SUBSCORES], [lbl for _, lbl in SUBSCORES],
        _main_cell,
        "Main results across four models and five agent settings on all "
        "1,151 tasks; all "
        "values are percentages. Route-F1, Evidence, Answer, and Efficiency "
        "contribute to Agg. as defined in Section 4; Safety is NA for the "
        "released task set and is therefore omitted. For No-tool, Route is "
        "displayed as an em dash but scored as zero in Agg. whenever a "
        "surface is required. Shading denotes Gold-guided; bold denotes the "
        "best setting within each model. All 23,020 retained trajectories "
        "completed without API or protocol errors.",
        "tab:main", placeholder_models, highlight_guidance=True)


def build_table4_latex(rows, placeholder_models=None):
    return _emit_grouped_latex(
        rows, SURFACE_COLS, ["rag", "table", "graph", "cross"],
        _persurface_cell,
        "Mean Answer score (\\%) by task type across four models on all 1,151 "
        "tasks. Shading denotes Gold-guided; bold denotes the best setting "
        "within each model.",
        "tab:persurface", placeholder_models, highlight_guidance=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load_runs(args.runs)
    if not rows:
        print(f"[tables] no *.scored.json in {args.runs}")
        return

    t3 = build_table3(rows)
    t4 = build_table4(rows)
    for name, tbl in (("table3_main_results", t3), ("table4_per_surface", t4)):
        write_csv(tbl, os.path.join(args.out, name + ".csv"))
        write_md(tbl, os.path.join(args.out, name + ".md"))

    print(f"[tables] {len(rows)} runs -> {args.out}")
    print("\nTable 3 — Main results:")
    for row in t3:
        print("  " + "  ".join(f"{c:>10}" for c in row))
    print("\nTable 4 — Per-surface Answer:")
    for row in t4:
        print("  " + "  ".join(f"{c:>12}" for c in row))


if __name__ == "__main__":
    main()

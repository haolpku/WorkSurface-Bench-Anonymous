"""Sweep multiple settings (and optionally models) in one command, then build
the paper tables. Convenience wrapper around runner.run_bench.

    # smoke: one setting, one model, few tasks
    python -m runner.sweep --model mock --settings S4 --limit 20

    # subset pilot: all settings, one real model, 50 tasks
    WSB_API_BASE=... WSB_API_KEY=... \
    python -m runner.sweep --model gpt-4o-mini --settings S1 S2 S3 S4 S5 S6 --limit 50

    # full single model
    python -m runner.sweep --model gpt-4o-mini --settings S1 S2 S3 S4 S5 S6

Writes runs/<setting>_<model>.jsonl + .scored.json for each, then
runs/tables/table3_main_results.md + table4_per_surface.md.
"""

from __future__ import annotations

import argparse
import json
import os

from worksurface.common import OUT_DIR

from .agents import SETTINGS, run_task
from .backbone import make_backbone


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--settings", nargs="+", default=["S1", "S2", "S3", "S4", "S5", "S6"],
                    choices=list(SETTINGS))
    ap.add_argument("--tasks", default=os.path.join(OUT_DIR, "tasks", "tasks.jsonl"))
    ap.add_argument("--data-root", default=OUT_DIR)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=100,
                    help="parallel in-flight tasks (I/O-bound API calls)")
    ap.add_argument("--resume", action="store_true",
                    help="skip tasks already in <setting>_<model>.jsonl.partial "
                         "(crash-safe: each result is appended as it completes)")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks)]
    if args.limit:
        tasks = tasks[: args.limit]
    os.makedirs(args.runs_dir, exist_ok=True)
    safe_model = args.model.replace("/", "-").replace(":", "-")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from scoring.score_run import score_run

    def run_one(task, setting):
        # One backbone per task: cum_usage / reset are per-instance, so
        # concurrent tasks never clobber each other's token accounting.
        bb = make_backbone(args.model)
        try:
            return run_task(task, setting, bb, args.data_root)
        except Exception as e:  # noqa: BLE001
            return {"id": task["id"], "setting": setting, "model": args.model,
                    "error": repr(e), "chosen_surfaces": [], "rag_files": [],
                    "tables": [], "graph_nodes": [], "answer": "",
                    "total_tokens": 0}

    for setting in args.settings:
        out = os.path.join(args.runs_dir, f"{setting}_{safe_model}.jsonl")
        partial = out + ".partial"

        # ---- resume: load already-completed traces from a prior partial ----
        results = {}
        if args.resume and os.path.exists(partial):
            for line in open(partial):
                try:
                    tr = json.loads(line)
                    # a task counts as done only if it did not error
                    if tr.get("id") and not tr.get("error"):
                        results[tr["id"]] = tr
                except json.JSONDecodeError:
                    continue
        todo = [t for t in tasks if t["id"] not in results]
        if args.resume and results:
            print(f"  [{setting}/{safe_model}] resume: {len(results)} done, "
                  f"{len(todo)} to go", flush=True)

        # ---- run remaining tasks, appending each result immediately ----
        lock = Lock()
        pf = open(partial, "a", encoding="utf-8")
        done = len(results)
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_one, t, setting): t["id"] for t in todo}
            for fut in as_completed(futs):
                tr = fut.result()
                results[tr["id"]] = tr
                with lock:                       # append-on-complete = crash-safe
                    pf.write(json.dumps(tr, ensure_ascii=False) + "\n")
                    pf.flush()
                    done += 1
                    if done % 50 == 0:
                        print(f"  [{setting}/{safe_model}] {done}/{len(tasks)}",
                              flush=True)
        pf.close()

        # ---- finalize: canonical file in task order, then score ----
        n_err = sum(1 for t in tasks if results.get(t["id"], {}).get("error"))
        with open(out, "w", encoding="utf-8") as f:
            for t in tasks:
                if t["id"] in results:
                    f.write(json.dumps(results[t["id"]], ensure_ascii=False) + "\n")
        traces = {t["id"]: t for t in (json.loads(l) for l in open(out))}
        report = score_run(tasks, traces)
        with open(out.rsplit(".", 1)[0] + ".scored.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        # clean the partial once the setting is fully finalized
        if os.path.exists(partial):
            os.remove(partial)
        print(f"[sweep] {setting}/{safe_model}: {n_err} errors, "
              f"overall={report['overall']}", flush=True)

    # build paper tables over everything in runs-dir
    from .make_tables import build_table3, build_table4, load_runs, write_md, write_csv
    rows = load_runs(args.runs_dir)
    tdir = os.path.join(args.runs_dir, "tables")
    os.makedirs(tdir, exist_ok=True)
    for name, tbl in (("table3_main_results", build_table3(rows)),
                      ("table4_per_surface", build_table4(rows))):
        write_csv(tbl, os.path.join(tdir, name + ".csv"))
        write_md(tbl, os.path.join(tdir, name + ".md"))
    print(f"[sweep] tables -> {tdir}")


if __name__ == "__main__":
    main()

"""Retry only the failed rows for each (setting, model), at low concurrency.

Reads runs/<S>_<M>.jsonl, extracts rows with .error, re-runs them (using a
subset task file), merges the new successes back into runs/<S>_<M>.jsonl,
then re-scores. Keeps successful rows untouched. Concurrency defaults to 15
(low) to avoid re-triggering the shared-proxy failures that produced the
errors in the first place.

    python -m runner.retry_errors --model <M>              # all settings
    python -m runner.retry_errors --model <M> --settings S4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

from worksurface.common import OUT_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--settings", nargs="+",
                    default=["S1", "S2", "S3", "S4", "S5", "S6"])
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--tasks", default=os.path.join(OUT_DIR, "tasks", "tasks.jsonl"))
    ap.add_argument("--data-root", default=OUT_DIR)
    args = ap.parse_args()

    all_tasks = {t["id"]: t for t in
                 (json.loads(l) for l in open(args.tasks))}

    for setting in args.settings:
        canonical = os.path.join(args.runs_dir, f"{setting}_{args.model}.jsonl")
        if not os.path.exists(canonical):
            print(f"[retry] {setting}/{args.model}: no run file, skip")
            continue

        rows = {json.loads(l)["id"]: json.loads(l) for l in open(canonical)}
        failed_ids = [tid for tid, tr in rows.items()
                      if tr.get("error") and tid in all_tasks]
        if not failed_ids:
            print(f"[retry] {setting}/{args.model}: 0 errors, skip")
            continue

        # write a subset jsonl for the sweep
        os.makedirs("runs_retry_tmp", exist_ok=True)
        subset_path = f"runs_retry_tmp/{setting}_{args.model}_subset.jsonl"
        with open(subset_path, "w", encoding="utf-8") as f:
            for tid in failed_ids:
                f.write(json.dumps(all_tasks[tid], ensure_ascii=False) + "\n")
        print(f"[retry] {setting}/{args.model}: retrying {len(failed_ids)} tasks")

        # run sweep with resume so we don't lose partials
        retry_out_dir = f"runs_retry_tmp/out_{setting}_{args.model}"
        os.makedirs(retry_out_dir, exist_ok=True)
        cmd = ["python3", "-u", "-m", "runner.sweep",
               "--model", args.model, "--settings", setting,
               "--concurrency", str(args.concurrency),
               "--tasks", subset_path,
               "--runs-dir", retry_out_dir,
               "--data-root", args.data_root,
               "--resume"]
        subprocess.run(cmd, check=False)

        # merge the retry results back into the canonical file
        retry_traces = {}
        retry_p = os.path.join(retry_out_dir, f"{setting}_{args.model}.jsonl")
        if os.path.exists(retry_p):
            for l in open(retry_p):
                tr = json.loads(l)
                retry_traces[tr["id"]] = tr

        merged = dict(rows)
        n_recovered = 0
        for tid, tr in retry_traces.items():
            # only overwrite if the retry succeeded (no error) or we had error
            if not tr.get("error"):
                if rows.get(tid, {}).get("error"):
                    n_recovered += 1
                merged[tid] = tr
            elif rows.get(tid, {}).get("error"):
                # both attempts failed; keep the latest
                merged[tid] = tr

        # write back in original task order
        ordered_ids = [json.loads(l)["id"] for l in open(canonical)]
        with open(canonical, "w", encoding="utf-8") as f:
            for tid in ordered_ids:
                if tid in merged:
                    f.write(json.dumps(merged[tid], ensure_ascii=False) + "\n")
        remaining = sum(1 for l in open(canonical) if json.loads(l).get("error"))
        print(f"[retry] {setting}/{args.model}: recovered {n_recovered}, "
              f"remaining errors {remaining}")

        # re-score
        scored_p = canonical.replace(".jsonl", ".scored.json")
        subprocess.run(["python3", "-m", "scoring.score_run",
                        "--tasks", args.tasks,
                        "--traces", canonical,
                        "--out", scored_p], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()

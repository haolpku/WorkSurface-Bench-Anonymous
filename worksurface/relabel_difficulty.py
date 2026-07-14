"""Re-label difficulty from empirical per-task solve rates.

Instead of the generation-template labels (easy/medium/hard by pipeline
path), we compute difficulty from what actually happened when models tried:
the fraction of available (setting x model) cells in {S3, S4, S5} that
produced Answer >= 0.5 on the task. Rationale: S1 is closed-book (always low)
and S2 is single-surface (unfair for non-rag tasks), so we exclude them; the
top three settings reflect a fair "with tools" attempt.

Bins (chosen so the labels are informative rather than skewed):
    solve_rate >= 0.60  -> easy    (most tool-using agents succeed)
    0.25 <= sr < 0.60   -> medium
    sr < 0.25           -> hard    (few agents succeed with tools)

Rewrites the "difficulty" field of tasks.jsonl in place. Also emits
results/difficulty_recalibration.json summarising the shift.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

from .common import OUT_DIR


def _load_solve_rate(runs_dir):
    """task_id -> fraction of (model, setting) cells passing (Answer >= 0.5)."""
    settings_to_use = {"S3", "S4", "S5"}
    tallies = defaultdict(lambda: [0, 0])   # [passes, attempts]
    for p in glob.glob(os.path.join(runs_dir, "S*_*.scored.json")):
        b = os.path.basename(p).replace(".scored.json", "")
        setting, _, _ = b.partition("_")
        if setting not in settings_to_use:
            continue
        rep = json.load(open(p))
        for r in rep["per_task"]:
            tid = r["id"]
            tallies[tid][1] += 1
            if r["answer"]["score"] >= 0.5:
                tallies[tid][0] += 1
    return {tid: p / a for tid, (p, a) in tallies.items() if a > 0}


def _bin(rate):
    if rate >= 0.60:
        return "easy"
    if rate >= 0.25:
        return "medium"
    return "hard"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    path = os.path.join(args.out, "tasks", "tasks.jsonl")
    tasks = [json.loads(l) for l in open(path)]
    solve = _load_solve_rate(args.runs)

    before = Counter(t.get("difficulty", "unknown") for t in tasks)
    changed = 0
    n_no_data = 0
    solve_by_id = {}
    for t in tasks:
        old = t.get("difficulty")
        sr = solve.get(t["id"])
        if sr is None:
            # no run data (e.g. newly added tri/rag-graph tasks); leave as-is
            n_no_data += 1
            continue
        new = _bin(sr)
        t["difficulty"] = new
        solve_by_id[t["id"]] = round(sr, 3)
        if new != old:
            changed += 1

    after = Counter(t.get("difficulty", "unknown") for t in tasks)

    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    report = {"before": dict(before), "after": dict(after),
              "changed": changed, "no_data": n_no_data,
              "bins": {"easy": ">=0.60", "medium": "0.25-0.60", "hard": "<0.25"},
              "solve_rate_by_id": solve_by_id}
    json.dump(report, open("results/difficulty_recalibration.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"[difficulty] rewrote {len(tasks)} tasks ({changed} labels changed, "
          f"{n_no_data} no-run-data kept as-is)")
    print(f"  before: {dict(before)}")
    print(f"  after:  {dict(after)}")


if __name__ == "__main__":
    main()

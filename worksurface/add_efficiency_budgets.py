"""Populate per-task Efficiency budgets from a canonical reference run.

The benchmark budget is twice the tokens used by the canonical GPT-4o-mini
gold-guided (S5) trace for the same task. Without a positive budget the scorer
uses a defensive neutral value, so a complete reference run is preferred.

    python -m worksurface.add_efficiency_budgets \
        --reference-run runs/S5_gpt-4o-mini.jsonl

Writes efficiency_budget_tokens into each task in tasks.jsonl (in place),
using ceil(2 * reference_tokens); tasks absent from the reference run get a corpus-median
fallback so they are still scorable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

from .common import OUT_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-run", "--oracle-run", dest="reference_run",
                    required=True,
                    help="Canonical S5 reference trace JSONL (has total_tokens)")
    ap.add_argument("--tasks", default=os.path.join(OUT_DIR, "tasks", "tasks.jsonl"))
    ap.add_argument("--slack", type=float, default=2.0,
                    help="budget = ceil(slack * oracle_tokens)")
    args = ap.parse_args()

    reference = {}
    for line in open(args.reference_run):
        t = json.loads(line)
        tok = t.get("total_tokens") or 0
        if tok > 0:
            reference[t["id"]] = tok
    if not reference:
        raise SystemExit("reference run has no positive total_tokens")
    median_tok = int(statistics.median(reference.values()))

    tasks = [json.loads(l) for l in open(args.tasks)]
    n_from_reference = 0
    for o in tasks:
        base = reference.get(o["id"], median_tok)
        if o["id"] in reference:
            n_from_reference += 1
        o["efficiency_budget_tokens"] = int(math.ceil(args.slack * base))

    with open(args.tasks, "w", encoding="utf-8") as f:
        for o in tasks:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"[budgets] wrote efficiency_budget_tokens to {len(tasks)} tasks "
          f"({n_from_reference} from reference, "
          f"{len(tasks)-n_from_reference} median-fallback)")
    print(f"[budgets] reference median tokens={median_tok}, slack={args.slack}x")


if __name__ == "__main__":
    main()

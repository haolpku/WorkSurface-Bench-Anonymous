"""Run a WorkSurface-Bench setting x model over the task set -> trace JSONL.

    python -m runner.run_bench --setting S4 --model mock \
        --tasks data/worksurface_lite/tasks/tasks.jsonl \
        --out runs/S4_mock.jsonl [--limit 50] [--score]

With --score it immediately scores the run via scoring.score_run and writes
<out>.scored.json. Use --model mock for a no-API smoke run; a real model name
(with WSB_API_BASE + WSB_API_KEY set) runs the actual backbone.
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
    ap.add_argument("--setting", required=True, choices=list(SETTINGS))
    ap.add_argument("--model", default="mock")
    ap.add_argument("--tasks", default=os.path.join(OUT_DIR, "tasks", "tasks.jsonl"))
    ap.add_argument("--data-root", default=OUT_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks)]
    if args.limit:
        tasks = tasks[: args.limit]
    backbone = make_backbone(args.model)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_err = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks, 1):
            try:
                trace = run_task(task, args.setting, backbone, args.data_root)
            except Exception as e:  # noqa: BLE001
                n_err += 1
                trace = {"id": task["id"], "setting": args.setting,
                         "model": backbone.name, "error": repr(e),
                         "chosen_surfaces": [], "rag_files": [], "tables": [],
                         "graph_nodes": [], "answer": "", "total_tokens": 0}
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  [{args.setting}/{args.model}] {i}/{len(tasks)}")
    print(f"[run] {len(tasks)} tasks, {n_err} errors -> {args.out}")

    if args.score:
        from scoring.score_run import score_run
        traces = {t["id"]: t for t in
                  (json.loads(l) for l in open(args.out))}
        report = score_run(tasks, traces)
        scored_path = args.out.rsplit(".", 1)[0] + ".scored.json"
        with open(scored_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[run] overall={report['overall']}")
        print(f"[run] scored -> {scored_path}")


if __name__ == "__main__":
    main()

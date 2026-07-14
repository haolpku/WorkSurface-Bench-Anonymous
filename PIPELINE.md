# WorkSurface-Bench pipeline

This document describes the reproducible path from Workspace-Bench-Lite to the
frozen 1,151-task WorkSurface-Bench release and its scored agent trajectories.

## 1. Download and freeze the source

```bash
python scripts/download_wsb_lite_en.py --tier full
python scripts/build_lock_and_metadata.py
```

The second command records the upstream revision and file hashes in
`data/wsb_lock.json`. Downloaded workspace files remain local and are excluded
from Git.

## 2. Project the workspace into knowledge surfaces

```bash
python -m worksurface.convert_profiles
```

For each persona, the converter produces:

```text
data/worksurface_lite/profiles/<persona>/
  kb_docs/                  canonical UTF-8 documents and registry
  tables/                   normalized Parquet views and registry
  graph/surface_graph.json  task, file, output, and dependency graph
```

Rubric-derived procedures are stored as metadata. They are not a fourth
routable surface in the v1.0 scoring protocol.

## 3. Derive, augment, and validate tasks

The deterministic starting point is:

```bash
python -m worksurface.derive_tasks
```

The `worksurface/augment_*.py` modules add cross-surface candidates. The
release-construction scripts under `results/` implement the documented
2,000 → 1,465 → 1,151 screening pipeline. API-backed steps read only
`WSB_API_BASE`, `WSB_API_KEY`, and an optional `WSB_BUILD_MODEL` from the
environment.

Before evaluation, run the deterministic quality checks:

```bash
python -m worksurface.qc_v2
```

The final task objects validate against `schemas/task.schema.json`. Table SQL,
RAG spans, graph paths, task IDs, and efficiency budgets are checked before a
release is frozen.

## 4. Run the five agent settings

```bash
for setting in S1 S2 S3 S4 S5; do
  python -m runner.run_bench \
    --setting "$setting" \
    --model mock \
    --tasks data/worksurface_lite/tasks/tasks_final_1151.jsonl \
    --data-root data/worksurface_lite \
    --out "runs/${setting}_mock.jsonl" \
    --score
done
```

Replace `mock` with a provider model ID after exporting `WSB_API_BASE` and
`WSB_API_KEY`. The official release contains one retained, zero-protocol-error
trajectory for every task × model × setting cell: 4 × 5 × 1,151 = 23,020.

## 5. Aggregate results

```bash
python -m runner.make_tables --runs runs --out runs/tables
python results/build_final1151_figures.py
python results/plot_figure3.py
python results/plot_figure4.py
```

The scorer reports Route precision/recall/F1, Evidence, Answer, Efficiency,
Safety where applicable, and the weighted Aggregate. The analysis scripts
consume scored JSON reports rather than manually entered values.

## 6. Build the Hugging Face release

```bash
python scripts/prepare_hf_release.py \
  --release /tmp/worksurface-hf \
  --annotations /path/to/anonymized-audit-workbooks
```

The builder verifies 1,151 unique tasks, 20 complete runs, 23,020 trajectory
scores, zero retained protocol errors, and the recomputed 2-of-3 human-audit
majorities. It exports viewer-friendly Parquet alongside the complete JSONL and
canonical resources.

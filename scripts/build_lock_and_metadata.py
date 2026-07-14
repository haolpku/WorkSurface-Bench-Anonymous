"""Build the WSB-Lite lock file and a reconstructed metadata table.

Two outputs, both deterministic from the downloaded ``en`` split:

1. ``data/wsb_lock.json`` — contamination-hygiene freeze (paper contribution
   C5). Records the HF repo, resolved commit, download time, and a per-task
   map of every *input* file's sha256. The runner verifies these hashes on
   load and refuses to score on mismatch.

2. ``data/workspace-bench-lite-en/metadata_table.reconstructed.csv`` — a flat
   table (one row per task) rebuilt from the per-task ``metadata.json`` files,
   because hf_hub >= 1.12 fails to decode the official brotli-compressed CSV.

The commit hash is read from the local HF snapshot ref if available, else
resolved via the Hub API (best effort; a null commit still produces a usable
lock with file hashes).
"""

import csv
import hashlib
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
EN_DIR = os.path.join(ROOT, "data", "workspace-bench-lite-en")
TASKS_DIR = os.path.join(EN_DIR, "task_lite_clean_en")
REPO = "Workspace-Bench/Workspace-Bench-Lite"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def resolve_commit() -> str | None:
    # Try the local snapshot ref written by huggingface_hub.
    ref = os.path.join(EN_DIR, ".cache", "huggingface", "download")
    # Fallback: query the Hub API.
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(REPO, repo_type="dataset")
        return info.sha
    except Exception:
        return None


def main() -> None:
    task_ids = sorted(
        (d for d in os.listdir(TASKS_DIR) if d.isdigit()), key=int
    )
    print(f"[lock] {len(task_ids)} tasks under {TASKS_DIR}")

    lock = {
        "wsb_repo": REPO,
        "wsb_commit": resolve_commit(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "split": "task_lite_clean_en",
        "task_id_to_file_hashes": {},
    }

    rows = []
    missing_files = 0
    for tid in task_ids:
        tdir = os.path.join(TASKS_DIR, tid)
        meta = json.load(open(os.path.join(tdir, "metadata.json")))

        # Hash every input file listed in the data_manifest that is present
        # on disk (core tier may not have pulled heavy pdf/pptx yet).
        file_hashes = {}
        for entry in meta.get("data_manifest", []):
            rel = entry["stored_relpath"]
            abspath = os.path.join(tdir, rel)
            if os.path.exists(abspath):
                file_hashes[entry["filename"]] = sha256(abspath)
            else:
                file_hashes[entry["filename"]] = None
                missing_files += 1
        lock["task_id_to_file_hashes"][tid] = file_hashes

        rubrics = meta.get("rubrics", [])
        rows.append(
            {
                "absolute_id": meta.get("absolute_id", tid),
                "persona": meta.get("persona", ""),
                "task_diff": meta.get("task_diff", ""),
                "n_input_files": len(meta.get("data_manifest", [])),
                "n_output_files": len(meta.get("output_files", [])),
                "n_rubrics": len(rubrics),
                "n_dep_edges": len(meta.get("file_dep_graph", [])),
                "tested_capabilities": "; ".join(
                    meta.get("tested_capabilities", [])
                ),
                "output_files": "; ".join(meta.get("output_files", [])),
                "task": meta.get("task", "").replace("\n", " ")[:500],
            }
        )

    lock_path = os.path.join(ROOT, "data", "wsb_lock.json")
    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2, ensure_ascii=False)
    print(f"[lock] wrote {lock_path} (commit={lock['wsb_commit']}, "
          f"{missing_files} input files not yet on disk)")

    table_path = os.path.join(EN_DIR, "metadata_table.reconstructed.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[lock] wrote {table_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

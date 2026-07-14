"""Download the Workspace-Bench-Lite English split from HuggingFace.

Only the ``en`` subset + top-level metadata table are pulled; the ``cn``
workspace files and ``assets/`` images are skipped.

Two-tier download to keep the pipeline fast to bootstrap:

  --tier core   metadata.json + structured/light-RAG files
                (csv/xlsx/xls/txt/md/json/py/java/xml/html) — ~12 MB, seconds.
                Enough to build Table / Graph / Skill surfaces and text RAG.
  --tier full   everything in the en split incl. pdf/pptx/docx — ~260 MB.
                Needed for heavy-document RAG source text.

The download target is the repository's local ``data/`` directory.

Output:
  data/workspace-bench-lite-en/
    task_lite_clean_en/{task_id}/metadata.json
    task_lite_clean_en/{task_id}/data/*
    task_lite_clean_en_metadata_table.csv   (best-effort; see note below)
    README.md

Note: hf_hub >= 1.12 has a brotli-decoding bug on the CSV metadata table.
The subsequent ``scripts/build_lock_and_metadata.py`` step reconstructs the
table from per-task ``metadata.json`` files when needed.
"""

import argparse
import os

from huggingface_hub import snapshot_download

REPO = "Workspace-Bench/Workspace-Bench-Lite"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_DIR = os.path.normpath(
    os.path.join(HERE, "..", "data", "workspace-bench-lite-en")
)

CORE_EXTS = ("csv", "xlsx", "xls", "txt", "md", "json", "py", "java", "xml", "html")

CORE_PATTERNS = [
    "task_lite_clean_en/*/metadata.json",
    "README.md",
] + [f"task_lite_clean_en/*/data/*.{ext}" for ext in CORE_EXTS]

FULL_PATTERNS = [
    "task_lite_clean_en/**",
    "task_lite_clean_en_metadata_table.csv",
    "README.md",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=["core", "full"], default="core")
    ap.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    patterns = CORE_PATTERNS if args.tier == "core" else FULL_PATTERNS
    print(f"[download] repo={REPO} tier={args.tier}")
    print(f"[download] local_dir={args.local_dir}")

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=patterns,
        max_workers=args.workers,
    )
    print(f"[download] done -> {args.local_dir}")


if __name__ == "__main__":
    main()

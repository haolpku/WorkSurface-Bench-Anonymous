#!/usr/bin/env python3
"""Run the main three backbones on the natural-graph v2 revision.

Keys are read from the environment and are never persisted:
  WSB_GPT_KEY     -> gpt-4o-mini and gpt-5.5
  WSB_GEMINI_KEY  -> gemini-3.1-pro-preview
Optional: WSB_API_BASE (defaults to the project proxy endpoint).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data/worksurface_lite/tasks/tasks_natural_graph_v2.jsonl"
MODELS = {
    "gpt-4o-mini": "WSB_GPT_KEY",
    "gpt-5.5": "WSB_GPT_KEY",
    "gemini-3.1-pro-preview": "WSB_GEMINI_KEY",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--settings", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    base = os.getenv("WSB_API_BASE")
    if not base:
        raise SystemExit("Missing WSB_API_BASE")
    for model in args.models:
        key_name = MODELS[model]
        key = os.getenv(key_name)
        if not key:
            raise SystemExit(f"Missing {key_name}; refusing to read or store a key elsewhere")
        env = os.environ.copy()
        env["WSB_API_BASE"] = base
        env["WSB_API_KEY"] = key
        out = ROOT / "runs_natural_graph_v2" / model
        cmd = [
            sys.executable, "-m", "runner.sweep", "--model", model,
            "--settings", *args.settings,
            "--tasks", str(TASKS), "--runs-dir", str(out),
            "--concurrency", str(args.concurrency), "--resume",
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

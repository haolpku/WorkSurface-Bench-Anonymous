"""Shared utilities for the WorkSurface-Bench conversion pipeline.

Everything that converts a Workspace-Bench-Lite task into canonical surfaces
routes through here for: locating the source split, loading per-task
metadata, stripping the content-hash prefix from stored filenames, and
normalizing identifiers for DuckDB views and graph nodes.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
EN_DIR = os.path.join(ROOT, "data", "workspace-bench-lite-en")
TASKS_DIR = os.path.join(EN_DIR, "task_lite_clean_en")
OUT_DIR = os.path.join(ROOT, "data", "worksurface_lite")

# Filenames on disk are prefixed with a 16-hex content hash, e.g.
# "a508c5c50582a72e_product_info.csv". The manifest's "filename" is the clean
# name; "stored_relpath" is the prefixed path.
HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{16}_")

# Persona display name -> filesystem-safe slug used for profile dir names.
PERSONA_SLUGS = {
    "Operations Manager": "operations_manager",
    "Logistics Manager": "logistics_manager",
    "Researcher": "researcher",
    "Backend Developer": "backend_developer",
    "Product Manager": "product_manager",
}

# File-type routing. A file's extension decides which surface(s) it can back.
TABULAR_EXTS = {".csv", ".xlsx", ".xls"}
DOC_EXTS = {".md", ".txt", ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".html"}
CODE_EXTS = {".py", ".java", ".xml", ".json"}


def persona_slug(persona: str) -> str:
    return PERSONA_SLUGS.get(
        persona, re.sub(r"[^a-z0-9]+", "_", persona.lower()).strip("_")
    )


def strip_hash_prefix(name: str) -> str:
    """product_info.csv from a508...._product_info.csv (basename only)."""
    base = os.path.basename(name)
    return HASH_PREFIX_RE.sub("", base)


def safe_ident(name: str) -> str:
    """A DuckDB-safe identifier: lowercase, alnum + underscore, no leading digit."""
    stem = os.path.splitext(strip_hash_prefix(name))[0]
    ident = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower()
    if not ident:
        ident = "t"
    if ident[0].isdigit():
        ident = "t_" + ident
    return ident


def normalize_col(col: str) -> str:
    """Header normalization: lowercase, strip, collapse whitespace/units."""
    c = str(col).strip().lower()
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^0-9a-z_]+", "", c)
    return c.strip("_") or "col"


@dataclass
class Task:
    """One Workspace-Bench-Lite source task, with resolved on-disk paths."""

    task_id: str
    dir: str
    meta: dict[str, Any]

    @property
    def persona(self) -> str:
        return self.meta.get("persona", "")

    @property
    def difficulty(self) -> str:
        return self.meta.get("task_diff", "")

    @property
    def rubrics(self) -> list[str]:
        return self.meta.get("rubrics", [])

    @property
    def rubric_types(self) -> list[str]:
        return self.meta.get("rubric_types", [])

    @property
    def output_files(self) -> list[str]:
        return self.meta.get("output_files", [])

    @property
    def dep_edges(self) -> list[dict[str, str]]:
        return self.meta.get("file_dep_graph", [])

    @property
    def capabilities(self) -> list[str]:
        return self.meta.get("tested_capabilities", [])

    def manifest(self) -> list[dict[str, str]]:
        """[{filename, stored_relpath, abspath, ext, exists}] for input files."""
        out = []
        for e in self.meta.get("data_manifest", []):
            abspath = os.path.join(self.dir, e["stored_relpath"])
            out.append(
                {
                    "filename": e["filename"],
                    "stored_relpath": e["stored_relpath"],
                    "abspath": abspath,
                    "ext": os.path.splitext(e["filename"])[1].lower(),
                    "exists": os.path.exists(abspath),
                }
            )
        return out


def load_tasks(only: list[str] | None = None) -> list[Task]:
    """Load every en-split task (or a subset by id), sorted numerically."""
    ids = sorted((d for d in os.listdir(TASKS_DIR) if d.isdigit()), key=int)
    if only:
        keep = set(only)
        ids = [i for i in ids if i in keep]
    tasks = []
    for tid in ids:
        tdir = os.path.join(TASKS_DIR, tid)
        meta = json.load(open(os.path.join(tdir, "metadata.json")))
        tasks.append(Task(task_id=tid, dir=tdir, meta=meta))
    return tasks


def tasks_by_persona(tasks: list[Task]) -> dict[str, list[Task]]:
    groups: dict[str, list[Task]] = {}
    for t in tasks:
        groups.setdefault(persona_slug(t.persona), []).append(t)
    return groups


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

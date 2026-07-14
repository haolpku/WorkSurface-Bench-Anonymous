"""Orchestrate the four surface converters into canonical profiles.

Entry point:  python -m worksurface.convert_profiles [--tasks 107 108 ...]

Output layout (data/worksurface_lite/):
  profiles/{persona_slug}/
    kb_docs/*.md
    tables/{view}.parquet + registry.json
    graph/surface_graph.json
  skills/{cluster}/SKILL.md          (shared, enterprise-wide)
  skills_meta.json
  task_skill_map.json
  gates.json                          (Appendix A2 go/no-go numbers)
  manifest.json                       (provenance + per-profile index)

Skills are shared across the enterprise, so they are derived once over all
tasks; RAG/Table/Graph are per-persona profiles.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

from .common import (
    OUT_DIR,
    load_tasks,
    persona_slug,
    tasks_by_persona,
    write_json,
)
from .convert_graph import build_graph
from .convert_rag import build_kb_docs
from .convert_skills import build_skills
from .convert_tables import build_tables


def _add_shared_artifact_edges(profiles_graphs: dict, hash_index: dict) -> int:
    """Second pass: link files that share a content hash across tasks/profiles.

    Adds a ``shared_artifact`` node + ``is_instance_of`` edges into whichever
    profile graph each file belongs to. Returns the number of shared nodes.
    """
    shared = {h: locs for h, locs in hash_index.items() if len(locs) >= 2}
    for h, locs in shared.items():
        node_id = f"shared::{h}"
        for persona, task_id, clean in locs:
            slug = persona_slug(persona)
            g = profiles_graphs.get(slug)
            if not g:
                continue
            if not any(n["id"] == node_id for n in g["nodes"]):
                g["nodes"].append(
                    {"id": node_id, "type": "shared_artifact", "hash": h,
                     "n_instances": len(locs)}
                )
            fid = f"t{task_id}::{clean}"
            g["edges"].append({"from": fid, "to": node_id,
                               "rel": "is_instance_of", "source": "phaseA"})
    return len(shared)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="restrict to these WSB task ids (pilot subset)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    tasks = load_tasks(only=args.tasks)
    print(f"[convert] {len(tasks)} tasks -> {args.out}")
    profiles_dir = os.path.join(args.out, "profiles")
    os.makedirs(profiles_dir, exist_ok=True)

    groups = tasks_by_persona(tasks)
    hash_index: dict = {}
    profiles_graphs: dict = {}
    manifest = {"profiles": {}, "n_source_tasks": len(tasks)}
    gates = {"per_profile": {}}

    for slug, ptasks in sorted(groups.items()):
        pdir = os.path.join(profiles_dir, slug)
        os.makedirs(pdir, exist_ok=True)
        print(f"\n[convert] profile={slug} ({len(ptasks)} tasks)")

        # Table first (its coverage decides RAG demotions).
        table_reg, demoted, tbl_cov = build_tables(ptasks, pdir)
        write_json(os.path.join(pdir, "tables", "registry.json"), table_reg)
        print(f"    tables: {tbl_cov['n_views']} views, "
              f"{tbl_cov['n_demoted_to_rag']} demoted, "
              f"coverage={tbl_cov['table_track_coverage']:.2f}")

        # RAG (picks up demoted sheets).
        kb_reg = build_kb_docs(ptasks, pdir, demote_to_rag=demoted)
        write_json(os.path.join(pdir, "kb_docs", "registry.json"), kb_reg)
        print(f"    kb_docs: {len(kb_reg)} canonical docs")

        # Graph (uses table columns; accumulates hashes for shared artifacts).
        graph, density = build_graph(
            ptasks, pdir, table_registry=table_reg,
            global_hash_index=hash_index,
        )
        profiles_graphs[slug] = graph
        print(f"    graph: {density['n_nodes']} nodes, {density['n_edges']} edges, "
              f"median_nontrivial={density['median_nontrivial_edges_per_task']}, "
              f"graph_only_eligible={density['graph_only_eligible']}")

        manifest["profiles"][slug] = {
            "n_tasks": len(ptasks),
            "task_ids": [t.task_id for t in ptasks],
            "n_kb_docs": len(kb_reg),
            "n_table_views": len(table_reg),
            "n_graph_nodes": density["n_nodes"],
            "n_graph_edges": density["n_edges"],
        }
        gates["per_profile"][slug] = {"table": tbl_cov, "graph": density}

    # shared-artifact second pass, then persist graphs
    n_shared = _add_shared_artifact_edges(profiles_graphs, hash_index)
    print(f"\n[convert] {n_shared} cross-task shared artifacts")
    for slug, g in profiles_graphs.items():
        write_json(os.path.join(profiles_dir, slug, "graph",
                                "surface_graph.json"), g)

    # skills: shared enterprise-wide
    skills_meta, task_skill_map = build_skills(tasks, args.out)
    write_json(os.path.join(args.out, "skills_meta.json"), skills_meta)
    write_json(os.path.join(args.out, "task_skill_map.json"), task_skill_map)
    n_retained = sum(1 for m in skills_meta.values() if m["retained"])
    print(f"[convert] skills: {n_retained}/{len(skills_meta)} clusters retained")

    # aggregate gate summary (Appendix A2 headline numbers)
    covs = [g["table"]["table_track_coverage"]
            for g in gates["per_profile"].values()]
    # Graph gate is task-level (solutions §1.2 Phase C): fraction of ALL Lite
    # tasks with >= 5 non-trivial edges. >= 0.30 keeps Graph in core-Lite.
    per_task_counts = {}
    for g in gates["per_profile"].values():
        per_task_counts.update(g["graph"]["per_task_nontrivial_counts"])
    all_counts = [per_task_counts.get(t.task_id, 0) for t in tasks]
    n_ge5 = sum(1 for c in all_counts if c >= 5)
    graph_task_pass = n_ge5 / len(all_counts) if all_counts else 0.0
    import statistics as _st
    gates["summary"] = {
        "table_coverage_min": min(covs) if covs else 0,
        "table_coverage_mean": sum(covs) / len(covs) if covs else 0,
        "graph_task_pass_fraction": graph_task_pass,
        "graph_median_nontrivial_all_tasks": _st.median(all_counts) if all_counts else 0,
        "graph_only_eligible_core": graph_task_pass >= 0.30,
        "skills_retained": n_retained,
        "skills_total": len(skills_meta),
        "shared_artifacts": n_shared,
    }
    write_json(os.path.join(args.out, "gates.json"), gates)
    write_json(os.path.join(args.out, "manifest.json"), manifest)

    print("\n[convert] gate summary:")
    for k, v in gates["summary"].items():
        print(f"    {k}: {v}")
    print(f"[convert] done -> {args.out}")


if __name__ == "__main__":
    main()

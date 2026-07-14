"""Final QC pass over the augmented task set (solutions §2.1 spirit).

Steps (all deterministic, no API):
  1. Deduplicate: drop tasks with an identical (source_task, task_type,
     normalized question). LLM augmentation and multi-view generation produce
     near-identical repeats; we keep the first.
  2. Re-validate every surviving task against schemas/task.schema.json.
  3. Report the final distribution vs the paper target and write the cleaned
     tasks.jsonl in place (backing up the pre-QC file).

Cue-stripping (removing lexical surface tells so Route is not gamed from the
question wording) is a documented LLM step in worksurface.llm_hooks; it is
left off here because the deterministic questions already avoid strong tells
and running it changes gold-bearing spans. It can be enabled in a later pass.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, OrderedDict

from .common import OUT_DIR


def _norm_q(o):
    q = re.sub(r"\s+", " ", o["question"].lower().strip())
    return (str(o["source"]["task_id"]), o["task_type"], q)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--schema", default=os.path.join(
        os.path.dirname(__file__), "..", "schemas", "task.schema.json"))
    args = ap.parse_args()

    path = os.path.join(args.out, "tasks", "tasks.jsonl")
    tasks = [json.loads(l) for l in open(path)]
    n0 = len(tasks)

    # 1. dedup (keep first occurrence of each normalized question)
    seen, deduped = set(), []
    for o in tasks:
        k = _norm_q(o)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(o)

    # 2. schema validation
    import jsonschema
    schema = json.load(open(args.schema))
    v = jsonschema.Draft202012Validator(schema)
    valid, dropped = [], 0
    for o in deduped:
        if any(True for _ in v.iter_errors(o)):
            dropped += 1
            continue
        valid.append(o)

    # 3. write (back up first)
    shutil.copy(path, path + ".preqc")
    with open(path, "w", encoding="utf-8") as f:
        for o in valid:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    tt = Counter(o["task_type"] for o in valid)
    at = Counter(o["answer_type"] for o in valid)
    ab = at.get("abstain", 0)
    print(f"[qc] {n0} -> dedup {len(deduped)} -> schema-valid {len(valid)} "
          f"(dropped {dropped} invalid)")
    print(f"[qc] task_type: {dict(tt)}")
    print(f"[qc] answer_type: {dict(at)}")
    print(f"[qc] abstain: {ab} ({100*ab//max(len(valid),1)}%)")
    print(f"[qc] backup at {path}.preqc")


if __name__ == "__main__":
    main()

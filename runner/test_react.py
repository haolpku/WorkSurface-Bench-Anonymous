"""Offline validation of the real ReAct path — no network, no API key.

A ScriptedBackbone mimics an OpenAI-compatible model: it inspects the running
message list and returns JSON actions that a competent model would emit for
the task's gold surfaces, then a final answer. This exercises react_loop, the
tool dispatch, JSON extraction, and cross-turn token accounting exactly as a
real APIBackbone would — so we can trust the wiring before spending money.

Run: python -m runner.test_react
"""

from __future__ import annotations

import json

from worksurface.common import load_tasks  # noqa: F401  (kept for parity)
from .agents import run_task
from .react import _extract_json


class ScriptedBackbone:
    """Deterministic fake API model. Emits gold-surface tool calls then answer.

    It simulates usage tokens so cum_usage accounting is exercised.
    """

    def __init__(self, task):
        self.name = "scripted"
        self.task = task
        self.cum_usage = {"input": 0, "output": 0}
        self.last_usage = {"input": 0, "output": 0}
        self._plan = self._build_plan(task)
        self._i = 0

    def reset(self):
        self.cum_usage = {"input": 0, "output": 0}
        self._i = 0

    def _build_plan(self, task):
        """One tool action per gold-evidence item, then a final answer."""
        actions = []
        for ev in task.get("gold_evidence", []):
            s = ev.get("surface")
            if s == "table" and ev.get("query"):
                actions.append({"tool": "table_query", "args": {"sql": ev["query"]}})
            elif s == "rag":
                actions.append({"tool": "kb_search",
                                "args": {"query": task["question"], "k": 3}})
            elif s == "graph":
                path = ev.get("graph_path") or []
                if path:
                    actions.append({"tool": "graph_traverse",
                                    "args": {"node": path[0]}})
        ga = task.get("gold_answer")
        actions.append({"final_answer": ga})
        return actions

    def _bill(self, msgs):
        # crude token proxy: chars/4
        inp = sum(len(m["content"]) for m in msgs) // 4
        self.last_usage = {"input": inp, "output": 20}
        self.cum_usage["input"] += inp
        self.cum_usage["output"] += 20

    def chat(self, system, user, *, max_tokens=1024):
        self._bill([{"content": system}, {"content": user}])
        # used by S3 router pick / S2 direct answer
        if "Pick exactly ONE surface" in system:
            return (self.task.get("required_surfaces") or ["rag"])[0]
        ga = self.task.get("gold_answer")
        return json.dumps(ga) if isinstance(ga, list) else str(ga)

    def chat_messages(self, messages, *, max_tokens=1024):
        self._bill(messages)
        act = self._plan[min(self._i, len(self._plan) - 1)]
        self._i += 1
        return json.dumps(act)


def main():
    # pick one task of each type that has table/graph/rag evidence
    tasks = load_tasks() if False else None  # noqa
    import json as _json
    all_tasks = [ _json.loads(l) for l in
                  open("data/worksurface_lite/tasks/tasks.jsonl") ]
    by_type = {}
    for t in all_tasks:
        by_type.setdefault(t["task_type"], t)
    from scoring.score_run import score_task

    print("Validating real ReAct path with a scripted OpenAI-style backbone:\n")
    for ttype, task in by_type.items():
        for setting in ("S3", "S4", "S5", "S6"):
            bb = ScriptedBackbone(task)
            trace = run_task(task, setting, bb, "data/worksurface_lite")
            scored = score_task(task, trace)
            assert trace["total_tokens"] > 0, "token accounting failed"
            print(f"  {ttype:13} {setting}  route_f1={scored['route']['f1']:.2f} "
                  f"ans={scored['answer']['score']:.2f} "
                  f"ev={scored['evidence']['score']:.2f} "
                  f"tokens={trace['total_tokens']} "
                  f"tools={len(trace['tool_trace'])}")
    # JSON extraction robustness
    assert _extract_json('```json\n{"final_answer": 42}\n```')["final_answer"] == 42
    assert _extract_json('sure! {"tool":"kb_search","args":{"query":"x"}} ok')["tool"] == "kb_search"
    assert _extract_json("no json here") is None
    print("\n  JSON extraction OK")
    print("\nreal ReAct path validated offline (no API spent)")


if __name__ == "__main__":
    main()

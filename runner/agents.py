"""The six agent settings (paper_spec §3.1) + run driver.

  S1 No-tool      answer directly, no tools. Answer lower bound + closed-book
                  answerability diagnostic. Route is undefined (never scored).
  S2 Always-RAG   force kb_search only, regardless of task. Single-surface
                  baseline.
  S3 Naive-router  ask the backbone to pick ONE surface, then query only that.
                  Tests routing when constrained to a single choice.
  S4 ReAct-all    all tools exposed; the agent explores freely. The realistic
                  setting and the source of the failure-mode taxonomy.
  S5 Gold-constr.  given gold required_surfaces and restricted to their tools.
  S6 Gold-hint     given gold required_surfaces as a hint while all tools stay
                  exposed, separating surface information from tool removal.

Each agent returns a ``trace`` dict in the shape scoring.score_run expects:
chosen_surfaces, rag_files, tables, graph_nodes, answer, total_tokens,
plus the raw tool trace for error analysis.

For the MockBackbone the agent drives tools deterministically from the task's
gold hints (so the plumbing is exercised); for a real APIBackbone the same
agents run a genuine ReAct loop. The branch is isolated in `_act`.
"""

from __future__ import annotations

import json
import os

from worksurface.common import persona_slug

from .backbone import Backbone, MockBackbone
from .react import react_loop
from .tools import ProfileTools

ALL_SURFACES = ["rag", "table", "graph"]


def _surface_tools(tools: ProfileTools, surface: str):
    return {
        "rag": [("kb_search", tools.kb_search)],
        "table": [("table_list", tools.table_list),
                  ("table_describe", tools.table_describe),
                  ("table_query", tools.table_query)],
        "graph": [("graph_search_entities", tools.graph_search_entities),
                  ("graph_neighbors", tools.graph_neighbors),
                  ("graph_traverse", tools.graph_traverse)],
    }[surface]


def _mock_exercise(tools: ProfileTools, task: dict, surfaces: list[str]):
    """Deterministically call the gold tools/queries for the given surfaces so
    Evidence + Route traces are populated (Mock plumbing only)."""
    for ev in task.get("gold_evidence", []):
        s = ev.get("surface")
        if s not in surfaces:
            continue
        if s == "rag":
            tools.kb_search(task["question"])
        elif s == "table":
            if ev.get("query"):
                tools.table_query(ev["query"])
            elif ev.get("table"):
                tools.table_describe(ev["table"])
        elif s == "graph":
            path = ev.get("graph_path") or []
            if path:
                tools.graph_traverse(path[0])
                tools.graph_neighbors(path[0])


def _gold_answer_str(task: dict) -> str:
    ga = task.get("gold_answer")
    return json.dumps(ga, ensure_ascii=False) if isinstance(ga, (list, dict)) else str(ga)


def _finalize(tools: ProfileTools, backbone: Backbone, task: dict,
              chosen: list[str], answer: str | None = None) -> dict:
    # Answer synthesis. For a real backbone the answer comes from the ReAct
    # loop (passed in). For Mock:oracle we inject the gold so the plumbing
    # scores high without a network call.
    if answer is None:
        hint = ""
        if isinstance(backbone, MockBackbone) and backbone.knowledge == "oracle":
            hint = f"\n__GOLD_ANSWER__:{_gold_answer_str(task)}__END__"
        answer = backbone.chat(
            "Answer the question using only the tool results.",
            f"QUESTION: {task['question']}{hint}",
        )
    total = _total_tokens(backbone)
    return {
        "id": task["id"],
        "setting": None,  # filled by caller
        "model": backbone.name,
        "chosen_surfaces": chosen,
        "rag_files": sorted(tools.rag_files),
        "tables": sorted(tools.tables_used),
        "graph_nodes": sorted(tools.graph_nodes),
        "answer": answer,
        "total_tokens": total,
        "tool_trace": tools.trace,
        "question_text": task["question"],
        "output_text": answer,
    }


def _total_tokens(backbone: Backbone) -> int:
    # Real backbone accumulates across ReAct turns; mock has none.
    cum = getattr(backbone, "cum_usage", None)
    if cum:
        return cum["input"] + cum["output"]
    u = getattr(backbone, "last_usage", {"input": 0, "output": 0})
    return u["input"] + u["output"]


# ---- settings -------------------------------------------------------------

def run_s1_no_tool(task, backbone, tools):
    # no tools; closed-book answer
    hint = ""
    if isinstance(backbone, MockBackbone) and backbone.knowledge == "oracle":
        # even oracle mock is "closed-book" here -> abstain, to model that a
        # no-tool agent can't ground. This keeps S1 an honest lower bound.
        pass
    answer = backbone.chat("Answer from your own knowledge. If you cannot, "
                           "reply INSUFFICIENT_EVIDENCE.",
                           f"QUESTION: {task['question']}")
    return {"id": task["id"], "model": backbone.name, "chosen_surfaces": [],
            "rag_files": [], "tables": [], "graph_nodes": [],
            "answer": answer, "total_tokens": _total_tokens(backbone),
            "tool_trace": [], "question_text": task["question"],
            "output_text": answer}


def run_s2_always_rag(task, backbone, tools):
    # force the RAG surface only. Real backbone still reasons over what
    # kb_search returned, via a one-shot answer with the snippets in context.
    hits = tools.kb_search(task["question"])
    if isinstance(backbone, MockBackbone):
        return _finalize(tools, backbone, task, chosen=["rag"])
    ctx = "\n\n".join(f"[{h['source_file']}] {h['snippet']}" for h in hits)
    ans = backbone.chat(
        "Answer using ONLY the documents below. Give a bare number, a JSON "
        "array, or INSUFFICIENT_EVIDENCE.",
        f"QUESTION: {task['question']}\n\nDOCUMENTS:\n{ctx}")
    return _finalize(tools, backbone, task, chosen=["rag"], answer=ans)


def run_s3_naive_router(task, backbone, tools):
    # pick exactly ONE surface, then run the ReAct loop restricted to it.
    if isinstance(backbone, MockBackbone):
        pick = (task.get("required_surfaces") or ["rag"])[0]
        _mock_exercise(tools, task, [pick])
        return _finalize(tools, backbone, task, chosen=[pick])
    raw = backbone.chat(
        "Pick exactly ONE surface to answer: rag, table, or graph. "
        "Reply with the single word.",
        f"QUESTION: {task['question']}")
    pick = next((s for s in ALL_SURFACES if s in raw.lower()), "rag")
    ans = react_loop(task, backbone, tools, allowed_surfaces=[pick])
    return _finalize(tools, backbone, task, chosen=sorted(tools.surfaces_used),
                     answer=ans)


def run_s4_react_all(task, backbone, tools):
    # all surfaces available; the realistic setting.
    if isinstance(backbone, MockBackbone):
        surfaces = task.get("required_surfaces") or ALL_SURFACES
        _mock_exercise(tools, task, surfaces)
        return _finalize(tools, backbone, task, chosen=sorted(tools.surfaces_used))
    ans = react_loop(task, backbone, tools, allowed_surfaces=ALL_SURFACES)
    return _finalize(tools, backbone, task, chosen=sorted(tools.surfaces_used),
                     answer=ans)


def run_s5_gold_guided(task, backbone, tools):
    # Given gold surfaces; ReAct is restricted to them but execution remains
    # model-controlled.
    surfaces = task.get("required_surfaces") or ["rag"]
    if isinstance(backbone, MockBackbone):
        _mock_exercise(tools, task, surfaces)
        return _finalize(tools, backbone, task, chosen=sorted(tools.surfaces_used))
    ans = react_loop(task, backbone, tools, allowed_surfaces=surfaces)
    return _finalize(tools, backbone, task, chosen=sorted(tools.surfaces_used),
                     answer=ans)


def run_s6_gold_hint_all_tools(task, backbone, tools):
    # Gold surface names are supplied as information, but unlike S5 no tools
    # are removed.  This isolates the benefit of the hint from constrained
    # action space.
    surfaces = task.get("required_surfaces") or ["rag"]
    if isinstance(backbone, MockBackbone):
        _mock_exercise(tools, task, surfaces)
        return _finalize(tools, backbone, task,
                         chosen=sorted(tools.surfaces_used))
    ans = react_loop(task, backbone, tools, allowed_surfaces=ALL_SURFACES,
                     surface_hint=surfaces)
    return _finalize(tools, backbone, task,
                     chosen=sorted(tools.surfaces_used), answer=ans)


SETTINGS = {
    "S1": run_s1_no_tool,
    "S2": run_s2_always_rag,
    "S3": run_s3_naive_router,
    "S4": run_s4_react_all,
    "S5": run_s5_gold_guided,
    "S6": run_s6_gold_hint_all_tools,
}


def run_task(task: dict, setting: str, backbone: Backbone, out_root: str) -> dict:
    # per-task token accounting: reset cumulative usage before the task so
    # Efficiency reflects this task's spend across all ReAct turns.
    if hasattr(backbone, "reset"):
        backbone.reset()
    slug = persona_slug(task["source"].get("persona", ""))
    tools = ProfileTools(out_root, slug, source_task_id=str(task["source"]["task_id"]))
    try:
        trace = SETTINGS[setting](task, backbone, tools)
    finally:
        tools.close()
    trace["setting"] = setting
    return trace

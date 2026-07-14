"""Text-based ReAct loop for real (API) backbones.

OpenAI-compatible endpoints vary in native tool-calling support (one-api,
vLLM, OpenRouter, company proxies all differ), so we drive tools through a
portable text protocol instead: each turn the model emits ONE JSON object,
either an action or a final answer, and we feed back the observation.

    {"tool": "table_query", "args": {"sql": "SELECT ..."}}
    {"tool": "kb_search",   "args": {"query": "..."}}
    {"final_answer": <value>}

Works on any chat endpoint. The loop caps steps, records every tool call into
the ProfileTools trace (so Route/Evidence come out of real behavior), and
accumulates token usage across turns via backbone.cum_usage.

``allowed_surfaces`` restricts the exposed toolset — S4 exposes all three,
S5 (gold-guided) exposes only the required surfaces, S3 (naive router) exposes only
the single surface the model pre-selected.
"""

from __future__ import annotations

import json
import re

TOOL_SPECS = {
    "rag": [
        ("kb_search", '{"tool":"kb_search","args":{"query":"<text>","k":3}}',
         "search the knowledge base; returns doc snippets"),
    ],
    "table": [
        ("table_list", '{"tool":"table_list","args":{}}',
         "list available tables with row counts"),
        ("table_describe", '{"tool":"table_describe","args":{"view":"<name>"}}',
         "show a table's columns"),
        ("table_query", '{"tool":"table_query","args":{"sql":"SELECT ..."}}',
         "run a read-only DuckDB SQL query"),
    ],
    "graph": [
        ("graph_search_entities",
         '{"tool":"graph_search_entities","args":{"query":"<text>"}}',
         "find graph nodes matching text"),
        ("graph_neighbors", '{"tool":"graph_neighbors","args":{"node":"<id>"}}',
         "list a node's direct neighbors + edge relations"),
        ("graph_traverse",
         '{"tool":"graph_traverse","args":{"node":"<id>","rel":"<optional>"}}',
         "traverse from a node (optionally by relation)"),
    ],
}


def _dispatch(tools, name: str, args: dict):
    fn = {
        "kb_search": tools.kb_search,
        "table_list": tools.table_list,
        "table_describe": tools.table_describe,
        "table_query": tools.table_query,
        "graph_search_entities": tools.graph_search_entities,
        "graph_neighbors": tools.graph_neighbors,
        "graph_traverse": tools.graph_traverse,
    }.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**(args or {}))
    except TypeError as e:
        return {"error": f"bad args for {name}: {e}"}


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model turn (tolerates prose/fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except ValueError:
            pass
    # first balanced-looking {...}
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


def _tool_menu(allowed_surfaces: list[str]) -> str:
    lines = []
    for s in allowed_surfaces:
        for name, sig, desc in TOOL_SPECS.get(s, []):
            lines.append(f"  {sig}  # {desc}")
    return "\n".join(lines)


def _truncate(obj, limit: int = 1500) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + " …(truncated)"


def react_loop(task: dict, backbone, tools, allowed_surfaces: list[str],
               max_steps: int = 8):
    """Run the loop; returns the model's final answer string. Side effect:
    tools.trace / surfaces_used / evidence sets are populated."""
    menu = _tool_menu(allowed_surfaces)
    # The agent is working on a known workspace task, so it knows its own
    # entry node in the dependency graph: task_<id>. File nodes look like
    # "t<id>::<filename>"; answers about files should use the bare filename.
    task_node = f"task_{task['source']['task_id']}"
    graph_hint = ""
    if "graph" in allowed_surfaces:
        graph_hint = (
            f"\nYour workspace task's graph node is \"{task_node}\". Start "
            f"graph exploration there (e.g. graph_neighbors on it). File nodes "
            f"are \"t<id>::<filename>\"; when answering with files, return the "
            f"bare <filename> only."
        )
    system = (
        "You are an enterprise data agent. Answer the question by calling "
        "tools over the available knowledge surfaces. Each turn, reply with "
        "EXACTLY ONE JSON object and nothing else:\n"
        "  an action: {\"tool\":\"<name>\",\"args\":{...}}\n"
        "  or finish: {\"final_answer\": <value>}\n"
        "Give final_answer as a bare number for numeric questions, or a JSON "
        "array for list questions. If the evidence is insufficient, use "
        "{\"final_answer\":\"INSUFFICIENT_EVIDENCE\"}.\n\n"
        f"Available tools:\n{menu}{graph_hint}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"QUESTION: {task['question']}"},
    ]
    for _ in range(max_steps):
        reply = backbone.chat_messages(messages, max_tokens=800)
        messages.append({"role": "assistant", "content": reply})
        action = _extract_json(reply)
        if action is None:
            messages.append({"role": "user", "content":
                             "Reply with a single JSON object only."})
            continue
        if "final_answer" in action:
            fa = action["final_answer"]
            return json.dumps(fa, ensure_ascii=False) if isinstance(fa, (list, dict)) else str(fa)
        name = action.get("tool")
        obs = _dispatch(tools, name, action.get("args", {}))
        messages.append({"role": "user",
                         "content": f"OBSERVATION: {_truncate(obs)}"})
    # ran out of steps: ask for a final answer with what it has
    messages.append({"role": "user", "content":
                     "Max steps reached. Reply now with {\"final_answer\": <value>}."})
    reply = backbone.chat_messages(messages, max_tokens=400)
    action = _extract_json(reply) or {}
    fa = action.get("final_answer", "INSUFFICIENT_EVIDENCE")
    return json.dumps(fa, ensure_ascii=False) if isinstance(fa, (list, dict)) else str(fa)

"""Surface tools over a WorkSurface-Bench profile.

These are the callable tools an agent uses; each records into a trace so the
scorer can measure Evidence and Route. Tools are grouped by surface:

  RAG    kb_search(query)             -> top-k canonical doc snippets
  Table  table_list()                -> view names + row counts
         table_describe(view)         -> column schema
         table_query(sql)             -> DuckDB result rows (read-only)
  Graph  graph_search_entities(q)     -> matching node ids
         graph_neighbors(node)        -> adjacent nodes + edge rels
         graph_traverse(node, rel)    -> nodes reachable by a relation

A ``ProfileTools`` instance is scoped to one persona profile and (optionally)
one source task, so table views and graph nodes are the ones actually
available for that task. Every call appends to ``self.trace`` and updates the
Evidence-facing sets (rag_files / tables / graph_nodes) + the Route-facing
set (surfaces_used).
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

from worksurface.convert_tables import connect_registry


class ProfileTools:
    def __init__(self, out_root: str, persona_slug: str,
                 source_task_id: str | None = None):
        self.profile_dir = os.path.join(out_root, "profiles", persona_slug)
        self.source_task_id = source_task_id
        self.trace = []
        self.surfaces_used: set[str] = set()
        self.rag_files: set[str] = set()
        self.tables_used: set[str] = set()
        self.graph_nodes: set[str] = set()
        self._load()

    # ---- loading ----
    def _load(self):
        # RAG registry + doc texts
        self.kb = {}
        kb_dir = os.path.join(self.profile_dir, "kb_docs")
        reg = os.path.join(kb_dir, "registry.json")
        names = json.load(open(reg)) if os.path.exists(reg) else {}
        for doc, meta in names.items():
            if self.source_task_id and meta["source_task"] != self.source_task_id:
                continue
            path = os.path.join(kb_dir, doc)
            if os.path.exists(path):
                self.kb[doc] = {"meta": meta,
                                "text": open(path, encoding="utf-8").read()}
        # Table registry (scoped to task if given)
        tables_dir = os.path.join(self.profile_dir, "tables")
        tasks = [self.source_task_id] if self.source_task_id else None
        self.con, self.views = connect_registry(tables_dir, tasks=tasks)
        # Graph
        gpath = os.path.join(self.profile_dir, "graph", "surface_graph.json")
        self.graph = json.load(open(gpath)) if os.path.exists(gpath) else \
            {"nodes": [], "edges": []}
        self._adj = defaultdict(list)
        for e in self.graph["edges"]:
            self._adj[e["from"]].append((e["to"], e["rel"]))
        self._nodes = {n["id"]: n for n in self.graph["nodes"]}

    def _log(self, tool, args, surface, result_summary):
        self.surfaces_used.add(surface)
        self.trace.append({"tool": tool, "args": args, "surface": surface,
                           "result": result_summary})

    # ---- RAG ----
    def kb_search(self, query: str, k: int = 3):
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        scored = []
        for doc, d in self.kb.items():
            text = d["text"].lower()
            score = sum(text.count(t) for t in terms)
            if score:
                scored.append((score, doc, d))
        scored.sort(reverse=True, key=lambda x: x[0])
        hits = []
        for score, doc, d in scored[:k]:
            self.rag_files.add(d["meta"]["source_file"])
            snippet = d["text"][:600]
            hits.append({"doc": doc, "source_file": d["meta"]["source_file"],
                         "score": score, "snippet": snippet})
        self._log("kb_search", {"query": query, "k": k}, "rag",
                  [h["source_file"] for h in hits])
        return hits

    # ---- Table ----
    def table_list(self):
        out = [{"table": v, "rows": m["rows"],
                "source_file": m["source_file"]} for v, m in self.views.items()]
        self._log("table_list", {}, "table", [o["table"] for o in out])
        return out

    def table_describe(self, view: str):
        m = self.views.get(view)
        if not m:
            self._log("table_describe", {"view": view}, "table", "not_found")
            return {"error": f"no such table {view}"}
        self.tables_used.add(view)
        cols = [c["name"] for c in m["columns"]]
        self._log("table_describe", {"view": view}, "table", cols)
        return {"table": view, "rows": m["rows"], "columns": cols}

    def table_query(self, sql: str):
        # read-only guard
        if not re.match(r"^\s*(select|with)\b", sql, re.I):
            self._log("table_query", {"sql": sql}, "table", "rejected_non_select")
            return {"error": "only SELECT/WITH queries allowed"}
        for v in self.views:
            if v in sql:
                self.tables_used.add(v)
        try:
            rows = self.con.execute(sql).fetchall()
            cols = [d[0] for d in self.con.description]
            result = [dict(zip(cols, r)) for r in rows[:100]]
            self._log("table_query", {"sql": sql}, "table",
                      {"n_rows": len(rows)})
            return {"columns": cols, "rows": result, "n_rows": len(rows)}
        except Exception as e:  # noqa: BLE001
            self._log("table_query", {"sql": sql}, "table", f"error:{e}")
            return {"error": str(e)}

    # ---- Graph ----
    def graph_search_entities(self, query: str):
        import re as _re
        q = query.lower().strip()
        # token set of the query, plus any bare numbers ("task 3" -> "3")
        qtokens = set(_re.findall(r"\w+", q))
        hits = []
        for nid, n in self._nodes.items():
            hay = (nid + " " + str(n.get("filename", ""))).lower()
            if q and q in hay:
                hits.append(nid)
                continue
            # match "task 3" against node id "task_3" via shared tokens
            ntokens = set(_re.findall(r"\w+", hay))
            if qtokens and qtokens <= ntokens:
                hits.append(nid)
        self._log("graph_search_entities", {"query": query}, "graph", hits[:10])
        for h in hits[:10]:
            self.graph_nodes.add(h)
        return hits[:10]

    def graph_neighbors(self, node: str):
        out = [{"to": to, "rel": rel} for to, rel in self._adj.get(node, [])]
        self.graph_nodes.add(node)
        for o in out:
            self.graph_nodes.add(o["to"])
        self._log("graph_neighbors", {"node": node}, "graph",
                  [o["to"] for o in out])
        return out

    def graph_traverse(self, node: str, rel: str | None = None, depth: int = 2):
        seen, frontier = set(), [node]
        for _ in range(depth):
            nxt = []
            for cur in frontier:
                for to, r in self._adj.get(cur, []):
                    if rel and r != rel:
                        continue
                    if to not in seen:
                        seen.add(to)
                        nxt.append(to)
            frontier = nxt
        self.graph_nodes.add(node)
        self.graph_nodes.update(seen)
        self._log("graph_traverse", {"node": node, "rel": rel, "depth": depth},
                  "graph", sorted(seen)[:20])
        return sorted(seen)

    def close(self):
        try:
            self.con.close()
        except Exception:  # noqa: BLE001
            pass

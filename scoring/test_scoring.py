"""Sanity tests for the scoring protocol. Run: python -m scoring.test_scoring"""

from .answer import (score_number, score_list, score_boolean, score_abstain,
                     score_answer)
from .route_evidence import score_route, score_evidence
from .efficiency_safety import score_efficiency, score_safety, aggregate


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_number():
    assert score_number(43, "43").score == 1.0
    assert score_number(43, 44).score == 0.0             # int <=100 exact
    assert score_number(1710971.47, 1710900).score == 1.0  # within 5%
    assert score_number(1710971.47, 1000000).score == 0.0
    assert score_number(16.89, "16.89%").score == 1.0     # strips %
    print("  number OK")


def test_list():
    r = score_list(["a", "b", "c"], ["c", "b", "a"])
    assert r.score == 1.0                                 # order-invariant
    r = score_list(["spring boot", "hibernate"], ["Spring Boot"])
    assert approx(r.score, 2 * 1.0 * 0.5 / 1.5, tol=1e-3)  # recall .5 P 1.0
    r = score_list(["junit"], ["junitt"])                 # edit dist 1
    assert r.score == 1.0
    # regression (bug-013): numbered filenames must NOT fuzzy-match each other,
    # and score must never exceed 1.0 even with far more gold than pred.
    gold = [f"dependency_item_{i}.md" for i in range(1, 39)]
    r = score_list(gold, ["dependency_item_34.md", "dependency_item_35.md",
                          "dependency_item_1.md"])
    assert r.score <= 1.0, r.score
    assert r.detail["tp"] == 3, r.detail            # only the 3 exact matches
    assert r.detail["precision"] <= 1.0
    print("  list OK")


def test_boolean_abstain():
    assert score_boolean(True, "yes").score == 1.0
    assert score_boolean(False, "true").score == 0.0
    assert score_abstain("INSUFFICIENT_EVIDENCE").score == 1.0
    assert score_abstain("42").score == 0.0
    print("  boolean/abstain OK")


def test_composite_string():
    task = {"gold_answer": "report.xlsx; 20", "answer_type": "string"}
    assert score_answer(task, '["report.xlsx", 20]').score == 1.0
    assert score_answer(task, "report.xlsx; 20").score == 1.0
    assert score_answer(task, '["report.xlsx", 21]').score == 0.0
    assert score_answer(task, '[20, "report.xlsx"]').score == 0.0
    # Ordinary strings retain the original normalized exact-match behavior.
    ordinary = {"gold_answer": "Report Ready", "answer_type": "string"}
    assert score_answer(ordinary, " report   ready ").score == 1.0
    print("  composite string OK")


def test_route():
    r = score_route(["rag", "table"], ["rag", "table"])
    assert r.f1 == 1.0
    r = score_route(["rag", "table"], ["rag"])            # missed table
    assert r.precision == 1.0 and r.recall == 0.5
    r = score_route(["rag"], ["rag", "graph", "table"])   # over-selected
    assert approx(r.precision, 1/3, tol=1e-3) and r.recall == 1.0
    # skill is never routable: passing it is ignored
    r = score_route(["rag"], ["rag", "skill"])
    assert r.f1 == 1.0
    print("  route OK")


def test_evidence():
    gold = [{"surface": "rag", "file": "report.md"},
            {"surface": "table", "table": "t1__orders"},
            {"surface": "table", "table": "t1__products"}]
    trace = {"rag_files": ["report.md"], "tables": ["t1__orders"],
             "graph_nodes": []}
    r = score_evidence(gold, trace)
    # 1 rag hit + 1 of 2 table hits = 2/3 items
    assert approx(r.score, 2/3, tol=1e-3), r.score
    assert r.per_surface["rag"] == 1.0 and r.per_surface["table"] == 0.5
    graph_gold = [{"surface": "graph",
                   "graph_path": ["task_1", "task_requires_file", "t1::a.csv"]}]
    assert score_evidence(graph_gold, {"graph_nodes": ["task_1"]}).score == 0.0
    assert score_evidence(graph_gold,
                          {"graph_nodes": ["task_1", "t1::a.csv"]}).score == 1.0
    set_gold = [{"surface": "graph",
                 "verified_complete_set": ["a.csv", "b.csv"]}]
    assert score_evidence(set_gold,
                          {"graph_nodes": ["t1::a.csv"]}).score == 0.0
    assert score_evidence(set_gold,
                          {"graph_nodes": ["t1::a.csv", "t1::b.csv"]}).score == 1.0
    print("  evidence OK")


def test_efficiency():
    assert score_efficiency(0, 1000) == 1.0
    assert score_efficiency(1000, 1000) == 0.5            # at budget
    assert score_efficiency(2000, 1000) == 0.0            # at 2x budget
    assert score_efficiency(3000, 1000) == 0.0            # beyond
    assert score_efficiency(1001, 1000) < 0.5             # continuous at budget
    assert score_efficiency(1500, 1000) == 0.25
    assert score_efficiency(500, None) == 1.0             # no budget -> neutral
    print("  efficiency OK")


def test_safety():
    task_na = {"safety_threats": []}
    assert score_safety(task_na, {}).applicable is False
    task = {"safety_threats": ["path_escape"]}
    bad = {"file_ops": [{"op": "write", "path": "/etc/passwd"}]}
    assert score_safety(task, bad).score == 0.0
    good = {"file_ops": [{"op": "write", "path": "output/report.md"}]}
    assert score_safety(task, good).score == 1.0
    print("  safety OK")


def test_aggregate():
    # The aggregate uses the four active benchmark dimensions.
    a = aggregate(1.0, 1.0, 1.0, 1.0, None)
    assert approx(a.final, 1.0)
    assert "safety" not in a.parts
    # Safety is diagnostic and does not alter the aggregate.
    a = aggregate(1.0, 1.0, 1.0, 1.0, 0.0)
    assert approx(a.final, 1.0)
    print("  aggregate OK")


def test_end_to_end():
    task = {"id": "x", "task_type": "cross_surface",
            "required_surfaces": ["rag", "table"],
            "gold_answer": 16.89, "answer_type": "number",
            "gold_evidence": [{"surface": "table", "table": "t1__x",
                               "query": "SELECT ..."}],
            "efficiency_budget_tokens": 8000}
    from .score_run import score_task
    trace = {"chosen_surfaces": ["rag", "table"], "tables": ["t1__x"],
             "answer": "16.89%", "total_tokens": 4000}
    r = score_task(task, trace)
    assert r["answer"]["score"] == 1.0
    assert r["route"]["f1"] == 1.0
    assert r["evidence"]["score"] == 1.0
    assert r["efficiency"] == 0.75                        # 4000 of 8000
    assert r["aggregate"] > 0.9
    print("  end_to_end OK")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all scoring tests passed")

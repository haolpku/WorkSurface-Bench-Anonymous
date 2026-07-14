"""Answer scoring, normalized per answer_type (paper_spec §C3, solutions §1.3).

  number   exact for integers <= 100; else |a-b|/max(|a|,|b|) <= 0.05
  list     order-invariant, case-insensitive, per-item fuzzy (edit dist <= 2),
           scored as set F1
  boolean  exact
  string   case-insensitive exact after whitespace normalization
  abstain  exact match on the INSUFFICIENT_EVIDENCE token
  freeform judge against operational anchors (needs an LLMClient); without one
           we fall back to token-overlap F1 and flag needs_judge=True

Every scorer returns a float in [0, 1] plus a small dict of detail so the
runner can log why a task scored the way it did.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"


@dataclass
class AnswerResult:
    score: float
    detail: dict = field(default_factory=dict)


def _norm_str(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_number(gold: float, pred: Any) -> AnswerResult:
    try:
        p = float(re.sub(r"[,$%\s]", "", str(pred)))
    except (TypeError, ValueError):
        return AnswerResult(0.0, {"reason": "unparseable", "pred": pred})
    g = float(gold)
    if float(g).is_integer() and abs(g) <= 100:
        ok = abs(p - g) < 0.5
    else:
        ok = abs(p - g) / max(abs(g), 1e-9) <= 0.05
    return AnswerResult(1.0 if ok else 0.0, {"gold": g, "pred": p})


def _fuzzy_eq(a: str, b: str) -> bool:
    """Item equality tolerant of tiny typos, but NOT of differing IDs.

    Edit-distance budget scales with length (max 1 per 8 chars, capped at 2)
    so "dependency_item_1.md" and "dependency_item_34.md" — which differ only
    in their numeric id — are NOT considered equal, while "junit"/"junitt" is.
    A trailing/interior digit difference is always a mismatch.
    """
    if a == b:
        return True
    # if the two strings share a stem but differ in digits, reject
    if re.sub(r"\d+", "#", a) == re.sub(r"\d+", "#", b) and a != b:
        return False
    budget = min(2, 1 + max(len(a), len(b)) // 8)
    return _edit_distance(a, b) <= budget


def score_list(gold: list, pred: Any) -> AnswerResult:
    if isinstance(pred, str):
        s = pred.strip()
        parsed = None
        # a model may emit a JSON array; parse it before falling back to
        # separator splitting (else brackets/quotes corrupt fuzzy matching).
        if s.startswith("[") and s.endswith("]"):
            try:
                import json as _json
                parsed = _json.loads(s)
            except ValueError:
                parsed = None
        if isinstance(parsed, list):
            pred = parsed
        else:
            parts = re.split(r"[,\n;]+", s)
            pred = [p.strip().strip('"\'[]') for p in parts if p.strip().strip('"\'[]')]
    if not isinstance(pred, (list, tuple)):
        return AnswerResult(0.0, {"reason": "not a list", "pred": pred})
    g = [_norm_str(x) for x in gold]
    p = [_norm_str(x) for x in pred]

    # Greedy one-to-one matching: each predicted item consumes at most one
    # gold item (and vice versa), so TP can never exceed min(|g|,|p|) and
    # precision/recall/F1 stay in [0,1].
    unmatched_gold = list(range(len(g)))
    tp = 0
    for pi in p:
        for gi_idx, gi in enumerate(unmatched_gold):
            if _fuzzy_eq(pi, g[gi]):
                tp += 1
                unmatched_gold.pop(gi_idx)
                break

    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return AnswerResult(f1, {"precision": round(precision, 3),
                             "recall": round(recall, 3), "tp": tp,
                             "n_gold": len(g), "n_pred": len(p)})


def score_boolean(gold: bool, pred: Any) -> AnswerResult:
    s = _norm_str(pred)
    truthy = s in {"true", "yes", "1", "y"}
    falsy = s in {"false", "no", "0", "n"}
    if not (truthy or falsy):
        return AnswerResult(0.0, {"reason": "not boolean", "pred": pred})
    return AnswerResult(1.0 if (truthy == bool(gold)) else 0.0, {})


def score_string(gold: str, pred: Any) -> AnswerResult:
    # Some cross-surface tasks have a typed, ordered composite answer stored in
    # the release format as ``item; value``.  ReAct models naturally emit the
    # same answer as a JSON array.  Compare the elements rather than the
    # serialization so equivalent structured answers are not marked wrong.
    if ";" in gold:
        gold_parts = [part.strip() for part in gold.split(";")]
        pred_parts = _parse_composite(pred)
        if pred_parts is not None:
            ok = (len(gold_parts) == len(pred_parts) and
                  all(_composite_item_equal(g, p)
                      for g, p in zip(gold_parts, pred_parts)))
            return AnswerResult(
                1.0 if ok else 0.0,
                {"method": "ordered_composite_exact",
                 "n_gold": len(gold_parts), "n_pred": len(pred_parts)},
            )
    return AnswerResult(1.0 if _norm_str(gold) == _norm_str(pred) else 0.0, {})


def _parse_composite(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return parsed
    if ";" in text:
        return [part.strip() for part in text.split(";")]
    return None


def _composite_item_equal(gold: Any, pred: Any) -> bool:
    """Strict element equality with numeric serialization normalization."""
    g = _norm_str(gold).strip("\"'")
    p = _norm_str(pred).strip("\"'")
    number = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
    if number.fullmatch(g) and number.fullmatch(p):
        return math.isclose(float(g), float(p), rel_tol=0.0, abs_tol=1e-9)
    return g == p


def score_abstain(pred: Any) -> AnswerResult:
    return AnswerResult(1.0 if _norm_str(pred) == _norm_str(ABSTAIN_TOKEN) else 0.0, {})


def _token_f1(gold: str, pred: str) -> float:
    g = set(_norm_str(gold).split())
    p = set(_norm_str(pred).split())
    if not g or not p:
        return 0.0
    tp = len(g & p)
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def score_freeform(gold: Any, pred: Any, anchors: list[str] | None = None,
                   judge=None) -> AnswerResult:
    """Judge against anchors if a judge + anchors are available; else token F1."""
    anchors = anchors or []
    if judge is not None and anchors:
        hits = 0
        for a in anchors:
            verdict = judge.complete(
                "Answer YES if the ANSWER satisfies the CRITERION, else NO.",
                f"CRITERION: {a}\nANSWER: {pred}",
            )
            hits += verdict.strip().upper().startswith("YES")
        return AnswerResult(hits / len(anchors), {"anchor_hits": hits,
                                                  "n_anchors": len(anchors)})
    return AnswerResult(_token_f1(str(gold), str(pred)),
                        {"needs_judge": True, "method": "token_f1"})


def score_answer(task: dict, pred: Any, anchors: list[str] | None = None,
                 judge=None) -> AnswerResult:
    at = task.get("answer_type", "string")
    gold = task.get("gold_answer")
    if at == "abstain" or gold == ABSTAIN_TOKEN:
        return score_abstain(pred)
    if at == "number":
        return score_number(gold, pred)
    if at == "list":
        return score_list(gold if isinstance(gold, list) else [gold], pred)
    if at == "boolean":
        return score_boolean(bool(gold), pred)
    if at == "freeform":
        return score_freeform(gold, pred, anchors, judge)
    return score_string(str(gold), pred)

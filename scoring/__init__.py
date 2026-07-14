"""WorkSurface-Bench scoring: Route / Evidence / Answer / Efficiency / Safety.

See paper_spec_zh.md §C3 and solutions_v0.md §2 for the protocol. The
aggregate weighting is:

    Final = 0.35 Answer + 0.30 Evidence + 0.25 Route
          + 0.10 Efficiency
"""

from .answer import score_answer, AnswerResult, ABSTAIN_TOKEN
from .route_evidence import score_route, score_evidence, RouteResult, EvidenceResult
from .efficiency_safety import (
    score_efficiency,
    score_safety,
    aggregate,
    WEIGHTS,
)
from .score_run import score_run, score_task

__all__ = [
    "score_answer", "AnswerResult", "ABSTAIN_TOKEN",
    "score_route", "score_evidence", "RouteResult", "EvidenceResult",
    "score_efficiency", "score_safety", "aggregate", "WEIGHTS",
    "score_run", "score_task",
]

"""Model-agnostic LLM hooks for the WorkSurface-Bench conversion pipeline.

The deterministic deriver (derive_tasks.py) produces a high-precision core
with self-verified gold answers and NO model calls. Three steps in the paper
plan are inherently model-assisted; solutions_v0 §6 says the pipeline stays
model-agnostic and the pilot picks a concrete model per step. This module is
that seam — a single ``LLMClient`` protocol plus the four call sites, so a
pilot wires in one backend (Claude / GPT / Gemini / open) without touching
the deterministic code.

Steps that route through here (all OFF by default; enable per pilot budget):

  1. Extractive rewriter (§1.3 Step 2) — turn a yes/no rubric into a clean
     question + verify the rewrite implies the rubric passes.
  2. Qualitative anchors (§1.3 Step 3) — attach 2-3 operational anchors to a
     qualitative rubric so a judge can score against anchors, not prose.
  3. Surface-cue stripping (§2.1) — paraphrase a question to remove surface
     tells ("row", "table", "graph") without changing the ask, so Route is
     not gameable from lexical cues.
  4. Answer judge (scoring) — score freeform answers against anchors.

Nothing here is called by the deterministic pilot; it exists so the LLM path
is a documented, testable interface rather than scattered inline calls.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal chat interface any backend can satisfy."""

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        ...


# ---- prompt templates (kept here so the pilot can audit them) -------------

REWRITE_SYSTEM = (
    "You convert a boolean grading rubric into a single, self-contained "
    "question whose correct answer is a concrete value. Return ONLY the "
    "question. Do not include the answer."
)

VERIFY_SYSTEM = (
    "You are given a rubric and a proposed (question, gold_answer). Answer "
    "YES only if answering the question correctly necessarily implies the "
    "rubric passes; otherwise NO. Return YES or NO."
)

CUE_STRIP_SYSTEM = (
    "Rewrite the question to remove any lexical hint about which data "
    "surface (documents, tables, or dependency graph) is needed, while "
    "keeping the ask identical. Return ONLY the rewritten question."
)

ANCHORS_SYSTEM = (
    "Given a qualitative rubric, produce 2-3 concrete, checkable operational "
    "anchors (one per line) that a grader can verify without subjective "
    "judgment. Return one anchor per line."
)


def rewrite_rubric(client: LLMClient, rubric: str) -> str:
    return client.complete(REWRITE_SYSTEM, rubric).strip()


def verify_rewrite(client: LLMClient, rubric: str, question: str,
                   gold_answer: str) -> bool:
    out = client.complete(
        VERIFY_SYSTEM,
        f"RUBRIC: {rubric}\nQUESTION: {question}\nGOLD: {gold_answer}",
    )
    return out.strip().upper().startswith("YES")


def strip_surface_cues(client: LLMClient, question: str) -> str:
    return client.complete(CUE_STRIP_SYSTEM, question).strip()


def make_anchors(client: LLMClient, rubric: str) -> list[str]:
    out = client.complete(ANCHORS_SYSTEM, rubric)
    return [ln.strip("-• \t") for ln in out.splitlines() if ln.strip()]

"""Model backbones for the runner.

``Backbone`` is the model-agnostic interface every agent setting calls. Two
implementations ship:

  MockBackbone   deterministic, no network. It reads the task's gold hints
                 to produce a plausible tool plan + answer, so the whole
                 harness (tools -> trace -> scorer) is testable end-to-end and
                 in CI without an API key. It is NOT a baseline — it is a
                 wiring fixture. Runs tagged mock are excluded from any
                 leaderboard.

  APIBackbone    thin wrapper around an OpenAI-compatible chat endpoint
                 (Anthropic / OpenAI / Gemini via compat shims / open models
                 via vLLM). Off unless WSB_API_KEY + WSB_API_BASE are set.

The pilot (paper_spec §3.1) picks the concrete backbone models; this file only
provides the seam.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol


class Backbone(Protocol):
    name: str

    def chat(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        ...


class MockBackbone:
    """Deterministic stand-in. Emits gold-informed tool plans and answers.

    ``knowledge`` controls how much it 'knows': 'oracle' answers correctly,
    'blind' guesses. The runner uses 'oracle' only to smoke-test the plumbing.
    """

    name = "mock"

    def __init__(self, knowledge: str = "oracle"):
        self.knowledge = knowledge

    def chat(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        # The runner drives tools directly for Mock; chat() is only used for
        # the final answer synthesis step, where we echo the injected gold.
        m = re.search(r"__GOLD_ANSWER__:(.*?)__END__", user, re.S)
        if m and self.knowledge == "oracle":
            return m.group(1).strip()
        return "INSUFFICIENT_EVIDENCE"


class APIBackbone:
    """OpenAI-compatible chat wrapper. Requires WSB_API_BASE + WSB_API_KEY."""

    def __init__(self, model: str, api_base: str | None = None,
                 api_key: str | None = None):
        self.name = model
        self.api_base = api_base or os.environ.get("WSB_API_BASE")
        self.api_key = api_key or os.environ.get("WSB_API_KEY")
        if not (self.api_base and self.api_key):
            raise RuntimeError("APIBackbone needs WSB_API_BASE and WSB_API_KEY")
        self.last_usage = {"input": 0, "output": 0}
        # cumulative over ALL calls since reset() — a ReAct task makes many
        # calls, and Efficiency scores the whole task's token spend.
        self.cum_usage = {"input": 0, "output": 0}

    def reset(self):
        self.cum_usage = {"input": 0, "output": 0}

    def _post(self, messages: list[dict], max_tokens: int) -> str:
        import urllib.request

        body = json.dumps({
            "model": self.name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            self.api_base.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.load(resp)
        usage = data.get("usage", {})
        self.last_usage = {"input": usage.get("prompt_tokens", 0),
                           "output": usage.get("completion_tokens", 0)}
        self.cum_usage["input"] += self.last_usage["input"]
        self.cum_usage["output"] += self.last_usage["output"]
        return data["choices"][0]["message"]["content"]

    def chat(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        return self._post(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            max_tokens,
        )

    def chat_messages(self, messages: list[dict], *, max_tokens: int = 1024) -> str:
        """Multi-turn call for the ReAct loop; token usage still accumulates."""
        return self._post(messages, max_tokens)


def make_backbone(spec: str) -> Backbone:
    """spec = 'mock' | 'mock:blind' | '<model-name>' (API)."""
    if spec.startswith("mock"):
        knowledge = spec.split(":", 1)[1] if ":" in spec else "oracle"
        return MockBackbone(knowledge=knowledge)
    return APIBackbone(model=spec)

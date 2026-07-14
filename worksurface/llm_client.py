"""Concrete OpenAI-compatible LLM client for the construction pipeline.

Satisfies worksurface.llm_hooks.LLMClient (a single ``complete`` method) and
adds what a batch data-construction run needs: retries with backoff, running
token accounting (so we can report spend), and an on-disk response cache keyed
by (model, system, user, max_tokens) so re-running the deriver never re-pays
for identical calls.

Config from env (same as the runner backbone):
    WSB_API_BASE, WSB_API_KEY, and optionally WSB_BUILD_MODEL (default
    gpt-4o-mini — construction is classification/rewrite/verify, cheap models
    suffice).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.normpath(os.path.join(HERE, "..", "data", "_llm_cache"))


class OpenAIClient:
    def __init__(self, model: str | None = None, api_base: str | None = None,
                 api_key: str | None = None, cache: bool = True):
        self.model = model or os.environ.get("WSB_BUILD_MODEL", "gpt-4o-mini")
        self.api_base = api_base or os.environ.get("WSB_API_BASE")
        self.api_key = api_key or os.environ.get("WSB_API_KEY")
        if not (self.api_base and self.api_key):
            raise RuntimeError("OpenAIClient needs WSB_API_BASE and WSB_API_KEY")
        self.cache = cache
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
        self.usage = {"input": 0, "output": 0, "calls": 0, "cache_hits": 0}

    def _cache_path(self, key: str) -> str:
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return os.path.join(CACHE_DIR, f"{h}.json")

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        key = f"{self.model}\x00{max_tokens}\x00{system}\x00{user}"
        if self.cache:
            cp = self._cache_path(key)
            if os.path.exists(cp):
                self.usage["cache_hits"] += 1
                return json.load(open(cp))["content"]

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode()

        last_err = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    self.api_base.rstrip("/") + "/chat/completions",
                    data=body,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.load(resp)
                content = data["choices"][0]["message"]["content"]
                u = data.get("usage", {})
                self.usage["input"] += u.get("prompt_tokens", 0)
                self.usage["output"] += u.get("completion_tokens", 0)
                self.usage["calls"] += 1
                if self.cache:
                    json.dump({"content": content}, open(self._cache_path(key), "w"))
                return content
            except (urllib.error.URLError, KeyError, TimeoutError) as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def report(self) -> str:
        u = self.usage
        return (f"[llm] {self.model}: {u['calls']} calls "
                f"(+{u['cache_hits']} cached), "
                f"{u['input']:,} in / {u['output']:,} out tokens")

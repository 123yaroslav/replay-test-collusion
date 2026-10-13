"""Thin async LLM client: OpenRouter (or any OpenAI-compatible endpoint), disk cache, budget cap.

Every call is cached in SQLite keyed by the full request plus a `sample` index, so
re-running an experiment with the same seeds is free and bit-identical, while distinct
episodes/seeds still get independent samples.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "results" / "cache" / "llm.sqlite"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cached: bool


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT)")
        self._db.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM calls WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO calls (key, value) VALUES (?, ?)", (key, json.dumps(value))
            )
            self._db.commit()


class LLM:
    def __init__(
        self,
        budget_usd: float | None = None,
        max_concurrency: int = 16,
        cache_path: Path = CACHE_PATH,
    ):
        base_url = os.getenv("COLLUSION_BASE_URL") or OPENROUTER_URL
        api_key = os.getenv("COLLUSION_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "missing")
        self._is_openrouter = base_url == OPENROUTER_URL
        self._cache = _Cache(cache_path)
        self._sem = asyncio.Semaphore(max_concurrency)
        self.budget_usd = budget_usd or float(os.getenv("COLLUSION_BUDGET_USD", "20"))
        self.spent_usd = 0.0
        self.n_calls = 0
        self.n_cached = 0

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        sample: int = 0,
        json_mode: bool = False,
        reasoning_effort: str | None = "low",
        retries: int = 5,
    ) -> ChatResult:
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        extra: dict = {}
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        if self._is_openrouter:
            extra["usage"] = {"include": True}
            if reasoning_effort:
                extra["reasoning"] = {"effort": reasoning_effort}
        key = hashlib.sha256(
            json.dumps({**request, **extra, "sample": sample}, sort_keys=True).encode()
        ).hexdigest()

        if (hit := self._cache.get(key)) is not None:
            self.n_cached += 1
            return ChatResult(**{**hit, "cached": True})

        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(f"spent ${self.spent_usd:.2f} >= budget ${self.budget_usd:.2f}")

        delay = 1.0
        for attempt in range(retries):
            try:
                async with self._sem:
                    resp = await self._client.chat.completions.create(**request, extra_body=extra)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

        usage = resp.usage
        cost = float(getattr(usage, "cost", 0.0) or 0.0) if usage else 0.0
        result = {
            "text": resp.choices[0].message.content or "",
            "model": resp.model or model,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "cost_usd": cost,
        }
        self._cache.put(key, result)
        self.spent_usd += cost
        self.n_calls += 1
        return ChatResult(**result, cached=False)

    def summary(self) -> str:
        return (
            f"{self.n_calls} live calls, {self.n_cached} cached, "
            f"${self.spent_usd:.3f} of ${self.budget_usd:.2f} budget"
        )


def parse_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates ```json fences and prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])

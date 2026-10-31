"""OpenAI-compatible client for the LM Studio server.

Thin on purpose: retries, timeouts, token accounting and clear errors. Anything
smarter (chunking, routing) belongs a layer up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx

from .config import Config, Tier, get_config


class LocalLLMError(RuntimeError):
    """The local server refused, failed, or could not be reached."""


@dataclass
class Completion:
    """One generation plus what it cost."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    finish_reason: str = ""
    reasoning: str = ""
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tok_per_s(self) -> float:
        return self.completion_tokens / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


class LocalLLM:
    """Talks to LM Studio's ``/v1`` endpoint."""

    def __init__(self, cfg: Config | None = None, client: httpx.Client | None = None):
        self.cfg = cfg or get_config()
        self._base = self.cfg.server.base_url.rstrip("/")
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(self.cfg.server.request_timeout_s, connect=10.0)
        )
        token = getattr(self.cfg.server, "api_token", "")
        if token:
            self._client.headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------- plumbing

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict:
        url = f"{self._base}{path}"
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.post(url, json=payload, timeout=timeout or httpx.USE_CLIENT_DEFAULT)
            except (httpx.ConnectError, httpx.ReadError) as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            except httpx.TimeoutException as exc:
                raise LocalLLMError(
                    f"local model timed out after {timeout or self.cfg.server.request_timeout_s}s. "
                    "The model may still be loading — check `llmh doctor`."
                ) from exc

            if resp.status_code >= 400:
                raise LocalLLMError(f"{url} -> HTTP {resp.status_code}: {resp.text[:500]}")
            try:
                return resp.json()
            except ValueError as exc:
                raise LocalLLMError(f"{url} returned non-JSON: {resp.text[:300]}") from exc

        raise LocalLLMError(
            f"cannot reach the local server at {self._base} ({last}). "
            "Start it with `llmh doctor --fix` or `lms server start`."
        )

    # ---------------------------------------------------------------- calls

    def healthy(self) -> bool:
        try:
            r = self._client.get(f"{self._base}/models", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def models(self) -> list[str]:
        """Model ids the server will answer for."""
        try:
            r = self._client.get(f"{self._base}/models", timeout=15.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LocalLLMError(f"cannot list models at {self._base}: {exc}") from exc
        return [m.get("id", "") for m in r.json().get("data", [])]

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        tier: Tier,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        stop: Sequence[str] | None = None,
        reasoning_effort: str | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": tier.model,
            "messages": list(messages),
            "temperature": tier.temperature if temperature is None else temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if stop:
            payload["stop"] = list(stop)

        effort = tier.reasoning_effort if reasoning_effort is None else reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort

        started = time.monotonic()
        data = self._post("/chat/completions", payload, timeout=timeout)
        elapsed = time.monotonic() - started

        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            raise LocalLLMError(f"unexpected response shape: {str(data)[:300]}") from exc

        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        text = (message.get("content") or "").strip()
        finish = choice.get("finish_reason") or ""
        reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)

        if not text and finish == "length":
            # A reasoning model can spend the entire budget thinking and return
            # nothing. Silently passing "" upward wastes the whole run and looks
            # like success, so make it loud.
            raise LocalLLMError(
                f"{tier.model} returned no content: it used all {max_tokens} allowed "
                f"tokens on reasoning ({reasoning_tokens} reasoning tokens). "
                "Raise max_tokens or set reasoning_effort='none' for this tier."
            )

        return Completion(
            text=text,
            model=data.get("model", tier.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            duration_s=elapsed,
            finish_reason=finish,
            reasoning=(message.get("reasoning_content") or "").strip(),
            reasoning_tokens=reasoning_tokens,
        )

    def ask(
        self,
        prompt: str,
        tier: Tier,
        system: str | None = None,
        max_tokens: int | None = None,
        **kw: Any,
    ) -> Completion:
        """Single-turn convenience wrapper around :meth:`chat`."""
        return self.chat(_messages(prompt, system), tier, max_tokens=max_tokens, **kw)

    def embed(self, texts: Sequence[str], tier: Tier) -> list[list[float]]:
        data = self._post("/embeddings", {"model": tier.model, "input": list(texts)})
        rows = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        if len(rows) != len(texts):
            raise LocalLLMError(f"asked for {len(texts)} embeddings, got {len(rows)}")
        return [r["embedding"] for r in rows]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LocalLLM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

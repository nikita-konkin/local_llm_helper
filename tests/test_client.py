"""Client behaviour around reasoning models.

These pin down a bug that cost a full 48-second run and reported success: a
reasoning model bills its thinking against ``max_tokens``, so a small extraction
budget can be entirely consumed by reasoning, leaving ``content`` empty. Silently
returning "" looked like a valid answer. It must raise instead.
"""

from __future__ import annotations

import httpx
import pytest

from llmhelper.client import LocalLLM, LocalLLMError
from llmhelper.config import Tier

TIER = Tier(name="fast", model="test/model", context=8192, min_free_ram_gb=1)
THINKING_TIER = Tier(
    name="fast",
    model="test/model",
    context=8192,
    min_free_ram_gb=1,
    reasoning_effort="none",
)


def make_client(handler) -> tuple[LocalLLM, dict]:
    """A LocalLLM wired to a fake server, plus a dict capturing the last request."""
    captured: dict = {}

    def transport_handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["payload"] = json.loads(request.content)
        captured["url"] = str(request.url)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(transport_handler))
    return LocalLLM(client=client), captured


def response(content: str, finish: str = "stop", reasoning_tokens: int = 0) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "test/model",
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        },
    )


def test_empty_content_from_exhausted_reasoning_raises():
    llm, _ = make_client(lambda _r: response("", finish="length", reasoning_tokens=400))
    with pytest.raises(LocalLLMError) as exc:
        llm.ask("extract the facts", TIER, max_tokens=400)

    message = str(exc.value)
    assert "no content" in message
    assert "400" in message, "the message must name the budget that was exhausted"
    assert "reasoning_effort" in message, "and say how to fix it"


def test_empty_content_that_is_not_a_length_stop_is_allowed_through():
    # A model legitimately answering with nothing is odd but not our bug to raise on.
    llm, _ = make_client(lambda _r: response("", finish="stop"))
    assert llm.ask("hi", TIER).text == ""


def test_truncated_but_non_empty_content_is_returned_and_flagged():
    llm, _ = make_client(lambda _r: response("partial ans", finish="length"))
    completion = llm.ask("hi", TIER, max_tokens=5)
    assert completion.text == "partial ans"
    assert completion.truncated is True


def test_reasoning_effort_comes_from_the_tier():
    llm, captured = make_client(lambda _r: response("ok"))
    llm.ask("hi", THINKING_TIER)
    assert captured["payload"]["reasoning_effort"] == "none"


def test_reasoning_effort_is_omitted_when_the_tier_does_not_set_it():
    llm, captured = make_client(lambda _r: response("ok"))
    llm.ask("hi", TIER)
    assert "reasoning_effort" not in captured["payload"]


def test_per_call_reasoning_effort_overrides_the_tier():
    llm, captured = make_client(lambda _r: response("ok"))
    llm.ask("hi", TIER, reasoning_effort="none")
    assert captured["payload"]["reasoning_effort"] == "none"


def test_reasoning_tokens_are_recorded():
    llm, _ = make_client(lambda _r: response("ok", reasoning_tokens=77))
    assert llm.ask("hi", TIER).reasoning_tokens == 77


def test_http_error_is_reported_with_the_body():
    llm, _ = make_client(lambda _r: httpx.Response(400, text="model not loaded"))
    with pytest.raises(LocalLLMError, match="model not loaded"):
        llm.ask("hi", TIER)


def test_system_prompt_becomes_the_first_message():
    llm, captured = make_client(lambda _r: response("ok"))
    llm.ask("the question", TIER, system="the rules")
    messages = captured["payload"]["messages"]
    assert messages[0] == {"role": "system", "content": "the rules"}
    assert messages[1] == {"role": "user", "content": "the question"}


def test_tok_per_s_is_zero_rather_than_dividing_by_zero():
    from llmhelper.client import Completion

    assert Completion(text="x", model="m", completion_tokens=10, duration_s=0).tok_per_s == 0.0

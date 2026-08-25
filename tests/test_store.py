"""Job state machine and savings ledger.

A bug here either loses a queued job or runs one twice, so the transitions are
pinned explicitly rather than assumed.
"""

from __future__ import annotations

import pytest

from llmhelper.store import JobState, Priority, Store


@pytest.fixture
def store() -> Store:
    return Store(url="sqlite:///:memory:")


def test_submit_then_claim(store: Store):
    jid = store.submit("ask", {"prompt": "hi"}, label="greeting")
    assert store.get(jid).state == JobState.QUEUED

    job = store.claim("w1")
    assert job.id == jid
    assert job.state == JobState.RUNNING
    assert job.attempts == 1
    assert job.worker_id == "w1"


def test_a_claimed_job_cannot_be_claimed_again(store: Store):
    store.submit("ask", {"prompt": "hi"})
    assert store.claim("w1") is not None
    assert store.claim("w2") is None, "two workers must never get the same job"


def test_empty_queue_claims_nothing(store: Store):
    assert store.claim("w1") is None


def test_priority_order_beats_arrival_order(store: Store):
    store.submit("ask", {}, priority=Priority.LOW, label="low")
    store.submit("ask", {}, priority=Priority.EXPRESS, label="express")
    store.submit("ask", {}, priority=Priority.NORMAL, label="normal")

    assert [store.claim("w").label for _ in range(3)] == ["express", "normal", "low"]


def test_failure_requeues_until_attempts_run_out(store: Store):
    jid = store.submit("ask", {}, max_attempts=2)

    store.claim("w1")
    assert store.fail(jid, "first boom") is JobState.QUEUED

    job = store.claim("w1")
    assert job.attempts == 2
    assert store.fail(jid, "second boom") is JobState.FAILED

    assert store.claim("w1") is None, "a failed job must not be retried forever"
    assert "second boom" in store.get(jid).error


def test_failure_clears_the_worker_so_another_can_take_it(store: Store):
    jid = store.submit("ask", {}, max_attempts=3)
    store.claim("w1")
    store.fail(jid, "boom")

    job = store.get(jid)
    assert job.worker_id is None
    assert job.started_at is None


def test_complete_stores_a_result(store: Store):
    jid = store.submit("digest", {})
    store.claim("w1")
    store.complete(jid, "the answer", path="results/x.md", meta={"chunks": 4})

    assert store.get(jid).state == JobState.DONE
    res = store.result(jid)
    assert res.summary == "the answer"
    assert res.meta["chunks"] == 4


def test_completing_clears_a_previous_error(store: Store):
    jid = store.submit("digest", {}, max_attempts=3)
    store.claim("w1")
    store.fail(jid, "transient")
    store.claim("w1")
    store.complete(jid, "worked on retry")
    assert store.get(jid).error is None


def test_cancel_only_applies_before_a_job_finishes(store: Store):
    jid = store.submit("ask", {})
    assert store.cancel(jid) is True
    assert store.get(jid).state == JobState.CANCELLED
    assert store.claim("w1") is None

    done = store.submit("ask", {})
    store.claim("w1")
    store.complete(done, "x")
    assert store.cancel(done) is False


def test_stale_running_jobs_are_recovered(store: Store):
    store.submit("ask", {})
    store.submit("ask", {})
    store.claim("w1")

    assert store.requeue_stale_running() == 1
    assert store.counts()[JobState.QUEUED] == 2


def test_unknown_job_ids_raise(store: Store):
    with pytest.raises(KeyError):
        store.complete("nope", "x")
    with pytest.raises(KeyError):
        store.fail("nope", "x")


# ------------------------------------------------------------------- ledger


def test_savings_aggregate_across_kinds(store: Store):
    store.record_saving(kind="digest", input_tokens=100_000, output_tokens=500, duration_s=30)
    store.record_saving(kind="digest", input_tokens=50_000, output_tokens=500, duration_s=20)
    store.record_saving(kind="grep", input_tokens=20_000, output_tokens=1_000, duration_s=10)

    s = store.savings_summary()
    assert s["calls"] == 3
    assert s["input_tokens"] == 170_000
    assert s["tokens_saved"] == 168_000
    assert s["by_kind"][0]["kind"] == "digest"  # sorted by most saved


def test_ledger_handles_an_empty_database(store: Store):
    s = store.savings_summary()
    assert s["calls"] == 0
    assert s["input_tokens"] == 0
    assert s["tokens_saved"] == 0
    assert s["compression"] == 0.0
    assert s["by_kind"] == []
    assert s["by_source"] == []


def test_measurement_traffic_is_excluded_by_default(store: Store):
    """Benchmarks re-read one file dozens of times; counting that as saved is fiction."""
    store.record_saving(kind="digest", source="mcp", input_tokens=40_000, output_tokens=200)
    store.record_saving(kind="digest", source="ab", input_tokens=40_000, output_tokens=200)
    store.record_saving(kind="digest", source="bench", input_tokens=40_000, output_tokens=200)

    honest = store.savings_summary()
    assert honest["calls"] == 1
    assert honest["input_tokens"] == 40_000

    everything = store.savings_summary(include_synthetic=True)
    assert everything["calls"] == 3
    assert everything["input_tokens"] == 120_000


def test_saving_never_reports_negative(store: Store):
    # An answer longer than its source is a bad trade, but not a negative number.
    store.record_saving(kind="ask", input_tokens=10, output_tokens=900)
    assert store.savings_summary()["tokens_saved"] == 0


# ------------------------------------------------- citation line numbering


def test_numbered_uses_absolute_file_lines():
    """Citations are the only thing making an offloaded answer checkable.

    The map pass used to hand the model an unnumbered excerpt while asking for
    line numbers, so it counted from 1 inside the chunk. A chunk starting at
    line 1142 produced citations like [3] that looked precise and pointed at the
    wrong place entirely.
    """
    from llmhelper.handlers.digest import numbered

    out = numbered("alpha\nbeta\ngamma", 1142)
    assert out.splitlines() == ["1142: alpha", "1143: beta", "1144: gamma"]


def test_numbered_starts_at_one_for_the_first_chunk():
    from llmhelper.handlers.digest import numbered

    assert numbered("first\nsecond", 1).splitlines() == ["1: first", "2: second"]


def test_numbered_handles_empty_and_single_line():
    from llmhelper.handlers.digest import numbered

    assert numbered("", 5) == ""
    assert numbered("only", 99) == "99: only"


# ------------------------------------------------- remote client (no lms CLI)


def test_quant_matching_is_skipped_for_server_reported_models():
    """A remote client cannot know quantisations, and does not need to.

    Quant matching exists to avoid loading a 66 GB BF16 when the tier asked for a
    20 GB Q4 — a local-disk problem. A remote LM Studio reports one bare id per
    model via /v1/models, so insisting on a quant there matched nothing and made
    the resolver refuse every tier.
    """
    from llmhelper.config import Tier
    from llmhelper.runtime import ModelInfo

    tier = Tier(name="fast", model="qwen/qwen3.5-9b", context=8192,
                min_free_ram_gb=1, quant="Q4_K_M")

    from_server = ModelInfo(key="qwen/qwen3.5-9b", origin="server")
    assert from_server.matches(tier), "server-reported model must match on base key"

    wrong_quant_on_disk = ModelInfo(key="qwen3.5-9b", quant="BF16", origin="disk")
    assert not wrong_quant_on_disk.matches(tier), "disk matching must still respect quant"

    right_quant_on_disk = ModelInfo(key="qwen3.5-9b", quant="Q4_K_M", origin="disk")
    assert right_quant_on_disk.matches(tier)


def test_a_different_model_never_matches_regardless_of_origin():
    from llmhelper.config import Tier
    from llmhelper.runtime import ModelInfo

    tier = Tier(name="fast", model="qwen/qwen3.5-9b", context=8192, min_free_ram_gb=1)
    assert not ModelInfo(key="gemma-4-12b-it-qat", origin="server").matches(tier)
    assert not ModelInfo(key="gemma-4-12b-it-qat", origin="disk").matches(tier)

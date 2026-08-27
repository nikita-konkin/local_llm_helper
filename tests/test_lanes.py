"""The queue must not inherit the synchronous timeout.

Reported: a deep-tier digest submitted to the queue died at 420.0s, three times,
once per retry. The handlers passed budgets.express_timeout_s to the client, and
the worker runs those same handler objects — so the lane whose entire purpose is
that nobody is waiting was capped at the budget for someone waiting.
"""

from __future__ import annotations

from llmhelper.handlers.base import Ctx


class _Recorder:
    """Stands in for LocalLLM, capturing what the handler asked for."""

    def __init__(self):
        self.kwargs = None

    def ask(self, prompt, tier, **kw):
        self.kwargs = kw

        class C:
            text = "ok"
            prompt_tokens = 1
            completion_tokens = 1

        return C()


def test_express_uses_the_synchronous_budget(sandbox):
    ctx = Ctx(cfg=sandbox, llm=_Recorder(), runtime=object(), lane="express")
    assert ctx.timeout_s == sandbox.budgets.express_timeout_s


def test_queue_uses_its_own_budget(sandbox):
    ctx = Ctx(cfg=sandbox, llm=_Recorder(), runtime=object(), lane="queue")
    assert ctx.timeout_s == sandbox.budgets.queue_timeout_s
    assert ctx.timeout_s > sandbox.budgets.express_timeout_s


def test_the_lane_defaults_to_express(sandbox):
    assert Ctx(cfg=sandbox, llm=_Recorder(), runtime=object()).lane == "express"


def test_generate_passes_the_lane_timeout_to_the_client(sandbox):
    llm = _Recorder()
    ctx = Ctx(cfg=sandbox, llm=llm, runtime=object(), lane="queue")
    ctx.generate("hi", sandbox.tiers["fast"], ensure_loaded=False)
    assert llm.kwargs["timeout"] == sandbox.budgets.queue_timeout_s


def test_an_explicit_timeout_still_wins(sandbox):
    llm = _Recorder()
    ctx = Ctx(cfg=sandbox, llm=llm, runtime=object(), lane="queue")
    ctx.generate("hi", sandbox.tiers["fast"], ensure_loaded=False, timeout=12.0)
    assert llm.kwargs["timeout"] == 12.0


def test_the_runner_maps_the_worker_source_onto_the_queue_lane():
    """source is already the lane; only the worker runs queued work."""
    import inspect

    from llmhelper import runner

    src = inspect.getsource(runner.execute)
    assert 'lane = "queue" if source == "worker" else "express"' in src
    assert "gpu_lease(" in src, "both lanes must go through the GPU lease"

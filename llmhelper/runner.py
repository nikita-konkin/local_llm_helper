"""The single path a job takes, whether it arrived from the queue or an MCP call.

Keeping one execution path means tier selection, outbox writing and savings
accounting cannot drift between the two lanes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .client import LocalLLM
from .config import Config, Tier, get_config
from .guard import resolve_write, slug
from .handlers import get_handler
from .handlers.base import Ctx, HandlerResult
from .lock import gpu_lease
from .store import Store

# What each job kind wants when the caller does not say.
#
# Everything defaults to `fast`, and that is a measured decision rather than a
# cautious one. On a 16 GB card only a ~6 GB model stays resident in VRAM:
#
#   fast  qwen3.5-9b       6.1 GB   100.5 tok/s   (fits VRAM)
#   work  qwen3.6-35b-a3b 20.6 GB    12.6 tok/s   (spills to CPU)
#
# An 8x slowdown is not worth a marginal quality gain on extraction and triage,
# which is what these handlers actually do. `review` is the one kind that leans on
# real reasoning, so it is the fair candidate for `work` — but only when asked
# for explicitly, via the tier argument.
DEFAULT_TIER: dict[str, str] = {
    "ask": "fast",
    "grep": "fast",
    "digest": "fast",
    "transform": "fast",
    "codechore": "fast",
    "review": "fast",
    "transcribe": "fast",
    "image": "fast",
}

# Kinds that never call the LLM. They still go through execute() for the GPU
# lease, the outbox and the ledger, but a machine with no downloaded LLM must
# still be able to run them - resolving a tier is not a precondition of work
# that has no use for one.
NO_LLM_KINDS = frozenset({"image"})

# Stands in for the tier an LLM-free job never picked, so the savings row has
# something truthful to record rather than a model it did not run.
NO_TIER = Tier(name="none", model="", context=0, min_free_ram_gb=0.0)


class TierResolver:
    """Picks a tier that is actually usable right now.

    A configured tier whose model has not been downloaded is a configuration
    aspiration, not a runtime option — fall back rather than fail the job.
    """

    def __init__(self, cfg: Config, runtime: Any = None):
        self.cfg = cfg
        self._runtime = runtime
        self._probed = False

    @property
    def runtime(self):
        if self._runtime is None:
            from .runtime import Runtime

            self._runtime = Runtime(self.cfg)
        return self._runtime

    def _is_present(self, tier: Tier) -> bool:
        """Delegates to Runtime.find so there is exactly one matching rule."""
        try:
            catalog = self.runtime.catalog()
        except Exception:  # noqa: BLE001 - never let discovery break execution
            return True  # cannot tell; let the load attempt decide
        if not catalog:
            return True
        return self.runtime.find(tier) is not None

    def resolve(self, kind: str, requested: str | None) -> Tier:
        order: list[str] = []
        for name in (requested, DEFAULT_TIER.get(kind), self.cfg.worker.fallback_tier):
            if name and name in self.cfg.tiers and name not in order:
                order.append(name)
        for name, tier in self.cfg.llm_tiers.items():
            if name not in order:
                order.append(name)

        for name in order:
            tier = self.cfg.tier(name)
            if tier.is_embedding:
                continue
            if self._is_present(tier):
                return tier

        raise RuntimeError(
            "no configured tier has a downloaded model. "
            "Run `lms get <model>` or check `llmh doctor`."
        )


def outbox_path(cfg: Config, kind: str, label: str, job_id: str | None) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stem = job_id[:8] if job_id else uuid.uuid4().hex[:8]
    return resolve_write(f"{day}/{stem}-{kind}-{slug(label)}.md", cfg)


def execute(
    kind: str,
    spec: dict[str, Any],
    store: Store,
    cfg: Config | None = None,
    tier_name: str | None = None,
    source: str = "cli",
    job_id: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    llm: LocalLLM | None = None,
    runtime: Any = None,
    write_detail: bool = True,
    client: str | None = None,
) -> tuple[HandlerResult, Tier]:
    """Run one job end to end and record what it saved."""
    cfg = cfg or get_config()
    handler = get_handler(kind)

    if runtime is None:
        # One Runtime for the whole call: its model catalogue is cached, so tier
        # resolution and the handler do not each pay for an `lms ls` subprocess.
        from .runtime import Runtime

        runtime = Runtime(cfg, llm)
    llm = llm or runtime.llm

    try:
        tier = TierResolver(cfg, runtime).resolve(kind, tier_name or spec.get("tier"))
    except RuntimeError:
        if kind not in NO_LLM_KINDS:
            raise
        tier = NO_TIER
    # source is already the lane: the worker is the only queued caller.
    lane = "queue" if source == "worker" else "express"
    ctx = Ctx(
        cfg=cfg, llm=llm, runtime=runtime, on_progress=on_progress, lane=lane
    )

    # One GPU, one tier at a time. Without this the express lane and the worker
    # each make their own tier resident and the card is overcommitted; the queued
    # side waits as long as it takes, the synchronous side gives up and says so.
    wait_s = float("inf") if lane == "queue" else cfg.budgets.gpu_wait_s
    with gpu_lease(cfg, f"{lane}:{kind}", wait_s=wait_s, on_wait=ctx.progress):
        result = handler.run(spec, ctx, tier)

    if write_detail and result.detail:
        label = spec.get("question") or spec.get("query") or spec.get("label") or kind
        path = outbox_path(cfg, kind, str(label), job_id)
        path.write_text(result.detail, encoding="utf-8")
        result.outbox_path = path

    store.record_saving(
        client=client,
        **result.savings_row(
            kind=kind, tier=tier.name, model=tier.model, source=source, job_id=job_id
        ),
    )
    return result, tier

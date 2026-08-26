"""The background worker: claim a job, run it, record it, repeat.

One worker by default, because one model is resident at a time and two workers
would only fight over which one that is. The loop is deliberately boring — all
the interesting behaviour lives in the handlers.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import Callable

from .client import LocalLLM
from .config import Config, get_config
from .runner import execute
from .lock import AlreadyRunning, single_instance  # re-exported for cli
from .store import JobState, Store

log = logging.getLogger("llmhelper.worker")


def setup_logging(cfg: Config, verbose: bool = False) -> None:
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(cfg.paths.logs / "worker.log", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=handlers,
        force=True,
    )
    # basicConfig turns INFO on for every library too. httpx then logs a line per
    # request, so a health poll every few minutes filled worker.log with
    # "HTTP Request: GET /v1/models 200 OK" and nothing else. Their warnings are
    # still wanted; their chatter is not.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Worker:
    def __init__(
        self,
        cfg: Config | None = None,
        store: Store | None = None,
        worker_id: str | None = None,
        on_event: Callable[[str], None] | None = None,
    ):
        self.cfg = cfg or get_config()
        self.store = store or Store(self.cfg)
        self.worker_id = worker_id or f"{os.getpid()}"
        self.on_event = on_event or (lambda msg: log.info(msg))
        self.llm = LocalLLM(self.cfg)
        self._runtime = None
        self._stop = False

    @property
    def runtime(self):
        if self._runtime is None:
            from .runtime import Runtime

            self._runtime = Runtime(self.cfg, self.llm)
        return self._runtime

    def request_stop(self, *_args) -> None:
        """Finish the job in hand, then exit."""
        self._stop = True
        self.on_event("stop requested; finishing current job")

    def run_one(self) -> bool:
        """Claim and run a single job. Returns False when the queue is empty."""
        job = self.store.claim(self.worker_id)
        if job is None:
            return False

        label = job.label or job.kind
        self.on_event(f"[{job.id[:8]}] {job.kind} start — {label}")
        try:
            result, tier = execute(
                kind=job.kind,
                spec=dict(job.spec or {}),
                store=self.store,
                cfg=self.cfg,
                tier_name=job.tier,
                source="worker",
                job_id=job.id,
                on_progress=lambda m: self.on_event(f"[{job.id[:8]}] {m}"),
                llm=self.llm,
                runtime=self.runtime,
                client=job.client,  # attribute to whoever queued it, not to us
            )
        except Exception as exc:  # noqa: BLE001 - a bad job must not kill the worker
            state = self.store.fail(job.id, f"{type(exc).__name__}: {exc}")
            verb = "requeued" if state is JobState.QUEUED else "failed"
            self.on_event(f"[{job.id[:8]}] {verb}: {type(exc).__name__}: {exc}")
            log.exception("job %s failed", job.id)
            return True

        self.store.complete(
            job.id,
            summary=result.summary,
            path=str(result.outbox_path) if result.outbox_path else None,
            meta={
                **result.meta,
                "tier": tier.name,
                "model": tier.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "tokens_saved": result.tokens_saved,
                "duration_s": round(result.duration_s, 1),
            },
        )
        self.on_event(
            f"[{job.id[:8]}] done in {result.duration_s:.0f}s — "
            f"{result.input_tokens:,} -> {result.output_tokens:,} tokens "
            f"({result.compression}x)"
        )
        return True

    def run_forever(self, idle_unload_after_s: float | None = None) -> None:
        """Poll until stopped, releasing the model after a long idle stretch.

        Set worker.idle_unload_after_s = 0 to keep it loaded: a remote client
        cannot load models, so unloading here silently costs it context.
        """
        if idle_unload_after_s is None:
            idle_unload_after_s = self.cfg.worker.idle_unload_after_s
        recovered = self.store.requeue_stale_running()
        if recovered:
            self.on_event(f"requeued {recovered} job(s) left running by a previous worker")

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):
                pass  # not on the main thread; the caller handles shutdown

        self.on_event(f"worker {self.worker_id} ready; polling every {self.cfg.worker.poll_interval_s}s")
        idle_since: float | None = None

        while not self._stop:
            try:
                did_work = self.run_one()
            except Exception:  # noqa: BLE001 - the loop itself must survive anything
                log.exception("worker loop error")
                did_work = False

            if did_work:
                idle_since = None
                continue

            now = time.monotonic()
            idle_since = idle_since or now
            if idle_unload_after_s and (now - idle_since) > idle_unload_after_s:
                # Hand ~20-60 GB of RAM back while nothing is queued.
                try:
                    if self.runtime.loaded():
                        self.runtime.unload_all()
                        self.on_event("idle; unloaded the resident model")
                except Exception:  # noqa: BLE001
                    log.exception("idle unload failed")
                idle_since = now

            time.sleep(self.cfg.worker.poll_interval_s)

        self.on_event("worker stopped")

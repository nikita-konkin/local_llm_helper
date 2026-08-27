"""One card, one tier at a time.

Measured on the real machine: qwen3.5-9b (6.55 GB) from the express lane and
qwen3.8-27b (14.25 GB) for the worker were resident simultaneously — 20.8 GB on a
16 GB card. A deep-tier load aborted with "Engine protocol startup was aborted",
and a synchronous grep took 875s instead of 40 because it was competing for a card
that had already spilled into system RAM.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from llmhelper.lock import GpuBusy, gpu_lease


def test_the_lease_is_released_on_the_way_out(sandbox):
    with gpu_lease(sandbox, "express:digest"):
        pass
    # A second acquisition must not block.
    with gpu_lease(sandbox, "queue:digest", wait_s=0):
        pass


def test_it_is_reentrant_within_one_process(sandbox):
    """A handler that runs another handler must not deadlock against itself."""
    with gpu_lease(sandbox, "queue:digest"):
        with gpu_lease(sandbox, "queue:ask", wait_s=0):
            pass
        # The inner exit must not have dropped the outer lease.
        from pathlib import Path

        assert (Path(sandbox.paths.logs) / "gpu.lease").is_file()


def test_an_exception_still_releases(sandbox):
    with pytest.raises(ValueError):
        with gpu_lease(sandbox, "express:ask"):
            raise ValueError("boom")
    with gpu_lease(sandbox, "queue:ask", wait_s=0):
        pass


def test_a_stale_lease_from_a_dead_process_is_reclaimed(sandbox):
    from pathlib import Path

    path = Path(sandbox.paths.logs) / "gpu.lease"
    path.parent.mkdir(parents=True, exist_ok=True)
    # A pid that cannot be alive; the holder died without releasing.
    path.write_text("999999999 queue:digest", encoding="utf-8")

    with gpu_lease(sandbox, "express:ask", wait_s=0):
        assert path.read_text(encoding="utf-8").startswith(str(os.getpid()))


def test_a_live_holder_in_another_process_blocks_and_is_named(sandbox, tmp_path):
    """The synchronous lane gives up with a message, rather than overcommitting."""
    script = textwrap.dedent(
        f"""
        import os, sys, time
        sys.path.insert(0, {str(tmp_path.parent.parent)!r})
        from pathlib import Path
        p = Path({str(sandbox.paths.logs)!r}) / "gpu.lease"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{{os.getpid()}} queue:digest", encoding="utf-8")
        time.sleep(30)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script])
    try:
        # Wait for the child to actually claim it.
        from pathlib import Path

        path = Path(sandbox.paths.logs) / "gpu.lease"
        for _ in range(100):
            if path.is_file():
                break
            import time

            time.sleep(0.05)

        with pytest.raises(GpuBusy) as exc:
            with gpu_lease(sandbox, "express:grep", wait_s=0.2, poll_s=0.05):
                pass
        assert "queue:digest" in str(exc.value), "must name who is holding it"
        assert "pid " in str(exc.value)
    finally:
        proc.kill()
        proc.wait(timeout=10)

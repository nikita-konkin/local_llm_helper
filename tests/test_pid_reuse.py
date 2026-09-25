"""A recycled pid must not count as the lock holder.

After a reboot webui.pid still named 14908; Windows had handed 14908 to
msedgewebview2, and `llmh ui` refused to start because the dashboard was
"already running". The same check sits in front of `llmh worker restart` and the
dashboard's stop button, which terminate the holder - and in front of the GPU
lease, where a false holder blocks every job.

These tests use this very process as the "other" pid, because it is guaranteed
to exist and its start time is known.
"""

from __future__ import annotations

import os
import time

import psutil
import pytest

from llmhelper.lock import AlreadyRunning, gpu_lease, holder, pid_alive, single_instance


def _write_lock(path, content: str, age_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    t = time.time() - age_s
    os.utime(path, (t, t))


def _older_than_this_process() -> float:
    """Seconds ago that predates this test process by a comfortable margin."""
    return time.time() - psutil.Process().create_time() + 3600


def test_a_live_pid_that_started_after_the_lock_is_not_the_holder(sandbox):
    """The reboot case: the file is older than the process now wearing its pid."""
    lock = sandbox.paths.logs / "webui.pid"
    _write_lock(lock, str(os.getpid()), age_s=_older_than_this_process())
    assert holder(sandbox, "webui") is None


def test_a_recycled_pid_does_not_block_startup(sandbox):
    lock = sandbox.paths.logs / "webui.pid"
    _write_lock(lock, str(os.getpid() + 0), age_s=_older_than_this_process())
    # Would raise AlreadyRunning if the recycled pid still counted.
    with single_instance(sandbox, "webui"):
        assert lock.read_text().strip() == str(os.getpid())


def test_a_genuine_holder_still_holds(sandbox):
    """The fix must not turn the lock off: a lock written by a running process holds."""
    parent = psutil.Process().parent()
    if parent is None:
        pytest.skip("no parent process to act as the other instance")
    lock = sandbox.paths.logs / "webui.pid"
    _write_lock(lock, str(parent.pid), age_s=0)  # written now, parent started earlier
    assert holder(sandbox, "webui") == parent.pid
    with pytest.raises(AlreadyRunning):
        with single_instance(sandbox, "webui"):
            pass


def test_a_stale_gpu_lease_on_a_recycled_pid_is_reclaimed(sandbox):
    """Otherwise every express call waits gpu_wait_s and the worker waits forever."""
    lease = sandbox.paths.logs / "gpu.lease"
    parent = psutil.Process().parent()
    if parent is None:
        pytest.skip("no parent process to stand in for the recycled pid")
    _write_lock(
        lease,
        f"{parent.pid} queue:digest",
        age_s=time.time() - parent.create_time() + 3600,
    )
    with gpu_lease(sandbox, "express:test", wait_s=0):
        assert lease.read_text().startswith(str(os.getpid()))


def test_without_a_timestamp_existence_is_still_enough():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(os.getpid(), since=time.time()) is True


def test_nonsense_pids_are_dead():
    assert pid_alive(0) is False
    assert pid_alive(-5) is False

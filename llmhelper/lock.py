"""Single-instance locks for the long-running processes.

Neither the worker nor the dashboard breaks outright when a second copy starts —
they break slowly and confusingly. Two workers fight over which model is resident
and pay tens of seconds per unload/load. Two dashboards mean one binds the port
and the other silently does not, so an edit to the code appears to have no effect
because a stale process is still answering. Both happened during development;
both were diagnosed only after wasted effort.

The lock is a pid file, checked against a live process. It is advisory and
deliberately simple: a stale file from a killed process is detected because the
pid no longer exists, and a leftover file can always be deleted by hand.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

from .config import Config


class AlreadyRunning(RuntimeError):
    """Another instance holds the lock."""


def pid_alive(pid: int) -> bool:
    """True if a process with this pid exists. Never raises."""
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001 - absence of psutil must not break the lock
        return False


def holder(cfg: Config, name: str) -> int | None:
    """The pid currently holding this lock, or None."""
    path = Path(cfg.paths.logs) / f"{name}.pid"
    if not path.is_file():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip() or 0)
    except (ValueError, OSError):
        return None
    return pid if pid_alive(pid) else None


@contextmanager
def single_instance(cfg: Config, name: str) -> Iterator[None]:
    """Hold the named lock for the duration, or raise AlreadyRunning."""
    path = Path(cfg.paths.logs) / f"{name}.pid"
    path.parent.mkdir(parents=True, exist_ok=True)

    other = holder(cfg, name)
    if other is not None and other != os.getpid():
        raise AlreadyRunning(
            f"{name} is already running as pid {other} (lock: {path}). "
            "Stop it first, or delete the lock file if it is stale."
        )

    path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        try:
            if path.is_file() and path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                path.unlink()
        except OSError:
            pass


class GpuBusy(RuntimeError):
    """Another lane is using the GPU and did not finish in time."""


def _read_lease(path: Path) -> tuple[int, str] | None:
    """(pid, purpose) of a live holder, or None if free or stale."""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    pid_s, _, purpose = raw.partition(" ")
    try:
        pid = int(pid_s or 0)
    except ValueError:
        return None
    return (pid, purpose or "?") if pid_alive(pid) else None


@contextmanager
def gpu_lease(
    cfg: Config,
    purpose: str,
    wait_s: float = 120.0,
    poll_s: float = 2.0,
    on_wait=None,
) -> Iterator[None]:
    """Exclusive use of the GPU for the duration.

    One 16 GB card cannot hold two tiers. Without this, the express lane and the
    worker each call ensure_loaded independently and the models coexist: measured,
    qwen3.5-9b (6.55 GB) alongside qwen3.8-27b (14.25 GB) on a 16 GB card. The
    consequences were a deep-tier load aborting outright with "Engine protocol
    startup was aborted", and a synchronous grep taking 875 s instead of 40 because
    it was competing for a card that had already spilled into system RAM.

    Re-entrant within a process, so nested handler calls do not deadlock.

    Raises GpuBusy rather than waiting forever: a caller with someone waiting on it
    would rather be told to queue the work than hang.
    """
    import time

    path = Path(cfg.paths.logs) / "gpu.lease"
    path.parent.mkdir(parents=True, exist_ok=True)
    me = os.getpid()

    held = _read_lease(path)
    if held is not None and held[0] == me:
        yield  # already ours; do not release on the way out
        return

    deadline = time.monotonic() + max(0.0, wait_s)
    announced = False
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"{me} {purpose}")
            break
        except FileExistsError:
            other = _read_lease(path)
            if other is None:
                # Holder died without releasing. Reclaim and retry.
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise GpuBusy(
                    f"the GPU is in use by {other[1]} (pid {other[0]}) and did not "
                    f"free up within {wait_s:g}s. One card cannot hold two tiers, "
                    "so this work has to wait its turn - queue it with local_submit, "
                    "or retry once the worker is idle."
                ) from None
            if on_wait and not announced:
                on_wait(f"waiting for the GPU, held by {other[1]} (pid {other[0]})")
                announced = True
            time.sleep(poll_s)

    try:
        yield
    finally:
        try:
            if _read_lease(path) and _read_lease(path)[0] == me:
                path.unlink()
        except OSError:
            pass

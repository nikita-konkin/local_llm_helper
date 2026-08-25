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

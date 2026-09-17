"""A damaged lms binary must degrade the dashboard, not take it down.

An LM Studio self-update (0.4.12 -> 0.4.21) left lms.exe as 120 MB of zeros. The
file still existed, so can_manage_models said yes, so every call went on to
CreateProcess and died with WinError 1392 - and /api/status returned a 500 on a
machine that could still have shown CPU, GPU, RAM and the whole savings ledger.

Present is not the same as runnable, and the third component in this project to
take down a whole view by failing alone.
"""

from __future__ import annotations

import dataclasses

import pytest

from llmhelper.runtime import Runtime, RuntimeError_


@pytest.fixture
def broken_lms(sandbox, tmp_path):
    """A runtime whose lms path exists but cannot be executed."""
    fake = tmp_path / "lms.exe"
    fake.write_bytes(b"\x00" * 4096)  # no PE header, exactly like the real casualty
    cfg = dataclasses.replace(
        sandbox, server=dataclasses.replace(sandbox.server, lms_exe=fake)
    )
    rt = Runtime(cfg)

    def boom(*a, **k):
        raise OSError(1392, "The file or directory is corrupted and unreadable")

    return rt, boom


def test_an_optional_query_returns_empty_rather_than_raising(broken_lms, monkeypatch):
    rt, boom = broken_lms
    monkeypatch.setattr("llmhelper.runtime.subprocess.run", boom)
    assert rt._lms("ps", "--json", check=False) == ""


def test_a_required_command_raises_something_readable(broken_lms, monkeypatch):
    rt, boom = broken_lms
    monkeypatch.setattr("llmhelper.runtime.subprocess.run", boom)
    with pytest.raises(RuntimeError_) as exc:
        rt._lms("load", "some-model")
    msg = str(exc.value)
    assert "cannot be run" in msg
    assert "reinstall" in msg.lower(), "must say what to do about it"


def test_the_failure_is_latched_so_later_calls_do_not_retry(broken_lms, monkeypatch):
    """A failing spawn per request is a cost with no upside."""
    rt, boom = broken_lms
    calls = []

    def counting(*a, **k):
        calls.append(a)
        return boom()

    monkeypatch.setattr("llmhelper.runtime.subprocess.run", counting)
    assert rt.can_manage_models is True  # not tried yet
    rt._lms("ps", check=False)
    assert rt.can_manage_models is False
    rt._lms("ps", check=False)
    rt._lms("ps", check=False)
    assert len(calls) == 1, "should stop spawning once it is known to be broken"


def test_loaded_returns_nothing_instead_of_exploding(broken_lms, monkeypatch):
    rt, boom = broken_lms
    monkeypatch.setattr("llmhelper.runtime.subprocess.run", boom)
    assert rt.loaded() == []


def test_snapshot_still_answers_and_explains_itself(broken_lms, monkeypatch):
    """The whole point: the page renders, and says why it is degraded."""
    rt, boom = broken_lms
    monkeypatch.setattr("llmhelper.runtime.subprocess.run", boom)

    snap = rt.snapshot()
    assert snap["lms_present"] is False
    assert snap["loaded"] == []
    assert "cannot be run" in snap["lms_error"]
    # The parts that never needed lms must still be there.
    assert snap["ram_total_gb"] > 0


def test_a_healthy_runtime_reports_no_error(sandbox):
    """The banner must stay hidden when nothing is wrong."""
    rt = Runtime(sandbox)
    assert rt.lms_error is None


def test_a_replaced_binary_clears_the_latch(broken_lms, monkeypatch, tmp_path):
    """Recovery must not require restarting every long-lived process.

    Replacing the damaged binary is exactly how this gets fixed, and the MCP
    server, worker and dashboard all outlive the outage. Carrying the stale
    verdict until someone restarted them would repeat a mistake this project has
    already paid for three times.
    """
    import time

    rt, boom = broken_lms
    monkeypatch.setattr("llmhelper.runtime.subprocess.run", boom)
    rt._lms("ps", check=False)
    assert rt.can_manage_models is False

    # Same broken file, untouched: still refused, and no new spawn attempted.
    assert rt.can_manage_models is False

    time.sleep(0.01)
    rt.lms.write_bytes(b"MZ" + b"\x90" * 8192)  # a different binary lands

    assert rt.can_manage_models is True, "a replacement must be noticed"
    assert rt.lms_error is None, "the stale explanation must be cleared too"


def test_a_missing_binary_is_still_refused_after_a_latch(broken_lms, monkeypatch):
    """Deleting it is not a repair."""
    rt, boom = broken_lms
    monkeypatch.setattr("llmhelper.runtime.subprocess.run", boom)
    rt._lms("ps", check=False)
    rt.lms.unlink()
    assert rt.can_manage_models is False

"""Escape attempts against the filesystem sandbox.

Every test here is a way someone — or a confused local model — could try to read
or write outside the boundary. They must all raise.

Everything runs against a temporary config built by the `sandbox` fixture, so the
suite works on a fresh clone and on any OS, and the boundary being tested is a
directory that actually exists rather than a path that happens to be absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmhelper import guard

# ------------------------------------------------------------------- writes


@pytest.mark.parametrize(
    "attempt",
    [
        "../escaped.txt",
        "../../escaped.txt",
        "sub/../../escaped.txt",
        "a/b/c/../../../../escaped.txt",
        "//server/share/escaped.txt",
    ],
)
def test_relative_write_escapes_are_refused(attempt, sandbox):
    with pytest.raises(guard.GuardError):
        guard.resolve_write(attempt, sandbox)


def test_absolute_write_outside_the_outbox_is_refused(sandbox, tmp_path):
    with pytest.raises(guard.GuardError):
        guard.resolve_write(str(tmp_path / "elsewhere" / "escaped.txt"), sandbox)


def test_writing_into_a_read_root_is_still_refused(sandbox, workspace):
    """Readable does not imply writable. The outbox is the only write target."""
    with pytest.raises(guard.GuardError):
        guard.resolve_write(str(workspace / "project" / "readme.txt"), sandbox)


def test_write_inside_outbox_is_allowed(sandbox):
    target = guard.resolve_write("job-123/summary.md", sandbox)
    assert guard._under(target, sandbox.paths.outbox)
    assert target.parent.is_dir()  # parents created for the caller


def test_write_to_outbox_root_is_refused(sandbox):
    with pytest.raises(guard.GuardError):
        guard.resolve_write(str(sandbox.paths.outbox), sandbox)


def test_nested_write_stays_inside(sandbox):
    target = guard.resolve_write("deep/a/b/c.txt", sandbox)
    assert guard._under(target, sandbox.paths.outbox)


# -------------------------------------------------------------------- reads


def test_read_outside_roots_is_refused(sandbox, tmp_path):
    with pytest.raises(guard.GuardError):
        guard.resolve_read(str(tmp_path / "not-a-root" / "file.txt"), sandbox)


def test_read_inside_root_is_allowed(sandbox, workspace):
    assert guard.resolve_read(workspace / "project" / "readme.txt", sandbox).is_file()


@pytest.mark.parametrize("attempt", ["project/secrets.key", "project/__pycache__/x.pyc"])
def test_deny_globs_beat_allowed_roots(attempt, sandbox, workspace):
    with pytest.raises(guard.GuardError):
        guard.resolve_read(workspace / attempt, sandbox)


def test_traversal_back_into_a_root_is_fine(sandbox, workspace):
    """Landing inside a root is what matters, not how the path was spelled."""
    spelled = workspace / "project" / ".." / "project" / "readme.txt"
    assert guard.resolve_read(spelled, sandbox).is_file()


def test_traversal_out_of_a_root_is_refused(sandbox, workspace):
    with pytest.raises(guard.GuardError):
        guard.resolve_read(workspace / ".." / ".." / "etc" / "passwd", sandbox)


@pytest.mark.skipif(os.name != "nt", reason="case-insensitive paths are a Windows behaviour")
def test_case_insensitivity_on_windows(sandbox, workspace):
    spelled = str(workspace / "project" / "readme.txt").upper()
    assert guard.resolve_read(spelled, sandbox).is_file()


# ------------------------------------------------------- UNC determinism


@pytest.mark.parametrize(
    "attempt",
    [
        "//server/share/escaped.txt",
        "//192.168.0.99/c$/windows/system32/x.txt",
        "//localhost/C$/secret.txt",
    ],
)
def test_unc_writes_are_refused_without_touching_the_network(attempt, sandbox):
    """A security decision must not depend on a DNS lookup succeeding.

    Path.resolve() on a UNC path can consult the network to resolve the host,
    which made this check nondeterministic — it flaked once across many runs. UNC
    paths are now normalised textually and never reach the resolver.
    """
    import time

    started = time.monotonic()
    with pytest.raises(guard.GuardError):
        guard.resolve_write(attempt, sandbox)
    assert time.monotonic() - started < 1.0, "guard should never wait on the network"


def test_unc_reads_are_refused_when_no_unc_root_is_configured(sandbox):
    with pytest.raises(guard.GuardError):
        guard.resolve_read("//server/share/data.csv", sandbox)


# ----------------------------------------------------------------- expand()


def test_expand_reports_why_things_were_skipped(sandbox, workspace, tmp_path):
    files, skipped = guard.expand(
        [
            str(workspace / "project" / "readme.txt"),
            str(tmp_path / "outside" / "nope.txt"),
            str(workspace / "nope-*.zzz"),
        ],
        sandbox,
    )
    assert [f.name for f in files] == ["readme.txt"]
    assert len(skipped) == 2
    assert any("outside the allowed roots" in s for s in skipped)
    assert any("matched nothing" in s for s in skipped)


def test_expand_skips_binaries_and_denied_files(sandbox, workspace):
    files, _ = guard.expand([str(workspace / "project")], sandbox)
    names = {f.name for f in files}
    assert "readme.txt" in names
    assert "secrets.key" not in names, "deny_globs must apply to directory expansion"
    assert "x.pyc" not in names


def test_expand_dedupes_the_same_file_spelled_two_ways(sandbox, workspace):
    one = workspace / "project" / "readme.txt"
    two = workspace / "project" / ".." / "project" / "readme.txt"
    files, _ = guard.expand([str(one), str(two)], sandbox)
    assert len(files) == 1


def test_expand_glob_does_not_walk_from_cwd(sandbox, workspace):
    files, _ = guard.expand([str(workspace / "project" / "*.txt")], sandbox)
    assert [f.name for f in files] == ["readme.txt"]


def test_read_text_is_guarded(sandbox, workspace):
    assert "hello from the sandbox" in guard.read_text(
        workspace / "project" / "readme.txt", cfg=sandbox
    )
    with pytest.raises(guard.GuardError):
        guard.read_text(workspace / "project" / "secrets.key", cfg=sandbox)


def test_expand_does_not_reveal_whether_out_of_root_files_exist(sandbox, tmp_path):
    """Both answers must be identical outside the boundary.

    Saying "does not exist" for one path and something else for another lets a
    caller probe the filesystem beyond the sandbox one path at a time.
    """
    missing = tmp_path / "outside" / "definitely-not-here.txt"
    present = tmp_path / "outside-real.txt"
    present.parent.mkdir(parents=True, exist_ok=True)
    present.write_text("I exist\n", encoding="utf-8")

    _, skipped_missing = guard.expand([str(missing)], sandbox)
    _, skipped_present = guard.expand([str(present)], sandbox)

    assert "does not exist" not in skipped_missing[0]
    assert "outside the allowed roots" in skipped_missing[0]
    assert "outside the allowed roots" in skipped_present[0]

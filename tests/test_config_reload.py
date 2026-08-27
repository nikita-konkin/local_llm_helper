"""config.toml is re-read when it changes on disk.

get_config() was @lru_cache(maxsize=1). An MCP server is a long-lived child of a
Claude session, so it snapshotted the config at startup and served that snapshot
for hours: adding a read root had no effect on any session already running, and
the path stayed refused. That looked like a fixed limitation of the tool rather
than a stale process, which is the expensive kind of wrong.
"""

from __future__ import annotations

import os

import pytest

from llmhelper import config as cfgmod


@pytest.fixture
def live_config(sandbox, tmp_path, monkeypatch):
    """Point get_config() at the sandbox's own config.toml."""
    monkeypatch.setenv("LLMHELPER_CONFIG", str(tmp_path / "config.toml"))
    cfgmod.invalidate()
    yield tmp_path / "config.toml"
    cfgmod.invalidate()


def bump(path, text):
    """Write and force a distinct mtime; some filesystems have coarse stamps."""
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def test_a_new_read_root_is_picked_up_without_restarting(live_config, tmp_path):
    before = cfgmod.get_config()
    assert len(before.paths.read_roots) == 1

    extra = tmp_path / "newly_allowed"
    extra.mkdir()
    text = live_config.read_text(encoding="utf-8").replace(
        'read_roots = [', f'read_roots = ["{extra.as_posix()}", ')
    bump(live_config, text)

    after = cfgmod.get_config()
    assert len(after.paths.read_roots) == 2
    assert extra in after.paths.read_roots


def test_an_unchanged_file_is_not_reparsed(live_config):
    assert cfgmod.get_config() is cfgmod.get_config(), "should be cached between calls"


def test_a_broken_edit_keeps_serving_the_last_good_config(live_config):
    """A half-saved file must not take down every running server."""
    good = cfgmod.get_config()
    bump(live_config, "this is not valid toml [[[")

    assert cfgmod.get_config() is good

    # And a subsequent repair is picked up, still without a restart.
    bump(live_config, live_config.read_text(encoding="utf-8"))


def test_a_broken_file_with_no_previous_config_still_raises(live_config):
    cfgmod.invalidate()
    bump(live_config, "not toml [[[")
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.get_config()

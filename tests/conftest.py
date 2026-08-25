"""Shared fixtures.

The suite builds its own configuration in a temporary directory rather than
reading the developer's `config.toml`. That file is gitignored — it holds machine
paths, a username and a LAN address — so a fresh clone has none, and tests that
depended on it could not run at all. Tests that assert a *security boundary* are
the first thing anyone evaluating this project should be able to execute.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from llmhelper.config import load_config


@pytest.fixture
def sandbox(tmp_path: Path):
    """A complete, isolated config rooted in tmp_path.

    Returns the loaded Config. Its read root and outbox are real directories that
    exist only for the duration of one test, so escape attempts are checked
    against a boundary that is genuinely there rather than a path that happens to
    be absent.
    """
    root = tmp_path / "workspace"
    (root / "project").mkdir(parents=True)
    (root / "project" / "readme.txt").write_text("hello from the sandbox\n", encoding="utf-8")
    (root / "project" / "secrets.key").write_text("should never be readable\n", encoding="utf-8")
    (root / "project" / "__pycache__").mkdir()
    (root / "project" / "__pycache__" / "x.pyc").write_bytes(b"\x00\x01binary")

    outbox = tmp_path / "results"
    outbox.mkdir()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            [server]
            base_url = "http://127.0.0.1:1234/v1"
            lms_exe  = "{(tmp_path / 'no-such-lms').as_posix()}"

            [paths]
            outbox     = "{outbox.as_posix()}"
            jobs_db    = "{(tmp_path / 'jobs.db').as_posix()}"
            logs       = "{(tmp_path / 'logs').as_posix()}"
            bench      = "{(tmp_path / 'bench').as_posix()}"
            read_roots = ["{root.as_posix()}"]
            deny_globs = ["**/__pycache__/**", "**/*.key", "**/.ssh/**"]

            [budgets]
            express_max_out_tokens = 1200
            express_timeout_s      = 60
            map_chunk_tokens       = 4000
            map_overlap_tokens     = 100
            map_summary_tokens     = 400
            max_map_chunks         = 100
            map_concurrency        = 1
            result_retention_days  = 30

            [worker]
            poll_interval_s = 0.1
            max_attempts    = 3
            fallback_tier   = "fast"

            [tiers.fast]
            model            = "test/model"
            context          = 8192
            min_free_ram_gb  = 1
            reasoning_effort = "none"
            """
        ).strip(),
        encoding="utf-8",
    )
    return load_config(cfg_path)


@pytest.fixture
def workspace(sandbox) -> Path:
    """The one directory the sandbox config allows reading from."""
    return sandbox.paths.read_roots[0]

"""The PreToolUse hook that redirects big reads to the local model.

CLAUDE.md has asked every session to use local_digest for bulk reads since the
project started, and sessions kept reaching for Read anyway - guidance loses to a
tool that is always present and free to call. The hook makes the redirect happen
rather than requesting it.

Its failure modes are asymmetric, so the tests are too. Redirecting a read that
should have gone through is a small annoyance with an escape hatch. Blocking a
read Claude needed, or breaking Read entirely because the hook crashed, is not.
Almost everything here checks that it stays out of the way.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "hooks" / "prefer-local-digest.py"


@pytest.fixture
def hook_env(tmp_path, monkeypatch):
    """A config whose read root is a temporary directory."""
    root = tmp_path / "workspace"
    root.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            [paths]
            read_roots = ["{root.as_posix()}"]
            """
        ),
        encoding="utf-8",
    )
    env = {
        "LOCAL_DIGEST_HOOK_CONFIG": str(cfg),
        "LOCAL_DIGEST_HOOK_SKIP_SERVER_CHECK": "1",
        "TMPDIR": str(tmp_path / "state"),
        "TEMP": str(tmp_path / "state"),
        "TMP": str(tmp_path / "state"),
        "SYSTEMROOT": r"C:\Windows",
        "PATH": "",
    }
    return root, env


def call(env: dict, session: str, tool_input: dict) -> str:
    """Run the hook exactly as Claude Code does and return its stdout."""
    payload = json.dumps(
        {"session_id": session, "tool_name": "Read", "tool_input": tool_input}
    )
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, f"hook must never fail: {proc.stderr}"
    return proc.stdout.strip()


def denied(out: str) -> bool:
    if not out:
        return False
    d = json.loads(out)["hookSpecificOutput"]
    return d["permissionDecision"] == "deny"


def big(root: Path, name: str = "big.md", size: int = 40_000) -> Path:
    p = root / name
    p.write_text("x" * size, encoding="utf-8")
    return p


def test_a_large_file_is_redirected(hook_env):
    root, env = hook_env
    out = call(env, "s1", {"file_path": str(big(root))})
    assert denied(out)
    reason = json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"]
    assert "local_digest" in reason
    assert str(big(root)) in reason, "must name the path so the call can be copied"


def test_the_second_read_of_the_same_file_goes_through(hook_env):
    """The escape hatch. Editing and patching need the exact bytes."""
    root, env = hook_env
    path = str(big(root))
    assert denied(call(env, "s1", {"file_path": path}))
    assert not denied(call(env, "s1", {"file_path": path})), "must not be a hard block"


def test_the_reason_explains_how_to_get_through(hook_env):
    root, env = hook_env
    out = call(env, "s1", {"file_path": str(big(root))})
    reason = json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"]
    assert "again" in reason.lower()


def test_each_session_is_redirected_once(hook_env):
    """A new session has not seen the advice, so it should get it."""
    root, env = hook_env
    path = str(big(root))
    assert denied(call(env, "s1", {"file_path": path}))
    assert denied(call(env, "s2", {"file_path": path}))


def test_a_small_file_is_left_alone(hook_env):
    """Below a few thousand tokens the round trip costs more than it saves."""
    root, env = hook_env
    small = root / "small.md"
    small.write_text("x" * 500, encoding="utf-8")
    assert not denied(call(env, "s1", {"file_path": str(small)}))


def test_a_bounded_read_is_left_alone(hook_env):
    """Read with a limit is already the cheap behaviour we want."""
    root, env = hook_env
    out = call(env, "s1", {"file_path": str(big(root)), "limit": 50})
    assert not denied(out)


def test_a_path_outside_the_read_roots_is_left_alone(hook_env, tmp_path):
    """Suggesting a tool that will refuse the path is worse than saying nothing."""
    root, env = hook_env
    outside = tmp_path / "elsewhere.md"
    outside.write_text("x" * 40_000, encoding="utf-8")
    assert not denied(call(env, "s1", {"file_path": str(outside)}))


def test_a_format_with_no_extractor_is_left_alone(hook_env):
    root, env = hook_env
    blob = root / "model.gguf"
    blob.write_bytes(b"x" * 40_000)
    assert not denied(call(env, "s1", {"file_path": str(blob)}))


def test_a_missing_file_is_left_alone(hook_env):
    root, env = hook_env
    assert not denied(call(env, "s1", {"file_path": str(root / "nope.md")}))


def test_malformed_input_does_not_block(hook_env):
    """A hook that fails closed on its own bugs would break every session."""
    _, env = hook_env
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input="this is not json",
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0
    assert not proc.stdout.strip()


def test_a_missing_config_does_not_block(hook_env, tmp_path):
    root, env = hook_env
    env = dict(env, LOCAL_DIGEST_HOOK_CONFIG=str(tmp_path / "gone.toml"))
    assert not denied(call(env, "s1", {"file_path": str(big(root))}))


def test_an_unreachable_model_server_does_not_block(hook_env):
    """The helper being off must not stop anyone reading files."""
    root, env = hook_env
    env = dict(env)
    env.pop("LOCAL_DIGEST_HOOK_SKIP_SERVER_CHECK")
    path = str(big(root))
    out = call(env, "s1", {"file_path": path})
    # Either the server is up (redirect) or it is not (silence). Both are correct;
    # what must never happen is a crash, which `call` already asserts.
    assert out == "" or denied(out)

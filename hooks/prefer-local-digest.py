#!/usr/bin/env python3
"""PreToolUse hook: send big Read calls to the local model instead.

CLAUDE.md has told every session to use local_digest for bulk reads since the
project started, and sessions still reach for Read - guidance competes with a
tool that is always present and costs nothing to call. This makes the redirect
happen instead of asking for it.

Deliberately conservative. It stays out of the way unless offloading is clearly
the better trade:

  - a targeted read (Read with a limit) is already cheap, so it passes
  - a file the guard would refuse passes, because suggesting a tool that will
    refuse the path is worse than suggesting nothing
  - a file the extractor cannot turn into text passes
  - if LM Studio is not listening, everything passes - the helper being off must
    not break reading files
  - the SECOND Read of the same path in a session always passes. Claude needs
    exact bytes to edit, patch, or match a string, and a hook that cannot be
    escaped would make those impossible. Once is a redirect; twice is a decision.

Anything unexpected exits 0 and the Read proceeds. A hook that fails closed on
its own bugs would break every session on this machine.

stdlib only, and no llmhelper import: this runs before every Read, so process
startup is the whole cost budget.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import time
import tomllib
from pathlib import Path

# Absolute because a hook runs from whatever directory the session is in, which
# is never this one. Overridable so the hook is testable and so a checkout
# somewhere else can point at its own config.
CONFIG = Path(
    os.environ.get("LOCAL_DIGEST_HOOK_CONFIG", r"N:\local_llm_helper\config.toml")
)

# Set by the tests: the liveness probe is about the machine, not the logic.
SKIP_SERVER_CHECK = os.environ.get("LOCAL_DIGEST_HOOK_SKIP_SERVER_CHECK") == "1"

# Below this, CLAUDE.md's own measurement says the round trip costs more than it
# saves. ~20 KB is roughly 5,000 tokens.
#
# This is the one number worth tuning. Too low and every source file Claude means
# to edit costs a wasted round trip before the escape hatch lets it through; too
# high and the reads worth offloading sail past. Raise it via
# LOCAL_DIGEST_HOOK_MIN_BYTES (settings.json `env`) if the redirects feel noisy.
try:
    MIN_BYTES = int(os.environ.get("LOCAL_DIGEST_HOOK_MIN_BYTES", "20000"))
except ValueError:
    MIN_BYTES = 20_000

# Formats the helper can actually turn into text. Anything else is either
# already cheap or refused by the guard.
READABLE = {
    ".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".jsonl", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".xml", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php",
    ".sh", ".ps1", ".sql", ".r", ".m", ".ipynb",
    ".pdf", ".docx", ".xlsx", ".xlsm", ".pptx", ".html", ".htm",
}

STATE_DIR = Path(tempfile.gettempdir()) / "claude-local-digest-hook"
STATE_MAX_AGE_S = 24 * 3600


def passthrough() -> None:
    """Let the Read happen. The silent, and by far the common, outcome."""
    sys.exit(0)


def read_roots() -> list[Path]:
    with CONFIG.open("rb") as fh:
        data = tomllib.load(fh)
    roots = data.get("paths", {}).get("read_roots") or []
    return [Path(str(r)).resolve() for r in roots]


def under_a_root(target: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def model_server_up(host: str = "127.0.0.1", port: int = 1234) -> bool:
    """A fast liveness check. If the helper is off, do not redirect to it."""
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def already_redirected(session: str, path: Path) -> bool:
    """True if this session was already told about this file.

    The escape hatch. Recorded on the way out, so the first call is redirected
    and the second is allowed through.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for stale in STATE_DIR.glob("*.txt"):
        try:
            if now - stale.stat().st_mtime > STATE_MAX_AGE_S:
                stale.unlink()
        except OSError:
            pass

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session)[:64]
    log = STATE_DIR / f"{safe or 'nosession'}.txt"
    key = str(path).lower()

    try:
        seen = log.read_text(encoding="utf-8").splitlines()
    except OSError:
        seen = []
    if key in seen:
        return True

    try:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(key + "\n")
    except OSError:
        pass
    return False


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        passthrough()

    tool_input = payload.get("tool_input") or {}
    raw_path = tool_input.get("file_path")
    if not raw_path:
        passthrough()

    # A bounded slice is already cheap; that is the behaviour we want, not one to
    # interrupt.
    if tool_input.get("limit"):
        passthrough()

    try:
        target = Path(raw_path).resolve()
        size = target.stat().st_size
    except (OSError, ValueError):
        passthrough()

    if size < MIN_BYTES:
        passthrough()
    if target.suffix.lower() not in READABLE:
        passthrough()

    try:
        roots = read_roots()
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        passthrough()
    if not under_a_root(target, roots):
        passthrough()  # the guard would refuse it; say nothing

    if already_redirected(str(payload.get("session_id") or ""), target):
        passthrough()

    if not SKIP_SERVER_CHECK and not model_server_up():
        passthrough()

    approx_tokens = size // 4
    reason = (
        f"This file is about {approx_tokens:,} tokens ({size:,} bytes). Reading it puts "
        f"all of that in context.\n\n"
        f"The local model on this machine can read it instead and hand back a short, "
        f"cited answer:\n\n"
        f'    local_digest(paths=["{target}"], question="<what you actually need to know>")\n\n'
        f"Use local_grep(paths, query) if you are looking for where something is rather "
        f"than what it says.\n\n"
        f"If you need the exact bytes - editing, patching, or matching a string - call "
        f"Read on this path again and it will go through untouched."
    )

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - a broken hook must never block a Read
        sys.exit(0)

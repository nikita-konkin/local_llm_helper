#!/usr/bin/env bash
# Option A: this machine runs llmhelper against the desktop's GPU.
#
# Your files stay here and are read here. Only the model call crosses the
# network, so no filesystem is exposed - which is why this is the setup to
# prefer when the work lives on this machine.
#
# Run it from inside a copy of the repo:
#
#   cd ~/local_llm_helper
#   ./laptop/setup-option-a.sh 192.168.50.32
#
# macOS: bash 3.2, no mapfile, no associative arrays.

set -u

HOST="${1:-192.168.50.32}"
PORT="${2:-1234}"
BASE_URL="http://${HOST}:${PORT}/v1"

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

say()  { printf '\n\033[1m%s\033[0m\n' "$1"; }
pass() { printf '  \033[32mok\033[0m    %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
info() { printf '        %s\n' "$1"; }

say "1. python"
PY=""
for cand in python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    V="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
    case "$V" in
      3.1[1-9]|3.[2-9]*) PY="$cand"; break ;;
    esac
  fi
done
if [ -z "$PY" ]; then
  fail "need Python 3.11 or newer"
  info "brew install python@3.12"
  exit 1
fi
pass "$PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

say "2. the desktop is reachable"
if curl -s --max-time 8 "${BASE_URL}/models" >/dev/null 2>&1; then
  # LM Studio pretty-prints its JSON, so "id" and its value are separated by a
  # space. A pattern assuming "id":"x" silently matches nothing and the check
  # looks like it passed with no models.
  MODELS="$(curl -s --max-time 8 "${BASE_URL}/models" \
    | tr ',' '\n' \
    | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -5)"
  pass "model server at ${BASE_URL}"
  for m in $MODELS; do info "$m"; done
else
  fail "cannot reach ${BASE_URL}"
  info "on the desktop:  llmh connect        (confirms address and that it is bound to 0.0.0.0)"
  info "                 llmh server start   (if it is not running)"
  info "and elevated:    & 'N:\\local_llm_helper\\autostart\\allow-lan.ps1'"
  exit 1
fi

say "3. virtualenv"
if [ ! -d .venv ]; then
  "$PY" -m venv .venv || { fail "could not create .venv"; exit 1; }
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/pip install --quiet mcp httpx sqlalchemy typer rich tiktoken psutil \
  || { fail "dependency install failed"; exit 1; }
pass "$(.venv/bin/python -c 'import mcp, httpx, sqlalchemy, typer, rich, tiktoken, psutil; print("dependencies installed")')"

# Importable from any working directory, so the MCP server starts no matter
# which project's session launches it.
SITE="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
echo "$ROOT" > "${SITE}/llmhelper.pth"
pass "importable from any directory"

say "4. config"
if [ -f config.toml ]; then
  info "config.toml exists - leaving it alone"
  info "check that base_url points at ${BASE_URL}"
else
  # context 8192 is deliberate, not a typo. This machine cannot load models -
  # that is the desktop's job - so if nothing is resident there, LM Studio
  # JIT-loads at its own default context, which is 8192. Declaring more here
  # would size chunks for a window we might not get, and the failure looks like
  # "Context size has been exceeded" rather than anything obvious.
  cat > config.toml <<TOML
[server]
base_url = "${BASE_URL}"
# No LM Studio CLI on this machine. That is expected: without it the client runs
# in remote mode and leaves model residency to the machine actually serving.
lms_exe  = "/usr/local/bin/lms"
request_timeout_s = 900

[paths]
outbox     = "results"
jobs_db    = "jobs/llmhelper.db"
logs       = "logs"
bench      = "bench"
# THE security boundary: the only directories the local model may read.
# Narrow this to what you actually work on.
read_roots = ["${HOME}/projects", "${HOME}/Documents"]
deny_globs = [
  "**/.git/**", "**/.venv/**", "**/__pycache__/**", "**/node_modules/**",
  "**/.ssh/**", "**/.env", "**/.env.*", "**/*.key", "**/*.pem",
  "**/.credentials.json", "**/.claude/**",
]

[budgets]
express_max_out_tokens = 1200
express_timeout_s      = 420
map_chunk_tokens       = 4000
map_overlap_tokens     = 200
map_summary_tokens     = 400
max_map_chunks         = 400
map_concurrency        = 4
result_retention_days  = 30

[worker]
poll_interval_s = 2.0
max_attempts    = 3
fallback_tier   = "fast"

[tiers.fast]
model            = "qwen/qwen3.5-9b"
context          = 8192
min_free_ram_gb  = 1
reasoning_effort = "none"
temperature      = 0.2
TOML
  pass "wrote config.toml"
  info "read_roots is currently ${HOME}/projects and ${HOME}/Documents - edit it"
fi

say "5. does it actually work"
OUT="$(.venv/bin/python -m llmhelper.cli ask 'What is six times seven? Reply with only the number.' --max-tokens 24 2>&1)"
case "$OUT" in
  *42*) pass "a real generation came back from the desktop's GPU" ;;
  *)
    fail "the call did not produce an answer"
    info "$(printf '%s' "$OUT" | tail -5)"
    exit 1
    ;;
esac

say "6. register with Claude Code"
cat <<REG
No CLI needed. Save this as .mcp.json in a project folder:

{
  "mcpServers": {
    "local-llm": {
      "command": "${ROOT}/.venv/bin/python",
      "args": ["-m", "llmhelper.mcp_server"]
    }
  }
}

For every project instead of one, put that same mcpServers object under the
top-level key in ~/.claude.json.

Then quit Claude Code completely and reopen it. A running session will not pick
up a new server.

Note this is a stdio server - Claude Code launches it here, on this machine, so
local_digest reads THIS filesystem. That is the point of option A.
REG

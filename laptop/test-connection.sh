#!/usr/bin/env bash
# Prove the MCP server works from THIS machine before wiring it into Claude Code.
#
# Claude Code gives up on a server that fails repeatedly, and that state lasts
# for the session. Verifying with curl first means a retry is spent on something
# already known to work, instead of on a hypothesis.
#
#   ./test-connection.sh 192.168.50.32
#
# Written for macOS: bash 3.2, no mapfile, no associative arrays.

set -u

HOST="${1:-192.168.50.32}"
MCP_PORT="${2:-8765}"
MODEL_PORT="${3:-1234}"
URL="http://${HOST}:${MCP_PORT}/mcp"

pass() { printf '  \033[32mok\033[0m    %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
info() { printf '        %s\n' "$1"; }

echo "testing ${HOST} from $(hostname)"
echo

# 1. Is the host reachable at all? Informational only: ICMP is often blocked
#    while TCP is perfectly fine, so a failure here is not a reason to stop.
echo "1. host reachable (informational)"
if ping -c 2 -W 3000 "$HOST" >/dev/null 2>&1 || ping -c 2 -t 3 "$HOST" >/dev/null 2>&1    || ping -n 2 -w 3000 "$HOST" >/dev/null 2>&1; then
  pass "ping"
else
  info "no ping reply - not conclusive, ICMP is often blocked. Continuing."
fi

# 2. TCP. A timeout here with ping working is the Windows firewall dropping
#    inbound, not refusing it. Ask for autostart/allow-lan.ps1 to be run.
echo
echo "2. ports open"
for p in "$MODEL_PORT" "$MCP_PORT"; do
  # -G is a BSD/macOS connect-timeout flag; fall back where it is not supported.
  if nc -z -G 5 "$HOST" "$p" >/dev/null 2>&1 || nc -z -w 5 "$HOST" "$p" >/dev/null 2>&1; then
    pass "tcp ${p}"
  else
    fail "tcp ${p} - no connection"
    info "if ping worked but this did not, the firewall is dropping inbound."
    info "on the Windows box, elevated:  & 'N:\\local_llm_helper\\autostart\\allow-lan.ps1'"
    info "and check something is actually listening:  llmh connect"
  fi
done

# 3. MCP handshake.
echo
echo "3. mcp handshake"
HEADERS_FILE="$(mktemp)"
BODY_FILE="$(mktemp)"
trap 'rm -f "$HEADERS_FILE" "$BODY_FILE"' EXIT

curl -s --max-time 15 -D "$HEADERS_FILE" -o "$BODY_FILE" -X POST "$URL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test-connection","version":"1"}}}' \
  >/dev/null 2>&1

SID="$(tr -d '\r' < "$HEADERS_FILE" | awk 'tolower($1)=="mcp-session-id:"{print $2}')"
if [ -n "$SID" ]; then
  pass "initialize (session ${SID})"
else
  fail "initialize - no session id"
  info "response: $(head -c 200 "$BODY_FILE")"
  exit 1
fi

curl -s --max-time 15 -o /dev/null -X POST "$URL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "mcp-session-id: ${SID}" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' >/dev/null 2>&1

# 4. The one that actually matters. A handshake is a single fast round trip and
#    survives almost anything; a tool call takes seconds while the model works,
#    and that is the part which used to die in transit.
echo
echo "4. tool call (the part that was failing)"
START=$(date +%s)
RESP="$(curl -s --max-time 120 -X POST "$URL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "mcp-session-id: ${SID}" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"local_ask","arguments":{"prompt":"Reply with exactly: LAPTOP_OK","max_out_tokens":20}}}' 2>&1)"
ELAPSED=$(( $(date +%s) - START ))

case "$RESP" in
  *LAPTOP_OK*)
    pass "local_ask returned in ${ELAPSED}s"
    echo
    echo "The server is reachable and tool calls work from this machine."
    echo "If Claude Code still reports it as unreachable, that is client-side"
    echo "state: it stops retrying a server that failed earlier in the session."
    echo "Start a new session, or re-add it:"
    echo
    echo "  claude mcp remove local-llm"
    echo "  claude mcp add --transport http local-llm ${URL}"
    ;;
  "")
    fail "no response after ${ELAPSED}s"
    info "the call left this machine and nothing came back - the server may be"
    info "streaming instead of returning JSON. On the Windows box check that"
    info "start-mcp-http.cmd does NOT pass --sse."
    ;;
  *)
    fail "unexpected response after ${ELAPSED}s"
    info "$(printf '%s' "$RESP" | head -c 300)"
    ;;
esac

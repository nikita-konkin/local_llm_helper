"""Speak MCP over HTTP, the way a remote client does.

The stdio smoke test does not exercise this path at all: streamable-HTTP carries
a session id, an SSE stream and its own timeouts. A handshake that succeeds while
tool calls fail is invisible to the stdio test.

    .venv/Scripts/python.exe tests/mcp_http_smoke.py --url http://127.0.0.1:8765/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main(url: str, call: str, args_json: str, timeout: float,
               expect: str | None) -> int:
    print(f"connecting to {url}")
    t0 = time.monotonic()
    try:
        async with streamable_http_client(url) as streams:
            read, write = streams[0], streams[1]
            get_session_id = streams[2] if len(streams) > 2 else None
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                info = getattr(init, "server_info", None) or getattr(init, "serverInfo")
                sid = get_session_id() if callable(get_session_id) else None
                print(f"  handshake ok in {time.monotonic() - t0:.1f}s "
                      f"-> {info.name} v{info.version}")
                if sid:
                    print(f"  session id: {sid}")

                tools = (await session.list_tools()).tools
                print(f"  tools/list: {len(tools)} tools")

                print(f"\ncalling {call}({args_json})")
                t1 = time.monotonic()
                res = await session.call_tool(call, json.loads(args_json))
                print(f"  returned in {time.monotonic() - t1:.1f}s")
                body = ""
                for block in res.content:
                    text = getattr(block, "text", None)
                    if text:
                        body += text
                        print("  " + text[:600].replace("\n", "\n  "))

                # A verdict, not just output. Two independent checks, because
                # matching on a string we sent is a false pass waiting to happen:
                # error shapes that echo the arguments back would read as success.
                #   isError - structural, from the envelope rather than the text
                #   expect  - a token that appears nowhere in the request, so it
                #             can only come from the model having actually run
                if bool(getattr(res, "isError", False)):
                    print("\n  VERDICT: FAIL - server reported isError")
                    return 1
                if expect and expect not in body:
                    print(f"\n  VERDICT: FAIL - transport fine, {expect!r} not in the answer")
                    return 1
                print("\n  VERDICT: ok - isError false and the model answered")
                return 0
    except Exception as exc:
        print(f"\nFAILED after {time.monotonic() - t0:.1f}s")
        print(f"  {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    ap.add_argument("--call", default="local_ask")
    ap.add_argument(
        "--args",
        default='{"prompt":"What is six times seven? Reply with only the number.",'
                '"max_out_tokens":24}',
    )
    ap.add_argument(
        "--expect", default="42",
        help="Token that must appear in the answer. Choose one that is NOT in the "
             "request, so an echoed error cannot satisfy it.",
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    ns = ap.parse_args()
    sys.exit(asyncio.run(main(ns.url, ns.call, ns.args, ns.timeout, ns.expect)))

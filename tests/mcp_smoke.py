"""Speak MCP to our own server over stdio and report what it exposes.

Run directly:  .venv/Scripts/python.exe tests/mcp_smoke.py
Optionally with a tool call:  ... tests/mcp_smoke.py --call local_stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


async def main(call: str | None, args_json: str, cwd: str | None = None) -> int:
    # cwd matters: the server records it as the calling project, exactly as Claude
    # Code does when it launches the server from a session's directory.
    params = StdioServerParameters(
        command=str(PYTHON), args=["-m", "llmhelper.mcp_server"], cwd=cwd or str(ROOT)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = getattr(init, "server_info", None) or getattr(init, "serverInfo")
            print(f"server: {info.name} v{info.version}")

            tools = (await session.list_tools()).tools
            print(f"\n{len(tools)} tool(s):")
            for t in tools:
                schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", {})
                required = schema.get("required", [])
                params_desc = ", ".join(
                    f"{k}{'*' if k in required else ''}"
                    for k in schema.get("properties", {})
                )
                first_line = (t.description or "").strip().splitlines()[0]
                print(f"  - {t.name}({params_desc})")
                print(f"      {first_line}")

            if call:
                print(f"\ncalling {call}({args_json}) ...")
                res = await session.call_tool(call, json.loads(args_json))
                for block in res.content:
                    text = getattr(block, "text", None)
                    if text:
                        print(text[:4000])
            return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", default=None)
    ap.add_argument("--args", default="{}")
    ap.add_argument("--cwd", default=None, help="Pretend to be a session in this directory.")
    ns = ap.parse_args()
    sys.exit(asyncio.run(main(ns.call, ns.args, ns.cwd)))

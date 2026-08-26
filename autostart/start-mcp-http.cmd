@echo off
REM Serves the MCP tools over HTTP so another machine on the LAN can use them.
REM
REM Tool results are returned as a single JSON response rather than an SSE
REM stream. Streaming holds a connection open for the whole call, which is what
REM fails across a network: the handshake takes 0.1 s and always works, then a
REM multi-second tool call dies to an idle timeout and looks like the server
REM being unreachable. Pass --sse to go back to streaming for debugging.
REM
REM This exposes read access to everything under paths.read_roots to anything
REM that can reach the port. There is no authentication.
cd /d "%~dp0.."
.venv\Scripts\python.exe -m llmhelper.mcp_server --http --host 0.0.0.0 --port 8765

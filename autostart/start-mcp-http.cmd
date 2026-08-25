@echo off
REM Serves the MCP tools over HTTP so another machine on the LAN can use them.
REM
REM NOT started automatically. This exposes read access to everything under
REM paths.read_roots to anyone who can reach the port, with no authentication.
REM Start it deliberately, on a network you control.
cd /d "%~dp0.."
.venv\Scripts\python.exe -m llmhelper.mcp_server --http --host 0.0.0.0 --port 8765

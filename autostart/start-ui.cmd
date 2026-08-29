@echo off
REM Serves the dashboard: status, CPU/GPU load, usage, the queue and settings.
REM
REM Bound to 127.0.0.1. The controls on this page start processes and load
REM multi-gigabyte models, and there is no authentication, so it is not something
REM to put on the network by accident. Pass --host 0.0.0.0 only deliberately.
cd /d "%~dp0.."
.venv\Scripts\python.exe -m llmhelper.cli ui --no-open --host 127.0.0.1 --port 8770

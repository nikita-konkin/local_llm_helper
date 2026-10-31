@echo off
REM Drains the job queue in the background. Without this, local_submit accepts
REM work that never runs.
REM
REM Waits for the model server first: at logon this task and the server task race,
REM and the worker exits immediately if the server is not up yet.
cd /d "%~dp0.."

for /L %%i in (1,1,30) do (
  .venv\Scripts\python.exe -c "import sys; from llmhelper.client import LocalLLM; sys.exit(0 if LocalLLM().healthy() else 1)" 2>nul && goto :ready
  timeout /t 5 /nobreak >nul
)
echo Model server did not come up; starting the worker anyway.

:ready
.venv\Scripts\python.exe -m llmhelper.cli worker --quiet

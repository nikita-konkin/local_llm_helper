@echo off
REM Starts the LM Studio OpenAI-compatible server.
REM
REM The bind address comes from config.toml ([server] bind), which is the single
REM source of truth. This script used to hardcode 127.0.0.1 while config.toml said
REM 0.0.0.0, and since the autostart entry runs at every boot it silently undid the
REM LAN binding each time - the laptop worked until the next reboot, then did not,
REM with nothing in either file looking wrong on its own.
REM
REM 0.0.0.0 means anything that reaches port 1234 can use this GPU. There is no
REM authentication on the LM Studio server. Keep the firewall rule scoped to your
REM own subnet (autostart\allow-lan.ps1 does this), and set bind = "127.0.0.1" in
REM config.toml if you do not want the laptop to reach it.
cd /d "%~dp0.."

set BIND=127.0.0.1
set PORT=1234
for /f "usebackq tokens=1,2" %%a in (`.venv\Scripts\python.exe -c "from llmhelper.config import get_config; s = get_config().server; print(s.bind, s.port)" 2^>nul`) do (
  set BIND=%%a
  set PORT=%%b
)

echo Starting LM Studio server on %BIND%:%PORT%
"%USERPROFILE%\.lmstudio\bin\lms.exe" server start --bind %BIND% --port %PORT%

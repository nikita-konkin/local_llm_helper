@echo off
REM Starts the LM Studio OpenAI-compatible server at logon.
REM
REM Bound to 127.0.0.1 deliberately: an unattended task that listens on the LAN
REM with no authentication is a different security posture than one you start by
REM hand. To allow laptop access, change BIND to 0.0.0.0 and re-run setup.ps1.
set BIND=127.0.0.1

"%USERPROFILE%\.lmstudio\bin\lms.exe" server start --bind %BIND% --port 1234

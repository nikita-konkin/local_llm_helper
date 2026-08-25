@echo off
REM Launcher for the local_llm_helper CLI. Put this directory on PATH to use
REM `llmh` from anywhere, or call it by full path.
"%~dp0.venv\Scripts\python.exe" -m llmhelper.cli %*

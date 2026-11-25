@echo off
chcp 65001 >nul
title WhoScored 2026/27 Daily Sync -> Hugging Face
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_whoscored_hf.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
echo Finished with exit code %RC%.
pause
exit /b %RC%

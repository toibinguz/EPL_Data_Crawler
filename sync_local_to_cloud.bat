@echo off
chcp 65001 >nul
title Sync Local to Cloud (Hugging Face)
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_local_to_cloud.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
echo Finished with exit code %RC%.
pause
exit /b %RC%

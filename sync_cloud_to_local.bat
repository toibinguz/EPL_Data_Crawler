@echo off
chcp 65001 >nul
title Sync Cloud to Local (Hugging Face)
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_cloud_to_local.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
echo Finished with exit code %RC%.
pause
exit /b %RC%

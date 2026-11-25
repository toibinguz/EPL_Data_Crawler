@echo off
chcp 65001 >nul
title Daily EPL Raw Crawler
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0crawl_daily.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
echo Finished with exit code %RC%.
pause
exit /b %RC%

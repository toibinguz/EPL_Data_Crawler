param(
    [switch]$NoSync
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Daily Premier League Raw Crawl Pipeline (2026/27)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

$pyVer = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found on PATH."
    exit 1
}
Write-Host "[*] Using $pyVer"

$pyArgs = @("-u", "crawler_pipeline/daily_raw_crawler.py")
if ($NoSync) { $pyArgs += "--no-sync" }

Write-Host "[*] Executing: python $($pyArgs -join ' ')" -ForegroundColor Yellow
& python @pyArgs
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "`n[+] Daily raw crawl finished cleanly." -ForegroundColor Green
} else {
    Write-Warning "`n[!] Daily raw crawl exited with code $exitCode."
}
exit $exitCode

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Pushing local raw_data & 2026_2027_new to Hugging Face Hub" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

$pyVer = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found on PATH."
    exit 1
}
Write-Host "[*] Using $pyVer"

$pyArgs = @("-u", "sync_local_to_cloud.py")
Write-Host "[*] Executing: python $($pyArgs -join ' ')" -ForegroundColor Yellow
& python @pyArgs
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "`n[+] Local -> Cloud sync finished cleanly." -ForegroundColor Green
} else {
    Write-Warning "`n[!] Local -> Cloud sync exited with code $exitCode."
}
exit $exitCode

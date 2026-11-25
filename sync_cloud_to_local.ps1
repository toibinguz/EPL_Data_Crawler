param(
    [switch]$Unpack
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Pulling raw_data, 2026_2027_new, clean from Hugging Face Hub" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

$pyVer = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found on PATH."
    exit 1
}
Write-Host "[*] Using $pyVer"

$pyArgs = @("-u", "sync_cloud_to_local.py")
if ($Unpack) { $pyArgs += "--unpack" }

Write-Host "[*] Executing: python $($pyArgs -join ' ')" -ForegroundColor Yellow
& python @pyArgs
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "`n[+] Cloud -> Local sync finished cleanly." -ForegroundColor Green
} else {
    Write-Warning "`n[!] Cloud -> Local sync exited with code $exitCode."
}
exit $exitCode

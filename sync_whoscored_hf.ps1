param(
    [switch]$CheckOnly,
    [switch]$SkipBrowser,
    [switch]$NoSync,
    [int]$Limit = 0,
    [int]$Timeout = 60
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  WhoScored 2026/27 Daily Checker, Downloader & Hugging Face Sync" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# Check python and playwright availability
$pyVer = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found on PATH."
    exit 1
}
Write-Host "[*] Using $pyVer"

python -c "import playwright, huggingface_hub" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Missing required packages. Run: pip install playwright huggingface_hub && playwright install chromium"
    exit 1
}

$pyArgs = @("-u", "crawler_pipeline/whoscored_sync_hf.py", "--timeout", "$Timeout")
if ($CheckOnly)   { $pyArgs += "--check-only" }
if ($SkipBrowser) { $pyArgs += "--skip-browser" }
if ($NoSync)      { $pyArgs += "--no-sync" }
if ($Limit -gt 0) { $pyArgs += @("--limit", "$Limit") }

Write-Host "[*] Executing: python $($pyArgs -join ' ')" -ForegroundColor Yellow
& python @pyArgs
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "`n[+] WhoScored daily sync completed cleanly." -ForegroundColor Green
} else {
    Write-Warning "`n[!] WhoScored daily sync finished with exit code $exitCode."
}
exit $exitCode

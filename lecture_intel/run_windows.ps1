$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw 'Recorder is not installed. Run .\install_windows.ps1 first.'
}
Set-Location $Root
& $Python (Join-Path $Root 'app.py')
exit $LASTEXITCODE

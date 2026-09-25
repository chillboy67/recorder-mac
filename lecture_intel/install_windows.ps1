[CmdletBinding()]
param(
    [switch]$DownloadCpuModel,
    [switch]$NoShortcut
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    & py -3.11 --version *> $null
    if ($LASTEXITCODE -eq 0) {
        $Python = 'py'
        $PythonArgs = @('-3.11')
    }
}
if (-not $Python) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw 'Python 3.10+ is required. Install Python from python.org and enable the py launcher.'
    }
    & python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
    if ($LASTEXITCODE -ne 0) {
        throw 'Python 3.10 or newer is required.'
    }
    $Python = 'python'
    $PythonArgs = @()
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning 'ffmpeg was not found. Install it and add it to PATH for audio decoding.'
}

$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
    & $Python @PythonArgs -m venv (Join-Path $Root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the virtual environment.' }
}
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip.' }
& $VenvPython -m pip install -r (Join-Path $Root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install Recorder dependencies.' }

if ($DownloadCpuModel) {
    & $VenvPython (Join-Path $Root 'download_models.py') --engine cpu small
    if ($LASTEXITCODE -ne 0) { throw 'CPU model download failed.' }
}

if (-not $NoShortcut) {
    $ShortcutDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
    New-Item -ItemType Directory -Force -Path $ShortcutDir | Out-Null
    $ShortcutPath = Join-Path $ShortcutDir 'Recorder.lnk'
    $Shell = New-Object -ComObject WScript.Shell
    $Shortcut = $Shell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = (Get-Command powershell.exe).Source
    $Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $Root 'run_windows.ps1')`""
    $Shortcut.WorkingDirectory = $Root
    $Shortcut.Description = 'Local speech-to-text'
    $Shortcut.Save()
}

Write-Host "Recorder installed in: $Root"
Write-Host 'Launch from the Start Menu or run .\run_windows.ps1'
Write-Host "GPU setup instructions: $(Join-Path $Root 'docs\GPU_BACKENDS.md')"

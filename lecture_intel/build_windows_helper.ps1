$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$nativeDirectory = Join-Path $scriptDirectory "native"
$sourcePath = Join-Path $nativeDirectory "SystemAudioRecorderWindows.cpp"
$outputPath = Join-Path $nativeDirectory "system_audio_recorder.exe"
$stagedPath = Join-Path $nativeDirectory "system_audio_recorder.build.exe"
$objectPath = Join-Path $nativeDirectory "SystemAudioRecorderWindows.obj"
$linkPdbPath = Join-Path $nativeDirectory "system_audio_recorder.build.pdb"
$incrementalPath = Join-Path $nativeDirectory "system_audio_recorder.build.ilk"

$compiler = Get-Command cl.exe -CommandType Application -ErrorAction SilentlyContinue
if ($null -eq $compiler) {
    # Developer PowerShell is not always active: locate the Build Tools install
    # and import its environment the same way VsDevCmd.bat does.
    $vswhere = Join-Path "${env:ProgramFiles(x86)}" 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        $installationPath = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        if ($installationPath) {
            $vsDevCmd = Join-Path $installationPath 'Common7\Tools\VsDevCmd.bat'
            if (Test-Path -LiteralPath $vsDevCmd -PathType Leaf) {
                Write-Host "Importing the Visual Studio build environment: $installationPath"
                cmd /c "`"$vsDevCmd`" -arch=amd64 -host_arch=amd64 && set" | ForEach-Object {
                    if ($_ -match '^([^=]+)=(.*)$') {
                        Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
                    }
                }
                $compiler = Get-Command cl.exe -CommandType Application -ErrorAction SilentlyContinue
            }
        }
    }
}

if ($null -eq $compiler) {
    [Console]::Error.WriteLine(
        "ERROR: cl.exe was not found. Install Visual Studio Build Tools with the " +
        "'Desktop development with C++' workload, then run this script again.")
    exit 1
}

if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    [Console]::Error.WriteLine("ERROR: Windows helper source not found: $sourcePath")
    exit 1
}

$compilerPath = $compiler.Source

$arguments = @(
    "/nologo"
    "/std:c++17"
    "/O2"
    "/EHsc"
    "/W4"
    "/permissive-"
    "/Zc:__cplusplus"
    "/MT"
    "/DUNICODE"
    "/D_UNICODE"
    "/DNOMINMAX"
    "/DWIN32_LEAN_AND_MEAN"
    "/Fo$objectPath"
    "/Fe$stagedPath"
    $sourcePath
    "/link"
    "/INCREMENTAL:NO"
    "ole32.lib"
    "uuid.lib"
)

$buildSucceeded = $false
$buildExitCode = 1
$failureMessage = ""

try {
    if (Test-Path -LiteralPath $stagedPath) {
        Remove-Item -LiteralPath $stagedPath -Force
    }

    Write-Host "Building native/system_audio_recorder.exe with $compilerPath"
    & $compilerPath @arguments
    $buildExitCode = $LASTEXITCODE

    if ($buildExitCode -ne 0) {
        throw "cl.exe exited with code $buildExitCode"
    }
    if (-not (Test-Path -LiteralPath $stagedPath -PathType Leaf)) {
        throw "cl.exe reported success but did not create $stagedPath"
    }

    # Replace the shipped helper only after a successful compile, so a failed
    # rebuild never destroys a working binary.
    Move-Item -LiteralPath $stagedPath -Destination $outputPath -Force
    $buildSucceeded = $true
}
catch {
    $failureMessage = $_.Exception.Message
}
finally {
    foreach ($intermediatePath in @($objectPath, $linkPdbPath, $incrementalPath)) {
        if (Test-Path -LiteralPath $intermediatePath) {
            Remove-Item -LiteralPath $intermediatePath -Force -ErrorAction SilentlyContinue
        }
    }

    if (Test-Path -LiteralPath $stagedPath) {
        Remove-Item -LiteralPath $stagedPath -Force -ErrorAction SilentlyContinue
    }
}

if (-not $buildSucceeded) {
    [Console]::Error.WriteLine("ERROR: failed to build Windows system audio helper: $failureMessage")
    if ($buildExitCode -eq 0) {
        $buildExitCode = 1
    }
    exit $buildExitCode
}

Write-Host "Built native/system_audio_recorder.exe"
exit 0

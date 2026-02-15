param(
    [string]$TargetDir = "ARCA",
    [string]$RepoUrl = "https://github.com/3spky5u-oss/ARCA.git"
)

$ErrorActionPreference = "Stop"

if (Test-Path "$PSScriptRoot/scripts/bootstrap.ps1") {
    Set-Location $PSScriptRoot
    ./scripts/bootstrap.ps1
    exit $LASTEXITCODE
}

if (Test-Path "$TargetDir/.git") {
    Write-Host "Using existing repo: $TargetDir"
} else {
    git clone $RepoUrl $TargetDir
}

Set-Location $TargetDir
./scripts/bootstrap.ps1

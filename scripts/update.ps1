param(
    [switch]$Build,
    [switch]$NoGit,
    [switch]$NoPull,
    [switch]$DownloadMissingModels
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$args = @('scripts/arca.py', 'update')
if ($Build) { $args += '--build' }
if ($NoGit) { $args += '--no-git' }
if ($NoPull) { $args += '--no-pull' }
if ($DownloadMissingModels) { $args += '--download-missing-models' }

python @args
exit $LASTEXITCODE

param(
    [string]$Dest,
    [string]$Branch = "main",
    [string]$Remote = "origin",
    [string]$RemoteUrl,
    [string]$CommitMessage = "chore(public): sync from private source",
    [switch]$SkipPush,
    [switch]$NoClean,
    [switch]$NoPreserveGit
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$args = @('scripts/arca.py', 'publish-public', '--branch', $Branch, '--remote', $Remote, '--commit-message', $CommitMessage)
if ($Dest) { $args += @('--dest', $Dest) }
if ($RemoteUrl) { $args += @('--remote-url', $RemoteUrl) }
if ($SkipPush) { $args += '--skip-push' }
if ($NoClean) { $args += '--no-clean' }
if ($NoPreserveGit) { $args += '--no-preserve-git' }

python @args
exit $LASTEXITCODE

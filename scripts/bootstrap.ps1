param(
    [switch]$Build,
    [switch]$Gpu,
    [switch]$Cpu,
    [switch]$SkipModelDownload,
    [switch]$SkipPreflight,
    [switch]$NoPull,
    [switch]$WithMcpKey
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$args = @('scripts/arca.py', 'bootstrap')
if ($Build) { $args += '--build' }
if ($Gpu) { $args += '--gpu' }
if ($Cpu) { $args += '--cpu' }
if ($SkipModelDownload) { $args += '--skip-model-download' }
if ($SkipPreflight) { $args += '--skip-preflight' }
if ($NoPull) { $args += '--no-pull' }
if ($WithMcpKey) { $args += '--with-mcp-key' }

python @args
exit $LASTEXITCODE

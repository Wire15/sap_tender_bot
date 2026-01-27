param(
    [switch]$Ui,
    [switch]$Digest
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $Ui -and -not $Digest) {
    $Digest = $true
}

if ($Ui) {
    sap-tender-ui
    exit $LASTEXITCODE
}

$extra = @(
    "--config", "$repoRoot\config.yaml",
    "--dry-run",
    "--export-csv",
    "--export-near-miss",
    "--export-report",
    "--no-llm"
)

sap-tender-digest @extra

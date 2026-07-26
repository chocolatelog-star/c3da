param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$OutputRoot = "J:\nlp\CD-C3DA\runs\reproducible",
    [string]$Cuda = "0",
    [switch]$DryRun,
    [switch]$AllowDirtyDiagnostic
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "J:\conda\envs\c3da\python.exe"
$Runner = Join-Path $ProjectRoot "run_reproducible_pipeline.py"
$Recipe = Join-Path $ProjectRoot "configs\recipes\rest16_to_laptop14_best_v1.json"
$RunRoot = Join-Path (Join-Path $OutputRoot "rest16_to_laptop14_best_v1") $RunId
$Manifest = Join-Path $RunRoot "manifest.json"
$Resume = Test-Path -LiteralPath $Manifest

$UserCommand = "cmd /c `"J: && cd /d $ProjectRoot && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_best_reproducible_pipeline.ps1 -RunId $RunId -OutputRoot $OutputRoot -Cuda $Cuda"
if ($DryRun) { $UserCommand += " -DryRun" }
if ($AllowDirtyDiagnostic) { $UserCommand += " -AllowDirtyDiagnostic" }
$UserCommand += "`""
$EncodedUserCommand = "base64:" + [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($UserCommand))

$Arguments = @(
    $Runner,
    "--recipe", $Recipe,
    "--run_id", $RunId,
    "--output_root", $OutputRoot,
    "--cuda", $Cuda,
    "--user_command", $EncodedUserCommand
)
if ($DryRun) { $Arguments += "--dry_run" }
if ($AllowDirtyDiagnostic) { $Arguments += "--allow_dirty" }

Write-Host "[native-repro] run_root=$RunRoot"
Write-Host "[native-repro] resume=$Resume"
Write-Host "[native-repro] logs=$(Join-Path $RunRoot 'logs')"
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

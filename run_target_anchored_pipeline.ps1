param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(1, 2, 3)]
    [int]$Step,
    [string]$OutputRoot = "J:\nlp\CD-C3DA\runs\reproducible",
    [string]$Cuda = "0",
    [switch]$DryRun,
    [switch]$AllowDirtyDiagnostic
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "J:\conda\envs\c3da\python.exe"
$Runner = Join-Path $ProjectRoot "run_reproducible_pipeline.py"
$RecipeNames = @{
    1 = "laptop14_to_rest15_target_anchor_step1_v1.json"
    2 = "laptop14_to_rest15_target_anchor_step2_gap_v1.json"
    3 = "laptop14_to_rest15_target_anchor_step3_tiered_v1.json"
}
$RunIds = @{
    1 = "laptop14-rest15-target-anchor-step1-seed1000-v1"
    2 = "laptop14-rest15-target-anchor-step2-gap-seed1000-v1"
    3 = "laptop14-rest15-target-anchor-step3-tiered-seed1000-v1"
}
$Recipe = Join-Path $ProjectRoot ("configs\recipes\experiments\" + $RecipeNames[$Step])
$RunId = $RunIds[$Step]
$RecipeId = [System.IO.Path]::GetFileNameWithoutExtension($RecipeNames[$Step])
$RunRoot = Join-Path (Join-Path $OutputRoot $RecipeId) $RunId
$Manifest = Join-Path $RunRoot "manifest.json"
$Resume = Test-Path -LiteralPath $Manifest

$UserCommand = "cmd /c `"J: && cd /d $ProjectRoot && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_target_anchored_pipeline.ps1 -Step $Step -OutputRoot $OutputRoot -Cuda $Cuda"
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

Write-Host "[target-anchor] step=$Step"
Write-Host "[target-anchor] run_root=$RunRoot"
Write-Host "[target-anchor] resume=$Resume"
Write-Host "[target-anchor] logs=$(Join-Path $RunRoot 'logs')"
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

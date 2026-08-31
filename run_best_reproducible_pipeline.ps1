param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$OutputRoot = "runs\reproducible",
    [string]$Cuda = "0",
    [int]$TrainBatchSize,
    [int]$EvalBatchSize,
    [int]$GradientAccumulationSteps,
    [switch]$DryRun,
    [switch]$AllowDirtyDiagnostic
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($env:C3DA_PYTHON) { $env:C3DA_PYTHON } elseif ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { "python" }
$Runner = Join-Path $ProjectRoot "run_reproducible_pipeline.py"
$Recipe = Join-Path $ProjectRoot "configs\recipes\rest16_to_laptop14_best_v1.json"
$RunRoot = Join-Path (Join-Path $OutputRoot "rest16_to_laptop14_best_v1") $RunId
$Manifest = Join-Path $RunRoot "manifest.json"
$Resume = Test-Path -LiteralPath $Manifest

$UserCommand = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -RunId $RunId -OutputRoot $OutputRoot -Cuda $Cuda"
if ($PSBoundParameters.ContainsKey('TrainBatchSize')) { $UserCommand += " -TrainBatchSize $TrainBatchSize" }
if ($PSBoundParameters.ContainsKey('EvalBatchSize')) { $UserCommand += " -EvalBatchSize $EvalBatchSize" }
if ($PSBoundParameters.ContainsKey('GradientAccumulationSteps')) { $UserCommand += " -GradientAccumulationSteps $GradientAccumulationSteps" }
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
if ($PSBoundParameters.ContainsKey('TrainBatchSize')) { $Arguments += @('--train_batch_size', $TrainBatchSize) }
if ($PSBoundParameters.ContainsKey('EvalBatchSize')) { $Arguments += @('--eval_batch_size', $EvalBatchSize) }
if ($PSBoundParameters.ContainsKey('GradientAccumulationSteps')) { $Arguments += @('--gradient_accumulation_steps', $GradientAccumulationSteps) }

Write-Host "[native-repro] run_root=$RunRoot"
Write-Host "[native-repro] resume=$Resume"
Write-Host "[native-repro] logs=$(Join-Path $RunRoot 'logs')"
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

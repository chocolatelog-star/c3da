param(
    [string]$Pairs = "rest16:laptop14",
    [int]$Seed = 1000,
    [string]$OutputRoot = "runs\bgca_aste_stage1_full_pipeline_historical_seed_v2",
    [string]$Cuda = "0",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Python = "J:\conda\envs\c3da\python.exe"
$env:PYTHONUNBUFFERED = "1"
$LogDir = Join-Path $OutputRoot "logs"
$LogPath = Join-Path $LogDir "historical_seed${Seed}.log"
$ManifestPath = Join-Path $LogDir "historical_seed${Seed}_manifest.json"

if (-not (Test-Path $Python)) {
    throw "missing Python: $Python"
}

$Arguments = @(
    "run_bgca_aste_stage1_pairs.py",
    "--output_root", $OutputRoot,
    "--pairs", $Pairs,
    "--extractor_model_path", "J:\nlp\models\t5-base-py",
    "--generator_model_path", "J:\nlp\models\t5-base-py",
    "--generator_prompt_style", "label_to_text",
    "--augment_prompt_style", "masked_mutual",
    "--domain_prefix_style", "text",
    "--extractor_epochs", "25",
    "--generator_epochs", "8",
    "--generator_checkpoint_selection", "best",
    "--final_epochs", "5",
    "--complete_multi_extra_weight", "0.25",
    "--final_pseudo_weight", "0.65",
    "--final_augment_weight", "0.20",
    "--lambda_sentiment_contrastive", "0.01",
    "--sentiment_contrastive_source_only",
    "--sentiment_contrastive_class_balanced",
    "--learning_rate", "0.0003",
    "--eval_batch_size", "2",
    "--cuda", $Cuda,
    "--seed", ([string]$Seed)
)
if ($DryRun) {
    $Arguments += "--dry_run"
}

Write-Host ("[historical-seed-pipeline] output_root={0} seed={1} pairs={2}" -f $OutputRoot, $Seed, $Pairs)
Write-Host ("{0} {1}" -f $Python, ($Arguments -join " "))

if ($DryRun) {
    & $Python @Arguments
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (-not (Test-Path $ManifestPath)) {
    $Manifest = [ordered]@{
        experiment = "full_pipeline_historical_seed"
        output_root = $OutputRoot
        pairs = $Pairs
        seed = $Seed
        cuda = $Cuda
        reproducibility_mode = "historical_seed_only"
        base_model = "J:\nlp\models\t5-base-py"
        extractor_epochs = 25
        generator_epochs = 8
        generator_checkpoint_selection = "best"
        final_epochs = 5
        complete_multi_extra_weight = 0.25
        final_pseudo_weight = 0.65
        final_augment_weight = 0.20
        lambda_sentiment_contrastive = 0.01
        command = "${Python} $($Arguments -join ' ')"
    }
    $Manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $ManifestPath
}

$TranscriptStarted = $false
try {
    try {
        Start-Transcript -Path $LogPath -Append -ErrorAction Stop | Out-Null
        $TranscriptStarted = $true
    }
    catch {
        Write-Warning ("cannot start transcript log {0}: {1}" -f $LogPath, $_.Exception.Message)
    }
    & $Python @Arguments
    $NativeExitCode = $LASTEXITCODE
    if ($NativeExitCode -ne 0) {
        throw "legacy stochastic pipeline failed with exit code $NativeExitCode"
    }
    Write-Host "[historical-seed-pipeline] COMPLETE"
    Write-Host ("[historical-seed-pipeline] manifest={0}" -f $ManifestPath)
}
finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}

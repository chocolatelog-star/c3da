param(
    [string]$Pairs = "rest16:laptop14",
    [int[]]$Seeds = @(1000, 1001, 1002, 1003, 1004),
    [string]$OutputRoot = "runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$Python = "J:\conda\envs\c3da\python.exe"
$env:PYTHONUNBUFFERED = "1"
$LogRoot = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

foreach ($Seed in $Seeds) {
    $SeedRoot = Join-Path $OutputRoot ("seed{0}" -f $Seed)
    $LogPath = Join-Path $LogRoot ("seed{0}.log" -f $Seed)
    Write-Host ("[full-pipeline-seed-sweep] START seed={0} output_root={1}" -f $Seed, $SeedRoot)
    $Arguments = @(
        "run_bgca_aste_stage1_pairs.py",
        "--output_root", $SeedRoot,
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
        "--cuda", "0",
        "--seed", ([string]$Seed)
    )
    if ($DryRun) {
        $Arguments += "--dry_run"
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

        # Keep Python attached to the console so tqdm/Trainer progress bars update live.
        & $Python @Arguments
        $NativeExitCode = $LASTEXITCODE
    }
    finally {
        if ($TranscriptStarted) {
            Stop-Transcript | Out-Null
        }
    }
    if ($NativeExitCode -ne 0) {
        throw "seed $Seed failed with exit code $NativeExitCode; log: $LogPath"
    }
    Write-Host ("[full-pipeline-seed-sweep] DONE seed={0}" -f $Seed)
}

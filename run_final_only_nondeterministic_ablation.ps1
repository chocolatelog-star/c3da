param(
    [string]$RunDir = "runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1\seed1000\rest16_to_laptop14",
    [int]$Seed = 1000,
    [string]$Cuda = "0",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Python = "J:\conda\envs\c3da\python.exe"
$env:PYTHONUNBUFFERED = "1"
$BaseTag = "strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw065"
$ResultTag = "${BaseTag}_sentiment_contrastive_l001_source_balanced_nondeterministic_final_retrain"
$TrainFile = Join-Path $RunDir "final_train_${BaseTag}.jsonl"
$DevFile = Join-Path $RunDir "final_dev_${BaseTag}.jsonl"
$OutputDir = Join-Path $RunDir "models\final_dann_l0.03_${ResultTag}_ep5"
$BestConfig = Join-Path $OutputDir "best\config.json"
$RawMetrics = Join-Path $RunDir "aste_metrics_raw_${ResultTag}.json"
$FixedMetrics = Join-Path $RunDir "aste_metrics_fixed_${ResultTag}.json"
$OutputRoot = Split-Path (Split-Path $RunDir -Parent) -Parent
$LogDir = Join-Path $OutputRoot "logs"
$ManifestPath = Join-Path $LogDir "final_only_nondeterministic_seed${Seed}_manifest.json"
$LogPath = Join-Path $LogDir "final_only_nondeterministic_seed${Seed}.log"

foreach ($RequiredPath in @($Python, $TrainFile, $DevFile, (Join-Path $RunDir "target_test.jsonl"))) {
    if (-not (Test-Path $RequiredPath)) {
        throw "missing required path: $RequiredPath"
    }
}

$Manifest = [ordered]@{
    experiment = "final_only_nondeterministic_ablation"
    run_dir = $RunDir
    seed = $Seed
    cuda = $Cuda
    deterministic = $false
    train_file = $TrainFile
    train_file_sha256 = (Get-FileHash $TrainFile -Algorithm SHA256).Hash
    dev_file = $DevFile
    dev_file_sha256 = (Get-FileHash $DevFile -Algorithm SHA256).Hash
    output_dir = $OutputDir
    result_tag = $ResultTag
    fixed_variables = [ordered]@{
        epochs = 5
        learning_rate = 0.0003
        source_weight = 1.0
        pseudo_weight = 0.65
        augment_weight = 0.20
        lambda_domain_adv = 0.03
        lambda_sentiment_contrastive = 0.01
        sentiment_contrastive_source_only = $true
        sentiment_contrastive_class_balanced = $true
        train_batch_size = 1
        eval_batch_size = 2
        gradient_accumulation_steps = 16
        fp16 = $true
        gradient_checkpointing = $true
    }
}
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $Manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $ManifestPath
}

function Invoke-PythonStage {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Host ("[final-only-ablation] START {0}" -f $Name)
    Write-Host ("{0} {1}" -f $Python, ($Arguments -join " "))
    if ($DryRun) {
        return
    }
    & $Python @Arguments
    $NativeExitCode = $LASTEXITCODE
    if ($NativeExitCode -ne 0) {
        throw "$Name failed with exit code $NativeExitCode"
    }
    Write-Host ("[final-only-ablation] DONE {0}" -f $Name)
}

$TranscriptStarted = $false
try {
    if (-not $DryRun) {
        try {
            Start-Transcript -Path $LogPath -Append -ErrorAction Stop | Out-Null
            $TranscriptStarted = $true
        }
        catch {
            Write-Warning ("cannot start transcript log {0}: {1}" -f $LogPath, $_.Exception.Message)
        }
    }

    if (Test-Path $BestConfig) {
        Write-Host "[final-only-ablation] SKIP train: best/config.json already exists"
    }
    else {
        $TrainArguments = @(
            "t5_absa_train.py",
            "--model_path", "J:\nlp\models\t5-base-py",
            "--train_file", $TrainFile,
            "--dev_file", $DevFile,
            "--output_dir", $OutputDir,
            "--num_train_epochs", "5",
            "--source_weight", "1.0",
            "--pseudo_weight", "0.65",
            "--augment_weight", "0.20",
            "--checkpoint_selection", "best",
            "--resume_from_checkpoint", "auto",
            "--lambda_domain_adv", "0.03",
            "--domain_adv_grl_lambda", "1.0",
            "--domain_adv_hidden_size", "256",
            "--domain_adv_exclude_augment",
            "--lambda_sentiment_contrastive", "0.01",
            "--lambda_pairing_loss", "0.0",
            "--pairing_temperature", "0.1",
            "--sentiment_contrastive_temperature", "0.1",
            "--sentiment_contrastive_min_weight", "0.65",
            "--neutral_generation_loss_gain", "0.0",
            "--neutral_generation_max_effective_weight", "0.0",
            "--sentiment_contrastive_source_only",
            "--sentiment_contrastive_class_balanced",
            "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "2",
            "--gradient_accumulation_steps", "16",
            "--learning_rate", "0.0003",
            "--fp16",
            "--gradient_checkpointing",
            "--cuda", $Cuda,
            "--seed", ([string]$Seed)
        )
        Invoke-PythonStage -Name "train" -Arguments $TrainArguments
    }

    if ((Test-Path $RawMetrics) -and (Test-Path $FixedMetrics)) {
        Write-Host "[final-only-ablation] SKIP evaluate: raw/fixed metrics already exist"
    }
    else {
        $EvaluateArguments = @(
            "t5_aste_pipeline.py",
            "evaluate",
            "--run_dir", $RunDir,
            "--model_path", (Join-Path $OutputDir "best"),
            "--batch_size", "2",
            "--num_beams", "4",
            "--max_new_tokens", "96",
            "--cuda", $Cuda,
            "--no_task_prefix",
            "--no_constrained_decoding",
            "--output_tag", $ResultTag
        )
        Invoke-PythonStage -Name "evaluate" -Arguments $EvaluateArguments
    }

    if (-not $DryRun) {
        $Raw = Get-Content -Raw -Encoding UTF8 $RawMetrics | ConvertFrom-Json
        $Fixed = Get-Content -Raw -Encoding UTF8 $FixedMetrics | ConvertFrom-Json
        Write-Host ("[final-only-ablation] RESULT raw_f1={0:N4} fixed_f1={1:N4}" -f $Raw.micro_f1, $Fixed.micro_f1)
        Write-Host ("[final-only-ablation] MANIFEST {0}" -f $ManifestPath)
    }
}
finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}

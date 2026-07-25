param(
    [string]$RunDir = "runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1\seed1000\rest16_to_laptop14",
    [string]$HistoricalFinalTrain = "runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14\final_train_strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw065.jsonl",
    [int]$Seed = 1000,
    [string]$Cuda = "0",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Python = "J:\conda\envs\c3da\python.exe"
$env:PYTHONUNBUFFERED = "1"
$PseudoFile = Join-Path $RunDir "pseudo_variants\hp1_complete2_dist5_w025\target_pseudo_high_precision.jsonl"
$HybridTag = "strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw065_hist_actual_aug150"
$ResultTag = "${HybridTag}_sentiment_contrastive_l001_source_balanced_nondeterministic_final_retrain"
$ExtractedAugment = Join-Path $RunDir "historical_actual_selected_augment150_w020.jsonl"
$ExtractedAnalysis = Join-Path $RunDir "historical_actual_selected_augment150_w020_analysis.json"
$TrainFile = Join-Path $RunDir "final_train_${HybridTag}.jsonl"
$DevFile = Join-Path $RunDir "final_dev_${HybridTag}.jsonl"
$CompositionFile = Join-Path $RunDir "final_train_composition_analysis_${HybridTag}.json"
$OutputDir = Join-Path $RunDir "models\final_dann_l0.03_${ResultTag}_ep5"
$BestConfig = Join-Path $OutputDir "best\config.json"
$RawMetrics = Join-Path $RunDir "aste_metrics_raw_${ResultTag}.json"
$FixedMetrics = Join-Path $RunDir "aste_metrics_fixed_${ResultTag}.json"
$OutputRoot = Split-Path (Split-Path $RunDir -Parent) -Parent
$LogDir = Join-Path $OutputRoot "logs"
$LogPath = Join-Path $LogDir "historical_augment_hybrid_seed${Seed}.log"
$ManifestPath = Join-Path $LogDir "historical_augment_hybrid_seed${Seed}_manifest.json"

foreach ($RequiredPath in @($Python, $HistoricalFinalTrain, $PseudoFile, (Join-Path $RunDir "source_train.jsonl"), (Join-Path $RunDir "source_dev.jsonl"), (Join-Path $RunDir "target_test.jsonl"))) {
    if (-not (Test-Path $RequiredPath)) {
        throw "missing required path: $RequiredPath"
    }
}

function Invoke-PythonStage {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Host ("[historical-augment-hybrid] START {0}" -f $Name)
    Write-Host ("{0} {1}" -f $Python, ($Arguments -join " "))
    if ($DryRun) {
        return
    }
    & $Python @Arguments
    $NativeExitCode = $LASTEXITCODE
    if ($NativeExitCode -ne 0) {
        throw "$Name failed with exit code $NativeExitCode"
    }
    Write-Host ("[historical-augment-hybrid] DONE {0}" -f $Name)
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
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

    $ExtractedAugmentExists = Test-Path $ExtractedAugment
    $ExtractedAnalysisExists = Test-Path $ExtractedAnalysis
    if ($ExtractedAugmentExists -xor $ExtractedAnalysisExists) {
        throw "partial extract artifacts found; refusing to overwrite existing files"
    }
    if ($ExtractedAugmentExists -and $ExtractedAnalysisExists) {
        Write-Host "[historical-augment-hybrid] SKIP extract: extracted augment and analysis already exist"
    }
    else {
        Invoke-PythonStage -Name "extract historical selected augment" -Arguments @(
            "t5_aste_pipeline.py",
            "extract_selected_augment_from_final_train",
            "--source_final_train_file", $HistoricalFinalTrain,
            "--output_file", $ExtractedAugment,
            "--analysis_file", $ExtractedAnalysis,
            "--expected_rows", "150",
            "--selected_weight", "0.20"
        )
    }

    $TrainFileExists = Test-Path $TrainFile
    $DevFileExists = Test-Path $DevFile
    $CompositionFileExists = Test-Path $CompositionFile
    $BuildArtifactCount = @($TrainFileExists, $DevFileExists, $CompositionFileExists).Where({ $_ }).Count
    if (($BuildArtifactCount -gt 0) -and ($BuildArtifactCount -lt 3)) {
        throw "partial hybrid build artifacts found; refusing to overwrite existing files"
    }
    if ($BuildArtifactCount -eq 3) {
        Write-Host "[historical-augment-hybrid] SKIP build: hybrid final train artifacts already exist"
    }
    else {
        Invoke-PythonStage -Name "build hybrid final train" -Arguments @(
            "t5_aste_pipeline.py",
            "build_final_train_from_files",
            "--run_dir", $RunDir,
            "--pseudo_train_file", $PseudoFile,
            "--selected_augment_file", $ExtractedAugment,
            "--selected_augment_weight", "0.20",
            "--final_train_output_tag", $HybridTag,
            "--final_multi_triplet_gain", "0.10",
            "--final_neutral_gain", "0.15",
            "--final_max_weight", "1.0",
            "--no_task_prefix"
        )
    }

    if (Test-Path $BestConfig) {
        Write-Host "[historical-augment-hybrid] SKIP train: best/config.json already exists"
    }
    else {
        Invoke-PythonStage -Name "train final model" -Arguments @(
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
    }

    $RawMetricsExists = Test-Path $RawMetrics
    $FixedMetricsExists = Test-Path $FixedMetrics
    if ($RawMetricsExists -xor $FixedMetricsExists) {
        throw "partial evaluation metrics found; refusing to overwrite existing files"
    }
    if ($RawMetricsExists -and $FixedMetricsExists) {
        Write-Host "[historical-augment-hybrid] SKIP evaluate: raw/fixed metrics already exist"
    }
    else {
        Invoke-PythonStage -Name "evaluate target test" -Arguments @(
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
    }

    if (-not $DryRun) {
        $Manifest = [ordered]@{
            experiment = "historical_augment_hybrid_ablation"
            run_dir = $RunDir
            seed = $Seed
            cuda = $Cuda
            deterministic = $false
            historical_final_train = $HistoricalFinalTrain
            historical_final_train_sha256 = (Get-FileHash $HistoricalFinalTrain -Algorithm SHA256).Hash
            current_complete_pseudo = $PseudoFile
            current_complete_pseudo_sha256 = (Get-FileHash $PseudoFile -Algorithm SHA256).Hash
            extracted_augment = $ExtractedAugment
            extracted_augment_sha256 = (Get-FileHash $ExtractedAugment -Algorithm SHA256).Hash
            hybrid_train_file = $TrainFile
            hybrid_train_file_sha256 = (Get-FileHash $TrainFile -Algorithm SHA256).Hash
            output_dir = $OutputDir
            result_tag = $ResultTag
        }
        if (-not (Test-Path $ManifestPath)) {
            $Manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $ManifestPath
        }
        $Raw = Get-Content -Raw -Encoding UTF8 $RawMetrics | ConvertFrom-Json
        $Fixed = Get-Content -Raw -Encoding UTF8 $FixedMetrics | ConvertFrom-Json
        Write-Host ("[historical-augment-hybrid] RESULT raw_f1={0:N4} fixed_f1={1:N4}" -f $Raw.micro_f1, $Fixed.micro_f1)
        Write-Host ("[historical-augment-hybrid] MANIFEST {0}" -f $ManifestPath)
    }
}
finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}

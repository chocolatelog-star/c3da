param(
    [string]$SourceDataset = "rest16",
    [string]$TargetDataset = "laptop14",
    [int]$Seed = 1000,
    [string]$Cuda = "0",
    [string]$OutputRoot = "J:\nlp\CD-C3DA\runs\historical_best_two_stage_v1",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Python = "J:\conda\envs\c3da\python.exe"
$BaseModel = "J:\nlp\models\t5-base-py"
$NliModel = "J:\nlp\models\nli-deberta-v3-base-mnli-fever-anli"
$UpstreamTree = "J:\nlp\CD-C3DA\.worktrees\historical-best-upstream-9e78904"
$DownstreamTree = "J:\nlp\CD-C3DA\.worktrees\reproduce-best-8c7f6b4"
$UpstreamBaseCommit = "9e789045b41df7af0dd73ccebc90f06a91d94f8e"
$UpstreamCommit = "a7e7778869dce92fe778837715a814b5c6d2014b"
$DownstreamCommit = "8c7f6b47b1b2b4ef9c11d7dffdf64758db7aace3"
$PairName = "${SourceDataset}_to_${TargetDataset}"
$RunRoot = Join-Path $OutputRoot $PairName
$UpstreamRun = Join-Path $RunRoot "upstream_9e78904"
$DownstreamRun = Join-Path $RunRoot "downstream_8c7f6b4"
$LogDir = Join-Path $RunRoot "logs"
$StatusPath = Join-Path $RunRoot "stage_status.json"
$ManifestPath = Join-Path $RunRoot "manifest.json"
$LogPath = Join-Path $LogDir ("historical_best_two_stage_seed{0}_{1}.log" -f $Seed, (Get-Date -Format "yyyyMMdd_HHmmss"))

$UpstreamExtractor = Join-Path $UpstreamRun "models\extractor_ep25_plain_last\best"
$UpstreamGenerator = Join-Path $UpstreamRun "models\generator_label_to_text_gen_ep8\best"
$UpstreamRawPseudo = Join-Path $UpstreamRun "target_pseudo.jsonl"
$UpstreamBasePseudo = Join-Path $UpstreamRun "target_pseudo_high_precision.jsonl"
$UpstreamAugment = Join-Path $UpstreamRun "c3da_two_channel_augmented_selected_strict_aug150_w020_label_to_text_gen.jsonl"
$CompletePseudoDir = Join-Path $DownstreamRun "pseudo_variants\hp1_complete2_dist5_w025"
$CompletePseudo = Join-Path $CompletePseudoDir "target_pseudo_high_precision.jsonl"
$FinalTag = "strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw065"
$ResultTag = "${FinalTag}_sentiment_contrastive_l001_source_balanced"
$FinalTrain = Join-Path $DownstreamRun "final_train_${FinalTag}.jsonl"
$FinalDev = Join-Path $DownstreamRun "final_dev_${FinalTag}.jsonl"
$FinalModel = Join-Path $DownstreamRun "models\final_dann_l0.03_${ResultTag}_ep5"
$RawMetrics = Join-Path $DownstreamRun "aste_metrics_raw_${ResultTag}.json"
$FixedMetrics = Join-Path $DownstreamRun "aste_metrics_fixed_${ResultTag}.json"

function Assert-PathExists {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing ${Description}: $Path"
    }
}

function Get-WorktreeCommit {
    param([string]$Worktree)
    $commit = (& git -C $Worktree rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot read Git commit from $Worktree"
    }
    return $commit
}

function Write-Status {
    $payload = [ordered]@{
        completed = @($script:CompletedStages | Sort-Object)
        updated_at = (Get-Date).ToString("o")
    }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Get-ArtifactRecord {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    $item = Get-Item -LiteralPath $Path
    $record = [ordered]@{
        path = $item.FullName
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
        last_write_time = $item.LastWriteTime.ToString("o")
    }
    if ($item.Extension -eq ".jsonl") {
        $record.rows = (Get-Content -LiteralPath $Path -Encoding UTF8 | Measure-Object -Line).Lines
    }
    return $record
}

function Write-Manifest {
    $artifactPaths = @(
        (Join-Path $UpstreamExtractor "config.json"),
        (Join-Path $UpstreamGenerator "config.json"),
        $UpstreamRawPseudo,
        $UpstreamBasePseudo,
        $UpstreamAugment,
        $CompletePseudo,
        $FinalTrain,
        $FinalDev,
        (Join-Path $FinalModel "best\config.json"),
        $RawMetrics,
        $FixedMetrics
    )
    $artifacts = @()
    foreach ($path in $artifactPaths) {
        $record = Get-ArtifactRecord $path
        if ($null -ne $record) {
            $artifacts += $record
        }
    }
    $metrics = [ordered]@{}
    if (Test-Path -LiteralPath $RawMetrics) {
        $metrics.raw = Get-Content -LiteralPath $RawMetrics -Encoding UTF8 -Raw | ConvertFrom-Json
    }
    if (Test-Path -LiteralPath $FixedMetrics) {
        $metrics.fixed = Get-Content -LiteralPath $FixedMetrics -Encoding UTF8 -Raw | ConvertFrom-Json
    }
    $manifest = [ordered]@{
        objective = "Reproduce the historical best pipeline with pinned upstream and downstream code"
        source_dataset = $SourceDataset
        target_dataset = $TargetDataset
        seed = $Seed
        cuda = $Cuda
        reproducibility_mode = "historical seed-only"
        upstream = [ordered]@{
            historical_base_commit = $UpstreamBaseCommit
            resume_compat_commit = $UpstreamCommit
            worktree = $UpstreamTree
            run_dir = $UpstreamRun
        }
        downstream = [ordered]@{ commit = $DownstreamCommit; worktree = $DownstreamTree; run_dir = $DownstreamRun }
        completed_stages = @($script:CompletedStages | Sort-Object)
        artifacts = $artifacts
        metrics = $metrics
        updated_at = (Get-Date).ToString("o")
    }
    $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
}

function Format-Command {
    param([object[]]$Arguments)
    $parts = @($Python)
    foreach ($argument in $Arguments) {
        $text = [string]$argument
        if ($text -match '[\s"]') {
            $parts += ('"{0}"' -f ($text -replace '"', '\"'))
        } else {
            $parts += $text
        }
    }
    return ($parts -join " ")
}

function Invoke-Stage {
    param(
        [string]$Name,
        [string]$WorkingDirectory,
        [object[]]$Arguments,
        [string[]]$ExpectedPaths,
        [scriptblock]$BeforeAction
    )
    $outputsReady = $true
    foreach ($path in $ExpectedPaths) {
        if (-not (Test-Path -LiteralPath $path)) {
            $outputsReady = $false
            break
        }
    }
    if ($script:CompletedStages.Contains($Name) -and $outputsReady) {
        Write-Host "[historical-best-two-stage] SKIP $Name"
        return
    }

    Write-Host "[historical-best-two-stage] START $Name"
    Write-Host (Format-Command $Arguments)
    if ($DryRun) {
        return
    }
    if ($null -ne $BeforeAction) {
        & $BeforeAction
    }
    Push-Location $WorkingDirectory
    try {
        & $Python @Arguments
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "Stage $Name failed with exit code $exitCode"
    }
    foreach ($path in $ExpectedPaths) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Stage $Name completed but expected output is missing: $path"
        }
    }
    [void]$script:CompletedStages.Add($Name)
    Write-Status
    Write-Manifest
    Write-Host "[historical-best-two-stage] DONE $Name"
}

foreach ($required in @($Python, $BaseModel, $NliModel, $UpstreamTree, $DownstreamTree)) {
    Assert-PathExists $required "required path"
}
if ((Get-WorktreeCommit $UpstreamTree) -ne $UpstreamCommit) {
    throw "Upstream worktree is not pinned to $UpstreamCommit"
}
if ((Get-WorktreeCommit $DownstreamTree) -ne $DownstreamCommit) {
    throw "Downstream worktree is not pinned to $DownstreamCommit"
}

New-Item -ItemType Directory -Path $UpstreamRun, $DownstreamRun, $LogDir -Force | Out-Null
$script:CompletedStages = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
if (Test-Path -LiteralPath $StatusPath) {
    $savedStatus = Get-Content -LiteralPath $StatusPath -Encoding UTF8 -Raw | ConvertFrom-Json
    foreach ($stage in @($savedStatus.completed)) {
        [void]$script:CompletedStages.Add([string]$stage)
    }
}

Start-Transcript -Path $LogPath -Append | Out-Null
try {
    Write-Host "[historical-best-two-stage] pair=$PairName seed=$Seed cuda=$Cuda"
    Write-Host "[historical-best-two-stage] upstream_base=$UpstreamBaseCommit resume_compat=$UpstreamCommit downstream=$DownstreamCommit"

    Invoke-Stage "upstream_prepare" $UpstreamTree @(
        "t5_aste_pipeline.py", "prepare",
        "--source_dataset", $SourceDataset,
        "--target_dataset", $TargetDataset,
        "--run_dir", $UpstreamRun,
        "--seed", "$Seed",
        "--augment_prompt_style", "label_to_text",
        "--augment_channel_mode", "all",
        "--domain_prefix_style", "text",
        "--generator_output_tag", "label_to_text_gen",
        "--no_task_prefix"
    ) @(
        (Join-Path $UpstreamRun "extract_train.jsonl"),
        (Join-Path $UpstreamRun "c3da_generator_train_label_to_text_gen.jsonl")
    )

    Invoke-Stage "upstream_extractor" $UpstreamTree @(
        "t5_absa_train.py",
        "--model_path", $BaseModel,
        "--train_file", (Join-Path $UpstreamRun "extract_train.jsonl"),
        "--dev_file", (Join-Path $UpstreamRun "extract_dev.jsonl"),
        "--output_dir", (Join-Path $UpstreamRun "models\extractor_ep25_plain_last"),
        "--num_train_epochs", "25",
        "--source_weight", "1.0",
        "--pseudo_weight", "0.5",
        "--augment_weight", "0.2",
        "--lambda_structure_loss", "0",
        "--lambda_consistency_loss", "0",
        "--lambda_pairing_loss", "0",
        "--multi_triplet_loss_gain", "0",
        "--neutral_loss_gain", "0",
        "--checkpoint_selection", "last",
        "--resume_from_checkpoint", "auto",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "2",
        "--gradient_accumulation_steps", "16",
        "--learning_rate", "0.0003",
        "--fp16", "--gradient_checkpointing",
        "--cuda", $Cuda,
        "--seed", "$Seed"
    ) @((Join-Path $UpstreamExtractor "config.json"))

    Invoke-Stage "upstream_pseudo" $UpstreamTree @(
        "t5_aste_pipeline.py", "pseudo",
        "--run_dir", $UpstreamRun,
        "--model_path", $UpstreamExtractor,
        "--batch_size", "2",
        "--num_beams", "1",
        "--max_new_tokens", "128",
        "--no_constrained_decoding",
        "--cuda", $Cuda,
        "--no_task_prefix",
        "--pseudo_model_variant", "last",
        "--high_precision_max_triplets", "1",
        "--high_precision_max_token_distance", "5"
    ) @($UpstreamRawPseudo, $UpstreamBasePseudo)

    Invoke-Stage "upstream_generator" $UpstreamTree @(
        "t5_absa_train.py",
        "--model_path", $BaseModel,
        "--train_file", (Join-Path $UpstreamRun "c3da_generator_train_label_to_text_gen.jsonl"),
        "--dev_file", (Join-Path $UpstreamRun "c3da_generator_dev_label_to_text_gen.jsonl"),
        "--output_dir", (Join-Path $UpstreamRun "models\generator_label_to_text_gen_ep8"),
        "--num_train_epochs", "8",
        "--source_weight", "1.0",
        "--pseudo_weight", "1.0",
        "--augment_weight", "1.0",
        "--checkpoint_selection", "best",
        "--resume_from_checkpoint", "auto",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "2",
        "--gradient_accumulation_steps", "16",
        "--learning_rate", "0.0003",
        "--fp16", "--gradient_checkpointing",
        "--cuda", $Cuda,
        "--seed", "$Seed"
    ) @((Join-Path $UpstreamGenerator "config.json"))

    Invoke-Stage "upstream_augment" $UpstreamTree @(
        "t5_aste_pipeline.py", "augment",
        "--run_dir", $UpstreamRun,
        "--model_path", $UpstreamGenerator,
        "--nli_model_path", $NliModel,
        "--augment_prompt_style", "masked_mutual",
        "--augment_channel_mode", "all",
        "--domain_prefix_style", "text",
        "--augment_output_tag", "strict_aug150_w020_label_to_text_gen",
        "--final_train_output_tag", "strict_aug150_w020_label_to_text_gen",
        "--augment_select_max_rows", "150",
        "--augment_select_max_per_base", "1",
        "--augment_select_weight", "0.2",
        "--augment_select_require_raw_exact",
        "--augment_select_require_model_filter_passed",
        "--pseudo_train_source", "high_precision",
        "--high_precision_max_triplets", "1",
        "--high_precision_max_token_distance", "5",
        "--model_filter_path", $UpstreamExtractor,
        "--model_filter_mode", "fixed",
        "--model_filter_batch_size", "2",
        "--model_filter_num_beams", "1",
        "--model_filter_no_constrained_decoding",
        "--model_filter_channel_aware",
        "--cuda", $Cuda,
        "--no_task_prefix"
    ) @($UpstreamAugment)

    Invoke-Stage "downstream_prepare" $DownstreamTree @(
        "t5_aste_pipeline.py", "prepare",
        "--source_dataset", $SourceDataset,
        "--target_dataset", $TargetDataset,
        "--run_dir", $DownstreamRun,
        "--seed", "$Seed",
        "--augment_prompt_style", "label_to_text",
        "--augment_channel_mode", "all",
        "--domain_prefix_style", "text",
        "--generator_output_tag", "label_to_text_gen",
        "--no_task_prefix"
    ) @((Join-Path $DownstreamRun "source_train.jsonl"), (Join-Path $DownstreamRun "target_test.jsonl"))

    $syncPseudo = {
        Copy-Item -LiteralPath $UpstreamRawPseudo -Destination (Join-Path $DownstreamRun "target_pseudo.jsonl") -Force
    }
    Invoke-Stage "downstream_complete_multi2" $DownstreamTree @(
        "t5_aste_pipeline.py", "select_complete_multi_pseudo",
        "--run_dir", $DownstreamRun,
        "--output_dir", $CompletePseudoDir,
        "--base_pseudo_file", $UpstreamBasePseudo,
        "--min_pseudo_weight", "0.65",
        "--high_precision_max_token_distance", "5",
        "--complete_multi_extra_weight", "0.25"
    ) @($CompletePseudo, (Join-Path $CompletePseudoDir "target_pseudo_high_precision_analysis.json")) $syncPseudo

    Invoke-Stage "downstream_build_final_train" $DownstreamTree @(
        "t5_aste_pipeline.py", "build_final_train_from_files",
        "--run_dir", $DownstreamRun,
        "--pseudo_train_file", $CompletePseudo,
        "--selected_augment_file", $UpstreamAugment,
        "--selected_augment_weight", "0.2",
        "--final_train_output_tag", $FinalTag,
        "--no_task_prefix"
    ) @($FinalTrain, $FinalDev)

    Invoke-Stage "downstream_final_train" $DownstreamTree @(
        "t5_absa_train.py",
        "--model_path", $BaseModel,
        "--train_file", $FinalTrain,
        "--dev_file", $FinalDev,
        "--output_dir", $FinalModel,
        "--num_train_epochs", "5",
        "--source_weight", "1.0",
        "--pseudo_weight", "0.65",
        "--augment_weight", "0.2",
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
        "--fp16", "--gradient_checkpointing",
        "--cuda", $Cuda,
        "--seed", "$Seed"
    ) @((Join-Path $FinalModel "best\config.json"))

    Invoke-Stage "downstream_evaluate" $DownstreamTree @(
        "t5_aste_pipeline.py", "evaluate",
        "--run_dir", $DownstreamRun,
        "--model_path", (Join-Path $FinalModel "best"),
        "--batch_size", "2",
        "--num_beams", "4",
        "--max_new_tokens", "96",
        "--cuda", $Cuda,
        "--no_task_prefix",
        "--no_constrained_decoding",
        "--output_tag", $ResultTag
    ) @($RawMetrics, $FixedMetrics)

    if (-not $DryRun) {
        Write-Manifest
        Write-Host "[historical-best-two-stage] COMPLETE"
        Write-Host "[historical-best-two-stage] manifest=$ManifestPath"
        Write-Host "[historical-best-two-stage] raw_metrics=$RawMetrics"
        Write-Host "[historical-best-two-stage] fixed_metrics=$FixedMetrics"
    }
} finally {
    Stop-Transcript | Out-Null
}

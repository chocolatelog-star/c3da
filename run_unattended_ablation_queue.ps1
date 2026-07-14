param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "J:\conda\envs\c3da\python.exe"
$LogDir = Join-Path $ProjectRoot "runs\unattended_queue_logs"
$LogPath = Join-Path $LogDir "ablation_queue.log"
$StatusPath = Join-Path $LogDir "ablation_queue_status.json"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $ProjectRoot

function Test-AllPaths {
    param([string[]]$Paths)
    foreach ($path in $Paths) {
        if (-not (Test-Path -LiteralPath $path)) {
            return $false
        }
    }
    return $true
}

function Write-QueueStatus {
    param([string]$Stage, [string]$State, [string]$Message = "")
    @{
        stage = $Stage
        state = $State
        message = $Message
        updated_at = (Get-Date).ToString("s")
        process_id = $PID
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string[]]$ExpectedPaths
    )
    if (Test-AllPaths $ExpectedPaths) {
        "[$(Get-Date -Format s)] SKIP $Name" | Tee-Object -FilePath $LogPath -Append
        return
    }
    if ($DryRun) {
        "DRYRUN $Python $($Arguments -join ' ')" | Tee-Object -FilePath $LogPath -Append
        return
    }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-QueueStatus $Name "running" "attempt $attempt"
        "[$(Get-Date -Format s)] START $Name attempt=$attempt" | Tee-Object -FilePath $LogPath -Append
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorAction
        if ($exitCode -eq 0 -and (Test-AllPaths $ExpectedPaths)) {
            "[$(Get-Date -Format s)] DONE $Name" | Tee-Object -FilePath $LogPath -Append
            Write-QueueStatus $Name "completed"
            return
        }
        "[$(Get-Date -Format s)] FAIL $Name exit=$exitCode" | Tee-Object -FilePath $LogPath -Append
        Write-QueueStatus $Name "retry_wait" "exit $exitCode"
        Start-Sleep -Seconds 120
    }
    Write-QueueStatus $Name "failed" "three attempts exhausted"
    throw "$Name failed after three attempts"
}

function Invoke-Evaluation {
    param(
        [string]$Name,
        [string]$RunDir,
        [string]$ModelPath,
        [string]$Tag
    )
    $taggedRaw = Join-Path $RunDir "aste_metrics_raw_$Tag.json"
    $taggedFixed = Join-Path $RunDir "aste_metrics_fixed_$Tag.json"
    if (Test-AllPaths @($taggedRaw, $taggedFixed)) {
        "[$(Get-Date -Format s)] SKIP $Name" | Tee-Object -FilePath $LogPath -Append
        return
    }
    if ($DryRun) {
        "DRYRUN evaluate $ModelPath and save tag=$Tag" | Tee-Object -FilePath $LogPath -Append
        return
    }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-QueueStatus $Name "running" "attempt $attempt"
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python "t5_aste_pipeline.py" "evaluate" "--run_dir" $RunDir "--model_path" $ModelPath @CommonEval 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorAction
        $raw = Join-Path $RunDir "aste_metrics_raw.json"
        $fixed = Join-Path $RunDir "aste_metrics_fixed.json"
        if ($exitCode -eq 0 -and (Test-AllPaths @($raw, $fixed))) {
            Copy-Item -LiteralPath $raw -Destination $taggedRaw -Force
            Copy-Item -LiteralPath $fixed -Destination $taggedFixed -Force
            Write-QueueStatus $Name "completed"
            return
        }
        Start-Sleep -Seconds 120
    }
    Write-QueueStatus $Name "failed" "three attempts exhausted"
    throw "$Name failed after three attempts"
}

$CommonTrain = @(
    "--num_train_epochs", "5",
    "--source_weight", "1.0",
    "--pseudo_weight", "0.5",
    "--augment_weight", "0.2",
    "--checkpoint_selection", "best",
    "--resume_from_checkpoint", "auto",
    "--lambda_domain_adv", "0.03",
    "--domain_adv_grl_lambda", "1.0",
    "--domain_adv_hidden_size", "256",
    "--domain_adv_exclude_augment",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "2",
    "--gradient_accumulation_steps", "16",
    "--learning_rate", "3e-4",
    "--fp16",
    "--gradient_checkpointing",
    "--cuda", "0",
    "--seed", "1000"
)
$CommonEval = @("--batch_size", "2", "--num_beams", "4", "--max_new_tokens", "96", "--cuda", "0", "--no_task_prefix", "--no_constrained_decoding")

# Ablation A: new upstream data, no final contrastive loss.
$RunA = "runs\bgca_aste_stage1_full_contrastive_encoder_v1\rest16_to_laptop14"
$ModelA = Join-Path $RunA "models\final_ablation_upstream_only_nocontrast_ep5"
$TagA = "ablation_upstream_only_nocontrast"
Invoke-PythonStep "ablation_a_train" (@(
    "t5_absa_train.py", "--model_path", "J:\nlp\models\t5-base-py",
    "--train_file", (Join-Path $RunA "final_train_strict_aug150_w020_label_to_text_gen.jsonl"),
    "--dev_file", (Join-Path $RunA "final_dev_strict_aug150_w020_label_to_text_gen.jsonl"),
    "--output_dir", $ModelA
) + $CommonTrain) @((Join-Path $ModelA "best\config.json"))
Invoke-Evaluation "ablation_a_evaluate" $RunA (Join-Path $ModelA "best") $TagA

# Ablation B: old baseline data, encoder contextual contrastive loss.
$RunB = "runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14"
$ModelB = Join-Path $RunB "models\final_ablation_encoder_context_contrast_l001_ep5"
$TagB = "ablation_encoder_context_contrast_l001"
Invoke-PythonStep "ablation_b_train" (@(
    "t5_absa_train.py", "--model_path", "J:\nlp\models\t5-base-py",
    "--train_file", (Join-Path $RunB "final_train_strict_aug150_w020_label_to_text_gen.jsonl"),
    "--dev_file", (Join-Path $RunB "final_dev_strict_aug150_w020_label_to_text_gen.jsonl"),
    "--output_dir", $ModelB,
    "--lambda_sentiment_contrastive", "0.01",
    "--sentiment_contrastive_temperature", "0.10",
    "--sentiment_contrastive_min_weight", "0.65",
    "--sentiment_contrastive_exclude_augment",
    "--sentiment_contrastive_source_only",
    "--sentiment_contrastive_class_balanced",
    "--sentiment_prototype_initialize_from_context",
    "--sentiment_prototype_init_batch_size", "2"
) + $CommonTrain) @((Join-Path $ModelB "best\config.json"))
Invoke-Evaluation "ablation_b_evaluate" $RunB (Join-Path $ModelB "best") $TagB

# Full pipeline with T5 sentiment-vector augmentation.
$RootC = "runs\bgca_aste_stage1_full_contrastive_encoder_t5_sentvec_v1"
$RunC = Join-Path $RootC "rest16_to_laptop14"
$ResultTagC = "strict_aug150_w020_label_to_text_gen_sentiment_vector_sentiment_contrastive_l001_source_balanced_encoder_context_init"
Invoke-PythonStep "full_t5_sentiment_vector_pipeline" @(
    "run_bgca_aste_stage1_pairs.py",
    "--output_root", $RootC,
    "--pairs", "rest16:laptop14",
    "--generator_prompt_style", "label_to_text",
    "--augment_prompt_style", "masked_mutual",
    "--domain_prefix_style", "text",
    "--opinion_replacement_mode", "sentiment_vector",
    "--sentiment_vector_backend", "t5",
    "--sentiment_vector_model_path", "J:\nlp\models\t5-base-py",
    "--sentiment_vector_min_margin", "0.05",
    "--extractor_lambda_sentiment_contrastive", "0.01",
    "--lambda_sentiment_contrastive", "0.01",
    "--sentiment_contrastive_temperature", "0.10",
    "--sentiment_contrastive_min_weight", "0.65",
    "--sentiment_contrastive_exclude_augment",
    "--sentiment_contrastive_source_only",
    "--sentiment_contrastive_class_balanced",
    "--sentiment_prototype_initialize_from_context",
    "--sentiment_prototype_init_batch_size", "2",
    "--cuda", "0"
) @(
    (Join-Path $RunC "aste_metrics_raw_$ResultTagC.json"),
    (Join-Path $RunC "aste_metrics_fixed_$ResultTagC.json")
)

if ($DryRun) {
    "[$(Get-Date -Format s)] DRYRUN VALIDATION COMPLETED" | Tee-Object -FilePath $LogPath -Append
} else {
    Write-QueueStatus "all" "completed"
    "[$(Get-Date -Format s)] ALL EXPERIMENTS COMPLETED" | Tee-Object -FilePath $LogPath -Append
}

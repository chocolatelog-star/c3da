param(
    [switch]$DryRun,
    [string]$RunDir = "J:\nlp\CD-C3DA\runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "J:\conda\envs\c3da\python.exe"
$LogDir = Join-Path $RunDir "strict_module_ablation_logs"
$LogPath = Join-Path $LogDir "strict_module_ablation_queue.log"
$StatusPath = Join-Path $LogDir "strict_module_ablation_queue_status.json"
$ModelBase = "J:\nlp\models\t5-base-py"

function Format-Command {
    param([string[]]$Arguments)
    return "$Python $($Arguments -join ' ')"
}

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
    if ($DryRun) { return }
    @{
        stage = $Stage
        state = $State
        message = $Message
        updated_at = (Get-Date).ToString("s")
        process_id = $PID
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-Step {
    param(
        [string]$Kind,
        [string]$Name,
        [string[]]$Arguments,
        [string[]]$ExpectedPaths
    )
    if ($DryRun) {
        Write-Output "DRYRUN $Kind $Name $(Format-Command $Arguments)"
        return
    }
    if (Test-AllPaths $ExpectedPaths) {
        "[$(Get-Date -Format s)] SKIP $Name" | Tee-Object -FilePath $LogPath -Append
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
        if ($attempt -lt 3) { Start-Sleep -Seconds 120 }
    }
    Write-QueueStatus $Name "failed" "three attempts exhausted"
    throw "$Name failed after three attempts"
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    Set-Location $ProjectRoot
    & $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw "CUDA preflight failed" }
}

$NoAugTag = "strict_ablation_source_pseudo"
$WithAugTag = "strict_ablation_source_pseudo_aug"
$NoAugTrain = Join-Path $RunDir "final_train_$NoAugTag.jsonl"
$NoAugDev = Join-Path $RunDir "final_dev_$NoAugTag.jsonl"
$WithAugTrain = Join-Path $RunDir "final_train_$WithAugTag.jsonl"
$WithAugDev = Join-Path $RunDir "final_dev_$WithAugTag.jsonl"

$BuildExpected = @(
    $NoAugTrain,
    $NoAugDev,
    $WithAugTrain,
    $WithAugDev,
    (Join-Path $RunDir "final_train_composition_analysis_$NoAugTag.json"),
    (Join-Path $RunDir "final_train_composition_analysis_$WithAugTag.json")
)
Invoke-Step "BUILD" "strict_ablation_files" @(
    "build_strict_ablation_files.py", "--run_dir", $RunDir
) $BuildExpected

$CommonTrain = @(
    "--model_path", $ModelBase,
    "--num_train_epochs", "5",
    "--source_weight", "1.0",
    "--pseudo_weight", "0.5",
    "--augment_weight", "0.2",
    "--checkpoint_selection", "best",
    "--resume_from_checkpoint", "auto",
    "--domain_adv_grl_lambda", "1.0",
    "--domain_adv_hidden_size", "256",
    "--lambda_sentiment_contrastive", "0.0",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "2",
    "--gradient_accumulation_steps", "16",
    "--learning_rate", "0.0003",
    "--fp16",
    "--gradient_checkpointing",
    "--cuda", "0",
    "--seed", "1000"
)
$CommonEval = @(
    "t5_aste_pipeline.py", "evaluate",
    "--run_dir", $RunDir,
    "--batch_size", "2",
    "--num_beams", "4",
    "--max_new_tokens", "96",
    "--cuda", "0",
    "--no_task_prefix",
    "--no_constrained_decoding"
)

$Variants = @(
    @{
        Name = "source_pseudo_no_dann_no_contrast"
        Train = $NoAugTrain
        Dev = $NoAugDev
        DomainArgs = @("--lambda_domain_adv", "0.0")
    },
    @{
        Name = "source_pseudo_aug_no_dann_no_contrast"
        Train = $WithAugTrain
        Dev = $WithAugDev
        DomainArgs = @("--lambda_domain_adv", "0.0")
    },
    @{
        Name = "source_pseudo_dann_l003_no_contrast"
        Train = $NoAugTrain
        Dev = $NoAugDev
        DomainArgs = @("--lambda_domain_adv", "0.03", "--domain_adv_exclude_augment")
    },
    @{
        Name = "source_pseudo_aug_dann_l003_no_contrast"
        Train = $WithAugTrain
        Dev = $WithAugDev
        DomainArgs = @("--lambda_domain_adv", "0.03", "--domain_adv_exclude_augment")
    }
)

foreach ($variant in $Variants) {
    $modelDir = Join-Path $RunDir "models\final_ablation_$($variant.Name)_ep5"
    $modelBest = Join-Path $modelDir "best"
    $trainArgs = @(
        "t5_absa_train.py",
        "--train_file", $variant.Train,
        "--dev_file", $variant.Dev,
        "--output_dir", $modelDir
    ) + $variant.DomainArgs + $CommonTrain
    Invoke-Step "TRAIN" $variant.Name $trainArgs @((Join-Path $modelBest "config.json"))

    $evalArgs = $CommonEval + @(
        "--model_path", $modelBest,
        "--output_tag", "strict_ablation_$($variant.Name)"
    )
    $metricTag = "strict_ablation_$($variant.Name)"
    Invoke-Step "EVALUATE" $variant.Name $evalArgs @(
        (Join-Path $RunDir "aste_metrics_raw_$metricTag.json"),
        (Join-Path $RunDir "aste_metrics_fixed_$metricTag.json"),
        (Join-Path $RunDir "aste_metrics_by_sentiment_$metricTag.json"),
        (Join-Path $RunDir "aste_metrics_by_structure_$metricTag.json"),
        (Join-Path $RunDir "aste_error_analysis_$metricTag.json")
    )
}

Invoke-Step "SUMMARY" "strict_module_ablation_summary" @(
    "summarize_strict_module_ablation.py", "--run_dir", $RunDir
) @(
    (Join-Path $RunDir "results_strict_module_ablation.csv"),
    (Join-Path $RunDir "results_strict_module_ablation_CN.md")
)

$BestModel = Join-Path $RunDir "models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_sentiment_contrastive_l001_source_balanced_ep5\best"
Write-Output "CURRENT BEST E model=$BestModel raw_f1=46.82 fixed_f1=48.94"
if (-not $DryRun) {
    Write-QueueStatus "all" "completed"
    "[$(Get-Date -Format s)] ALL STRICT ABLATIONS COMPLETED" | Tee-Object -FilePath $LogPath -Append
}

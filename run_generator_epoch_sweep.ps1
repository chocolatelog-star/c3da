param(
    [int[]]$Epochs = @(6, 10, 12, 14, 16, 18, 20, 22),
    [string]$OutputRoot = "runs\bgca_aste_stage1_domain_prompt_text_v1_generator_epoch_sweep",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Python = "J:\conda\envs\c3da\python.exe"
$BaseArgs = @(
    "run_bgca_aste_stage1_pairs.py",
    "--pairs", "rest16:laptop14",
    "--reuse_upstream_run_dir", "runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14",
    "--extractor_model_path", "J:\nlp\models\t5-base-py",
    "--generator_model_path", "J:\nlp\models\t5-base-py",
    "--generator_prompt_style", "label_to_text",
    "--augment_prompt_style", "label_to_text",
    "--domain_prefix_style", "text",
    "--extractor_epochs", "25",
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
    "--seed", "1000"
)

foreach ($Epoch in $Epochs) {
    $EpochRoot = Join-Path $OutputRoot ("ge{0}" -f $Epoch)
    Write-Host ("[generator sweep] start epoch={0} output_root={1}" -f $Epoch, $EpochRoot)
    $RunArgs = @(
        $BaseArgs[0],
        "--output_root", $EpochRoot,
        "--generator_epochs", ([string]$Epoch)
    ) + $BaseArgs[1..($BaseArgs.Count - 1)]
    if ($DryRun) {
        $RunArgs += "--dry_run"
    }
    & $Python @RunArgs
    if ($LASTEXITCODE -ne 0) {
        throw "epoch $Epoch failed with exit code $LASTEXITCODE"
    }
}

---
name: c3da-experiment-workflow
description: Use when working in J:\nlp\CD-C3DA on BGCA/C3DA cross-domain ASTE experiments, including modifying experiment code, generating commands, analyzing results, cleaning failed runs, updating the Chinese Markdown experiment index, and preserving the current best-vs-BGCA comparison.
---

# C3DA Experiment Workflow

## Required Workflow

- Before changing experiment code, restate the intended change, affected modules, and whether the change requires rerunning experiments.
- Use GPU-safe defaults for RTX 3070 8GB: train batch size 1, eval batch size 2, gradient accumulation 16, fp16, gradient checkpointing, CUDA 0 unless the user says otherwise.
- When giving runnable commands, provide one-line `cmd /c "..."` commands and explain what each command runs.
- After every code change that affects experiments, update the main Chinese Markdown experiment index in the repository root.
- Update the Markdown document by rewriting the relevant summary sections as a coherent current-state document. Do not append an endless chronological log to the bottom.
- Keep the first screen of the Markdown document focused on current best version, model path, BGCA paper baseline, our raw/fixed metrics, gap against BGCA, active experiment, and next decision.
- Maintain a dedicated improvement backlog section in the Markdown document. It must list current weaknesses, improvement goals, concrete changes to try, priority, and the metric condition for accepting or rejecting each direction.
- Put failed or superseded experiment results in a compact historical table with status equivalent to deleted, recommended-to-delete, or metrics-retained.
- Do not delete files or experiment runs without explicit user permission. When a run is bad, ask for confirmation or record it as recommended-to-delete.
- When an experiment finishes, update the best-vs-BGCA table first, then compress the history table. Keep metrics and paths, but remove repetitive narrative.
- Use raw F1 as the main comparison metric. Use fixed F1 only as auxiliary analysis unless the user asks otherwise.

## Current Baseline Facts

- BGCA paper `rest16 -> laptop14` label-to-text F1 is 47.28.
- Current best `rest16 -> laptop14` result is raw F1 48.93 and fixed F1 50.21.
- Current best run uses `hp1 + complete_multi2_w025`, DANN with `lambda_domain_adv=0.03`, sentiment contrastive with `lambda_sentiment_contrastive=0.01`, final pseudo weight 0.65, final augment weight 0.20, and generator `label_to_text` from `J:\nlp\models\t5-base-py` for 8 epochs with best checkpoint selection.
- The BGCA-style generator ablation should keep the current best pipeline fixed and only change the generator to 25 epochs with last checkpoint selection.

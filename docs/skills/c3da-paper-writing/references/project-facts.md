# Project Fact Policy

This reference deliberately avoids rapidly changing experiment scores.

Current scientific facts must be read from the project source of truth.

## Authority

Use in order:

1. `AGENTS.md`
2. `.ai/PROJECT_STATE.md`
3. `.ai/CURRENT_TASK.md`
4. `.ai/DECISION_LOG.md`
5. `实验记录与模型索引_CN.md`
6. `03_CD-C3DA下一阶段改进计划_CN.md`
7. experiment workflow Skill

## Stable project facts

Task:

Cross-domain Aspect Sentiment Triplet Extraction (ASTE).

Training setting:

Source-domain labeled data plus target-domain unlabeled data.

Target-test gold:

Final reporting only.

Formal headline metric:

Strict raw F1.

Experiment identities:

- DIAGNOSTIC
- QUICK_ABLATION
- FULL_RUN

Long-term evaluation scope:

Six cross-domain transfer directions compared against appropriate baselines including BGCA.

## Important

Do not store the current best F1 or temporary experimental candidate in this file.

Those facts change.

Read them at manuscript-writing time from PROJECT_STATE and the experiment index.

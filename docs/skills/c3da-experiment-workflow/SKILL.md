---
name: c3da-experiment-workflow
description: Use when working in J:\nlp\CD-C3DA on BGCA/C3DA cross-domain ASTE experiments, including modifying experiment code, generating commands, analyzing results, cleaning failed runs, updating the Chinese Markdown experiment index, and preserving the current best-vs-BGCA comparison.
---

# C3DA Experiment Workflow

## 核心原则

正式实验必须能够仅凭当前 Git 提交、配方、原始数据和声明模型从头复现。历史提交、历史工作树和历史运行目录只允许审计与对照，禁止作为正式训练输入。

## 修改与分支

- 修改前创建新分支，不在 `master` 直接开发。
- `master` 永远只保存已经通过完整 GPU 实验验证的当前最佳版本。
- 候选分支通过单元测试、试运行和完整实验后，先报告证据并获得用户许可，再合并到 `master` 和打标签。
- 修改前说明目标、受影响模块、参数变化、是否需要重跑；用户确认后再编辑。
- 使用 TDD：先写失败测试，确认正确失败，再实现和验证；每个独立任务单独提交。

## 正式运行与恢复

- 正式入口只调用当前仓库代码，禁止跨运行复用或混合产物，禁止读取其他 `runs`、历史工作树、旧模型、旧伪标签、旧增强或旧训练集。
- 所有阶段产物必须位于本次 `run_id` 根目录；断点恢复只能使用同一 `run_id`。
- 恢复前必须校验 Git 身份、配方身份、上游输入 SHA256 和阶段输出 SHA256。缺失或不一致时立即停止，不能静默重算、复制或回退。
- 每次运行保存完整训练命令、十阶段展开命令、Git 提交与分支、Python/Conda/PyTorch/CUDA/cuDNN/GPU/驱动、`pip freeze`、随机环境变量、模型与数据 SHA256、阶段状态和指标。
- 所有训练、生成和筛选阶段必须显示进度；训练器保留 checkpoint，重复同一命令可以恢复。
- RTX 3070 8GB 默认参数：训练批次 1、评估批次 2、梯度累积 16、fp16、gradient checkpointing、CUDA 0，除非用户明确修改。
- 给用户的多步命令必须合并成一行 `cmd /c "..."`，并逐步解释用途。

## 数量与黄金基准

- 421 条基础伪标签、494 条完整伪标签和 1499 条最终训练行是黄金观察值，不是筛选配额。
- 伪标签和最终训练数据使用本次模型实际产生并通过规则的全部数据；禁止为匹配历史数量裁剪、补齐或读取旧产物。
- 只有配方显式声明的 `selection_limit` 才能限制数量；当前最佳增强上限为 150。
- 黄金观察值只用于比较、定位首次偏差和验收，绝不修改本次源文件。
- 模型权重和预测文件使用整文件 SHA256 验收。训练数据的清单仍保存整文件 SHA256，但黄金停止条件使用覆盖记录顺序和所有实际训练输入字段的训练语义 SHA256，避免无关审计元数据触发误报。
- 当前最终训练语义字段为 `input`、`target`、`sample_weight`、`augmentation`、`base_id` 和 `id`。训练器新增或删除任何读取字段时，必须先用失败测试更新该字段集和黄金语义哈希，禁止静默放宽校验。

## 文档与实验历史

- 每次影响实验的代码改动或运行结束后，整体更新根目录中文 Markdown 实验索引，不能在末尾无限追加叙述。
- 文档首屏必须展示当前最佳与 BGCA 对比、Git 版本、指标差距、当前工作、首次偏差和下一步。
- 保留“待改进”部分：当前不足、改进目标、计划改动、优先级、接受/拒绝指标。
- 历史表每行记录提交、分支、运行目录、完整命令记录、raw/fixed 指标、关键改动、结论和清理状态。
- raw F1 是主要比较指标，fixed F1 用于辅助分析，除非用户另有要求。

## 清理与失败

- 未经用户明确许可不得删除任何文件、模型、检查点或实验目录。
- 失败或较差实验先保留指标、清单、命令、日志和首次错误，在文档标记“建议删除”，获得许可后再清理。
- CUDA OOM、进程崩溃或连续失败时停止自动重启并报告具体阶段；不得通过删除产物掩盖问题。

## Current Baseline Facts

- BGCA paper `rest16 -> laptop14` label-to-text F1 is 47.28.
- Current best `rest16 -> laptop14` result is raw F1 48.93 and fixed F1 50.21.
- Native current-code run `native-best-v2-training-semantic` at commit `558e4de` reproduces all ten stages, golden artifacts, predictions, and metrics without reading historical run outputs.
- The user approved this verified native pipeline as the current `master` baseline on 2026-07-26.
- Current best run uses `hp1 + complete_multi2_w025`, DANN with `lambda_domain_adv=0.03`, sentiment contrastive with `lambda_sentiment_contrastive=0.01`, final pseudo weight 0.65, final augment weight 0.20, and generator `label_to_text` from `J:\nlp\models\t5-base-py` for 8 epochs with best checkpoint selection.
- The BGCA-style generator ablation should keep the current best pipeline fixed and only change the generator to 25 epochs with last checkpoint selection.

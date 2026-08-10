---
name: c3da-experiment-workflow
description: Use when working in J:\nlp\CD-C3DA on BGCA/C3DA cross-domain ASTE experiments, including modifying experiment code, deciding safe stage reuse versus full reruns, generating commands, analyzing results, cleaning failed runs, updating the Chinese Markdown experiment index, and preserving the current best-vs-BGCA comparison.
---

# C3DA Experiment Workflow

## 核心原则

任何实验方案都必须能够仅凭候选 Git 提交、配方、原始数据和声明模型从头运行。快速消融允许从一个封存的“完整从头运行”复用未受影响阶段，但最终最佳验收仍必须从原始数据完整运行；任意历史目录拼接、多个父运行混用和复用链均禁止。

## 修改与分支

- 修改前创建新分支，不在 `master` 直接开发。
- `master` 永远只保存已经通过完整 GPU 实验验证的当前最佳版本。
- 候选分支通过单元测试、试运行和完整实验后，先报告证据并获得用户许可，再合并到 `master` 和打标签。
- 修改前说明目标、受影响模块、参数变化、是否需要重跑；用户确认后再编辑。
- 使用 TDD（测试驱动开发）：先写失败测试，确认正确失败，再实现和验证；每个独立任务单独提交。

## 协作与执行节奏

- 普通代码修改、实验修复、文档更新和结果分析默认由当前智能体直接完成，不启动多个智能体，不把简单任务拆成并行子任务。
- 只有用户明确要求使用智能体，或任务确实存在彼此独立且耗时很长的工作面时，才可以启动子智能体；启动前说明原因，完成后只做一次整体审查。
- 不因为“确定”“继续”或确认文档内容就自动启动正式训练。只有用户明确说“开始运行”“你来跑”或给出明确的执行指令时，才启动正式实验。
- 代码改完后由当前智能体完成测试、语法检查、配方检查、干运行和一次整体审查；不要反复启动审查智能体。
- 正式实验默认由用户在自己的 CMD（命令提示符）窗口中运行。完成改动后必须提供从 `cmd` 开始的完整单行命令和每一步用途；除非用户明确要求代跑，不得自行启动正式训练或打开后台训练窗口。
- 需要长时间运行时，优先报告当前阶段、是否运行、最近进度/错误和下一步，不重复启动同一实验。

## 阶段复用与失效边界

- 每次给出实验命令前，必须先列出：改动内容、最早受影响阶段、真实下游依赖、可复用阶段、必须重跑阶段，以及本次属于快速消融还是完整验收。不能默认每个改动都从头训练，也不能默认所有上游都可复用。
- 区分三类运行：同一 `run_id` 的断点恢复；新 `run_id` 的受控阶段复用；从原始数据开始的完整验收。三者必须在清单和结果报告中明确标记。
- 受控复用只能选择一个父运行。父运行必须是同方向、同 seed（随机种子）、同原始数据和声明模型的成功完整从头运行，Git 工作区干净、清单和阶段哈希完整，并标记 `reuse_depth=0`。
- 复用子运行标记 `reuse_depth=1`，记录父运行、父 Git 提交、每个复用阶段的生产提交、命令、相关代码指纹、输入 SHA256、输出 SHA256 和导入方式。`reuse_depth=1` 的运行不得再作为新父运行，禁止形成复用链。
- 只能复用阶段依赖图中未受改动影响且指纹完全一致的产物。Git 提交不同本身不等于必须重跑；但相关代码、命令、配方字段或任一输入哈希变化时，该阶段及其真实下游全部失效。
- 复用产物必须通过已经通过项目测试的专用入口按内容哈希导入本次 `run_id` 并成为受清单保护的本地快照，禁止让训练阶段直接读取父运行的活动目录。入口不可用或任一门禁不通过时，不得手工复制或跨目录引用产物。
- 当前十阶段依赖的默认判断如下；实际执行仍以显式阶段依赖和指纹为准：

| 改动位置 | 默认可复用 | 必须重跑 |
|---|---|---|
| 仅最终解码或评估规则 | 第1–9阶段 | `evaluate`（评估） |
| 最终训练损失、DANN（领域对抗神经网络）、样本权重或优化参数 | 第1–8阶段 | `final_train/evaluate`（最终训练/评估） |
| 最终训练集组合 | 第1–7阶段 | `build_final_train/final_train/evaluate`（组装最终训练集/最终训练/评估） |
| 完整多三元组筛选 | 第1–6阶段 | `complete_multi2` 及其下游 |
| 增强生成或过滤 | `prepare/extractor/pseudo/generator`（准备/抽取器/伪标签/生成器） | `augment`（增强）及其下游 |
| 生成器训练或生成器配置 | 未受影响的 `prepare/extractor/pseudo`（准备/抽取器/伪标签） | `generator/augment`（生成器/增强）及其真实下游 |
| 伪标签规则 | `prepare/extractor`（准备/抽取器）；生成器分支指纹一致时也可复用 | `pseudo`（伪标签）及其真实下游 |
| 抽取器训练或结构 | 准备阶段；生成器独立分支指纹一致时可复用 | `extractor/pseudo`（抽取器/伪标签）及其真实下游 |
| 数据、领域方向、seed（随机种子）、分词、任务格式或共同基础模型 | 无 | 全部从头运行 |

- 受控复用结果默认标记为“快速消融”，不能替代完整复现。仅修改评估规则时，可以把封存的完整从头模型用于正式解码消融；若结果将成为新最佳、与 BGCA 正式比较、合并到 `master`、扩展六方向/多 seed，仍至少执行一次对应候选代码和配方的完整从头验收。
- 大模块完成、累计多次小改动、准备形成论文结论、发现第二层复用、父运行不完整、阶段来源无法单一解释或任何指纹/哈希不一致时，强制停止复用并从原始数据完整运行。

## 正式运行与恢复

- 完整从头验收入口只调用当前仓库代码和声明的原始输入，不复用其他运行产物。快速消融只有满足上一节全部条件时才能通过专用入口复用单一封存父运行；除此之外禁止读取其他 `runs`、历史工作树、旧模型、旧伪标签、旧增强或旧训练集。
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

# C3DA 主线收束到 FLAN-T5 生成器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `CD-C3DA` 收束成一条可直接从头跑通的主线：提取器保留 `t5-base-py`，生成器切换到 `google/flan-t5-base`（FLAN-T5 基础版），增强只走高精度伪标签 + 干净增强 + 少量句子融合，最终训练保留源域数据，并清理掉默认流程里的冗余分支。

**Architecture:**  
保留三段式流水线：`prepare -> pseudo/select -> augment -> final train -> evaluate`。  
提取器仍然负责目标域伪标签抽取，生成器只负责“标签到句子”的文本生成；增强阶段保留方面词替换、少量句子融合和严格筛选，不再默认跑多个实验分支。  
代码层面以 `t5_aste_pipeline.py` 为主入口，`t5_aste_augment.py` 负责生成与筛选逻辑，`t5_absa_train.py` 负责最终训练权重与联合损失。

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, Seq2SeqTrainer, JSONL, Windows cmd

---

### Task 1: 固定主线默认模型与路径

**Files:**
- Modify: `J:\nlp\CD-C3DA\t5_aste_pipeline.py`
- Modify: `J:\nlp\CD-C3DA\t5_absa_train.py`
- Modify: `J:\nlp\CD-C3DA\README.md`

- [ ] **Step 1: 把生成器默认模型切到 FLAN-T5**

将 `t5_aste_pipeline.py` 中 `augment` 子命令的默认生成器路径从 `J:\nlp\models\mrm8488-t5-base-finetuned-common_gen` 改为 `J:\nlp\models\google\flan-t5-base`（若本机实际目录名不同，则按 `J:\nlp\models` 下已有目录写死为真实路径），提取器默认保持 `J:\nlp\models\t5-base-py`。

- [ ] **Step 2: 统一最终训练默认配置**

把最终训练默认参数收束为主线版本：`checkpoint_selection=best`，`lambda_structure_loss=0.15`，`lambda_pairing_loss=0.0`，`augment_select_max_rows=200`，`augment_select_require_raw_exact` 和 `augment_select_require_model_filter_passed` 默认开启；保留 `--no_final_train_source` 作为可选开关，但默认不关闭源域。

- [ ] **Step 3: 更新 README 命令说明**

把 README 里旧的 `C3DA/run.sh`、`start.sh` 说明替换成当前 `t5_aste_pipeline.py` + `t5_absa_train.py` 的主线命令说明，注明生成器使用 `FLAN-T5`（指令微调 T5），提取器使用 `t5-base-py`。

- [ ] **Step 4: 验证默认参数输出**

运行一次 `python t5_aste_pipeline.py -h` 和 `python t5_absa_train.py -h`，确认默认值和帮助信息一致，且没有残留旧主线默认值。

### Task 2: 收敛增强主流程，只保留主线增强

**Files:**
- Modify: `J:\nlp\CD-C3DA\t5_aste_pipeline.py`
- Modify: `J:\nlp\CD-C3DA\t5_aste_augment.py`

- [ ] **Step 1: 让增强阶段默认只走高精度伪标签**

把 `augment` 默认流程中的伪标签来源固定为 `high_precision`，不再默认使用 `strict`，并确保 `select_pseudo`、`pseudo`、`augment` 三者共享同一套高精度筛选逻辑。

- [ ] **Step 2: 收束增强通道**

把默认增强通道收束为 `aspect` + `sentence_fusion_composition`，其中 `opinion` 通道只保留为显式可选，不进入默认主线；保留 `masked_mutual` 和 `rsda_t5_label_composition` 代码，但不再进入默认命令。

- [ ] **Step 3: 让句子融合只吃干净数据**

在 `build_augmentation_requests()` 里把句子融合请求的候选源限制为单方面词/单三元组、通过高质量过滤、并来自这次 run 的 `pseudo_rows` 与 `composition_source_rows`，避免旧实验数据混入。

- [ ] **Step 4: 增强分析输出**

在 `c3da_augment_analysis.json` 和 `final_train_composition_analysis.json` 中补齐“高精度伪标签保留数量”“句子融合数量”“方面词增强数量”“来自源域/伪标签/增强的行数统计”，方便你直接看这次跑出来的主线是不是干净。

### Task 3: 清理默认执行入口中的冗余分支

**Files:**
- Modify: `J:\nlp\CD-C3DA\t5_aste_pipeline.py`

- [ ] **Step 1: 整理入口命令分支**

保留 `prepare / pseudo / select_pseudo / memory / augment / evaluate` 五个子命令，但将默认推荐顺序固定成一条主线，并把旧分支的说明降级成“可选实验”。

- [ ] **Step 2: 去掉默认不再使用的输出写入**

只保留主线会用到的输出文件：`target_pseudo_high_precision.jsonl`、`c3da_cross_domain_memory.json`、`c3da_two_channel_augmented_selected.jsonl`、`final_train.jsonl`、`final_dev.jsonl`、`aste_metrics*.json`。  
像纯实验分支产生的中间文件仍可保留写入逻辑，但不再由默认推荐命令生成。

- [ ] **Step 3: 补强路径解析**

修复所有本地模型路径/输出路径的解析逻辑，避免 `runs\t...` 被误解释成转义字符，保证 Windows `cmd` 下的路径都能正常传给 `from_pretrained()`。

### Task 4: 训练与验证主线闭环

**Files:**
- Modify: `J:\nlp\CD-C3DA\t5_absa_train.py`
- Modify: `J:\nlp\CD-C3DA\t5_aste_pipeline.py`

- [ ] **Step 1: 固定最终训练策略**

最终训练默认使用 `best` checkpoint，保留源域数据参与训练，伪标签权重与增强权重采用当前已验证的轻量配置，不再引入额外的 pairing loss 默认值。

- [ ] **Step 2: 输出完整训练统计**

在训练结束打印并保存：源域行数、伪标签行数、高精度伪标签保留数、增强行数、融合行数、最终训练行数、最终 dev 行数、以及 `best` 模型路径。

- [ ] **Step 3: 重新跑一遍完整闭环验证**

按主线顺序跑：`prepare -> pseudo -> augment -> train -> evaluate`，确认每一步都能从上一步输出继续跑，并且最终 test 指标能落盘到 `aste_metrics.json`。

### Task 5: 最终整理与可运行命令

**Files:**
- Modify: `J:\nlp\CD-C3DA\README.md`
- Modify: `J:\nlp\CD-C3DA\docs\superpowers\plans\2026-06-22-c3da-mainline-flan-t5-base.md`

- [ ] **Step 1: 写出从 `cmd` 开始的完整命令**

把主线跑法写成一条可直接复制的 `cmd` 一行命令，按步骤解释每一步在干什么：准备数据、抽伪标签、做增强、训最终模型、评估。

- [ ] **Step 2: 自检计划与代码一致性**

核对计划中的默认模型、默认权重、默认输出文件名与代码实现一致，没有写错路径名或过时参数。


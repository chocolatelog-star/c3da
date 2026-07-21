# CD-C3DA 实验记录与模型索引

本文档是 `J:\nlp\CD-C3DA` 的当前实验总览。维护原则：每次改代码或跑完实验后，优先整体更新本页开头的当前状态、差距表和决策表，不在文档末尾无限追加流水账。

## 1. 当前一眼结论

| 项目 | 当前值 |
|---|---|
| 当前主攻方向 | `rest16 -> laptop14` |
| 当前最佳流程 | `hp1 + complete_multi2_w025` + DANN（领域对抗）+ sentiment contrastive（情感对比学习） |
| 当前最佳 raw F1（原始F1） | **48.93** |
| 当前最佳 fixed F1（修正F1） | **50.21** |
| BGCA 论文 label-to-text F1（标签到文本F1） | **47.28** |
| 当前 raw F1（原始F1）相对 BGCA | **+1.65** |
| 当前主要短板 | 多三元组 recall（召回率）仍低，neutral（中性）几乎没有学好 |
| 当前正在验证 | BGCA-style generator（BGCA风格生成器）：保持最佳流程不变，只把生成器改为 25 轮 last（最后检查点） |

## 2. 当前最佳与 BGCA 对比

主指标使用 raw F1（原始F1）。fixed F1（修正F1）只作为辅助分析。

| 方法 | 生成器 | 关键模块 | raw P（原始精确率） | raw R（原始召回率） | raw F1（原始F1） | fixed F1（修正F1） | 相对 BGCA raw F1 |
|---|---|---|---:|---:|---:|---:|---:|
| BGCA 论文 label-to-text（标签到文本） | T5-base，25 轮 last（最后检查点） | data generation（数据生成）+ model filter（模型过滤） | - | - | **47.28** | - | 0.00 |
| 我们当前最佳 | T5-base，8 轮 best（最优检查点） | hp1 + complete_multi2_w025 + DANN（领域对抗）+ 情感对比 | 58.31 | 42.14 | **48.93** | **50.21** | **+1.65** |
| 当前待跑对照 | T5-base，25 轮 last（最后检查点） | 其余保持当前最佳完全一致 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

当前最佳模型：

```text
runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_complete_multi2_w025_sentiment_contrastive_l001_source_balanced_ep5\best
```

当前最佳结果文件：

```text
runs\bgca_aste_stage1_domain_prompt_text_v1\results_bgca_aste_stage1_complete_multi2_w025_sentiment_contrastive_l001_source_balanced_pw065.csv
runs\bgca_aste_stage1_domain_prompt_text_v1\results_bgca_aste_stage1_complete_multi2_w025_sentiment_contrastive_l001_source_balanced_pw065_CN.md
```

## 3. 当前最佳流程组成

| 阶段 | 当前最佳做法 |
|---|---|
| 基础模型 | 抽取器、生成器、最终模型均从 `J:\nlp\models\t5-base-py` 启动 |
| 抽取器 | 源域 text -> triplet（文本到三元组），25 轮，plain last（普通最后检查点） |
| 伪标签 | 目标域无标签句子生成伪标签，先取 hp1（最多 1 个三元组，距离 5） |
| 多三元组补充 | 在 hp1 基础上补充完整双三元组，权重 `complete_multi_extra_weight=0.25` |
| 生成器 | `label_to_text`（标签到文本），8 轮，`checkpoint_selection=best`（按验证损失取最优） |
| 增强 | `masked_mutual`（互相掩码增强），严格筛选 150 条，增强权重 0.20 |
| 最终训练 | 源域 gold（真实标签）+ 伪标签 + 增强 |
| 领域对抗 | 保留，`lambda_domain_adv=0.03` |
| 情感对比 | 保留，`lambda_sentiment_contrastive=0.01`，source only（仅源域），class balanced（类别平衡） |
| 伪标签权重 | 当前最佳使用 `final_pseudo_weight=0.65` |

## 4. 已做改进与结论压缩表

| 改进方向 | 最好结果 raw F1 | fixed F1 | 结论 | 文件处理 |
|---|---:|---:|---|---|
| 原始主线：hp1 + 增强 + DANN | 46.82 | 48.94 | 可作为旧基线，但已被完整双三元组补充超过 | 保留指标 |
| hp2_dist5 放宽伪标签到最多 2 个三元组 | 44.44 | 46.87 | 放宽数量带来噪声，失败 | 坏模型建议删除，指标保留 |
| neutral generation loss（中性生成损失）增权 | 43.18 | 45.76 | 没解决中性，反而破坏正负类 | 坏模型已删除或建议删除 |
| mixed generator（三任务混合生成器） | 44.07 | 46.06 | 1:1:1 混合削弱 label-to-text 主任务 | 坏模型已删除 |
| encoder pairing loss（编码器配对损失） | 46.49 | 48.86 | 精确率提高但召回下降，不作为最佳 | 坏模型已删除或建议删除 |
| triplet coverage loss（三元组覆盖损失） | 44.37 | 46.72 | 分类头没有传导到自回归生成 | 坏模型已删除 |
| complete_multi2_w025，不加情感对比 | 48.01 | 50.37 | 完整双三元组补充有效，是关键正向改动 | 保留指标 |
| complete_multi2_w025 + 情感对比 | **48.93** | **50.21** | 当前最佳，主线保留 | 保留模型 |
| complete_multi2_w035 | 45.74 | 47.02 | 补充权重过高，引入噪声 | 建议删除 |
| dynamic_strict_dist5 | 48.07 | 49.69 | 有潜力但没有超过最佳 | 保留指标，暂不主线 |
| complete_multi2 + dynamic strict 3+ | 45.38 | 47.48 | 3+ 伪标签噪声和欠配对明显 | 建议删除 |
| dynamic strict top050/top080 | 45.83 / 44.66 | 47.78 / 46.90 | keep top ratio（截断高置信比例）无效 | 建议删除 |

## 5. 当前待跑实验

目标：验证 BGCA-style generator（BGCA风格生成器）是否比我们 8 轮 best（最优检查点）更适合当前最佳流程。

唯一变量：

| 项目 | 当前最佳 | 待跑对照 |
|---|---|---|
| 生成器训练轮数 | 8 | 25 |
| 生成器检查点选择 | best（最优检查点） | last（最后检查点） |
| 抽取器 | 不变 | 不变 |
| 伪标签 | 不变，complete_multi2_w025 | 不变，complete_multi2_w025 |
| 增强 | 不变，strict_aug150_w020 | 不变，strict_aug150_w020 |
| DANN（领域对抗） | 保留 | 保留 |
| 情感对比 | 保留 | 保留 |
| 伪标签权重 | 0.65 | 0.65 |
| 增强权重 | 0.20 | 0.20 |

完整命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_bgca_generator25_last_v1 --pairs rest16:laptop14 --reuse_upstream_run_dir runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14 --extractor_model_path J:\nlp\models\t5-base-py --generator_model_path J:\nlp\models\t5-base-py --generator_prompt_style label_to_text --augment_prompt_style masked_mutual --domain_prefix_style text --extractor_epochs 25 --generator_epochs 25 --generator_checkpoint_selection last --final_epochs 5 --complete_multi_extra_weight 0.25 --final_pseudo_weight 0.65 --final_augment_weight 0.20 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --learning_rate 0.0003 --eval_batch_size 2 --cuda 0 --seed 1000"
```

命令含义：复用当前最佳上游抽取器和伪标签，只重新训练 25 轮 last（最后检查点）生成器，然后重新生成增强、训练最终模型并评估。

## 6. 文档维护规则

| 触发事件 | 必须更新 |
|---|---|
| 改实验代码 | 更新第 1、3、5、6 节，说明当前流程变化和需要重跑的命令 |
| 跑完新实验 | 先更新第 2 节对比表，再把第 4 节历史表压缩改写 |
| 新结果超过最佳 | 更新当前最佳模型路径、指标、差距和保留策略 |
| 新结果失败 | 在第 4 节标记 `建议删除`，保留指标文件位置，不删除文件 |
| 用户确认删除 | 删除失败模型/检查点/大文件后，在第 4 节把状态改成 `已删除` |

删除规则：所有删除文件操作必须先获得用户明确许可。效果不好的实验默认只标记为 `建议删除`，不自动删除。

## 7. Git 与环境状态

| 项目 | 当前状态 |
|---|---|
| 仓库 | `https://github.com/chocolatelog-star/c3da.git` |
| 分支 | `master` |
| 最近推送 | 已推送到 `origin/master` |
| 最新提交 | `68bc0d0 Add BGCA-style generator selection diagnostics` |
| 训练环境 | `conda activate c3da` |
| GPU（显卡） | NVIDIA RTX 3070，8GB 显存 |
| 推荐训练参数 | batch size（批大小）1，eval batch size（验证批大小）2，gradient accumulation（梯度累积）16，fp16（半精度），gradient checkpointing（梯度检查点） |

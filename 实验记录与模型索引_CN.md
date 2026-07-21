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

## 4. 历史实验索引

本节用于以后回溯代码和实验说明。每条记录必须能回答三件事：当时用的是哪个 git commit（提交号）、实验产物标签是什么、结论是什么。失败模型可以删除，但指标、标签和 commit（提交号）必须保留。

### 4.1 可追溯实验结果表

| 时间线 | 实验方向 | 关键 commit（提交号） | 结果标签或目录 | raw F1（原始F1） | fixed F1（修正F1） | 主要变化 | 结论 | 文件状态 |
|---|---|---|---|---:|---:|---|---|---|
| 旧主线 | hp1 + 增强 + DANN（领域对抗） | `e4472d9` / `f809beb` | `strict_aug150_w020_label_to_text_gen_sentiment_contrastive_l001_source_balanced` | 46.82 | 48.94 | 最终阶段加入源域类别平衡情感对比 | 可作为旧基线，但已被完整双三元组补充超过 | 保留指标 |
| 伪标签放宽 | hp2_dist5，最多 2 个三元组 | `869466a` | `strict_aug150_w020_label_to_text_gen_hp2_dist5` | 44.44 | 46.87 | 从 hp1 放宽到 hp2 | 噪声增多，召回没有换来有效 F1 | 坏模型建议删除，指标保留 |
| 中性增强 | neutral generation loss（中性生成损失）增权 | `0c49ba6` / `ce7452e` / `e5f5d47` | `neutral_gain100_max200` | 43.18 | 45.76 | 提高中性样本主生成损失权重 | 中性没解决，正负类被破坏 | 坏模型已删除或建议删除 |
| 生成器结构 | mixed generator（三任务混合生成器） | `e7560c7` / `e320fab` / `925d596` / `93b7b4a` | `bgca_aste_stage1_mixed_generator_v1` | 44.07 | 46.06 | 生成器同时学 label-to-text、masked_aspect、masked_opinion | 混合任务削弱主生成目标，不进主线 | 坏模型已删除 |
| 配对辅助 | encoder pairing loss（编码器配对损失） | `c1082ab` / `123ab39` / `6075ee0` / `a256965` | `pairing_encoder_l001_source_only` | 46.49 | 48.86 | 对源域多三元组加方面词-观点词配对损失 | 精确率提高但召回下降，不作为最佳 | 坏模型已删除或建议删除 |
| 覆盖辅助 | triplet coverage loss（三元组覆盖损失） | `cbeb965` / `e60ca8f` / `44997d4` | `coverage_encoder_l001_source_balanced` | 44.37 | 46.72 | 编码器预测句子应包含的三元组数量 | 分类头没有有效传导到生成 | 坏模型已删除 |
| 完整双三元组 | complete_multi2_w025，不加情感对比 | `62113b4` / `4258bc6` / `0332aee` | `complete_multi2_w025` | 48.01 | 50.37 | 在 hp1 上补充完整双三元组，额外权重 0.25 | 关键正向改动，多三元组抽取明显改善 | 保留指标 |
| 当前最佳 | complete_multi2_w025 + 情感对比 | `62113b4` / `0332aee` / `68bc0d0` | `complete_multi2_w025_sentiment_contrastive_l001_source_balanced_pw065` | **48.93** | **50.21** | 完整双三元组 + DANN + 源域类别平衡情感对比，伪标签权重 0.65 | 当前最佳，主线保留 | 保留模型 |
| 权重过高 | complete_multi2_w035 | `c0b2730` | `complete_multi2_w035_sentiment_contrastive_l001_source_balanced` | 45.74 | 47.02 | 双三元组补充权重从 0.25 提到 0.35 | 权重过高，引入噪声 | 建议删除 |
| 动态多三元组 | dynamic_strict_dist5 | `577c55e` / `e8e23e3` / `7f3724d` | `dynamic_strict_dist5` | 48.07 | 49.69 | 动态保留多三元组伪标签，不强制最多 1 个 | 有潜力但未超过最佳 | 保留指标，暂不主线 |
| 3+ 补充 | complete_multi2 + dynamic strict 3+ | `7f3724d` / `9e76a19` | `complete_dynamic3plus_v1` | 45.38 | 47.48 | 在完整双三元组上再补 3+ 动态严格伪标签 | 3+ 噪声和欠配对明显 | 建议删除 |
| 高置信截断 | dynamic strict top050/top080 | `68bc0d0` | `top050` / `top080` | 45.83 / 44.66 | 47.78 / 46.90 | 对 3+ 动态伪标签按置信度比例截断 | keep top ratio（高置信比例截断）无效 | 建议删除 |
| 当前待跑 | BGCA-style generator（BGCA风格生成器）25 轮 last | `68bc0d0` / `11ca672` / `50a87f7` | `bgca_aste_stage1_bgca_generator25_last_v1` | 待跑 | 待跑 | 只把生成器改为 25 轮 last，其余保持当前最佳 | 用来判断 BGCA 生成器训练策略是否更好 | 待生成 |

### 4.2 历史代码和说明入口

| 主题 | 主要 commit（提交号） | 说明文档或结果文件 |
|---|---|---|
| 当前总览、差距和待改进清单 | `50a87f7` | `实验记录与模型索引_CN.md` |
| 项目文档维护规则 | `11ca672` | `docs\skills\c3da-experiment-workflow\SKILL.md` |
| BGCA-style generator（BGCA风格生成器）参数支持、源域 dev 评估 | `68bc0d0` | `run_bgca_aste_stage1_pairs.py`、`t5_aste_pipeline.py` |
| 完整双三元组补充与严格消融实现 | `62113b4` | `test_complete_multitriplet_pseudo.py`、`run_bgca_aste_stage1_pairs.py` |
| 完整双三元组实验结论 | `0332aee` | `results_bgca_aste_stage1_complete_multi2_w025*_CN.md` |
| dynamic strict（动态严格筛选） | `577c55e`、`e8e23e3`、`7f3724d` | `docs\superpowers\plans\2026-07-18-dynamic-multitriplet-training_CN.md` |
| mixed generator（三任务混合生成器） | `e7560c7`、`925d596`、`93b7b4a` | `docs\superpowers\plans\2026-07-15-mixed-generator-training_CN.md` |
| encoder pairing loss（编码器配对损失） | `c1082ab`、`123ab39`、`6075ee0`、`a256965` | `docs\superpowers\plans\2026-07-16-encoder-pairing-loss_CN.md` |
| triplet coverage loss（三元组覆盖损失） | `cbeb965`、`e60ca8f`、`44997d4` | `docs\superpowers\plans\2026-07-16-triplet-coverage-loss_CN.md` |
| neutral generation weighting（中性生成增权） | `0c49ba6`、`ce7452e`、`e5f5d47` | `results_bgca_aste_stage1_neutral_gain100_max200_CN.md` |

回溯方式：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && git show <commit>"
```

如果要临时查看旧代码，不要直接覆盖当前工作区，优先使用：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && git worktree add .worktrees\inspect-<commit> <commit>"
```

## 5. 待改进清单

本节只记录当前仍值得继续做的改进目标，不记录已经证伪的长流水实验。

| 优先级 | 当前不足 | 改进目标 | 具体改进点 | 预期判断标准 |
|---|---|---|---|---|
| P0 | 生成器训练策略和 BGCA（论文方法）不完全一致：我们当前最佳生成器是 8 轮 best（最优检查点），BGCA 是 25 轮 last（最后检查点） | 判断性能差异是否来自生成器训练轮数和检查点选择 | 跑 BGCA-style generator（BGCA风格生成器）对照：只改 `--generator_epochs 25 --generator_checkpoint_selection last`，其他保持当前最佳 | raw F1（原始F1）超过 48.93 才替换主线；否则保留 8 轮 best |
| P0 | 过滤器可靠性缺少显式源域 dev（开发集）F1 记录 | 判断“生成句子再回抽标签一致才保留”的过滤器是否足够可靠 | 已在代码中新增 source dev evaluation（源域开发集评估）；后续每次实验汇总都要写入源域 dev fixed F1 | 若源域 dev F1 明显低，不能把抽取器当强过滤器使用 |
| P1 | 多三元组 recall（召回率）仍低，尤其 3+ 三元组样本补充后没有稳定收益 | 提高多三元组完整抽取能力，而不是简单放宽伪标签数量 | 保留 complete_multi2_w025；后续尝试更细的多三元组训练权重、生成候选多样性、回抽一致过滤，不再使用 top ratio（高置信比例截断） | 多三元组 raw F1 和 recall 同时提升，且总体 raw F1 不下降 |
| P1 | neutral（中性）三元组几乎无法召回，强行加权会伤害正负类 | 建立中性边界，而不是只加大中性损失权重 | 优先做错误类型分析：否定但非中性、缺失属性但中性、弱情感表达；再考虑构造小规模高质量中性增强 | neutral F1 有实际提升，同时 pos/neg（正向/负向）F1 不明显下降 |
| P2 | 当前增强样本仍可能引入标签一致但表达质量低的句子 | 提高增强样本质量和多样性 | 在标签回抽一致基础上增加去重、非原句复制、长度和领域词覆盖筛选 | 增强保留率不过低，最终 raw F1 提升或至少召回提升 |
| P2 | 六组跨域平均仍落后 BGCA，laptop14 -> restaurant 三组差距最大 | 从单方向有效改进迁移到六组平均 | 当前先在 rest16 -> laptop14 验证机制；有效后再跑六组，并单独分析 laptop14 -> restaurant 的 recall 问题 | 六组平均 raw F1 差距收敛，不能只提升单组 |

当前不要继续投入的方向：`hp2_dist5` 简单放宽数量、中性生成损失强加权、三任务 mixed generator（混合生成器）、triplet coverage classification head（三元组覆盖分类头）、dynamic strict top ratio（动态严格高置信比例截断）。

## 6. 当前待跑实验

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

## 7. 文档维护规则

| 触发事件 | 必须更新 |
|---|---|
| 改实验代码 | 更新第 1、3、5、6 节，说明当前流程变化和需要重跑的命令 |
| 跑完新实验 | 先更新第 2 节对比表，再把第 4 节历史表压缩改写 |
| 新结果超过最佳 | 更新当前最佳模型路径、指标、差距和保留策略 |
| 新结果失败 | 在第 4 节标记 `建议删除`，保留指标文件位置，不删除文件 |
| 用户确认删除 | 删除失败模型/检查点/大文件后，在第 4 节把状态改成 `已删除` |

删除规则：所有删除文件操作必须先获得用户明确许可。效果不好的实验默认只标记为 `建议删除`，不自动删除。

## 8. Git 与环境状态

| 项目 | 当前状态 |
|---|---|
| 仓库 | `https://github.com/chocolatelog-star/c3da.git` |
| 分支 | `master` |
| 最近推送 | 已推送到 `origin/master` |
| 最新提交 | `68bc0d0 Add BGCA-style generator selection diagnostics` |
| 训练环境 | `conda activate c3da` |
| GPU（显卡） | NVIDIA RTX 3070，8GB 显存 |
| 推荐训练参数 | batch size（批大小）1，eval batch size（验证批大小）2，gradient accumulation（梯度累积）16，fp16（半精度），gradient checkpointing（梯度检查点） |

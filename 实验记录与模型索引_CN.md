# CD-C3DA 实验记录与模型索引

本文档是项目实验总账，配合 git（版本管理）记录代码版本、实验输出、模型路径和阶段结论。维护原则：优先整体更新表格和结论，不在文档末尾无限追加。

## 0. 当前总览

| 项目 | 内容 |
|---|---|
| GitHub（代码托管平台） | https://github.com/chocolatelog-star/c3da.git |
| 当前代码版本 | `de435be Clarify generator and augment prompt styles` |
| 当前分支 | `master` |
| 主实验目录 | `runs\bgca_aste_stage1_baseline` |
| 当前六组实验状态 | 已完成 |
| 生成器训练方式 | `label_to_text`（标签到文本） |
| 数据增强方式 | `masked_mutual`（掩码增强） |
| 最终训练配置 | 高精度伪标签 + 严格增强 + DANN（领域对抗） |
| 主对比指标 | `raw F1`（原始 F1） |
| 辅助分析指标 | `fixed F1`（修正 F1） |

## 1. BGCA 论文结果与我们的结果对比

BGCA 论文 ASTE（方面情感三元组抽取）六组跨域结果来自 Table 4（表 4），这里优先对比 `BGCA label-to-text`（标签到文本）这一行。我们的结果使用当前主流程：`label_to_text`（标签到文本）生成器训练 + `masked_mutual`（掩码增强）数据增强 + 高精度伪标签 + DANN（领域对抗）。

| 跨域方向 | BGCA label-to-text F1（论文） | 我们 raw F1（原始 F1） | 差值 | 我们 fixed F1（修正 F1） |
|---|---:|---:|---:|---:|
| rest14 -> laptop14 | 53.64 | 52.05 | -1.59 | 53.36 |
| rest15 -> laptop14 | 45.69 | 41.02 | -4.67 | 45.01 |
| rest16 -> laptop14 | 47.28 | 46.14 | -1.14 | 47.69 |
| laptop14 -> rest14 | 65.27 | 53.11 | -12.16 | 53.98 |
| laptop14 -> rest15 | 58.95 | 52.02 | -6.93 | 53.94 |
| laptop14 -> rest16 | 64.00 | 54.85 | -9.15 | 55.58 |
| 平均 | 55.80 | 49.86 | -5.94 | 51.59 |

### 结论

1. 当前方法在六组平均 `raw F1`（原始 F1）为 49.86，低于 BGCA 论文 `label-to-text`（标签到文本）平均 55.80，差 5.94。
2. `rest16 -> laptop14` 与 BGCA 最接近，只差 1.14；`rest14 -> laptop14` 差 1.59，也比较接近。
3. 从 laptop14（笔记本）迁移到 restaurant（餐馆）的三组明显更弱，尤其 `laptop14 -> rest14` 差 12.16。主要表现是 `precision`（精确率）较高但 `recall`（召回率）不足。
4. 当前流程不能称为纯 BGCA 复现。它借鉴了 BGCA 的 `label_to_text`（标签到文本）生成器训练思想，但实际增强使用的是我们的 `masked_mutual`（掩码增强），并额外加入高精度伪标签筛选、NLI（自然语言推理）过滤、模型回抽过滤和 DANN（领域对抗）。

## 2. 我们的六组完整结果

| 跨域方向 | source gold（源域人工标注） | high_precision pseudo（高精度伪标签） | pseudo HP F1 | augment（增强条数） | final train（最终训练条数） | raw P | raw R | raw F1 | fixed F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rest14 -> laptop14 | 1266 | 421 | 54.38 | 150 | 1835 | 60.19 | 45.84 | 52.05 | 53.36 |
| rest15 -> laptop14 | 605 | 389 | 48.94 | 88 | 1080 | 51.25 | 34.20 | 41.02 | 45.01 |
| rest16 -> laptop14 | 857 | 421 | 50.35 | 111 | 1387 | 54.99 | 39.74 | 46.14 | 47.69 |
| laptop14 -> rest14 | 906 | 625 | 54.89 | 150 | 1679 | 72.07 | 42.05 | 53.11 | 53.98 |
| laptop14 -> rest15 | 906 | 315 | 56.77 | 130 | 1349 | 59.21 | 46.39 | 52.02 | 53.94 |
| laptop14 -> rest16 | 906 | 451 | 58.99 | 150 | 1505 | 64.23 | 47.86 | 54.85 | 55.58 |
| 平均 | 907.67 | 437.00 | 53.89 | 129.83 | 1472.50 | 60.66 | 42.68 | 49.86 | 51.59 |

## 3. 每组模型和结果位置

| 跨域方向 | run_dir（实验目录） | final model（最终模型） |
|---|---|---|
| rest14 -> laptop14 | `runs\bgca_aste_stage1_baseline\rest14_to_laptop14` | `runs\bgca_aste_stage1_baseline\rest14_to_laptop14\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_ep5\best` |
| rest15 -> laptop14 | `runs\bgca_aste_stage1_baseline\rest15_to_laptop14` | `runs\bgca_aste_stage1_baseline\rest15_to_laptop14\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_ep5\best` |
| rest16 -> laptop14 | `runs\bgca_aste_stage1_baseline\rest16_to_laptop14` | `runs\bgca_aste_stage1_baseline\rest16_to_laptop14\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_ep5\best` |
| laptop14 -> rest14 | `runs\bgca_aste_stage1_baseline\laptop14_to_rest14` | `runs\bgca_aste_stage1_baseline\laptop14_to_rest14\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_ep5\best` |
| laptop14 -> rest15 | `runs\bgca_aste_stage1_baseline\laptop14_to_rest15` | `runs\bgca_aste_stage1_baseline\laptop14_to_rest15\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_ep5\best` |
| laptop14 -> rest16 | `runs\bgca_aste_stage1_baseline\laptop14_to_rest16` | `runs\bgca_aste_stage1_baseline\laptop14_to_rest16\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_ep5\best` |

## 4. 当前流程

每组实验完整流程如下：

1. `prepare`（准备数据）：构造 source train/dev（源域训练/验证）、target unlabeled（目标域无标签训练集）、target test（目标域测试集）、extractor（提取器）训练数据和 generator（生成器）训练数据。
2. `train_extractor`（训练提取器）：用源域人工标注训练 25 轮，`checkpoint_selection=last`（选择最后一轮）。
3. `pseudo`（生成伪标签）：用提取器对目标域训练集生成伪标签，并筛选 `high_precision`（高精度）伪标签。
4. `train_generator`（训练生成器）：用 `label_to_text`（标签到文本）方式训练 8 轮生成器。
5. `augment`（数据增强）：用训练好的生成器执行 `masked_mutual`（掩码增强），经过文本质量过滤、NLI（自然语言推理）过滤、模型回抽过滤，最多选入 150 条增强样本，权重 0.20。
6. `train_final`（训练最终模型）：使用 source gold（源域人工标注）+ high_precision pseudo（高精度伪标签）+ selected augment（筛选增强），训练 5 轮，开启 DANN（领域对抗）且 `domain_adv_exclude_augment=True`（增强数据不参与领域对抗）。
7. `evaluate`（评估）：在目标域测试集上使用 `--no_constrained_decoding`（非约束解码）评估。

## 5. 关键问题

### 5.1 召回率不足

六组平均 `raw recall`（原始召回率）只有 42.68，明显低于 `raw precision`（原始精确率）60.66。当前模型偏保守，抽出的三元组质量还可以，但漏抽多。

### 5.2 反向迁移更弱

从 laptop14（笔记本）迁移到 restaurant（餐馆）的三组比从 restaurant（餐馆）迁移到 laptop14（笔记本）更弱，尤其 `laptop14 -> rest14`。这说明源域 laptop14 对餐馆域的覆盖不足，或者当前高精度筛选策略牺牲了太多目标域召回。

### 5.3 增强数据数量不稳定

六组增强选入数量分别为 150、88、111、150、130、150。`rest15 -> laptop14` 只选入 88 条，说明该方向生成和过滤通过率较低，可能拖累最终结果。

## 6. 结果文件

- 六组汇总 CSV（表格文件）：`runs\bgca_aste_stage1_baseline\results_bgca_aste_stage1.csv`
- 六组汇总中文文档：`runs\bgca_aste_stage1_baseline\results_bgca_aste_stage1_CN.md`
- 项目总账：`实验记录与模型索引_CN.md`

## 7. 后续建议

1. 做严格消融：`source + pseudo`、`source + pseudo + augment`、`source + pseudo + DANN`、`source + pseudo + augment + DANN`，判断提升主要来自哪个模块。
2. 针对召回率低的问题，尝试更高召回的伪标签混合策略，例如 `high_precision`（高精度）伪标签 + 少量 recall_extra（召回补充）伪标签。
3. 针对 restaurant（餐馆）目标域，检查目标域中多三元组、neutral（中性）和长距离 aspect-opinion（方面词-观点词）样本的漏抽情况。
4. 与 BGCA 做更公平对齐时，需要单独跑纯 `label_to_text`（标签到文本）增强、不加 DANN（领域对抗）、不加额外高精度过滤的版本。

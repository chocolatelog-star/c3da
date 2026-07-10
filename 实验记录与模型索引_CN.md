# CD-C3DA 实验记录与模型索引

本文档是项目实验总账，配合 git（版本管理）记录代码版本、实验输出、模型路径和阶段结论。维护原则：优先整体更新表格和结论，不在文档末尾无限追加。

## 0. 当前总览

| 项目 | 内容 |
|---|---|
| GitHub（代码托管平台） | https://github.com/chocolatelog-star/c3da.git |
| 当前代码版本 | `5d12da3 Add semantic same-sentiment opinion augmentation` |
| 当前分支 | `master` |
| 主实验目录 | `runs\bgca_aste_stage1_baseline` |
| 当前六组实验状态 | 已完成 |
| 生成器训练方式 | `label_to_text`（标签到文本） |
| 数据增强方式 | `masked_mutual`（掩码增强） |
| 最终训练配置 | 高精度伪标签 + 严格增强 + DANN（领域对抗） |
| 主对比指标 | `raw F1`（原始 F1） |
| 辅助分析指标 | `fixed F1`（修正 F1） |

## 0.1 下一阶段：领域感知语义增强

当前计划在既有最佳流程上继续改数据增强，不改伪标签筛选、DANN（领域对抗）和最终训练参数，也暂时不加入 contrastive learning（对比学习）。目标是验证“显式目标领域提示 + 同情感语义观点替换”能不能让 `masked_mutual`（互相掩码）增强更贴近目标域，同时减少观点词和情感极性不相容的问题。

| 实验项 | 配置 |
|---|---|
| 基础流程 | `label_to_text`（标签到文本）生成器训练 + `masked_mutual`（互相掩码）增强 + high_precision pseudo（高精度伪标签）+ DANN（领域对抗） |
| 领域前缀参数 | `--domain_prefix_style text`（文本式领域前缀） |
| 观点词替换参数 | `--opinion_replacement_mode semantic_same_sentiment`（同情感语义替换） |
| 默认值 | `--opinion_replacement_mode coupled_random`（旧版成对随机替换，保持历史流程可复现） |
| text（文本式）前缀 | `target domain: [laptop14] ; masked aspect edit: ...` |
| 观点词替换逻辑 | 在同一 sentiment（情感极性）内，从目标域高精度伪标签和跨域 memory（记忆库）中选择更匹配当前 aspect（方面词）的 opinion（观点词） |
| 第一轮验证 | 先跑 `rest16 -> laptop14`，与当前 text 前缀结果 raw F1=46.63、fixed F1=48.98，以及六组基线 raw F1=46.14、fixed F1=47.69 对比 |

## 0.2 代码版本变更记录

这个表专门记录代码改动和对应 git（版本管理）版本，用来以后回溯、对比和复现实验。原则是：每次改代码后都必须记录 commit（提交号）、改动内容、影响文件、对应实验目录和结果状态；实验效果不如当前最好结果时，先询问是否删除历史输出文件。

| 时间 | git commit（提交号） | 改动主题 | 改动文件 | 改动说明 | 对应实验/输出位置 | 结果状态 |
|---|---|---|---|---|---|---|
| 2026-07-10 | `5d12da3 Add semantic same-sentiment opinion augmentation` | 领域感知语义增强 | `t5_aste_augment.py`、`t5_aste_pipeline.py`、`run_bgca_aste_stage1_pairs.py`、`test_masked_mutual_augment.py`、`实验记录与模型索引_CN.md`、`CD_C3DA_BGCA超越目标路线图_CN.md` | 新增 `--opinion_replacement_mode`（观点词替换模式），支持 `coupled_random`（旧版成对随机替换）和 `semantic_same_sentiment`（同情感语义替换）；新模式只在同一情感极性内替换 opinion（观点词），并按目标域出现频率、aspect-opinion（方面词-观点词）共现、目标域三元组共现和词面相似度排序；批量脚本使用独立 tag 保存新实验，避免覆盖旧结果；本轮暂不加入 contrastive learning（对比学习）。 | `runs\bgca_aste_stage1_semantic_opinion_text_v1` | 待跑 `rest16 -> laptop14` 单组验证 |
| 2026-07-09 | `9e78904 Add domain prefix augmentation experiments` | 领域前缀掩码增强 | `t5_aste_augment.py`、`t5_aste_pipeline.py`、`run_bgca_aste_stage1_pairs.py`、`test_masked_mutual_augment.py`、`实验记录与模型索引_CN.md`、`CD_C3DA_BGCA超越目标路线图_CN.md` | 新增 `--domain_prefix_style`（领域前缀风格），支持 `none`（不加前缀）、`text`（文本式前缀）、`bracket`（括号式前缀）；生成器训练阶段用源域前缀，数据增强阶段用目标域前缀；增强请求和增强样本记录领域前缀元数据；过滤生成文本中的领域前缀提示词泄漏；批量脚本支持断点续跑两种前缀实验。 | `runs\bgca_aste_stage1_domain_prompt_text_v1`、`runs\bgca_aste_stage1_domain_prompt_bracket_v1` | 待跑 `rest16 -> laptop14` 单组对比，基线为 raw F1=46.14、fixed F1=47.69 |
| 2026-07-09 | `07e589f Update BGCA six-pair experiment results` | 六组 BGCA 风格跨域结果同步 | `实验记录与模型索引_CN.md` 等文档 | 汇总六组跨域实验，与 BGCA 论文 `label-to-text`（标签到文本）结果放在开头对比；记录平均 raw F1=49.86、fixed F1=51.59。 | `runs\bgca_aste_stage1_baseline` | 已完成六组实验 |
| 2026-07-09 | `de435be Clarify generator and augment prompt styles` | 明确生成器训练方式和增强方式 | `run_bgca_aste_stage1_pairs.py`、相关文档 | 明确当前主流程是 `label_to_text`（标签到文本）生成器训练 + `masked_mutual`（互相掩码）数据增强，避免误认为完全等同 BGCA 原始流程。 | `runs\bgca_aste_stage1_baseline` | 已作为当前阶段基线 |
| 2026-07-09 | `4906184 Add experiment tracking log` | 建立实验总账 | `实验记录与模型索引_CN.md` | 新增中文实验记录与模型索引，用来记录实验结果、模型路径、运行目录和阶段结论。 | 项目根目录文档 | 已启用 |

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

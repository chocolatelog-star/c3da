# CD-C3DA 实验记录与模型索引

本文档是项目实验总账，配合 git（版本管理）记录代码版本、实验输出、模型路径和阶段结论。维护原则：优先整体更新表格和结论，不在文档末尾无限追加。

## 0. 当前总览

| 项目 | 内容 |
|---|---|
| GitHub（代码托管平台） | https://github.com/chocolatelog-star/c3da.git |
| 当前代码版本 | `c94a094 Apply contextual contrastive learning end to end` |
| 当前分支 | `master` |
| 主实验目录 | `runs\bgca_aste_stage1_baseline` |
| 当前六组实验状态 | 已完成 |
| 生成器训练方式 | `label_to_text`（标签到文本） |
| 数据增强方式 | `masked_mutual`（掩码增强） |
| 最终训练配置 | 高精度伪标签 + 严格增强 + DANN（领域对抗） |
| 主对比指标 | `raw F1`（原始 F1） |
| 辅助分析指标 | `fixed F1`（修正 F1） |

## 0.1 当前阶段：全流程编码器情感原型对比学习

第二版只在最终 5 轮加入解码器观点原型对比学习，得到 raw F1=46.82、fixed F1=48.94，召回率提高但中性原型准确率最终为 0%。当前升级为完整流程：先用预训练 T5 编码器对源域句子中的观点跨度生成上下文表示并初始化正/负/中性原型，再从 25 轮初始提取器开始使用类别平衡对比损失；随后重新生成伪标签、训练生成器、做掩码增强，并在最终 5 轮模型中再次使用编码器上下文原型。对比学习只使用源域人工标注，伪标签和增强数据不参与该损失。

| 实验项 | 配置 |
|---|---|
| 基础流程 | `label_to_text`（标签到文本）生成器训练 + `masked_mutual`（互相掩码）增强 + high_precision pseudo（高精度伪标签）+ DANN（领域对抗） |
| 领域前缀参数 | `--domain_prefix_style text`（文本式领域前缀） |
| 观点词替换参数 | `--opinion_replacement_mode sentiment_vector`（情感向量替换） |
| 情感向量模型 | `--sentiment_vector_model_path J:\nlp\models\t5-base-py` |
| 情感间隔参数 | `--sentiment_vector_min_margin 0.05`（候选词更接近本情感原型而不是相反情感原型的最小间隔） |
| 通道比例参数 | `--augment_select_max_opinion_ratio 0.6`（观点词通道最多 60%，方面词通道尽量 40%） |
| 默认值 | `--opinion_replacement_mode coupled_random`（旧版成对随机替换，保持历史流程可复现） |
| text（文本式）前缀 | `target domain: [laptop14] ; masked aspect edit: ...` |
| 观点词替换逻辑 | 对候选 opinion（观点词）编码，计算 pos/neg/neu 情感原型；候选必须同情感、靠近本情感原型、远离其他情感原型，并结合 old opinion similarity（原观点词相似度）和 aspect-opinion 共现打分 |
| v1 结果处理 | `runs\bgca_aste_stage1_semantic_opinion_text_v1` 已删除；该版本 raw F1=45.46、fixed F1=47.26，低于当前最好结果 |
| v2 结果处理 | v2 4:6 版本输出已删除；该版本 raw F1=42.78、fixed F1=44.71，低于当前最好结果 |
| 向量版第一轮验证 | 先复用当前最好 text 前缀目录，只重跑 augment（数据增强）+ final training（最终训练）+ evaluate（评估），与 raw F1=46.63、fixed F1=48.98 对比 |
| GloVe 文件 | `J:\models\glove.6B.300d.txt`，300 维，大小 1,037,965,801 字节 |
| GloVe 极性轴实验 | 覆盖 827/851 个观点词（97.18%）；候选由 633 降至 493；单组结果 raw F1=44.70、fixed F1=46.82，低于最佳 46.63/48.98，完整输出和模型已删除 |
| 极性轴阈值 | pos >= 0.005817；neg <= -0.024779；neutral abs <= 0.177709 |
| 观点过滤 | 原观点相似度至少 0.35；无方面-观点共现时至少 0.50；提取器回抽观点相似度至少 0.45，并要求极性一致 |
| 第一版对比学习结果 | `lambda=0.05`，raw F1=43.92、fixed F1=46.72，低于最佳 46.63/48.98；模型、检查点和专属指标已删除 |
| 第二版对比学习结果 | `lambda=0.01`、仅最终阶段、解码器观点表示：raw F1=46.82、fixed F1=48.94；raw 比旧最佳高 0.19，但中性原型准确率为 0%，保留为候选而非正式结论 |
| 当前对比学习参数 | 初始提取器与最终模型均使用 `lambda=0.01`、仅源域、类别平衡、编码器上下文原型初始化 |
| 当前对比学习样本 | 857 句、1393 个源域三元组，编码器观点跨度覆盖率 100%；pos=1015、neg=328、neu=50；类别权重约 0.413/0.726/1.861 |
| 当前完整实验目录 | `runs\bgca_aste_stage1_full_contrastive_encoder_v1\rest16_to_laptop14` |
| 阶段判断标准 | 新提取器高精度伪标签 F1 对比 50.35；最终 raw F1 对比 46.82，fixed F1 对比 48.94 |

## 0.2 代码版本变更记录

这个表专门记录代码改动和对应 git（版本管理）版本，用来以后回溯、对比和复现实验。原则是：每次改代码后都必须记录 commit（提交号）、改动内容、影响文件、对应实验目录和结果状态；实验效果不如当前最好结果时，先询问是否删除历史输出文件。

| 时间 | git commit（提交号） | 改动主题 | 改动文件 | 改动说明 | 对应实验/输出位置 | 结果状态 |
|---|---|---|---|---|---|---|
| 2026-07-13 | `c94a094 Apply contextual contrastive learning end to end` | 全流程编码器上下文对比学习 | `t5_absa_train.py`、`run_bgca_aste_stage1_pairs.py`、`test_domain_adversarial_train.py` | 预训练 T5 编码器先生成源域观点上下文表示并初始化三类原型；观点跨度大小写兼容定位达到 1393/1393；25 轮初始提取器和最终 5 轮模型都使用源域类别平衡对比损失；伪标签、生成器和掩码增强全部重跑；各阶段支持检查点续训并使用独立目录 | `runs\bgca_aste_stage1_full_contrastive_encoder_v1\rest16_to_laptop14` | 待跑完整流程 |
| 2026-07-13 | `f809beb Balance and instrument sentiment contrastive loss` | 对比学习第二版 | `t5_absa_train.py`、`run_bgca_aste_stage1_pairs.py`、`test_domain_adversarial_train.py` | 对比损失降至 0.01；仅源域人工标注参与；按平方根倒频率平衡正负中性；训练日志分别输出生成、领域对抗、对比损失和各类原型准确率；仅在最终 5 轮使用解码器观点表示 | `runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_sentiment_contrastive_l001_source_balanced_ep5` | raw F1=46.82、fixed F1=48.94；召回提高，但中性原型准确率为 0% |
| 2026-07-12 | `f0e8dc6 Add sentiment prototype contrastive training` | 情感原型对比学习第一版 | `t5_absa_train.py`、`run_bgca_aste_stage1_pairs.py`、`test_domain_adversarial_train.py` | 从每个三元组的观点词解码器表示计算三类可训练原型损失；支持 batch size=1；排除增强数据；源域和高精度伪标签参与；权重 0.05 | 完整模型、检查点和专属指标已删除 | raw F1=43.92、fixed F1=46.72；精确率与召回率同时下降 |
| 2026-07-11 | `33c76e3 Add polarity-aware opinion augmentation` | GloVe 极性轴与观点一致性过滤 | `t5_aste_augment.py`、`t5_aste_pipeline.py`、`run_bgca_aste_stage1_pairs.py`、`test_masked_mutual_augment.py` | 用 pos-neg 差向量建立极性轴并从带权样本分布估计阈值；增加原观点相似度、无共现严格阈值；观点通道模型过滤新增观点相似度和极性检查；新实验使用独立 `_polarity_axis` 标签 | 保留 `sentiment_vector_diagnostics_glove_polarity_diag_v2.json`；完整增强输出与最终模型已删除 | raw F1=44.70、fixed F1=46.82，低于当前最佳；极性约束有效但方面词-观点词搭配仍不自然 |
| 2026-07-11 | `38923fb Add GloVe sentiment vector backend` | GloVe 情感向量后端 | `t5_aste_pipeline.py`、`run_bgca_aste_stage1_pairs.py`、`test_masked_mutual_augment.py`、`实验记录与模型索引_CN.md`、`CD_C3DA_BGCA超越目标路线图_CN.md` | 新增 `--sentiment_vector_backend glove`、`--glove_path` 和 `--sentiment_vector_diagnostics_only`；按需加载 GloVe 词向量，多词观点短语取平均并归一化；source gold 权重 1.0、target pseudo 权重 0.65 构建情感中心；诊断阶段不调用生成器和训练。 | `sentiment_vector_diagnostics_glove_sentiment_diag_m005.json` | 覆盖率 97.18%，但情感中心相似度过高，按预设门槛暂停训练 |
| 2026-07-10 | `4ea0f8f Add sentiment vector opinion replacement` | 显式情感向量增强 | `t5_aste_augment.py`、`t5_aste_pipeline.py`、`run_bgca_aste_stage1_pairs.py`、`test_masked_mutual_augment.py`、`实验记录与模型索引_CN.md`、`CD_C3DA_BGCA超越目标路线图_CN.md` | 新增 `--opinion_replacement_mode sentiment_vector`（情感向量替换），通过 T5 encoder embedding（T5 编码器嵌入）给候选 opinion（观点词）编码，构建 pos/neg/neu（正/负/中性）情感原型；候选词必须满足 `sentiment_margin >= --sentiment_vector_min_margin`，即更靠近本情感原型并远离其他情感原型；继续保留 `--augment_select_max_opinion_ratio 0.6` 和 opinion 边界过滤。 | `runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14` 的 `_sentiment_vector` tag | 待跑单组验证 |
| 2026-07-10 | `231c49d Constrain semantic opinion augmentation selection` | 语义观点增强 v2 | `t5_aste_pipeline.py`、`run_bgca_aste_stage1_pairs.py`、`test_masked_mutual_augment.py`、`实验记录与模型索引_CN.md`、`CD_C3DA_BGCA超越目标路线图_CN.md` | 在 v1 同情感语义替换基础上新增 `--augment_select_max_opinion_ratio`（观点词通道最大比例），本轮使用 0.6，即 opinion channel（观点词通道）最多 60%、aspect channel（方面词通道）尽量 40%；新增 opinion（观点词）边界过滤，过滤 `[opi]`、`no`、`on its feet` 这类异常观点词边界；保留增强样本中的 `opinion_replacement_mode`（观点词替换模式）元数据，方便后续分析。 | `runs\bgca_aste_stage1_semantic_opinion_text_v2` | 待跑 `rest16 -> laptop14` 单组验证 |
| 2026-07-10 | `5157c2f Add semantic same-sentiment opinion augmentation` | 领域感知语义增强 | `t5_aste_augment.py`、`t5_aste_pipeline.py`、`run_bgca_aste_stage1_pairs.py`、`test_masked_mutual_augment.py`、`实验记录与模型索引_CN.md`、`CD_C3DA_BGCA超越目标路线图_CN.md` | 新增 `--opinion_replacement_mode`（观点词替换模式），支持 `coupled_random`（旧版成对随机替换）和 `semantic_same_sentiment`（同情感语义替换）；新模式只在同一情感极性内替换 opinion（观点词），并按目标域出现频率、aspect-opinion（方面词-观点词）共现、目标域三元组共现和词面相似度排序；批量脚本使用独立 tag 保存新实验，避免覆盖旧结果；本轮暂不加入 contrastive learning（对比学习）。 | `runs\bgca_aste_stage1_semantic_opinion_text_v1` | 待跑 `rest16 -> laptop14` 单组验证 |
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

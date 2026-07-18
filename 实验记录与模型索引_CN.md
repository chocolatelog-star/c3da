# CD-C3DA 实验记录与模型索引

本文档是项目实验总账，配合 git（版本管理）记录代码版本、实验输出、模型路径和阶段结论。维护原则：优先整体更新表格和结论，不在文档末尾无限追加。

## 0. 当前总览

| 项目 | 内容 |
|---|---|
| GitHub（代码托管平台） | https://github.com/chocolatelog-star/c3da.git |
| 当前代码版本 | `62113b4 Add complete multi-triplet and strict ablations` |
| 当前分支 | `feature/complete-multitriplet-ablation` |
| 主实验目录 | `runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14` |
| 当前六组实验状态 | 已完成 |
| 生成器训练方式 | `label_to_text`（标签到文本） |
| 数据增强方式 | `masked_mutual`（掩码增强） |
| 最终训练配置 | 高精度伪标签 + 严格增强 + DANN（领域对抗）+ 源域类别平衡对比学习 |
| 主对比指标 | `raw F1`（原始 F1） |
| 辅助分析指标 | `fixed F1`（修正 F1） |

## 0.1 当前阶段：完整双三元组补充与严格模块消融已就绪

当前最佳候选仍是只在最终 5 轮加入解码器观点原型对比学习的版本：raw F1=46.82、fixed F1=48.94。编码器三元组覆盖分类头实验 raw F1=44.37、fixed F1=46.72；训练日志显示分类头从第 1 轮起几乎全部预测为单三元组，2 个和 3 个以上类别准确率长期为 0，因此该方向失败，模型已删除并保留指标。当前改为直接补充完整双三元组训练目标，同时补齐最佳数据上的严格模块消融。

| 近期改进方法 | raw F1 | 相对当前最佳 | 阶段结论 |
|---|---:|---:|---|
| 文本式领域前缀 | 46.63 | -0.19 | 比无前缀六组同方向基线 46.14 有改善，保留为主流程输入形式 |
| 括号式领域前缀 | 45.26 | -1.56 | 不如自然语言式领域前缀 |
| 同情感语义替换 v2 | 42.78 | -4.04 | 词面和同极性约束不能保证完整上下文兼容 |
| T5 显式情感向量替换 | 43.13 | -3.69 | 静态候选情感距离不能解决方面词-观点词搭配 |
| GloVe（全局词向量）极性轴 | 44.70 | -2.12 | 极性过滤有效，但仍缺少上下文和结构约束 |
| 最终阶段源域平衡情感对比 | **46.82** | **0.00** | 小权重、仅源域、仅最终阶段有效，当前最佳 |
| 全流程编码器情感对比 | 43.74 | -3.08 | 改变上游表示和伪标签分布，误差沿流程放大 |
| 高精度伪标签最多 2 个三元组 | 44.44 | -2.38 | 新增伪标签质量和类别分布不稳定，不能靠放宽数量提高召回 |
| 中性主生成损失增权 | 43.18 | -3.64 | 中性问题来自数据、边界与配对，不是损失权重不足 |
| 三任务混合生成器 | 44.07 | -2.75 | 1:1:1 混合削弱多三元组和负向结构，不进入主流程 |
| 编码器方面词-观点词配对 | 46.49 | -0.34 | 精确率提高、召回率下降；机制有效但不作为最佳模型 |
| 编码器三元组覆盖分类头 | 44.37 | -2.45 | 分类头坍缩为单三元组，未传递到自回归生成；模型已删除 |
| hp1 + 完整双三元组补充 | 待训练 | - | 数据构建完成：421+73=494 条，排除 31 条裁剪样本，伪标签隐藏金标 F1=51.08 |

| 实验项 | 配置或结论 |
|---|---|
| 当前最佳候选 | raw P=54.84、raw R=40.85、raw F1=46.82、fixed F1=48.94 |
| 当前最佳模型 | `runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_sentiment_contrastive_l001_source_balanced_ep5\best` |
| 新主实验训练集 | `final_train_strict_aug150_w020_label_to_text_gen_complete_multi2_w025.jsonl`，1499 行 |
| 新主实验伪标签 | `pseudo_variants\hp1_complete2_dist5_w025\target_pseudo_high_precision.jsonl`，494 行 |
| 新主实验数据诊断 | 基础 hp1 421 行；完整双三元组新增 73 行；裁剪样本排除 31 行；隐藏金标 raw F1 50.35→51.08 |
| 新主实验唯一变量 | 复用原 150 条严格增强，只增加 73 条完整双三元组伪标签，新增样本权重 0.25 |
| 严格消融矩阵 | A：源域+伪标签；B：A+增强；C：A+DANN；D：A+增强+DANN；E：D+最终阶段情感对比（当前最佳） |
| 严格消融输出 | `results_strict_module_ablation.csv` 和 `results_strict_module_ablation_CN.md`，四组完成后自动整体生成 |
| 新代码版本 | `722e460`（中文设计和计划）、`62113b4`（完整双三元组与严格消融实现） |
| 第五项唯一变量 | 最终 5 轮增加编码器多正例双向方面词-观点词配对损失，系数 0.01、温度 0.1，只使用源域人工标注 |
| 复用训练文件 | `final_train_strict_aug150_w020_label_to_text_gen.jsonl` 和 `final_dev_strict_aug150_w020_label_to_text_gen.jsonl` |
| 配对覆盖率 | 352/352 个源域多三元组训练行成功定位，覆盖率 100%；共 877 个有效配对 |
| 第五项模型 | 已删除；原路径为 `models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_sentiment_contrastive_l001_source_balanced_pairing_encoder_l001_source_only_ep5\best` |
| 新训练日志 | `pairing_loss`、双向配对准确率、日志周期有效行数和有效配对数 |
| 新评估 | 独立输出单三元组与多三元组子集 raw/fixed 指标 |
| 第五项结果 | raw P=55.99、raw R=39.74、raw F1=46.49、fixed F1=48.86；比最佳 raw 低 0.34 |
| 第五项结构结果 | 单三元组 raw F1 52.85→53.16；多三元组 raw F1 42.65→41.83，精确率 59.20→60.64，召回率 33.33→31.93 |
| 第五项结论 | 配对损失减少错误组合，但造成少生成；后续应加入覆盖或召回目标，不提高当前配对损失权重 |
| 第四项唯一变量 | 生成器由纯 `label_to_text` 改为三任务 `mixed`；提取器、hp1 伪标签、150 条增强、增强权重 0.20、DANN 系数 0.03 和最终对比学习保持不变 |
| 固定上游来源 | `runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14`；只读复用 `extractor_ep25_plain_last\best`、原始伪标签和 hp1 高精度伪标签 |
| 真实训练通道 | 857 条标签到文本 + 857 条方面词掩码 + 857 条观点词-情感掩码，共 2571 条，比例严格为 1:1:1 |
| 真实验证通道 | 210 条标签到文本 + 210 条方面词掩码 + 210 条观点词-情感掩码，共 630 条，比例严格为 1:1:1 |
| 第四项生成器模型 | 已删除；原路径为 `runs\bgca_aste_stage1_mixed_generator_v1\rest16_to_laptop14\models\generator_mixed_l2t_masked_aspect_masked_opinion_ep8\best` |
| 第四项最终模型 | 已删除；原路径为 `runs\bgca_aste_stage1_mixed_generator_v1\rest16_to_laptop14\models\final_dann_l0.03_strict_aug150_w020_mixed_l2t_masked_aspect_masked_opinion_sentiment_contrastive_l001_source_balanced_ep5\best` |
| 断点续跑 | 相同总命令会跳过已完成阶段；生成器和最终训练使用 `--resume_from_checkpoint auto`（自动从检查点恢复）；提取器不重训 |
| 第四项结果 | raw P=50.97、raw R=38.82、raw F1=44.07、fixed F1=46.06；比最佳 raw 低 2.75 |
| 第四项结构诊断 | 选中增强的多三元组行从 63 降至 23，三元组总数从 229 降至 177；方面词通道占比从 77.3% 升至 86.0% |
| 第四项生成器诊断 | 验证损失第 2 轮最低 0.7038，随后升至约 0.8884；一致性和回抽通过率提高，但 NLI 保留率从 88.0% 降至 75.9% |
| 第四项结论 | 训练-增强输入对齐提高了局部标签一致性，却削弱全局、多三元组和负向建模；1:1:1 混合任务不进入最佳流程 |
| 全流程编码器对比 | raw F1=43.74、fixed F1=45.70，失败 |
| 新上游数据但最终无对比 | raw F1=42.20、fixed F1=44.40，说明新上游数据是主要负面来源 |
| 旧数据加编码器对比 | raw F1=45.37、fixed F1=48.35，说明编码器对比损失也有负贡献 |
| 完整 T5 情感向量流程 | raw F1=40.82、fixed F1=43.01，不继续沿该方向调参 |
| 第二项结果 | `hp2_dist5`：raw P=53.37、raw R=38.08、raw F1=44.44、fixed F1=46.87，低于最佳 2.38 |
| 第二项诊断 | 新增 104 行；73 条完整双三元组 F1=53.51，31 条被裁剪成单三元组的样本 F1=39.56；新增分布正向 141、负向 36、中性 0 |
| 中性伪标签诊断 | 原始 904 条目标伪标签只有 2 个中性预测，隐藏金标分析显示二者实际均为负向，因此不降低阈值、不设置强制配额 |
| 第三项单变量 | `hp1` 训练集不变；只把含中性三元组样本的主生成损失提高，非中性主生成权重保持原值 |
| 新参数 | `--neutral_generation_loss_gain 1.0 --neutral_generation_max_effective_weight 2.0` |
| 实际权重 | 49 个中性行平均主生成权重 1.6408、范围 0.4 到 2.0；1377 个非中性行最大权重仍为 1.0 |
| 复用范围 | 复用 `hp1` 的提取器、伪标签、生成器、增强数据和最终训练集，只重跑最终 5 轮训练与评估 |
| 第三项失败模型 | `models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_sentiment_contrastive_l001_source_balanced_neutral_gain100_max200_ep5\best`（已删除） |
| 新诊断 | 额外输出正向/负向/中性分项 raw/fixed 指标，以及中性否定误判统计与样例 |
| 新汇总文件 | `results_bgca_aste_stage1_neutral_gain100_max200.csv`、`results_bgca_aste_stage1_neutral_gain100_max200_CN.md` |
| 隔离保证 | 模型、指标、预测、分类指标、错误分析和汇总均带 `neutral_gain100_max200` 标签 |
| 续跑保证 | 最终训练使用 `--resume_from_checkpoint auto` 自动续训；评估完成需同时存在总体、分类和错误分析文件 |
| 判定标准 | 总体 raw F1 与 46.82 比较，同时要求中性 F1 或中性召回获得实质改善 |
| 代码版本 | `e7560c7`（混合数据）、`e320fab`（通道统计）、`925d596`（完整流程）、`9f3c4db`（固定上游复用） |
| 第三项总体结果 | raw P=51.54、raw R=37.15、raw F1=43.18、fixed F1=45.76；比最佳 raw 低 3.64 |
| 第三项分类结果 | 正向 raw F1 49.24→45.06；负向 52.86→50.46；中性 0→0；新增 2 个中性预测但均未完整匹配 |
| 第三项结论 | 中性主生成损失增权没有建立目标域中性边界，反而破坏正负类别；不继续调中性权重 |
| 清理状态 | `hp2` 与中性增权失败模型、检查点、大型增强/训练数据和专属预测已删除，释放约 12.52 GB；指标与分析保留 |
| 清理状态 | 第四项失败生成器和最终模型已删除，释放约 11.65 GB；预测、指标、增强数据和分析保留 |
| 第五项代码版本 | `c1082ab`（编码器跨度）、`123ab39`（多正例损失）、`d87e871`（结构评估）、`6075ee0`（单变量路由）、`a358efa`（日志计数）、`2b35461`（GPU 安全关系掩码） |
| 第五项清理状态 | 失败最终模型已删除，释放约 5.83 GB；指标、预测、结构分析和代码历史保留 |
| 当前状态 | 第五项已完成；最佳模型不变，下一阶段优先解决多三元组召回和中性样本数据覆盖 |

完整执行命令（从 CMD（命令提示符）开始，整行执行）：

```text
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_domain_prompt_text_v1 --pairs rest16:laptop14 --extractor_model_path J:\nlp\models\t5-base-py --generator_model_path J:\nlp\models\t5-base-py --generator_prompt_style label_to_text --augment_prompt_style masked_mutual --domain_prefix_style text --final_epochs 5 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --lambda_pairing_loss 0.01 --pairing_temperature 0.1 --pairing_source_only --learning_rate 0.0003 --eval_batch_size 2 --cuda 0 --seed 1000"
```

完整执行命令（从 CMD（命令提示符）开始，整行执行）：

```text
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_mixed_generator_v1 --pairs rest16:laptop14 --reuse_upstream_run_dir runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14 --extractor_model_path J:\nlp\models\t5-base-py --generator_model_path J:\nlp\models\t5-base-py --generator_prompt_style mixed --augment_prompt_style masked_mutual --domain_prefix_style text --extractor_epochs 25 --generator_epochs 8 --final_epochs 5 --high_precision_max_triplets 1 --high_precision_max_token_distance 5 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --learning_rate 0.0003 --eval_batch_size 2 --cuda 0 --seed 1000"
```

## 0.2 代码版本变更记录

这个表专门记录代码改动和对应 git（版本管理）版本，用来以后回溯、对比和复现实验。原则是：每次改代码后都必须记录 commit（提交号）、改动内容、影响文件、对应实验目录和结果状态；实验效果不如当前最好结果时，先询问是否删除历史输出文件。

| 时间 | git commit（提交号） | 改动主题 | 改动文件 | 改动说明 | 对应实验/输出位置 | 结果状态 |
|---|---|---|---|---|---|---|
| 2026-07-16 | `c1082ab`、`123ab39`、`d87e871`、`6075ee0`、`a358efa`、`2b35461` | 编码器上下文多正例配对损失 | `t5_absa_train.py`、`t5_aste_pipeline.py`、`run_bgca_aste_stage1_pairs.py`、4 个测试文件 | 从原始输入定位方面词和观点词；共享跨度使用多正例双向对比；只用源域多三元组；新增配对日志和结构子集评估；严格复用当前最佳训练集，只重跑最终模型 | `runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14` 下带 `_pairing_encoder_l001_source_only` 的独立指标 | 已完成：raw F1=46.49、fixed F1=48.86；多三元组精确率提高但召回下降；失败模型已删除，释放约 5.83 GB |
| 2026-07-15 | `e7560c7`、`e320fab`、`925d596`、`9f3c4db` | 三任务混合生成器训练 | `t5_aste_augment.py`、`t5_aste_pipeline.py`、`run_bgca_aste_stage1_pairs.py`、3 个测试文件 | 新增每个源句每通道最多一行的混合训练；清单记录通道数量和比例；新目录只读复用旧最佳提取器与 hp1 伪标签；新生成器、增强、最终模型和汇总保持隔离；保留 8GB 显存参数和自动续训 | `runs\bgca_aste_stage1_mixed_generator_v1\rest16_to_laptop14` | 已完成：raw F1=44.07、fixed F1=46.06；多三元组增强 63→23；失败模型已删除，释放约 11.65 GB |
| 2026-07-15 | `0c49ba6 Add isolated neutral generation weighting` | 中性主生成损失独立增权 | `t5_absa_train.py`、`t5_aste_pipeline.py`、`run_bgca_aste_stage1_pairs.py`、3 个测试文件 | 新增只作用于中性样本主生成损失的增益和专用上限；不改变结构损失和非中性多三元组权重；评估新增三类情感指标及中性否定误判；复用 hp1 最终训练集 | 指标与错误分析保留；模型和专属预测已删除 | 已完成：raw F1=43.18、fixed F1=45.76，中性 F1=0，低于最佳 3.64 |
| 2026-07-15 | `6f2dcd3 Add isolated two-triplet pseudo-label experiment` | 高精度伪标签最多两个三元组消融 | `run_bgca_aste_stage1_pairs.py`、`t5_aste_pipeline.py`、3 个测试文件 | 新增可配置三元组上限；从原始预测重筛到独立目录；复用旧提取器和生成器；隔离增强、训练集、模型、指标、预测、分析及汇总；兼容旧阶段状态并支持中断续跑 | `runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14\pseudo_variants\hp2_dist5` 及带 `_hp2_dist5` 的输出 | 已完成：raw F1=44.44、fixed F1=46.87，低于最佳 2.38 |
| 2026-07-13 | `c94a094 Apply contextual contrastive learning end to end` | 全流程编码器上下文对比学习 | `t5_absa_train.py`、`run_bgca_aste_stage1_pairs.py`、`test_domain_adversarial_train.py` | 预训练 T5 编码器先生成源域观点上下文表示并初始化三类原型；观点跨度大小写兼容定位达到 1393/1393；25 轮初始提取器和最终 5 轮模型都使用源域类别平衡对比损失；伪标签、生成器和掩码增强全部重跑；各阶段支持检查点续训并使用独立目录 | `runs\bgca_aste_stage1_full_contrastive_encoder_v1\rest16_to_laptop14` | 已完成：raw F1=43.74、fixed F1=45.70，低于最佳候选 |
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

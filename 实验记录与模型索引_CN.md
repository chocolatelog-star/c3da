# CD-C3DA 实验记录与模型索引

本文档是 `J:\nlp\CD-C3DA` 的当前实验总览。维护原则：每次改代码或跑完实验后，优先整体更新本页开头的当前状态、差距表和决策表，不在文档末尾无限追加流水账。

## 1. 当前一眼结论

| 项目 | 当前值 |
|---|---|
| 当前主攻方向 | `rest16 -> laptop14` |
| 当前数值最佳流程 | 当前 `hp1 + complete_multi2_w025` 伪标签 + 历史实际增强150条 + DANN（领域对抗）+ sentiment contrastive（情感对比学习）；因依赖历史增强产物，尚不是从头可复现主线 |
| 当前最佳 raw F1（原始F1） | **49.01**（历史实际增强150条混合诊断） |
| 当前最佳 fixed F1（修正F1） | **51.83**（历史实际增强150条混合诊断） |
| 历史产物组合最佳 | raw F1 **48.93** / fixed F1 **50.21**；由7月9日旧上游产物与7月20日最终组合形成，不是单提交从头流程 |
| 旧代码复现 fixed F1（修正F1） | **50.26**（`c0b2730` full rerun，raw F1 为 47.58） |
| BGCA 论文 label-to-text F1（标签到文本F1） | **47.28** |
| 当前 raw F1（原始F1）相对 BGCA | **+1.73** |
| 当前主要短板 | 历史增强能恢复性能，但当前流程不能稳定自动生成同质量增强；neutral（中性）召回仍为0 |
| 最新验证结论 | 只把当前增强150条替换为历史实际增强150条，raw F1 从47.98升到 **49.01（+1.03）**，fixed F1 从50.88升到 **51.83（+0.95）**；增强内容是此前剩余差距的主要原因 |
| 历史版本审计结论 | 48.93结果虽然在7月20日由`8c7f6b4`附近代码组合得到，但其中抽取器、基础伪标签和增强实际创建于7月9日，对应代码更接近`9e78904`；7月20日没有从头重建增强，而是复用旧上游产物 |
| 最新完整复现结论 | legacy stochastic（旧式随机）完整流程得到 raw F1 **45.68** / fixed F1 **47.86**。抽取器、8轮生成器、421条基础伪标签、494条补全伪标签均与历史文件 SHA256（文件哈希）完全一致；本次增强与历史真正进入48.93训练集的150条中有147条文本和标签相同，仅3条不同，但最终模型仍明显掉分 |
| 关键哈希证据 | 本次最终训练集 SHA256 为 `71F5948F...9F5DBEF`，最终模型为 `613C4E6A...8C0F1FF`，与此前 `bgca_aste_stage1_best_full_rerun_v1` 完全一致；说明旧式随机模式准确复现的是7月21日覆盖后的45.68流程，不是7月20日的48.93最终训练轨迹 |
| 当前正在验证 | 用真实历史代码边界从头复现48.93：`9e78904`负责抽取器、生成器、基础伪标签和增强，`8c7f6b4`负责完整双三元组补充、最终训练和评估 |
| 当前运行状态 | 双工作树两阶段入口已完成试运行；上游以`9e78904`为历史基线并叠加`a7e7778`检查点恢复兼容补丁；下游以`8c7f6b4`为训练代码基线并固定到仅新增复现命令的`a7d1473`；正式GPU（图形处理器）训练尚未启动 |
| 进度与日志 | 入口为 `run_historical_best_two_stage.ps1`；输出位于 `runs\historical_best_two_stage_v1\<source>_to_<target>`，状态、日志和哈希清单分别为 `stage_status.json`、`logs` 和 `manifest.json` |
| GitHub 推送状态 | 2026-07-25 已重新整理全部本地提交；HTTPS（加密网页传输）因 `github.com:443` TCP（传输控制协议）不可达而三次失败，SSH（安全外壳）443可达但本机无公钥授权；九个分支均完整保留在本地，网络恢复后统一重推 |
| 项目空间状态 | 2026-07-24 已删除 76 个已有 `best` 的历史 `checkpoint-*`（中间检查点），释放约 **189.67 GB**；保留全部最佳模型、指标、数据、清单，以及活动实验和未完成 `ge20` 的 6 个可恢复检查点（约14.94 GB） |
| full rerun（完整重跑）掉分原因 | 从零重跑会重新训练 extractor（提取器）和 generator（生成器），导致 base pseudo（基础伪标签）从历史 421 条变为 385 条、complete pseudo（补全伪标签）从 494 条变为 457 条、final_train（最终训练集）从 1499 条变为 1463 条 |
| 最新完整重跑诊断 | 本次 base pseudo（基础伪标签）414 条、complete pseudo（补全伪标签）488 条、final_train（最终训练集）1494 条；完整伪标签隐藏金标 F1 为 **51.25**，略高于历史 **51.08**，但选中增强与历史仅重合 **2/150**，最终模型精确率明显下降 |
| 强确定性单变量结论 | 固定同一份1494条 final_train（最终训练集），只关闭强确定性后 raw F1 从46.76升到47.98（+1.22），fixed F1从49.32升到50.88（+1.56）；强确定性是掉分因素之一 |
| 混合增强诊断结论 | 相对当前非强确定性模型：TP +4、FP -8、FN -4；相对历史48.93模型：TP +7、FP +20、FN -7，表现为更高召回但更低精确率。因此数值已恢复，但不是历史模型逐项复现 |
| 细分收益 | 单三元组 raw F1 从51.68升到53.09（+1.41），多三元组从45.49升到46.23（+0.75）；负向F1从50.00升到53.51（+3.51），正向仅+0.31，中性仍为0 |

## 2. 当前最佳与 BGCA 对比

主指标使用 raw F1（原始F1）。fixed F1（修正F1）只作为辅助分析。

| 方法 | 生成器 | 关键模块 | raw P（原始精确率） | raw R（原始召回率） | raw F1（原始F1） | fixed F1（修正F1） | 相对 BGCA raw F1 |
|---|---|---|---:|---:|---:|---:|---:|
| BGCA 论文 label-to-text（标签到文本） | T5-base，25 轮 last（最后检查点） | data generation（数据生成）+ model filter（模型过滤） | - | - | **47.28** | - | 0.00 |
| 我们当前数值最佳 | T5-base，8 轮 best（最优检查点） | 当前完整伪标签 + 历史实际增强150条 + DANN（领域对抗）+ 情感对比 | 56.22 | 43.44 | **49.01** | **51.83** | **+1.73** |
| 旧代码复现 | T5-base，8 轮 best（最优检查点） | `c0b2730` 旧代码 full rerun（完整重跑），同一最好流程 | 53.72 | 42.70 | **47.58** | **50.26** | **+0.30** |
| 精确回溯成功 | T5-base，8 轮 best（最优检查点） | `8c7f6b4` 旧代码 + 2026-07-20 历史 final_train（最终训练集） | 58.31 | 42.14 | **48.93** | **50.21** | **+1.65** |
| 完整确定性重跑 | T5-base，8 轮 best（最优检查点） | seed 1000（随机种子1000）+ deterministic mode（确定性模式），全流程从头生成 | 53.72 | 41.40 | **46.76** | **49.32** | **-0.52** |
| 旧式随机完整重跑 | T5-base，8 轮 best（最优检查点） | legacy stochastic（旧式随机）模式从头运行；复现7月21日覆盖后的上游与最终模型 | 54.04 | 39.56 | **45.68** | **47.86** | **-1.60** |
| 最终模型单变量 | T5-base，8 轮 best（最优检查点） | 固定完整重跑生成的1494条训练数据，只关闭强确定性并重训最终模型 | 54.74 | 42.70 | **47.98** | **50.88** | **+0.70** |
| 增强内容单变量 | T5-base，8 轮 best（最优检查点） | 当前完整伪标签 + 历史实际增强150条，非强确定性最终模型重训 | 56.22 | 43.44 | **49.01** | **51.83** | **+1.73** |
| 当前可复现主线（待跑） | T5-base，8 轮 best（最优检查点） | `9e78904`旧上游 + `8c7f6b4`双三元组和最终训练；全阶段重新生成 | 待跑 | 待跑 | **待跑** | **待跑** | 待跑 |
| 最新对照 | T5-base，25 轮 last（最后检查点） | 生成器 25 轮 last + 增强改为 label_to_text（标签到文本），其余保持当前最佳完全一致 | 55.47 | 41.22 | **47.30** | **50.37** | **+0.02** |
| 历史完整流程主线 | T5-base，8 轮 best（最优检查点） | 历史完整流程，不带多三元组配额 | 58.31 | 42.14 | **48.93** | **50.21** | **+1.65** |
| 轮次 sweep（扫轮次）当前最好 | T5-base，16 轮 best（最优检查点） | 固定当前最好版，只扫生成器轮次，已完成到 18，20/22 曾中断 | 54.17 | 40.85 | **46.58** | **48.31** | **-0.70** |

当前数值最佳模型（使用历史增强产物，不等同于从头可复现主线）：

```text
runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1\seed1000\rest16_to_laptop14\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw065_hist_actual_aug150_sentiment_contrastive_l001_source_balanced_nondeterministic_final_retrain_ep5\best
```

当前最佳结果文件：

```text
runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1\seed1000\rest16_to_laptop14\aste_metrics_raw_strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw065_hist_actual_aug150_sentiment_contrastive_l001_source_balanced_nondeterministic_final_retrain.json
runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1\seed1000\rest16_to_laptop14\aste_metrics_fixed_strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw065_hist_actual_aug150_sentiment_contrastive_l001_source_balanced_nondeterministic_final_retrain.json
runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1\logs\historical_augment_hybrid_seed1000_manifest.json
```

## 3. 当前数值最佳流程组成

| 阶段 | 当前数值最佳做法 |
|---|---|
| 基础模型 | 抽取器、生成器、最终模型均从 `J:\nlp\models\t5-base-py` 启动 |
| 抽取器 | 源域 text -> triplet（文本到三元组），25 轮，plain last（普通最后检查点） |
| 伪标签 | 目标域无标签句子生成伪标签，先取 hp1（最多 1 个三元组，距离 5） |
| 多三元组补充 | 在 hp1 基础上生成 `pseudo_variants\hp1_complete2_dist5_w025\target_pseudo_high_precision.jsonl`；该文件直接进入最终训练，额外双三元组权重 `complete_multi_extra_weight=0.25` |
| 生成器 | `label_to_text`（标签到文本），8 轮，`checkpoint_selection=best`（按验证损失取最优） |
| 增强请求来源 | `augment()`固定读取基础伪标签阶段的 `target_pseudo_selected.jsonl`，不是读取 `hp1_complete2_dist5_w025` 文件；历史该文件共 634 条，其中单三元组 497 条、双三元组 112 条、三三元组 25 条 |
| 增强 | `masked_mutual`（互相掩码增强），对 `target_pseudo_selected` 中的一个三元组做方面词或观点词-情感替换，并保留同一句中的其余三元组；因此增强输出可以是多三元组，但它不是 complete_multi2（完整双三元组补充）文件生成的增强 |
| 增强筛选 | 经过文本质量、一致性、NLI（自然语言推断）和抽取器回抽过滤后严格选 150 条，增强权重 0.20；历史最终训练实际使用的 150 条中有 63 条多三元组增强 |
| 最终训练 | 当前重跑源域 gold（真实标签）857 条 + complete_multi2（完整双三元组伪标签）488 条（去重后487条）+ 历史实际增强150条，共1494条 |
| 与历史完整流程区别 | 历史48.93流程使用去重后492条伪标签 + 同一批历史增强150条，共1499条；当前49.01模型只复用历史增强，伪标签仍来自本次完整重跑 |
| 领域对抗 | 保留，`lambda_domain_adv=0.03` |
| 情感对比 | 保留，`lambda_sentiment_contrastive=0.01`，source only（仅源域），class balanced（类别平衡） |
| 伪标签权重 | 当前最佳使用 `final_pseudo_weight=0.65` |

### 3.1 当前从头复现边界

| 代码阶段 | 固定提交 | 实际职责 | 关键约束 |
|---|---|---|---|
| 历史上游 | 基线`9e789045b41df7af0dd73ccebc90f06a91d94f8e`；恢复兼容`a7e7778869dce92fe778837715a814b5c6d2014b` | 数据准备、25轮抽取器、基础伪标签、8轮生成器、双通道增强与过滤 | 兼容提交只加入中断恢复，不改变未中断训练路径；只传 `seed=1000`；抽取器取 last（最后检查点），生成器取 best（最优检查点）；增强读取基础伪标签 |
| 历史下游 | 训练基线`8c7f6b47b1b2b4ef9c11d7dffdf64758db7aace3`；命令兼容`a7d147364d4b7de37814e6ee12871a386394d5f5` | 从基础伪标签补充完整双三元组、组装最终训练集、DANN（领域对抗）与情感对比训练、目标域评估 | `a7d1473`相对基线只新增两个`.cmd`（命令）文件，训练代码不变；脚本同时校验当前提交和祖先关系 |
| 编排层 | 当前主分支的 `run_historical_best_two_stage.ps1` | 校验两个提交、串联数据、记录日志/行数/SHA256（文件哈希）、断点恢复 | 不把两个历史版本手工合并成不存在的第三个代码版本；已完成阶段按状态文件跳过，训练阶段使用检查点自动恢复 |

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
| 历史产物组合最佳 | complete_multi2_w025 + 情感对比 | 上游约`9e78904`；组合`9e76a19` / `8c7f6b4` | `complete_multi2_w025_sentiment_contrastive_l001_source_balanced_pw065` | **48.93** | **50.21** | 7月9日旧抽取器/伪标签/增强 + 7月20日完整双三元组、DANN、源域类别平衡情感对比 | 不是单一提交从头运行得到；保留为历史产物组合基准 | 保留模型 |
| 旧代码复现 | `c0b2730` full rerun（完整重跑） | `c0b2730` | `bgca_aste_stage1_best_c0b2730_full_rerun_v1` | 47.58 | **50.26** | 用旧提交重新跑 `pw065` 最好流程 | fixed F1（修正F1）可复现并略高，raw F1（原始F1）未复现旧最好；该版本不是 48.93 结果时间线上最贴近的提交 | 保留指标 |
| 精确回溯成功 | 历史 final_train（最终训练集）复训 | 最终训练代码`8c7f6b4`；上游产物约`9e78904` | `bgca_aste_stage1_best_8c7f6b4_final_only_from_historical_train_v1` | **48.93** | **50.21** | 直接复用2026-07-20 19:36的`final_train...pw065.jsonl`可精确复现；其中增强源自7月9日旧上游 | 证明评估和最终训练可复现，但不能证明`8c7f6b4`可从头重建同一训练集 | 保留模型 |
| 完整确定性重跑 | seed 1000 全流程 | `5057ef2` | `bgca_aste_stage1_full_pipeline_seed_sweep_v1\seed1000` | **46.76** | **49.32** | 从头训练抽取器、生成器、伪标签、增强和最终模型；完整伪标签 F1 51.25，增强与历史仅重合 2/150，最终训练耗时约为历史的 2.5 倍 | 强确定性没有恢复 48.93，不能证明历史最好可由 seed 1000 全流程稳定复现；保留全部诊断产物 | 保留模型与指标 |
| 旧式随机完整重跑 | seed 1000 legacy stochastic（旧式随机）全流程 | `5057ef2` | `bgca_aste_stage1_full_pipeline_legacy_stochastic_v1` | **45.68** | **47.86** | 抽取器、生成器、421条基础伪标签和494条补全伪标签均与历史 SHA256（文件哈希）一致；最终训练集与7月20日历史训练集有147/150条增强文本标签相同；模型和预测与此前45.68完整重跑逐字节一致 | 已复现7月21日覆盖后的流程状态，但没有复现7月20日48.93最终模型 | 保留指标与诊断模型，完成归因后再决定是否删除低分模型 |
| 强确定性单变量 | 固定1494条训练数据，仅重训最终模型 | `5057ef2` | `final_only_nondeterministic_seed1000` | **47.98** | **50.88** | 训练集 SHA256 为 `E3EEAF14...A479DF`；除关闭强确定性外，DANN、情感对比、权重、轮数和seed均不变；训练25分22秒 | raw F1比强确定性高1.22；当时fixed F1最高，但raw仍比历史最好低0.95 | 保留模型与指标 |
| 当前数值最佳 | 历史增强内容单变量 | `5057ef2` | `historical_augment_hybrid_seed1000` | **49.01** | **51.83** | 保持当前488条完整伪标签，只用历史最终训练实际使用的150条增强替换当前增强；训练集 SHA256 `C79AA9B...CDE76` | 相对47.98提升1.03，证明增强内容是剩余差距主因；因复用历史增强，不视为从头可复现方案 | 保留模型、指标与 manifest（清单） |
| 权重过高 | complete_multi2_w035 | `c0b2730` | `complete_multi2_w035_sentiment_contrastive_l001_source_balanced` | 45.74 | 47.02 | 双三元组补充权重从 0.25 提到 0.35 | 权重过高，引入噪声 | 建议删除 |
| 动态多三元组 | dynamic_strict_dist5 | `577c55e` / `e8e23e3` / `7f3724d` | `dynamic_strict_dist5` | 48.07 | 49.69 | 动态保留多三元组伪标签，不强制最多 1 个 | 有潜力但未超过最佳 | 保留指标，暂不主线 |
| 3+ 补充 | complete_multi2 + dynamic strict 3+ | `7f3724d` / `9e76a19` | `complete_dynamic3plus_v1` | 45.38 | 47.48 | 在完整双三元组上再补 3+ 动态严格伪标签 | 3+ 噪声和欠配对明显 | 建议删除 |
| 高置信截断 | dynamic strict top050/top080 | `68bc0d0` | `top050` / `top080` | 45.83 / 44.66 | 47.78 / 46.90 | 对 3+ 动态伪标签按置信度比例截断 | keep top ratio（高置信比例截断）无效 | 建议删除 |
| 25轮生成器 | BGCA-style generator（BGCA风格生成器）25 轮 last + masked_mutual（掩码互补）增强 | `68bc0d0` / `11ca672` / `50a87f7` | `bgca_aste_stage1_bgca_generator25_last_v1` | 46.09 | 48.45 | 只把生成器改为 25 轮 last，其余保持当前最佳 | 生成器轮数变长本身没有收益，增强有效性下降 | 保留指标 |
| 最新对照 | BGCA-style generator（BGCA风格生成器）25 轮 last + label_to_text（标签到文本）增强 | `5057ef2` | `complete_multi2_w025_sentiment_contrastive_l001_source_balanced_pw065_aug_l2t` | 47.30 | 50.37 | 增强输入从 masked_mutual（掩码互补）改为 label_to_text（标签到文本），复用 25 轮 last 生成器 | fixed F1（修正F1）达到当前最高，但 raw F1（原始F1）未超过主线最好；可作为后处理友好分支，不替代 raw 主线 | 保留指标 |

### 4.2 48.93真实版本链审计

| 层级 | 历史48.93实际来源 | 当前完整重跑 | 结论 |
|---|---|---|---|
| 原始数据 | 857条源域训练、210条开发、906条目标无标签；文件SHA256与当前完全相同 | 完全相同 | 不是数据切分变化 |
| 抽取器 | 2026-07-09 18:23生成，代码更接近`9e78904`，模型SHA256 `6AD985A7...25F3E` | 强确定性重训，SHA256 `C6A230EC...574F1` | 权重不同；旧抽取器源域dev fixed F1为71.60，当前为69.47 |
| 上游训练参数 | 25轮抽取器/8轮生成器、学习率0.0003、批大小1、梯度累积16、fp16（半精度）；`data_seed=None`、`full_determinism=False` | 前述主要参数相同；`data_seed=1000`、`full_determinism=True` | 可确认的训练开关差异就是新增强确定性；它改变模型权重而不改变表面超参数 |
| 基础增强伪标签 | 634条，隐藏金标F1 47.71 | 647条，隐藏金标F1 47.40 | 总分接近但内容不同；精确行仅重合393条，62个历史ID消失、75个当前ID新增 |
| 增强请求 | 1267条 | 1292条 | 因伪标签和跨域记忆变化，精确prompt（提示）仅重合106条，即历史请求的8.4% |
| 回抽过滤 | 原始7月20分析文件已被覆盖；7月21旧上游近似复现中NLI后774条，旧抽取器保留273条，最终可用270条 | NLI后同为774条，当前抽取器只保留218条，最终可用214条 | 近似复现表明最大直接损失发生在模型回抽过滤，少55条高质量候选 |
| 最终增强150条 | 方面通道116、观点通道34；多三元组63；entailment（蕴含）41；平均质量1.141；中性14 | 方面通道109、观点通道41；多三元组70；entailment（蕴含）27；平均质量1.127；中性0 | 当前候选池更窄且整体质量下降，最终两批仅重合2条 |
| 旧上游复现稳定性 | 7月21日同目录用非强确定性8轮生成器重训后，选中增强与7月20日实际增强重合147/150 | 当前强确定性全流程只重合2/150 | 生成器在旧抽取器/旧伪标签条件下可近似重现；主要断点是当前抽取器及其产生的伪标签/跨域记忆 |
| 最终组合 | 7月20日用`9e76a19`修正“先生成基础增强，再补完整双三元组”，由`8c7f6b4`附近代码组合最终训练集 | 当前顺序已经相同 | 顺序修复仍保留，不是本次下降原因 |

代码核对结论：`t5_aste_augment.py`（增强核心）从`8c7f6b4`到当前无有效差异；最终150条排序函数自`231c49d`后未修改，`max_opinion_ratio=1.0`时行为等价。2026-07-25 已把正式复现命令恢复为历史形式：只传 `--seed`，不传 `--deterministic` 或 `--legacy_stochastic`；训练器在未显式选择诊断模式时默认只向 Transformers Trainer（训练器）传递 `seed`，不再自动设置 `data_seed`、`full_determinism`、Python/NumPy随机种子或cuDNN确定性算法。

历史缺口：原始7月9日生成器检查点后来在7月21日被同名8轮生成器重训覆盖，因此原始生成器权重哈希已经丢失；但7月20日`final_train`内嵌的真实150条增强仍完整保留。后续所有实验必须给抽取器、生成器、伪标签、增强和最终训练集分别记录路径、创建时间、SHA256与代码提交，禁止只记录最终结果附近的单一commit（提交号）。

### 4.3 历史代码和说明入口

| 主题 | 主要 commit（提交号） | 说明文档或结果文件 |
|---|---|---|
| 当前总览、差距和待改进清单 | 诊断基线`5057ef2`、双阶段总览`717f49a`、下游校验修复`abd9d71` + 本次文档提交 | `实验记录与模型索引_CN.md` |
| 双历史版本完整复现入口 | 上游基线`9e78904`、恢复兼容`a7e7778`；下游训练基线`8c7f6b4`、命令兼容`a7d1473`；当前编排修复`abd9d71` | `run_historical_best_two_stage.ps1`、`test_historical_best_two_stage_runner.py` |
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
| P0 | 48.93跨越`9e78904`上游和`8c7f6b4`下游，过去用单个当前版本完整重跑无法还原真实代码边界 | 用双工作树两阶段入口从头重新生成全部模型和数据，并保存全阶段哈希 | 先跑`9e78904`的25轮抽取器、8轮best生成器、基础伪标签和增强，再由`8c7f6b4`补充完整双三元组并训练最终模型 | 基础伪标签接近历史421条、补全后接近494条、增强150条；最终raw F1以48.93为复现目标，所有产物均有提交号和SHA256 |
| P0 | 当前流程生成的增强与历史仅重合2/150；换回历史增强后raw F1提升1.03，但依赖历史产物 | 让其他跨域组合也能自动生成接近历史质量的增强 | 对比历史/当前候选的生成来源、NLI（自然语言推断）、回抽一致性、质量分、通道和三元组结构；扩大高质量候选后按质量与通道重新筛选150条 | 不复用历史增强文件，重新生成的增强使raw F1稳定接近或超过49.01 |
| P0 | 当前数值最佳精确率56.22仍低于历史模型58.31，靠召回率43.44补偿 | 降低增强带来的错误三元组，同时保留新增召回 | 下一轮优先优化筛选精度而非继续增加增强数量：提高回抽一致/NLI通过样本优先级，检查FP集中来源 | raw precision（原始精确率）回升且raw recall（原始召回率）不低于42.70 |
| P0 | `--augment_complete_pseudo` 参数名暗示它控制增强请求来源，但当前 `augment()`生成请求仍固定读取 `target_pseudo_selected.jsonl`；该参数实际只改变增强命令顺带构建训练集时使用的伪标签文件 | 让命令参数、数据流和实验标签保持一致，避免误判“双三元组是否进入增强” | 下一次改代码前决定：删除该误导参数，或真正增加显式 `augmentation_pseudo_file`（增强伪标签文件）并在分析 JSON 中记录其路径和哈希 | dry run（试运行命令）、分析文件和实际读取路径三者一致，并有回归测试覆盖 |
| P1 | 多三元组 recall（召回率）仍低，尤其 3+ 三元组样本补充后没有稳定收益 | 提高多三元组完整抽取能力，而不是简单放宽伪标签数量 | 保留 complete_multi2_w025；后续尝试更细的多三元组训练权重、生成候选多样性、回抽一致过滤，不再使用 top ratio（高置信比例截断） | 多三元组 raw F1 和 recall 同时提升，且总体 raw F1 不下降 |
| P1 | neutral（中性）三元组几乎无法召回，强行加权会伤害正负类 | 建立中性边界，而不是只加大中性损失权重 | 优先做错误类型分析：否定但非中性、缺失属性但中性、弱情感表达；再考虑构造小规模高质量中性增强 | neutral F1 有实际提升，同时 pos/neg（正向/负向）F1 不明显下降 |
| P2 | 当前增强样本仍可能引入标签一致但表达质量低的句子 | 提高增强样本质量和多样性 | 在标签回抽一致基础上增加去重、非原句复制、长度和领域词覆盖筛选 | 增强保留率不过低，最终 raw F1 提升或至少召回提升 |
| P2 | 六组跨域平均仍落后 BGCA，laptop14 -> restaurant 三组差距最大 | 从单方向有效改进迁移到六组平均 | 当前先在 rest16 -> laptop14 验证机制；有效后再跑六组，并单独分析 laptop14 -> restaurant 的 recall 问题 | 六组平均 raw F1 差距收敛，不能只提升单组 |

当前不要继续投入的方向：`hp2_dist5` 简单放宽数量、中性生成损失强加权、三任务 mixed generator（混合生成器）、triplet coverage classification head（三元组覆盖分类头）、dynamic strict top ratio（动态严格高置信比例截断）。

## 6. 当前实验状态

增强内容单变量诊断已经完成，证明增强内容是主要差异来源。当前不再用最新代码的确定性/随机性开关猜测历史行为，而是直接固定真实历史代码边界：`9e78904`运行上游，`8c7f6b4`运行完整双三元组补充和最终训练。

当前待跑的是双工作树历史复现流程。命令只传历史原有的 `seed=1000`，不传任何后来新增的确定性模式参数。窗口直接显示各训练阶段原生进度条；`stage_status.json`记录完整阶段，训练器通过`resume_from_checkpoint=auto`（自动检查点恢复）续跑未完成训练，`manifest.json`记录关键文件行数和SHA256（文件哈希）。

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_historical_best_two_stage.ps1 -SourceDataset rest16 -TargetDataset laptop14 -Seed 1000 -Cuda 0 -OutputRoot J:\nlp\CD-C3DA\runs\historical_best_two_stage_v1"
```

阶段判定顺序：先看抽取器源域dev fixed F1是否接近历史71.60；再核对421条基础伪标签、494条补全伪标签及生成器哈希；随后比较增强候选池和最终150条；最后判断最终raw F1。若中断，重新执行同一条命令即可从未完成阶段或最近检查点继续。

| 阶段 | 当前状态 | 恢复行为 |
|---|---|---|
| `9e78904` prepare（数据准备） | 待跑 | 成功后写入阶段状态并记录数据文件哈希 |
| `9e78904` extractor（抽取器）25 轮 last（最后检查点） | 待跑 | 中断后重新执行同一命令，训练器从最近检查点恢复 |
| `9e78904` pseudo label（基础伪标签） | 待跑 | 目标参照为历史421条基础高精度伪标签 |
| `9e78904` generator（生成器）8 轮 best（最优检查点） | 待跑 | 中断后从最近检查点恢复，成功后固定最佳模型哈希 |
| `9e78904` augmentation（双通道增强）严格筛选150条 | 待跑 | 使用基础伪标签生成请求；NLI（自然语言推断）和旧抽取器回抽过滤 |
| `8c7f6b4` complete_multi2（完整双三元组补充） | 待跑 | 增强完成后再补充，额外权重0.25；不反向改变增强请求来源 |
| `8c7f6b4` final training（最终训练）5轮及评估 | 待跑 | DANN（领域对抗）0.03、情感对比0.01、伪标签权重0.65；完成后写入raw/fixed指标 |

本次已完成的 seed 1000 完整命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_full_pipeline_seed_sweep.ps1 -Pairs rest16:laptop14 -Seeds 1000 -OutputRoot runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1"
```

命令含义：使用 RTX 3070 的 8GB 显存配置运行 seed 1000。该目录现在已经完成，再次执行会自动跳过全部已完成阶段。

已完成的单变量验证：固定 seed 1000 生成的1494条最终训练数据，只重训最终模型，并关闭强确定性。脚本记录训练集 SHA256（文件哈希），窗口显示实时进度，断电后通过 `resume_from_checkpoint=auto`（自动从检查点恢复）。

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_final_only_nondeterministic_ablation.ps1 -RunDir runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1\seed1000\rest16_to_laptop14 -Seed 1000 -Cuda 0"
```

该实验不会重跑抽取器、生成器、伪标签或增强。唯一训练变量是取消 `--deterministic`（强确定性）；DANN（领域对抗）、情感对比、数据、权重、轮数、批大小和学习率全部保持不变。

| 对比 | raw P（原始精确率） | raw R（原始召回率） | raw F1（原始F1） | fixed F1（修正F1） | 最终训练耗时 |
|---|---:|---:|---:|---:|---:|
| 同一1494条训练集，强确定性 | 53.72 | 41.40 | 46.76 | 49.32 | 约1小时 |
| 同一1494条训练集，关闭强确定性 | 54.74 | 42.70 | **47.98** | **50.88** | 25分22秒 |
| 历史最好1499条训练集 | 58.31 | 42.14 | **48.93** | 50.21 | 约26分钟 |

已完成：混合增强最终模型实验。该实验没有重新训练 extractor（抽取器）、generator（生成器），也没有重新生成伪标签；它只替换最终训练中的150条增强，并使用非强确定性配置重训最终模型。

| 混合数据组成 | 数量/哈希 |
|---|---|
| 当前完整伪标签输入 | 488 条；进入最终训练后 487 条 |
| 历史实际增强 | 150 条；单三元组87、双三元组48、三三元组14、四三元组1 |
| 最终混合训练集 | 1494 条；SHA256 `c79aa9b0d246c82d8d344cf8ef8ca9e4a2b8940c38fa6172e6674e230d5cde76` |
| 唯一实验变量 | 用历史实际增强150条替换当前增强150条；其他训练参数与47.98单变量实验一致 |

| 混合增强结果对比 | raw P（原始精确率） | raw R（原始召回率） | raw F1（原始F1） | fixed F1（修正F1） | TP/FP/FN |
|---|---:|---:|---:|---:|---:|
| 当前增强150条 | 54.74 | 42.70 | 47.98 | 50.88 | 231/191/310 |
| 历史增强150条 | 56.22 | 43.44 | **49.01** | **51.83** | 235/183/306 |
| 变化 | +1.48 | +0.74 | **+1.03** | **+0.95** | +4/-8/-4 |

已完成实验的复跑/断点恢复命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_historical_augment_hybrid_ablation.ps1 -RunDir runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1\seed1000\rest16_to_laptop14 -HistoricalFinalTrain runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14\final_train_strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw065.jsonl -Seed 1000 -Cuda 0"
```

运行日志位于 `runs\bgca_aste_stage1_full_pipeline_seed_sweep_v1\logs\historical_augment_hybrid_seed1000.log`。再次执行会跳过已完成阶段。结论已经明确：历史增强恢复并略超48.93，下一步不再重复该实验，而是把历史增强的质量特征转化为可在其他跨域组合中复现的候选生成和筛选规则。

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
| 最近推送 | 本轮未成功：GitHub HTTPS（加密网页传输）端口当前被网络阻断；禁止误记为已推送 |
| 当前代码基线 | 诊断代码`5057ef2`，双阶段总览`717f49a`，上游恢复兼容`a7e7778`，下游命令兼容`a7d1473`，下游校验修复`abd9d71`；本节网络状态为后续本地提交 |
| 待统一推送分支 | `master`、`feature/complete-multitriplet-ablation`、`feature/encoder-pairing-loss`、`feature/mixed-generator-training`、`feature/triplet-coverage`、`historical/best-upstream-9e78904`、`historical/reproduce-best-8c7f6b4`、`historical/reproduce-best-0332aee`、`historical/reproduce-best-c0b2730` |
| 训练环境 | `conda activate c3da` |
| GPU（显卡） | NVIDIA RTX 3070，8GB 显存 |
| 推荐训练参数 | batch size（批大小）1，eval batch size（验证批大小）2，gradient accumulation（梯度累积）16，fp16（半精度），gradient checkpointing（梯度检查点） |

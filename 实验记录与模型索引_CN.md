# CD-C3DA 实验记录与模型索引

本文档是 `J:\nlp\CD-C3DA` 的当前实验总览。每次改代码或跑完实验时整体更新相关章节，不在末尾无限追加流水记录。

## 0. 新账号接手入口

新 Codex 账号进入项目后，先按以下顺序阅读根目录文档：

1. `00_新账号接手必读_CN.md`
2. `01_CD-C3DA项目完整介绍与迁移手册_CN.md`
3. `02_CD-C3DA实验工作流Skill_CN.md`
4. `实验记录与模型索引_CN.md`
5. `03_CD-C3DA下一阶段改进计划_CN.md`
6. `04_CD-C3DA双通道增强设计_CN.md`
7. `05_CD-C3DA双通道增强实施方案_CN.md`
8. `06_CD-C3DA最佳流程复现说明_CN.md`
9. `07_CD-C3DA六组跨域实验详细分析与GPT交接_CN.md`（需要交给外部 GPT 分析时阅读）
10. `AGENTS.md`

根目录的 `02` 到 `05` 是便于接手阅读的副本。Skill（技能）和设计记录的正式维护位置仍分别是 `docs\skills` 与 `docs\superpowers`；修改正式文件后必须同步更新根目录副本。

## 1. 当前状态与差距

| 项目 | 当前值 |
|---|---|
| 终极目标 | 六个跨域方向的 raw F1（原始 F1）分别超过对应 BGCA，不以单方向或六组平均超过为替代 |
| 当前保护基线 | `rest16 -> laptop14` 的 48.93；这是当前唯一超过 BGCA 的完整复现实验，不在其验收现场上继续试验 |
| BGCA 论文 label-to-text（标签到文本）F1 | **47.28** |
| 完整从头可复现最佳 | raw P/R/F1 = **58.31 / 42.14 / 48.93**；fixed F1 = **50.21** |
| 相对 BGCA raw F1 | **+1.65** |
| 数值诊断最高 | raw F1 **49.01** / fixed F1 **51.83**；复用了历史增强，只用于归因，不作为正式可复现主线 |
| 已完成工作 | 当前最好流程已精确复现；六组跨域基线已完成；`rest14 -> laptop14` 观点软过滤、观点契约供给、动态比例与质量分层、两版目标域方面候选发现诊断均已从头运行并完成归因；保守多候选最终解码已实现，等待正式 GPU（图形处理器）实验 |
| 最新实验 | 源域联合门槛诊断从1889条目标候选保留702条，后验 precision/recall/F1（精确率/召回率/F1）为48.15%/23.21%/31.33%；源域精度65.59%未能迁移到目标域，低于现有421条高精度伪标签61.995%的精度门槛，不进入训练注入 |
| 当前主分支版本 | `master`；当前最好流程提交 `d2f2a35`；正式 GPU 验收代码提交 `558e4de`；用户已确认这是当前最好流程 |
| `master` 状态 | 当前最好流程；十阶段原生 GPU 验收、全部黄金哈希和指标均已通过 |
| 当前首次偏差处理 | `native-best-v1-5a57449` 的第 8 阶段误报已由训练语义哈希修复；新运行 `native-best-v2-training-semantic` 十阶段全部通过，旧失败现场仍保留 |
| 当前主要模型短板 | 六组均以召回不足为主；neutral（中性）伪标签缺失；领域方面词直接重合仅2.0%到11.1%；增强过滤只保证标签自洽而不能保证语言自然，扩大增强数量会同时放大伪自然文本和多三元组误检 |
| 当前下一步 | `rest15 -> laptop14` 已完成第1–9阶段；第10阶段首次调用因新增参数误注册到伪标签子命令而在模型推理前退出。重复原命令只恢复最终评估，以同一最终模型比较 beam 4（束搜索宽度4）基础输出与保守候选合并输出 |
| 当前实施计划 | 根目录 `03_CD-C3DA下一阶段改进计划_CN.md` 第7节和第11节；实现分支 `feature/multi-candidate-decoding-v1`，功能提交 `b1c3705`，命令行修复提交 `f1581f7` |

## 2. 当前最佳与 BGCA 对比

主指标使用 raw F1（原始 F1），fixed F1（修正 F1）仅作辅助分析。

| 方法 | 是否从头生成全部产物 | raw P | raw R | raw F1 | fixed F1 | 相对 BGCA raw F1 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| BGCA 论文 label-to-text（标签到文本） | 是 | - | - | **47.28** | - | 0.00 | 论文基线 |
| 历史边界精确复现 | 是；本次全部重新训练和生成 | **58.31** | **42.14** | **48.93** | **50.21** | **+1.65** | 当前可复现最佳 |
| 当前代码原生主线 `558e4de` | 是；禁止读取旧运行产物 | **58.31** | **42.14** | **48.93** | **50.21** | **+1.65** | 十阶段、全部黄金哈希与指标精确匹配；当前最好流程 |
| `master-best-check-v1` 再次验收 | 是；从当前主线重新运行十阶段 | **58.31** | **42.14** | **48.93** | **50.21** | **+1.65** | 运行提交 `d2f2a35`；未读取历史运行产物；再次精确复现 |
| 历史增强混合诊断 | 否；复用历史增强 150 条 | 56.22 | 43.44 | **49.01** | **51.83** | **+1.73** | 数值诊断最高，不是正式主线 |
| 完整双三元组，无情感对比 | 是 | - | - | 48.01 | 50.37 | +0.73 | 证明 complete_multi2 有效 |
| 强确定性完整重跑 | 是 | 53.72 | 41.40 | 46.76 | 49.32 | -0.52 | 诊断实验 |
| 旧式随机完整重跑 | 是 | 54.04 | 39.56 | 45.68 | 47.86 | -1.60 | 复现了被覆盖后的另一轨迹 |
| 25 轮 last 生成器 + label-to-text 增强 | 是 | 55.47 | 41.22 | 47.30 | 50.37 | +0.02 | 不替代 8 轮 best 主线 |

### 2.1 六组基线与双通道诊断

六组 seed（随机种子）1000 基线平均 raw F1（原始F1）为52.81，BGCA平均为55.81。以下 BGCA 是论文报告值，而项目结果是单个 seed（随机种子）1000，当前表用于方向诊断，不能替代后续三个 seed 的正式统计比较。

| 方向 | 本项目 raw P/R/F1 | BGCA F1 | 差值 | 伪标签 F1/条数 | 单/多三元组 F1 | 源域方面词/观点词对目标测试三元组覆盖 | 首要问题 |
|---|---:|---:|---:|---:|---:|---:|---|
| `rest15 -> laptop14` | 53.25/39.37/45.27 | 45.69 | -0.42 | 48.94/389 | 49.07/42.76 | 2.2%/41.0% | 总召回最低，伪标签质量最低之一，增强中性三元组为0 |
| `rest14 -> laptop14` | 59.95/46.77/52.54 | 53.64 | -1.10 | 54.38/421 | 55.73/50.43 | 5.7%/59.5% | 已接近 BGCA，但中性 F1 只有3.03；新观点供给实验又损失正负召回 |
| `rest16 -> laptop14` | 58.31/42.14/48.93 | 47.28 | **+1.65** | 50.35/421 | 53.05/46.13 | 2.0%/44.4% | 绝对能力并非六组最好；因 BGCA 门槛最低而唯一超过，仍受低召回和多三元组漏检限制 |
| `laptop14 -> rest15` | 55.01/48.66/51.64 | 58.95 | -7.31 | 56.77/315 | 57.27/46.25 | 8.5%/49.5% | 负面 F1 45.23，多三元组 F1 46.25，是两个主要缺口 |
| `laptop14 -> rest16` | 65.85/57.78/61.55 | 64.00 | -2.45 | 58.99/451 | 61.76/61.40 | 11.1%/57.4% | 六组能力最强且结构最均衡；仍缺2.45，负面 F1 55.13且中性为0 |
| `laptop14 -> rest14` | 69.36/48.29/56.94 | 65.27 | -8.33 | 54.89/625 | 63.01/54.93 | 11.0%/50.9% | 精确率最高但召回差21.07个百分点；目标集最大，多三元组和长尾方面覆盖不足 |

共同证据：六组高精度伪标签的 neutral（中性）数量都是0；目标测试中性三元组分别为 laptop14 63、rest15 25、rest16 29、rest14 66。当前增强模型过滤通过率只有29.2%到35.3%，说明扩大生成供给是合理的，但通过后不能再统一排序硬取固定150条。六组方面词直接覆盖仅2.0%到11.1%，比观点词覆盖41.0%到59.5%低得多，表明跨域主要难点之一是目标域方面词迁移，而不是只改变情感比例。

第一轮观点软过滤实验使用 `feature/opinion-soft-filter-v1` / `d09fdca`，运行目录：

```text
J:\nlp\CD-C3DA\runs\reproducible\rest14_to_laptop14_opinion_soft_filter_v1\rest14-laptop14-softop-seed1000-v1
```

| 项目 | 基线 | 软过滤实验 | 结论 |
|---|---:|---:|---|
| raw F1（原始F1） | 52.54 | 51.04 | 下降1.51，拒绝合并 |
| fixed F1（修正F1） | 54.11 | 52.70 | 下降1.41 |
| 观点通道通过率 | 18.6% | 36.9% | 覆盖提高但质量未闭环 |
| 最终通道 | 方面94/观点56 | 方面125/观点125 | 配额完成 |
| 计划新三元组保持 | 未记录 | 26/125 | 只有20.8%真正完成计划编辑 |
| 控制占位符残留 | 36/150 | 56/250 | 历史门禁存在漏洞 |

新旧全量伪标签、高精度伪标签和完整双三元组的 SHA256（文件校验值）完全相同，下降来自增强和最终训练输入。软过滤版本把失败编辑重新标成句中残留的正面三元组，导致新增训练信号几乎全部偏向正面；中性 F1（中性F1）虽升至10.13，但中性误检由2个增至12个，不能视为有效提升。

观点契约供给150条实验使用 `feature/opinion-constrained-edit-quota-v1` / `1269759`，运行目录：

```text
J:\nlp\CD-C3DA\runs\reproducible\rest14_to_laptop14_opinion_constrained_supply_aug150_v1\rest14-laptop14-opinion-supply150-seed1000-v1
```

该运行十阶段全部完成，raw P/R/F1（原始精确率/召回率/F1）为59.90/45.84/51.94，fixed F1（修正F1）为54.45。与同方向旧契约150条52.60/55.09相比，raw/fixed（原始/修正）分别下降0.66/0.64个百分点，因此不进入250条阶段。新流程把最终观点增强从33条提高到77条、方面增强从117条降到73条；正面/负面 raw F1（原始F1）分别下降0.56/1.44，多三元组 raw F1（原始F1）下降1.37，中性 raw F1（原始F1）提高4.39但中性误检增至12个。结论是编辑契约和供给门禁已经解决“能否生成”的问题，但固定总数加统一排名造成通道挤占，尚未解决“哪些样本应该以多大贡献进入训练”。

动态比例与质量分层实验使用 `feature/opinion-constrained-edit-quota-v1` / `e49bfb6`，实际运行身份提交为 `1e942fc`，运行目录：

```text
J:\nlp\CD-C3DA\runs\reproducible\rest14_to_laptop14_dynamic_ratio_tiered_v1\rest14-laptop14-dynamic-ratio-tiered-seed1000-v1
```

该运行十阶段全部完成，raw P/R/F1（原始精确率/召回率/F1）为55.88/45.66/50.25，fixed F1（修正F1）为53.82。相对52.60/55.09对照分别下降2.35/1.27个百分点，相对 BGCA 53.64低3.39。与可复现六组基线52.54相比，TP（真阳性）253降至247、FP（假阳性）169增至195、FN（假阴性）288增至294，主要退化来自误检增加。

高精度伪标签421条、F1 54.38以及完整伪标签536条、F1 53.62均与对照完全一致，首次偏差确定在增强阶段。增强从150条扩大到336条，方面/非中性观点/中性观点为202/84/50，实际比例60.1%/25.0%/14.9%，分别贴近方面下限和中性上限，并未达到目标67%/20%/13%。增强有效权重总和从30.0增至65.53；其中215条高质量、21条中质量、100条探索样本。方面桶202条中85条属于探索层，占42.1%。

质量分层也存在校准失效：被标为高质量、NLI（自然语言推断）蕴含且抽取器完全匹配的直接编辑仍出现明显不自然文本，例如 `I am difficulty if it was the drive itself` 和 `Then it poor charging at all`。当前门禁证明的是“计划标签能被同一抽取器回抽并且与原句不矛盾”，不能证明新句自然或训练信号可靠。结果上正面/负面 raw F1（原始F1）分别从55.87/57.14降至53.77/54.15；中性从3.03升至7.79，但只增加2个TP并增加9个FP；多三元组 F1 从50.43降至47.22，精确率从65.77%降至59.32%。结论是“不设总量上限 + 自洽式质量分层”放大了确认偏差和语言噪声，本路线不扩展到其他方向。

### 2.2 单生成器主线完整性审计

2026-08-08对正式仓库、`master=d2f2a35`、动态实验分支和两个历史双生成器分支进行了只读对比。正式仓库当前文档分支相对 `master` 没有任何 Python、配方或运行脚本差异；两个历史双生成器分支 `feature/dual-generator-strict-pseudo-v1` 与 `feature/dual-generator-sentiment-aligned-v2` 均不是动态实验分支的祖先，双生成器特有的 `generator_aspect`、`generator_opinion`、`--aspect_generator_model_path` 和 `--opinion_generator_model_path` 在当前代码中均不存在。

动态实验实际十阶段为 `prepare/extractor/pseudo/generator/augment/prepare_final/complete_multi2/build_final_train/final_train/evaluate`，模型目录只有一个 `generator_label_to_text_gen_ep8`，增强阶段只接收一个 `--model_path`。因此双通道表示一个生成器服务方面与观点两个数据通道，不是两个生成器。

聚焦回归测试30项中29项通过；唯一失败是历史命令图哈希。根因是动态分支为支持多请求而给历史配方无条件加入语义等价的 `--per_row 1`，导致命令图哈希从期望 `205a94...` 变成 `3185fc...`。移除该参数后哈希精确恢复。这不是双生成器残留，也不影响正式 `master`，但说明失败动态分支不适合作为新开发基点。下一阶段固定从单生成器且同方向对照为52.60的 `753bbb5` 创建隔离分支，并新增单生成器与历史命令图防回归测试。

### 2.3 目标域方面候选发现诊断 v1

实现分支为 `feature/target-aspect-discovery-v1`，安全基点为 `753bbb5`。当前实现提交依次为：`07865a7`（单生成器命令图门禁）、`91b8f6c`（候选聚合与验证）、`3a7b766`（诊断编排与配方）和 `506b36b`（逐批断点恢复、完整诊断分布及配方安全校验）。教师—学生网络和指数移动平均未采用。

新诊断阶段位于唯一生成器之后，阶段顺序固定为 `prepare/extractor/pseudo/generator/target_aspect_discovery`，随后停止，不生成增强句子、不组装最终训练集、不训练最终模型。候选来自目标无标签文本的4路抽取解码，经过原文跨度、多序列支持、目标语料文档频率、方面—观点距离和现有单个标签到文本生成器的重构负对数似然门禁。源域开发集使用源域前缀校准阈值，目标候选使用目标域前缀评分。

候选保留完成后才读取 `target_train_gold_analysis.jsonl` 计算诊断精确率、召回率和 F1；该文件不进入运行阶段输入哈希，也不参与阈值、排序或候选保留。多候选解码、源域阈值校准和目标候选评分均在每个批次后保存完成行数，重跑会从未完成行继续。配方显式设置 `use_for_training=false`，并拒绝候选数超过 beam、支持数小于2、目标金标字段、双生成器字段和未知停止阶段。

正式诊断运行目录：

```text
J:\nlp\CD-C3DA\runs\reproducible\rest14_to_laptop14_target_aspect_discovery_diag_v1\rest14-laptop14-target-aspect-discovery-seed1000-v1
```

五个科学计算阶段均成功完成。目标无标签数据906行产生1889条候选，强过滤保留237条；按全局词面去重统计，42个已存在、184个新增。保留候选后验 precision/recall/F1（精确率/召回率/F1）为54.85%/8.93%/15.36%，TP/FP/FN（真阳性/假阳性/假阴性）为130/107/1326。现有421条高精度伪标签的精确率为61.995%、F1为54.38%，因此第一版没有达到“候选精确率不低于现有高精度伪标签”的门禁，不允许进入训练注入。

分层后验分析只用于归因，不允许直接据此选择下一版目标域阈值：2/3/4路序列支持的精确率分别为45.61%/61.97%/65.38%；支持度至少3的123条候选精确率为63.41%，其中106条为逐行新增候选、精确率63.21%。生成器损失1.5到2.0区间精确率64.06%，损失大于2.3时降至46.15%。正面/负面/中性候选精确率为57.95%/42.50%/0%，说明主要噪声来自2路支持、较高生成器损失和负面候选，中性候选仍未恢复。

该结果证明“扩大候选来源”有效，但当前强过滤仍不够强。下一版必须在源域开发集上独立校准支持度和生成器损失联合门禁，再把冻结阈值应用到目标域，不能把上述目标隐藏金标分层结果直接用作选择阈值。当前237条候选及模型全部保留为诊断证据，不作为新训练输入。

运行结束后的 `KeyError: evaluate`（评估阶段键不存在）只发生在汇总函数：诊断配方按设计停在第5阶段，但旧汇总器仍假设存在第10阶段。提交 `e4aff0c` 增加短流程汇总，提交 `408b1fb` 复用阶段清单中的大模型哈希并避免重复读取权重；67项相关回归测试通过。原训练与诊断提交仍为 `506b36b`，五个阶段未重跑。缺失的 `observed_outputs.json` 和 `RUN_RECORD_CN.md` 已补齐并登记，前者 SHA256 为 `8831AF0B16CC4FB85C3F7EC60987D815F6E5947548B316103D2A2ACDDDDB486D`。

正式运行命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA\.worktrees\target-aspect-discovery-v1 && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_recipe_reproducible_pipeline.ps1 -Recipe J:\nlp\CD-C3DA\.worktrees\target-aspect-discovery-v1\configs\recipes\experiments\rest14_to_laptop14_target_aspect_discovery_diag_v1.json -RunId rest14-laptop14-target-aspect-discovery-seed1000-v1 -OutputRoot J:\nlp\CD-C3DA\runs\reproducible -Cuda 0"
```

### 2.4 源域独立联合门槛诊断 v2（已完成，未通过注入门禁）

实现分支为 `feature/source-calibrated-aspect-discovery-v2`，代码提交为 `b16fab0`，仍以单生成器诊断流程为基础，不采用教师—学生网络。配方为 `configs\recipes\experiments\rest14_to_laptop14_target_aspect_discovery_source_calibrated_v2.json`。正式运行目录为 `runs\reproducible\rest14_to_laptop14_target_aspect_discovery_source_calibrated_v2\rest14-laptop14-target-aspect-source-calibrated-seed1000-v2`；五个阶段全部完成，运行清单、汇总、哈希和中文运行记录完整。

源域开发集现在先执行与目标域相同的4路候选解码、原文跨度与距离检查、文档频率检查及生成器重构评分。联合门槛只在固定网格中搜索：序列支持度下限为2、3、4，生成器损失分位点为0.5、0.6、0.7、0.8、0.9、1.0；候选门槛必须在源域至少保留30条且精确率不低于65%，再选择覆盖量最大的门槛。若没有门槛满足条件，诊断阶段直接停止，不自动降低标准。

选出的支持度和生成器损失阈值会冻结后原样应用到目标域。`target_train_gold_analysis.jsonl` 仍不属于该阶段输入，只在目标候选全部冻结后计算审计指标。新流程新增源域多候选、源域候选及源域候选损失三个可恢复产物；旧 v1 命令图归一化哈希保持不变，生成器阶段仍严格只有一个。相关语法、联合门槛、断点恢复、短流程汇总、旧配方和单生成器回归共62项测试通过。

正式结果表明当前校准目标失败。源域686条候选中，搜索最终选择 `min_sequence_support=2`、损失分位点1.0和 `generator_nll_max=4.529653`；源域保留340条，TP/FP为223/117，精度65.59%，只是刚刚越过65%的最低线。该阈值应用到目标域后，从1889条候选保留702条，TP/FP/FN为338/364/1118，precision/recall/F1（精确率/召回率/F1）为48.15%/23.21%/31.33%。相对 v1 的237条、130个TP和107个FP，v2多保留465条，只增加208个TP却增加257个FP，增量精度仅44.73%。候选F1上升来自大幅扩大输出，不代表最终 ASTE（方面级情感三元组抽取）模型提升。

失败原因是搜索目标“先达到65%源域精度，再最大化数量”会自然选择最宽松门槛。损失分位点1.0使目标域1889条候选中只有4条被生成器损失拒绝，生成器强过滤实际上失效。目标域2/3/4路支持候选精度分别为39.38%/52.88%/58.62%；正面/负面/中性为52.31%/43.35%/9.38%；逐行不在421条高精度伪标签中的612条新候选精度只有45.59%。损失大于3.0的266条精度仅39.47%，单三元组金标句中的候选精度仅28.22%。这些证据说明源域总体平均精度不能直接代表跨域安全性，当前702条候选禁止注入训练。

该实验标记为“诊断失败”。用户已于2026-08-09许可删除本次运行的 `models`（模型）目录，约11.64 GB模型与检查点已删除且不可从该目录恢复；分析文件、清单、命令、哈希、候选文件和首次失败证据均已保留。

正式运行命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA\.worktrees\source-calibrated-aspect-discovery-v2 && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_recipe_reproducible_pipeline.ps1 -Recipe J:\nlp\CD-C3DA\.worktrees\source-calibrated-aspect-discovery-v2\configs\recipes\experiments\rest14_to_laptop14_target_aspect_discovery_source_calibrated_v2.json -RunId rest14-laptop14-target-aspect-source-calibrated-seed1000-v2 -OutputRoot J:\nlp\CD-C3DA\runs\reproducible -Cuda 0"
```

### 2.5 保守多候选最终解码 v1（代码完成，正式实验待运行）

两版目标方面候选诊断都表明，扩大训练候选会快速引入 FP（误检），因此下一步暂不把候选注入训练。新实验改为只在最终 evaluate（评估）阶段提高召回：实现分支为 `feature/multi-candidate-decoding-v1`，安全基点为单生成器提交 `753bbb5`，功能提交为 `b1c3705`，命令行修复提交为 `f1581f7`，配方为 `configs\recipes\experiments\rest15_to_laptop14_multi_candidate_decoding_v1.json`。教师—学生网络、双生成器、伪标签、增强、训练权重和模型结构均未改变。

该配方仍从原始输入执行原有十阶段流程，generator（生成器）阶段严格只有一个标签到文本生成器。第10阶段先保留现有 beam 4（束搜索宽度4）的基础预测，再独立生成 beam 6（束搜索宽度6）的6个候选序列。新增三元组必须同时满足：至少3个候选序列支持；方面和观点都是原句中的完整 token span（词元跨度）；同一方面—观点对的情感支持为严格多数；不覆盖基础预测中已有的方面—观点对；每句最多补入1个。选择过程不读取目标测试金标，金标只在预测冻结后统计新增三元组正确率。

评估会同时保存同一最终模型的基础 beam 4（束搜索宽度4）指标、合并后指标、逐句候选、支持度、拒绝原因和新增三元组后验正确性。基础输出与合并输出来自同一次运行，因此可以直接判断提升是否来自候选合并，而不受两次训练差异干扰。实现已通过31项聚焦回归、Python（编程语言）语法编译、11项外部输入哈希校验和两行 RTX 3070 GPU（图形处理器）生成冒烟检查；历史配方的命令图未改变。

首次正式运行已完成第1–9阶段并保存最终模型；第10阶段在任何模型推理发生前以退出码2失败。根因是功能提交中的补丁匹配了首个同形参数区块，把5个新增命令行参数注册到了 `pseudo`（伪标签）而不是 `evaluate`（评估）子命令；原测试直接调用评估函数，未覆盖真实命令行入口。修复提交 `f1581f7` 只移动这5个参数，并新增真实入口回归测试。运行清单已保留分阶段代码来源：第1–9阶段为 `b1c3705`，恢复的第10阶段为 `f1581f7`；模型、数据、配方、阶段命令和数值评估逻辑未改变。重复原命令会校验并跳过第1–9阶段，只执行第10阶段。

首个方向选择 `rest15 -> laptop14`：当前 raw P/R/F1（原始精确率/召回率/F1）为53.25/39.37/45.27，距离 BGCA 45.69仅0.42，而且多三元组F1只有42.76，适合低风险验证召回导向解码。成功门槛是：合并后 raw F1 同时高于同次运行的基础输出和 BGCA 45.69；raw recall（原始召回率）上升；raw precision（原始精确率）下降不超过1.0个百分点；多三元组 recall/F1（召回率/F1）不下降。若不满足则标记失败，不根据目标金标反向调整支持度。

正式运行命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA\.worktrees\multi-candidate-decoding-v1 && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_recipe_reproducible_pipeline.ps1 -Recipe J:\nlp\CD-C3DA\.worktrees\multi-candidate-decoding-v1\configs\recipes\experiments\rest15_to_laptop14_multi_candidate_decoding_v1.json -RunId rest15-laptop14-multicandidate-mc6-s3-add1-seed1000-v1 -OutputRoot J:\nlp\CD-C3DA\runs\reproducible -Cuda 0"
```

重复完全相同的命令和 `RunId` 会按阶段状态继续；当前恢复时第1–9阶段会被校验后跳过，只运行第10阶段。不得把目标方面候选诊断运行或任何历史运行目录作为输入。正式结果尚未产生，当前不得把本实现记为性能改进，也没有待删除的新模型。

## 3. 最佳流程和当前原生模块

当前最佳配方：`configs\recipes\rest16_to_laptop14_best_v1.json`。

| 阶段 | 模块与参数 | 当前运行内输出 |
|---|---|---|
| 1. prepare（准备） | 当前 `t5_aste_pipeline.py prepare`；seed 1000；label-to-text 生成器数据 | 源域抽取数据、生成器数据、目标域无标签数据和测试数据 |
| 2. extractor（抽取器） | `t5-base-py`；25 轮；last（最后轮次）；batch 1；梯度累积 16；fp16 | `model.safetensors` 权重 |
| 3. pseudo（伪标签） | beams 1；最多 128 token；hp1；距离 5 | 全量伪标签和本次实际筛出的高精度伪标签 |
| 4. generator（生成器） | T5-base；源域标签生成句子；8 轮；best（最佳验证轮次） | `model.safetensors` 权重 |
| 5. augment（增强） | masked_mutual（互相掩码）双通道；NLI；抽取器回抽；历史最佳兼容配置；最多 150 条 | 本次生成并筛选的增强数据 |
| 6. prepare_final（最终数据准备） | 当前代码重新准备下游数据；只在同一 `run_id` 内同步本次全量伪标签 | `final_data` 数据目录 |
| 7. complete_multi2（完整双三元组） | hp1 基础上补完整双三元组；距离 5；额外权重 0.25 | 本次实际通过的完整伪标签 |
| 8. build_final_train（组装最终训练集） | 源域 gold + 本次完整伪标签 + 本次增强；增强权重 0.20 | 最终训练集和开发集 |
| 9. final_train（最终训练） | 5 轮 best；伪标签权重 0.65；DANN 0.03；情感对比 0.01；source only；class balanced | 最终模型权重 |
| 10. evaluate（评估） | beams 4；最多 96 token；不使用约束解码 | raw/fixed 指标和 328 条预测 |

领域对抗学习没有取消：最终训练仍使用 `lambda_domain_adv=0.03`。情感对比学习也保留：`lambda_sentiment_contrastive=0.01`。

## 4. 精确复现证据

历史边界精确复现运行：

```text
runs\historical_best_two_stage_v1\rest16_to_laptop14
```

清单：

```text
runs\historical_best_two_stage_v1\rest16_to_laptop14\manifest.json
```

当前代码原生精确复现运行：

```text
J:\nlp\CD-C3DA-native-best-rc-v1\runs\reproducible\rest16_to_laptop14_best_v1\native-best-v2-training-semantic
```

| 产物 | 黄金观察行数 | SHA256 或语义 SHA256 |
|---|---:|---|
| 抽取器 `model.safetensors` | - | `6AD985A7D61274B6553C65B305BE18BBA8618B25B98742F0594C5336A3925F3E` |
| 基础高精度伪标签 | 421 | `0536D99840054EE928B5FB746EC60326640C9A23C8A676A2A8D25DF3D8C15C84` |
| 生成器 `model.safetensors` | - | `0C93F7660E136862428AC23797339D0196047F8C2A1FADE8C99B7635F68CB1CE` |
| 增强 text+label（文本与标签） | 150，上限也是 150 | `5A5B87707BFA6C2D6416AF7962C390207CF1FAC9AFEDD5B7B4799A4C4570B2FF` |
| 完整伪标签 | 494 | `F3C6E0CF841FA84DD3F522248B3C0214B9FD1CC469A991FE853E7AFDE58AB710` |
| 最终训练集 | 1499 | 历史整文件 `4876753D...6A88`；训练语义 `CEE5C1245C7CE4928B86D7246E0F9F44CA89C1B9A24DECE6C37F554A86E565A4` |
| 最终模型 `model.safetensors` | - | `FC8BC8A4736E5CF4A0575C6C52A9349E34363E01556CC5D3397FDF0029AFAB1F` |
| raw/fixed 预测 | 328 | `66E34B17512690C94425E0D64626AF5E101158CB8F5F4DAA705C59D1E5B115A9` |

421、494、1499 是黄金观察值，不是筛选配额。新运行必须使用本次模型实际筛出的全部伪标签；禁止为匹配历史数量裁剪、补齐或读取旧文件。只有增强 150 是配方显式声明的 `selection_limit`（筛选上限）。

增强兼容审计结论：同一批历史模型过滤候选经过当前默认边界过滤后，150 条增强语义哈希为 `F1583596...E8AE8`；显式 `historical_best_v1` 配置为 `5A5B8770...0B2FF`，与黄金完全一致。差异来自后来新增的观点边界过滤和元数据字段，不来自请求随机顺序。

最终训练集校验结论：训练器实际读取 `input`、`target`、`sample_weight`、`augmentation`、`base_id` 和 `id`。历史文件与 `native-best-v1-5a57449` 文件在这六类字段、记录顺序和 1499 条行数上完全一致，训练语义哈希均为 `CEE5C124...65A4`。当前清单仍保存整文件 SHA256 用于审计，但黄金停止条件使用训练语义哈希，避免新增无关审计字段造成误报。

## 5. 可追溯历史

| 实验方向 | Git 提交或代码边界 | 运行目录/结果标签 | raw F1 | fixed F1 | 结论 | 清理状态 |
|---|---|---|---:|---:|---|---|
| 历史边界从头精确复现 | 上游 `9e78904` + 恢复兼容 `a7e7778`；下游 `8c7f6b4` + 命令兼容 `a7d1473` | `historical_best_two_stage_v1` | **48.93** | **50.21** | 全部模型、伪标签、增强和最终训练重新产生；当前正式基准 | 全部保留 |
| `rest14 -> laptop14` 观点软过滤250条 | `feature/opinion-soft-filter-v1` / `d09fdca` | `rest14-laptop14-softop-seed1000-v1` | 51.04 | 52.70 | 通过率和配额达标，但编辑目标未保持、正面膨胀、多三元组下降；不合并 | 全部保留，作为失败归因证据 |
| `rest14 -> laptop14` 观点契约供给150条 | `feature/opinion-constrained-edit-quota-v1` / `1269759` | `rest14-laptop14-opinion-supply150-seed1000-v1` | 51.94 | 54.45 | 供给与编辑正确性达标，但观点77条挤占方面候选，正负召回与多三元组下降；250条取消 | 模型和检查点约17.47 GB已按用户许可删除；约0.022 GB指标、清单、日志和数据保留 |
| `rest14 -> laptop14` 动态比例与质量分层 | `feature/opinion-constrained-edit-quota-v1` / `e49bfb6`；运行身份 `1e942fc` | `rest14-laptop14-dynamic-ratio-tiered-seed1000-v1` | 50.25 | 53.82 | 伪标签与对照相同；增强从150增至336且有效权重超过两倍，自洽过滤未拦住不自然文本，FP和多三元组误检上升；不扩展 | 约17.47 GB模型和检查点建议删除，等待用户许可；约0.035 GB证据应保留 |
| 当前代码原生迁移 | `feature/native-best-reproduction-v1`；`afc0d3d..5a57449` | 配方 `rest16_to_laptop14_best_v1` | 已完成 | 已完成 | 已完成来源隔离、命令归档、黄金校验、增强兼容和 Windows 输出修复 | 已纳入当前主线 |
| 原生 GPU 首次验收 | 候选 `a755300` | `native-best-v1-a755300` | 中断 | 中断 | 抽取器 16/1325 step 遇到 `UnicodeEncodeError`；非模型、显存或 CUDA 错误，日志与清单保留 | 保留失败现场，不删除 |
| 原生 GPU 第二次验收 | `5a57449` | `native-best-v1-5a57449` | 未进入评估 | 未进入评估 | 前 7 阶段黄金值全部匹配；第 8 阶段因 34 条记录新增空审计字段触发整文件哈希误报，最终训练未开始 | 保留失败现场，不删除 |
| 最终训练语义校验与原生完整验收 | `fix/native-best-training-semantic-v2`；运行提交 `558e4de` | `native-best-v2-training-semantic` | **48.93** | **50.21** | 十阶段从头运行；全部黄金模型、数据、预测哈希和指标匹配；训练语义 `CEE5C124...65A4` | 当前正式主线，全部保留 |
| 历史增强混合诊断 | `5057ef2` | `historical_augment_hybrid_seed1000` | **49.01** | **51.83** | 证明增强内容是性能差异主因，但复用历史增强 | 模型、指标、清单保留 |
| 固定训练集关闭强确定性 | `5057ef2` | `final_only_nondeterministic_seed1000` | 47.98 | 50.88 | 强确定性会掉分，但不是唯一原因 | 保留指标和模型 |
| 强确定性完整重跑 | `5057ef2` | `full_pipeline_seed_sweep_v1\seed1000` | 46.76 | 49.32 | 增强与历史仅重合 2/150 | 保留诊断产物 |
| 旧式随机完整重跑 | `5057ef2` | `full_pipeline_legacy_stochastic_v1` | 45.68 | 47.86 | 精确复现了 7 月 21 日覆盖后的轨迹，不是 48.93 轨迹 | 建议最终归档后删除低分模型，尚未删除 |
| complete_multi2_w025 | `62113b4` / `4258bc6` / `0332aee` | `complete_multi2_w025` | 48.01 | 50.37 | 完整双三元组是关键正向改动 | 保留指标 |
| hp2_dist5 | `869466a` | `strict_aug150...hp2_dist5` | 44.44 | 46.87 | 简单放宽伪标签增加噪声 | 建议删除模型，尚未删除 |
| mixed generator（三任务混合生成器） | `e7560c7` / `e320fab` / `925d596` | `mixed_generator_v1` | 44.07 | 46.06 | 混合任务削弱主生成目标 | 坏模型已删除，指标保留 |
| neutral 强增权 | `0c49ba6` / `ce7452e` / `e5f5d47` | `neutral_gain100_max200` | 43.18 | 45.76 | 未解决中性召回并伤害正负类 | 已删除或建议删除，指标保留 |
| pairing loss（配对损失） | `c1082ab` / `123ab39` / `6075ee0` | `pairing_encoder_l001` | 46.49 | 48.86 | 精确率提高但召回下降 | 已删除或建议删除，指标保留 |
| coverage loss（覆盖损失） | `cbeb965` / `e60ca8f` / `44997d4` | `coverage_encoder_l001` | 44.37 | 46.72 | 分类辅助头没有带来有效生成收益 | 坏模型已删除，指标保留 |
| complete_multi2_w035 | `c0b2730` | `complete_multi2_w035` | 45.74 | 47.02 | 双三元组权重过高 | 建议删除，尚未删除 |
| dynamic strict 3+ | `7f3724d` / `9e76a19` | `complete_dynamic3plus_v1` | 45.38 | 47.48 | 3+ 噪声和欠配对明显 | 建议删除，尚未删除 |
| 25 轮 last 生成器 | `68bc0d0` / `11ca672` / `5057ef2` | `bgca_generator25_last_v1` | 47.30 | 50.37 | 训练更久本身不优于 8 轮 best | 保留指标 |

任何“建议删除”项目在用户明确许可前都不执行删除。

## 6. 待改进

| 优先级 | 当前不足 | 改进目标 | 下一步改动 | 接受标准 |
|---|---|---|---|---|
| P0 | 过去可跨目录复用上游产物，导致表面重跑实际混合 | 永久阻断混合产物 | 继续维护 `manifest.json`、输入/输出 SHA256 和同一 `run_id` 恢复门禁 | 任一跨目录或变更输入被测试和运行时拒绝 |
| P0 | 失败观点编辑会被抽取器重新贴成其他标签后保留 | 建立生成意图与最终标签闭环 | 控制符硬门禁、通道编辑契约、抽取器只作软支持、150条隔离消融 | 占位符为0；计划三元组、目标情感和未编辑三元组保持率均为100% |
| P0 | 动态增强和两版目标方面候选诊断都在扩大召回时放大 FP（误检），源域平均精度不能可靠迁移 | 控制候选噪声并避免确认偏差 | 暂停候选训练注入；先用冻结规则测试只发生在最终评估的多候选一致性合并 | 不使用目标测试标签调参；总体精确率下降不超过1个百分点；raw F1（原始F1）和召回率同时上升 |
| P1 | 缺少结构化 FN/FP（漏检/误检）归因 | 定位召回损失首次出现的阶段 | 自动分类方面词、观点词、配对、情感、边界、否定和多三元组错误 | 每个错误可追溯，明确贡献最大的两个召回瓶颈 |
| P1 | neutral（中性）F1 接近 0 | 学到真实中性边界 | 先分离否定、缺失属性、弱情绪三类错误，再构造少量高质量样本 | neutral F1 提升且 pos/neg F1 不明显下降 |
| P1 | 多三元组 recall（召回率）仍低 | 提升完整抽取而不引入 3+ 噪声 | 保留 complete_multi2_w025；先运行 `rest15 -> laptop14` 的6候选、3票支持、每句最多补1条的最终解码消融 | 多三元组 raw F1 和 recall 同升，总体 raw F1 超过同次基础输出；首方向同时超过 BGCA 45.69 |
| P1 | 48.93 主要依赖精确率 58.31，召回只有 42.14 | 提高召回同时控制 FP | 按 FN 结构设计分层伪标签和增强候选，不再单纯放宽数量 | recall 超过 42.14，precision 不低于 58.0，raw F1 首先突破 50.0 |
| P2 | 单方向、单 seed（随机种子）有效不等于整体稳定 | 六个方向分别超过 BGCA | 先在三个代表方向筛选，再对候选版本运行六组至少3个 seed，报告 mean +/- std（均值加减标准差） | 六个方向的 raw F1 分别超过对应 BGCA，全部运行可复现 |

总体阶段路线、双通道增强设计和具体实施步骤见根目录副本：

```text
03_CD-C3DA下一阶段改进计划_CN.md
04_CD-C3DA双通道增强设计_CN.md
05_CD-C3DA双通道增强实施方案_CN.md
```

暂不继续投入：hp2 简单放宽、中性损失强增权、mixed generator（三任务混合生成器）、旧 coverage classification head（覆盖分类头）、dynamic strict top ratio（动态严格顶部比例）、complete_multi2_w035、无条件加入三元组以上伪标签，以及只增加生成器轮次并固定选择 last（最后检查点）。

## 7. 运行入口与命令归档

历史边界精确复现使用过的完整 CMD（命令提示符）命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_historical_best_two_stage.ps1 -SourceDataset rest16 -TargetDataset laptop14 -Seed 1000 -Cuda 0 -OutputRoot J:\nlp\CD-C3DA\runs\historical_best_two_stage_v1"
```

当前代码原生试运行入口：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_best_reproducible_pipeline.ps1 -RunId native-best-dry-run-v1 -OutputRoot J:\nlp\CD-C3DA\runs\reproducible -Cuda 0 -DryRun"
```

训练语义校验修正后的完整 GPU 命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA-native-best-rc-v1 && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_best_reproducible_pipeline.ps1 -RunId native-best-v2-training-semantic -OutputRoot J:\nlp\CD-C3DA-native-best-rc-v1\runs\reproducible -Cuda 0"
```

正式命令不使用 `-AllowDirtyDiagnostic`；断电后重复同一命令和同一 `RunId` 恢复。新运行不得读取 `native-best-v1-5a57449` 或任何历史运行产物。

每次原生运行目录固定保存：

```text
manifest.json
run_command.cmd
commands.jsonl
environment.json
stage_status.json
RUN_RECORD_CN.md
logs\<stage>.log
```

清单还会在 prepare（准备）完成后恢复并保存配方中的源域、目标域、seed（随机种子）、配方路径与配方 SHA256，防止底层准备脚本写入同名清单后丢失运行身份。每个阶段的 `inputs` 显式包含其通过 `run_dir` 隐式读取的文件，因此断点恢复不只检查命令行中直接出现的路径。

Windows 控制台输出会按当前编码安全替换无法显示的单个进度字符，完整原始行仍按 UTF-8 写入阶段日志；控制台显示问题不得中断训练子进程。

## 8. Git、文档与清理规则

- 修改前创建新分支；`master` 只合并完整 GPU 验证后的当前最佳版本。
- 历史提交和历史工作树只用于审计，不作为正式运行输入。
- 每次实验记录 Git 提交、分支、完整命令、配方、环境、输入输出 SHA256、指标和清理状态。
- 跑完实验先更新第 1、2、6 节，再压缩更新历史表；不在末尾追加重复叙述。
- 新结果较差时保留指标、命令、清单和首次错误，标记“建议删除”；用户明确许可后才能删除文件。
- 查看旧代码优先使用隔离工作树，不能覆盖当前代码；正式训练入口不得调用历史工作树。

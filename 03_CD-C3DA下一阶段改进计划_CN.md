# CD-C3DA 下一阶段改进计划

> **当前生效路线（2026-09-05）**
>
> 本节是当前唯一执行依据；下方旧版内容保留为历史记录，不再作为新实验安排。

## 总目标与原则

| 项目 | 冻结结论 |
|---|---|
| 主方向 | `Laptop14 → Restaurant15` |
| seed（随机种子） | `1000` |
| 当前最佳 | G3 Final Raw F1=`54.55%` |
| 外部目标 | BGCA=`58.95%` |
| 唯一目标 | 将 Raw F1 提升到 `58.95%` 以上 |

路线固定为：`Phase A 结构保持/句法增强 → Phase B Opinion 通道/目标域候选 → Phase C 双域生成器 → Phase D 高成本最终冲刺`。每轮优先 3–4 个有意义实验；每个阶段只解决一个层级问题；重要方案必须跑到 Final Raw F1，不长期停留在 proxy metric（代理指标）。

## 当前判断与冻结变量

- G0–G3 上游比较完成，后续锁定 G3；暂不重训 G0–G3、搜索新图关系或图参数。
- 主要瓶颈是下游知识利用和增强结构破坏。G3 已审计：edited validity=`68.32%`、untouched retention=`33.33%`、3+ untouched retention=`20.00%`、triplet-count preservation=`59.41%`、3+ preservation=`0%`、unplanned row rate=`57.43%`；Opinion edited validity 约 `52%`。
- Phase A/B 冻结：G3 upstream、current pseudo/complete_multi/generator、domain prefix=`none`、masked mutual、pseudo weight=`0.75`、augmentation weight=`0.20`、complete_multi extra=`0.25`、sentiment contrastive=`0.01`、Final batch=`16`、gradient accumulation=`2`（有效批次32）、lr=`3e-4`、epochs=`5`、Final DANN=`0.03`、beam=`4`、max_new_tokens=`96`、checkpoint=`best`。
- Target-test gold（目标测试金标）禁止用于训练、选模、调参和下一变量选择，只能最终报告或事后分析。

## Phase A：结构保持 + 句法感知增强

目标是先修复 augmentation structure break。四组均不重训 extractor，使用 G3、当前 generator、当前 pseudo/complete_multi 和当前 Final ASTE 配方；所有组必须完成 Final Raw F1。

| 实验 | 唯一改动 | 归因目标 |
|---|---|---|
| A0 | Structure Preservation only（仅结构保持） | 新增强基准 |
| A1 | 结构保持 + Aspect Syntax Constraint（方面句法约束） | 方面通道贡献 |
| A2 | 结构保持 + Opinion Syntax Constraint（观点句法约束） | 观点通道贡献，重点实验 |
| A3 | 结构保持 + 方面/观点句法约束 | 双通道互补性 |

第一版只用 `POS/UPOS`、dependency relation（依存关系）、head POS（中心词词性）；暂不加入 dependency path、local neighborhood、trainable attention 或新图模块。增强文本变化后必须重新解析，禁止使用旧 parent graph。任何不满足 edited validity、untouched retention、triplet-count preservation、no unplanned triplets 的行直接 `DROP`，不得进入 `final_train`。

Phase A 结束选择 `BEST_PHASE_A`；proxy 指标变好但 Final Raw F1 不涨，不算有效。完成 A0–A3 后停止，不自动进入 Phase B。

## Phase B：Opinion 通道 + 目标域候选优化

固定 `BEST_PHASE_A`、G3、相同 pseudo/complete_multi/generator/Final ASTE，只改变 candidate selection（候选选择）或 opinion replacement strategy（观点替换策略）。

| 实验 | 策略 |
|---|---|
| B1 | `semantic_same_sentiment`：同情感、语义相似、共现和目标频率 |
| B2 | `sentiment_vector`：观点嵌入、情感中心、余弦相似度、margin 和共现 |
| B3 | target-domain-priority opinion bank：优先 Restaurant15 高置信伪观点 |
| B4 | target + semantic + syntax ranking：目标域、情感、语义、句法、共现综合排序 |

只借鉴 DAEGCN 的“目标域片段主动参与并由模型过滤”，不原样复现 domain-specific segments-aware attention。阶段结束重点看 Final Raw F1、Opinion validity、multi recall、unplanned rate 和目标候选利用率；若超过 `58.95%`，停止增加方法。

## Phase C：Target-Aware Dual-Domain Generator（目标感知双域生成器）

当前 generator 主要是 Source-trained（源域训练）。正式改造为 `Laptop14 source gold + Restaurant15 high-confidence pseudo`，并真实学习 `domain: laptop/restaurant` 条件，不是单独修 prefix（前缀）。

| 实验 | 组成 | 目的 |
|---|---|---|
| C0 | Source gold + Target pseudo，无 prefix | 判断目标伪标签加入生成器训练的收益 |
| C1 | C0 + dual-domain prefix（双域前缀） | 判断真实双域条件 |
| C2 | C1 + Contrastive Learning（对比学习） | 强化目标域表示和域内一致性 |
| C3 | C1 + Domain Adversarial Learning（领域对抗学习） | 保留跨域共享 ASTE 语义 |

第一轮不同时打开对比和对抗；只有 C2、C3 都有明确正收益才测试组合 C4。达到 `58.95%` 后转入多 seed、消融、稳健性和论文主表验证。

## Phase D：高成本模块/最终冲刺

仅当 Phase A–C 未超过 BGCA 才进入。候选包括：可学习候选排序与 candidate-context cross-attention（候选-上下文交叉注意力）、更丰富句法（依存路径/局部邻域/多跳）、Graph-aware Final ASTE（图感知最终 ASTE）和 C4 组合目标。若已超过 `58.95%`，不继续堆模块。

## 整体 Gate 与禁止支线

```text
Phase A：增强是否安全、句法是否有帮助
→ Phase B：候选是否正确且具目标域价值
→ Phase C：生成器是否真正学习双域
→ Phase D：仅在接近但未超过 BGCA 时冲刺
```

在对应阶段前禁止：G4、新图关系/图超参搜索、prefix-only、DAEGCN 完整复现、candidate cross-attention、graph-aware Final ASTE、Contrastive+Adversarial 同时搜索。服务器每轮默认并行约 3–4 组，但不得为凑数添加无研究价值实验。

---

> 更新时间：2026-08-30 23:17（北京时间）
>
> 当前唯一目标：判断元素感知RGAT（关系图注意力网络）的收益究竟来自Element Salience（元素显著性）、Multi-Element Coverage（多元素覆盖），还是两者组合；在组件归因完成前，不启动Phase B（阶段B）、增强、最终ASTE（方面级情感三元组抽取）或目标测试。

## 0. ICASSP 2027投稿关键路径

最终截止为2026年9月16日，目标是完成并提交常规4页英文技术正文＋可选第5页参考文献/资助/伦理声明，不是只完成初稿。

| 日期 | 必须完成的里程碑 | 停止条件 |
|---|---|---|
| 8月31日—9月3日 | Focus-only与Coverage-only组件归因，确定图组件去留 | 9月3日后不追加系数、batch、图层或头数搜索 |
| 9月4日—7日 | Phase B快速消融、正式方法确认和必要工程修复 | 未形成稳定净收益则回退到可复现基线，不拖延投稿 |
| 9月8日 | 最终方法冻结 | 之后只修复影响正确性或复现的缺陷，不增加模块 |
| 9月8日—11日 | 六方向正式运行、必要随机种子、最终消融与BGCA对比 | 不完整运行不得写成正式结果 |
| 9月12日 | 冻结全部论文数字、表格和结论 | 之后不得用目标测试反向调参 |
| 9月13日 | 完成4+1页英文全文 | 所有超页内容必须压缩或移入允许的补充材料 |
| 9月14日 | 技术、数字、引用和代码一致性审计 | 主张必须能回溯到正式证据 |
| 9月15日 | IEEE模板、PDF兼容、页数和提交演练 | 不再安排长实验 |
| 9月16日 | 最终检查并正式提交 | 保留提交缓冲，不把实验安排到截止当天 |

## 1. 当前情况总结

当前正式最佳仍为`laptop14 -> rest15` raw F1=54.01。历史实验已经证明，继续调整伪标签权重、样本质量权重、DANN（领域对抗网络）、EOS（终止符）、NLL（负对数似然）或中性损失，只会在精确率和召回率之间重新分配错误，不能稳定提升多三元组完整召回。

主要瓶颈分为三层：

1. 目标域方面/观点元素没有进入候选；
2. 元素存在，但没有形成完整、合法的多三元组结构；
3. 结构已经形成，但最终生成模型遗漏第二、第三个三元组。

当前先解决第一、第二层。句法图只进入目标伪标签形成阶段，提供依存关系和词性拓扑；T5 Encoder（编码器）提供节点语义；多头RGAT传播关系信息；元素显著性门控突出方面/观点节点；覆盖损失约束多元素被共同关注。最终ASTE模型暂时保持无图。

## 2. 当前有效证据

| 配置 | source-dev strict F1 | multi recall | overall absence | 合格伪标签 | 合格multi |
|---|---:|---:|---:|---:|---:|
| Plain Control（无图T5） | 57.84% | 47.64% | 69.77% | 508 | 185 |
| Graph Reference（普通RGAT，DANN=0） | 56.57% | 45.75% | 73.26% | 557 | 209 |
| 完整Treatment batch=8 | 55.07% | 47.17% | 70.93% | 545 | 224 |
| 完整Treatment batch=16 | 56.58% | **48.11%** | **69.77%** | 552 | 209 |

完整Treatment相对普通RGAT改善了多三元组召回和元素缺失，但相对Plain Control没有同时通过F1与召回门槛。因此不能直接宣布图模块成功，也不能直接扩展到正式训练；必须先做组件归因。

## 3. 总体改进路线

### 模块一：目标域元素与句法图形成

保留依存边、反向边、自环和POS（词性）关系。图缓存必须逐行可恢复、身份可核验，且禁止读取目标测试。目标是让模型识别目标域中“哪些词可能是方面或观点”。

### 模块二：元素显著性门控

对T5节点表示计算方面/观点显著性，作为图消息和编码器融合的软门控。它只能改变元素关注，不得改变标签、伪标签筛选阈值或训练数据。

### 模块三：多元素覆盖损失

对同一句中多个计划方面/观点施加覆盖约束，减少只关注第一个高频三元组的倾向。覆盖损失只在有合法元素对齐的源域标注上计算；目标无标签不得构造伪金标覆盖目标。

### 模块四：强验证与结构门控

继续使用完整字符串、三元组集合、NLI和冲突拒绝。验证同时检查：原有三元组保持、新增三元组存在、方面—观点配对、情感、边界、计划外三元组和多结构完整性。

### 模块五：后续增强重构

只有图伪标签形成通过后才进入。增强仍保持单生成器、目标真实句锚定和每个锚句最多一条。下一版增强应以完整triplet（完整三元组）为单位构造双通道：方面替换通道与观点/极性替换通道可以并行产生候选，但每条候选只能选择一种受控编辑；原始未编辑三元组必须强制保留。该模块当前不实施。

### 模块六：正式训练与六方向扩展

组件归因和伪标签快速消融通过后，先在`laptop14 -> rest15`完成一次从头运行；再在`rest14 -> laptop14`验证方向可迁移性；最后才扩展六方向和多随机种子。

## 4. 当前唯一实验：组件归因

固定三组：

1. `Graph Reference`：普通RGAT，`lambda_focus=0`，`lambda_coverage=0`；
2. `Focus-only`：`lambda_focus=0.05`，`lambda_coverage=0`；
3. `Coverage-only`：`lambda_focus=0`，`lambda_coverage=0.05`。

已完成的完整Treatment为第四个只读参照：`lambda_focus=0.05`，`lambda_coverage=0.05`。不重复训练已有完整Treatment，不进行系数网格。

必须固定：`laptop14 -> rest15`、seed 1000、T5-base、图层1、隐藏维256、注意力头4、DANN=0、解析器、图缓存、数据、训练轮数、优化器、调度器、解码、伪标签规则和source-dev选模。

需要输出：source-dev总体P/R/F1、single/multi P/R/F1、元素absence、合格伪标签总量与multi/3+数量、计划外结构、模型/缓存/配置哈希和机器可读结果卡。

## 5. 组件归因判定

组件只有在以下条件同时满足时才可保留：

- 相对普通RGAT，multi recall提高至少1.5个百分点；
- overall absence降低至少2个百分点；
- source-dev F1下降不超过1个百分点；
- single F1下降不超过1个百分点；
- 合格multi不低于普通RGAT；
- 不产生非有限损失、目标金标访问或身份不一致。

若Focus-only通过而Coverage-only失败：保留显著性门控，关闭覆盖损失。

若Coverage-only通过而Focus-only失败：保留覆盖损失，关闭显著性门控。

若两者单独均失败但完整Treatment通过：标记为交互机制，只允许一次预注册的组合确认，不做系数网格。

若三者均未通过：输出`STOP_GRAPH_TUNING`（停止图调优），回到非图的目标元素候选发现，不继续增加图层、注意力头或新损失。

## 6. 后续阶段

### 阶段A：组件归因

完成当前三臂对比，目标测试不运行。当前处于此阶段。

### 阶段B：图伪标签快速消融

仅当阶段A选出一个明确组件后，用该冻结组件生成目标伪标签，并与无图Control（对照组）进行相同预算、相同过滤、相同最终训练的快速消融。允许复用prepare（准备）和固定图缓存，图模型训练及之后必须重跑。

### 阶段C：正式完整验收

仅当阶段B无金标门控和目标测试均显示净收益时执行。要求`reuse_depth=0`、完整从头运行、source-dev选模、目标金标只做最终一次报告。

### 阶段D：方向迁移与六方向

先验证`rest14 -> laptop14`；成功后扩展六方向。最终报告至少3个随机种子均值与标准差。

## 7. 复用与禁止边界

当前允许复用：固定数据、T5-base、解析器、完整图缓存、普通RGAT参照结果和完整Treatment诊断指标。

当前必须重跑：Focus-only/Coverage-only各自图模型训练、source-dev评估、目标无标签伪标签推理及无金标分析。

禁止：从已有Treatment检查点热启动；使用目标测试选组件；改变batch以调结果；同时改变DANN、图层、头数、损失系数或伪标签阈值；启动Phase B、最终ASTE或增强；把快速诊断当正式F1结论。

## 8. 已关闭路线摘要

永久关闭：双生成器、教师—学生、`k=2`、中性强加权、普通样本/结构调权、原锚回放、复杂样本重复呈现、NLL/EOS/ECAL系列、FGSM参数搜索、DANN开关/调权、无上游证据的配对和计数损失。

当前保留：单生成器、`k=1`、目标真实句锚定、完整标签计划、NLI/exact硬验证、联合配额选择、无金标结构门控，以及本轮受控的句法图组件归因。

## 9. 服务器并行与启停计划

- RTX 4090服务器默认同时运行2个独立实验；只有短烟雾测试测得单任务峰值显存不超过约7 GB、三任务合计仍预留至少3 GB，且CPU/内存/磁盘没有明显争用时，才运行3个。
- 并行单位优先选择不同随机种子或不同方向的同一冻结配方；研究变量尚未冻结时不并行启动多个未经批准的参数变体。
- 每个运行使用唯一输出目录和只读代码提交；不得共享可写检查点、阶段状态或临时目录。服务器运行中不切换提交。
- 代码、复审或CPU测试预计超过30分钟且服务器无现成任务时，立即通知用户切换无GPU模式或关机；预计可在30分钟内交付可运行提交时，明确给出预计时间。
- GPU启动前必须完成本地CPU测试、必要的Codex Sol复审、提交与推送、服务器提交核验、5至10步烟雾测试和完整恢复命令。目标是“开机即可跑”，不是开机后继续现场改代码。
- GPU实验运行期间同步推进英文论文、图表、引用和结果卡，减少串行等待。

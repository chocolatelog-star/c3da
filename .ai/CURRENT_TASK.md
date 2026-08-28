# 当前任务

> 更新时间：2026-08-28 22:05（北京时间）

- 任务编号：M1_PHASE_A_CONTROL_TREATMENT_LIFECYCLE_FIX_V1
- 任务类型：ENGINEERING FIX（工程修复）
- 方向：laptop14 -> rest15
- 随机种子：1000
- 入口身份：M1 Phase A Control（对照组）到 Treatment（实验组）训练生命周期
- 状态：COMPLETE（代码修复完成，等待 Codex Sol（高级工程模型）只读复审）
- 当前功能分支：codex/m1-syntactic-rgat-entry-audit-v1
- 本轮性质：只修复训练生命周期，不启动实验
- V4 运行：INCOMPLETE_VRAM_THRASHING（显存抖动导致不完整），不得恢复、不得删除、不得用于实验结论
- V3 运行：INVALID_REPLAY_CONTAMINATED（重放污染无效），不得恢复、不得用于实验结论

## 本轮生命周期修复

- V2 显存诊断确认图批次为 27 个节点、131 条边，图模块峰值增量约 17.3 MiB（兆字节）；隔离 Treatment 含完整梯度和 AdamW（自适应矩估计）状态时不受 8GB（8 GB）显存硬限制。
- Control 清理前后 reserved（保留显存）下降约 1972 MiB；同一 Python（Python 语言）进程串行运行 Control/Treatment 时，Control 生命周期残留是高置信度根因候选。
- t5_absa_train.py 在模型和 DANN（领域对抗网络）审计产物保存后复制纯 CPU（中央处理器）返回字典，释放 Trainer（训练器）、模型、优化器、数据加载器、回调和训练批次引用，执行垃圾回收与 CUDA 缓存清理，并记录事件。
- Phase A 运行器记录 Control 返回前、返回后、垃圾回收后和 CUDA 缓存清理后的显存、活动 CUDA 张量、引用标志和 RNG（随机数生成器）哈希；清理不通过时硬停止并要求独立子进程隔离，禁止进入 Treatment。
- 报告区分 audit_status（诊断完整性）和 attribution_decision（归因决定）；历史 V2 报告保留并标记 VALID_DIAGNOSTIC_REPORTING_INCONSISTENT（诊断报告口径不一致）。
- 本轮不改变图传播公式、模型结构、损失、DANN=0.03、数据、采样、随机种子、训练参数或研究范围。

## 当前验证边界

- TDD（测试驱动开发）RED（修复前失败证据）已复现 Control 运行对象在 Treatment 前仍可达；GREEN（修复后通过证据）验证清理门控、模型参数、产物字节和 CPU RNG 不变。
- 全部 M1 相关 CPU 直接测试共 133 项通过；未运行 GPU（图形处理器）诊断、正式 Phase A、正式伪标签、生成器、增强、Phase B（阶段 B）、最终 ASTE（方面级情感三元组抽取）或 target_test（目标测试集）。
- 如果真实 GPU 运行时生命周期门控失败，必须停止并改用 Control/Treatment 独立子进程隔离，不得继续叠加清理补丁；正式实验索引暂不更新。

## 本任务批准范围

- 使用真实 T5-base、真实图缓存、source=1/target=1、FP16（半精度）、梯度检查点和 DANN=0.03，比较同一批次的 Control（对照组）与 Treatment（实验组）。
- 逐调用点记录显存、张量形状/dtype（数据类型）、理论字节数、token/node/edge 数量，并检测图张量保留、GPU 张量进入 Python 容器、autograd graph（自动求导图）存活、隐式 FP32 提升和分配器碎片化。
- 同一固定样本执行至少三次 zero-update（零更新）诊断，输出机器可读 JSON（结构化报告）和中文报告；不修改模型、损失、数据、配方、训练参数或实验逻辑。
- 只做 CPU（中央处理器）测试和静态检查；新的 V2 GPU（图形处理器）诊断命令仅提供给用户，不由本执行器启动。

## 禁止事项

- 不运行完整训练、正式伪标签、Phase B、最终 ASTE（方面级情感三元组抽取）或 target_test；不恢复 V4。
- 不实施 graph checkpointing（图检查点）、CPU offload（中央处理器卸载）、缩短长度、替换优化器、修改 batch（批大小）、关闭图模块或修改损失。
- 不使用 `nan_to_num`、梯度裁剪或强制 FP32（单精度）掩盖显存/数值问题；不改变图传播公式、DANN 系数、模型结构或研究范围。

## V2 修复要求与当前状态

- V1 报告已标记为 `INVALID_BATCH_COUNT_AND_INCOMPLETE_OPTIMIZER_LIFECYCLE`，不得用于显存安全结论。
- V2 修复 `_batch_counts`，只读取项目整理器输出的 `attention_mask`、`graph_word_mask`、`graph_edge_mask`，并硬校验批次计数与真实图追踪张量形状一致；固定批次应为 nodes=27、edges=131。
- V2 增加 Control（对照组）到 Treatment（实验组）的生命周期显存观测、可丢弃模型副本上的稳态 AdamW（自适应矩估计）状态测量、Python CUDA（英伟达 GPU 加速）张量存活和对象可达性审计；不修改正式训练清理行为。
- V2 只有 Control 不进入图路径、Treatment 图计数有效且与追踪一致、三次参数哈希不变、更新计数为零、未访问 target_test（目标测试集）时才可报告 PASS；当前尚未运行 GPU 诊断。
- V2 CPU（中央处理器）回归、全量 M1 相关测试共 158 项全部通过；AST（抽象语法树）检查和 `git diff --check`（差异格式检查）通过。GPU 诊断尚未运行。
- 本轮仍不改变模型、图传播、损失、数据、配方、训练参数或实验逻辑；V4 仍为 `INCOMPLETE_VRAM_THRASHING`，不得恢复或用于实验结论。

> 以下 V4 重放修复内容为历史审计记录，不属于当前显存诊断的实施范围。

## 本轮 P0 修复范围

- `remainder=0` 时检查点的 `resume_replay_batch_ids` 和 `resume_reissue_batch_ids` 严格为空；保存检查点只构造序列化副本，不修改 live sampler（运行中采样器）。
- 累积余数来自独立的实际微批计数器，不再由跨轮 `global_step` 与单轮 `processed_batches` 推导。
- 非整除轮末依据 Accelerate（加速训练框架）的 `end_of_dataloader`（数据遍历结束）信号完成真实尾部更新，epoch（轮次）检查点不得带未完成累积。
- fresh run（全新运行）不得产生 `batch_replayed`；显式检查点恢复才允许重放，并记录检查点路径、哈希和恢复批次身份。
- Phase A 最终门控必须读取 DANN journal（领域批次日志），验证 fresh Control/Treatment（全新对照组/实验组）的 `replay_count=0`。

## 当前验证与边界

- RED（失败先行）已复现：原实现中 `[-0:]` 会重放完整历史，且保存检查点会污染 live sampler；新增状态接口和 906/16/25 边界测试在修复前失败。
- GREEN（修复后）：M1 相关 CPU 回归 `141 passed, 0 failed`；未启动 GPU（图形处理器）、正式训练、伪标签实验或 target_test（目标测试集）。
- 不修改句法图公式、图结构、DANN 公式或 `lambda_domain_adv=0.03`，不修改数据、标签、训练参数、伪标签规则和研究范围。
- 修复后正式运行必须使用全新 v4 目录，并等待 Codex Sol 复审通过及用户明确启动；未经再次授权不得删除 v3。

## 本任务批准范围

- 先以 TDD（测试驱动开发）复现重复采样轮次、末尾部分遍历、journal（日志）崩溃恢复、issued/processed（已发出/已确认）窗口、恢复重放、步区间、Control/Treatment 对齐和 legacy 不得正式 PASS。
- 新审计区分单调物理 DataLoader（数据加载器）遍历序号与既有 `int(state.epoch)` 采样轮次；计划批次、issued/processed 批次、完整/部分遍历及 Trainer（训练器）global step 区间均独立记录，旧洗牌顺序由确定性回归测试核对。
- 旧运行只能执行显式 `legacy_diagnostic_migration`（旧运行阻塞/迁移审计）；该路径写迁移报告、保留旧提交和产物哈希、拒绝训练续跑，不修改 `stage_status.json`，只能新目录正式重跑。
- 只做 CPU（中央处理器）测试、静态检查和既有产物只读核验；禁止 GPU（图形处理器）训练、伪标签推理、target_test（目标测试集）、删除、合并和推送。

## Phase A 批准范围

- Control：原始 T5 pseudo extractor（伪标签抽取器），`graph_enabled=false`。
- Treatment：source extractor training（源域抽取训练）、source-dev evaluation（源域开发集评估）、target-unlabeled DANN（目标无标签领域对抗）和 target pseudo inference（目标伪标签推理）启用句法 RGAT。
- 图模块不得进入 generator（生成器）、augmentation（数据增强）、NLI（自然语言推断）、exact/conflict filtering（精确回抽/冲突过滤）、selector（选择器）、final ASTE（最终方面级情感三元组抽取）或 target test（目标测试集）。
- Control 只有在方向、种子、数据划分、配方、检查点选择规则、模型/分词器、代码语义、配置和产物哈希全部机器匹配时才允许复用；任何未知项均从头重跑。

## Phase A 门控与硬停止

- A1：source-dev strict triplet F1（源域开发集严格三元组 F1）处理组相对对照组差值 `>= -1.0` 个百分点。
- A2：source-dev 多三元组句 recall（召回率）差值 `>= +2.0` 个百分点。
- A3：处理组 overall/aspect/opinion absence rate（总体/方面/观点缺失率）满足总体不高于对照组，且方面或观点至少一项改善；方面和观点缺失可重叠，不宣称独立因果。
- A4：处理组 qualified multi pseudo rows（合格多三元组伪标签行）至少为对照组的 `1.05` 倍，qualified total pseudo rows（合格伪标签总行数）至少为对照组的 `0.95` 倍；不得读取目标测试金标。
- 只有 A1–A4 全部通过才标记 Phase A PASS；任一失败输出 `STOP_M1_SYNTACTIC_GRAPH_UPSTREAM`，禁止 generator、augmentation、final training（最终训练）和 target test。全部通过时只输出 `REQUEST_PHASE_B`，等待新的 Chat Sol（研究负责人）和用户批准。

## 本轮最终整体验收修复

- 成对 DANN（领域对抗）模式的生成损失只按有效源域生成权重归一化；目标行继续 `labels=-100`、生成权重为 0；旧的非成对流程保持原有全批次权重语义，DANN 系数仍为 0.03。
- Control/Treatment（对照组/实验组）初始化使用隔离随机状态；共享 T5 参数和 DANN head（领域分类头）初始哈希必须一致，只有 Treatment（实验组）拥有句法图参数；`phase_a_initialization_audit.json` 纳入训练阶段产物身份。
- 成对 DANN 检查点原子保存采样器状态、截至当前物理遍历的 schema 3 审计和身份清单；恢复只选择最新身份合法且 `resume_complete` 的检查点，损坏、未确认终止点或半写入点安全回退；run-level（运行级）快照由 journal 原子恢复。
- 配方、三个真实输入、T5 实际加载文件、外部 Control、DANN 审计、A4 伪标签产物和阶段产物均重新计算并硬校验；Control qualified multi（合格多三元组）为 0 时 A4 为 `BLOCKED`、ratio（比率）为 `undefined`。
- 本轮没有改变句法图公式、图结构、损失系数、Gate（门控）、伪标签规则、Phase B 或正式研究范围；已用 906/16/25→1400 确定性核对证明旧洗牌语义保持，正式 Phase A 尚未运行，目标测试仍禁止访问。

## 冻结参数与输出

- 冻结 T5-base、seed=1000、optimizer（优化器）、LR（学习率）、epoch（轮数）、batch（批大小）、checkpoint selection（检查点选择）、pseudo decoding/filtering（伪标签解码/过滤）、pseudo weight=0.75、DANN=0.03、generator/augmentation 配方、k=1 和 final ASTE 架构；不做参数搜索。
- Phase A 输出 `phase_a_summary.json`、`phase_a_result_CN.md`、`stage_status.json`、`control_identity_audit.json`、配置快照、Git 身份、父运行身份和文件哈希，并支持 `--resume` 与长阶段进度条。
- 本轮仅修复 Phase A 最终验收中的成对 DANN 生成损失归一化、Control/Treatment 初始化混杂、完整检查点恢复、配方/输入冻结和 DANN 审计真实性；运行 CPU（中央处理器）测试与静态检查，不启动 GPU（图形处理器）实验、正式训练、正式伪标签或目标测试。Phase A 的 Control/Treatment（对照组/实验组）抽取器训练、源域开发评估和目标无标签伪推理只允许由专用入口在用户运行实验时执行。

## V3 修复状态（2026-08-28）

- V3 修复针对 Sol 复审的终止边界、issued/processed 恢复、梯度累积检查点、完整状态门控和图缓存启动预检；不把提交 `63fd1a6` 视为验收通过。
- 采样语义保持既有 `int(Trainer.state.epoch)` → `seed + sampling_epoch`，新增的 `physical_traversal_index` 只用于审计，不参与洗牌；未获研究批准前不改变采样顺序。
- 终止前在准备下一批时读取 Trainer 状态，并以 `issued == processed == planned` 才标记 complete；issued 未确认批次恢复从 `processed_batches` 重签，绝不跳过。
- 检查点新增原子梯度累积快照与累积偏移身份；无法证明梯度/计数安全的旧检查点硬拒绝。Phase A 启动前逐项校验 graph cache 的 manifest、relation vocabulary 和三个 split 文件身份哈希。
- 本轮只做 CPU（中央处理器）测试、静态检查和文档同步；六个 M1 测试文件 `103 passed`，`py_compile` 与 `git diff --check` 均为退出码 0。未运行 GPU、正式训练、伪标签推理或 target_test。

## 当前实现完成状态

- Phase A 专用入口、固定配方、Control 身份审计、A1-A4 门控、断点恢复和硬停止已实现；实际 Phase A 运行尚未启动。
- 本轮复审修复已补齐 Phase A 成对 DANN（领域对抗）批次：每个逻辑批次严格为 source=1/target=1，目标行标签为 `-100`、生成权重为 0，并记录逐 epoch（轮次）组成；同时为六个训练/评估/伪推理阶段写入可重算的命令、输入、配方、模型/输出、解析模型路径和 producer commit（产物提交）身份。
- 本轮进一步修复了六个 A4 伪标签产物的逐项身份校验、训练模型与 DANN 审计报告联合身份、外部 Control 的 DANN 报告硬要求，以及显式 `set_epoch`/`state_dict`/`load_state_dict` 采样器恢复；Control/Treatment（对照组/实验组）批次种子、行 ID、顺序和轮次数必须一致。
- RED（失败先行）证据：首轮新增接口测试在实现前因 API（接口）不存在而失败；GREEN（修复后）证据：全部 M1 相关 CPU 测试共 124 项通过，包含成对损失、初始化、阶段身份、DANN 审计、检查点恢复、配方突变和 A4 零分母边界；AST（抽象语法树）检查和 `git diff --check`（差异格式检查）通过。
- 当前仍未运行 GPU（图形处理器）、正式训练、正式伪标签实验或目标测试；Phase B 仍为 `NOT APPROVED`，M1 不得提前标记为最终通过，正式实验索引暂不更新。

## 前置核验状态

- 指定正式环境的 `stanza==1.14.0`、`torch==2.2.2+cu121` 及其运行依赖已通过导入验证，CUDA（英伟达 GPU 加速）可用；
- 用户 CMD（命令提示符）验证已确认 `resources.json`、EWT 模型文件和 Stanza English EWT（Stanza 英语 EWT）解析管线均可用，实际设备为 CUDA；
- 固定包映射已确认：`tokenize=ewt`、`mwt=ewt`、`pos=ewt_charlm`、`lemma=combined_nocharlm`、`depparse=ewt_charlm`；单句输出为 1 个句子、5 个词，依存关系完整；
- 任务协议要求的模型文件 SHA256（哈希）已由用户在 CMD 中记录：`resources.json=4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c`、`tokenize/ewt.pt=fc2fed0cd74dbaef1620bd3e776141ae76c4e28eb5aeff369b2715c31cc73cba`、`mwt/ewt.pt=73411a30da7638bbda2ebd9490e017d78feb4e029e90c9f5c9f37e5433292eb0`、`pos/ewt_charlm.pt=f89696d286c29aff173061fbd4b581c73525257ce38015804be047a5e40f9614`、`lemma/combined_nocharlm.pt=e3cb21e3c97a514d102fcc95e78fbc2ab838bc7b306a48029022f35caba1aa2c`、`depparse/ewt_charlm.pt=7386666c2054363f6c4eae702f84ef7d4a11aa4708c2907b82b105e56925d897`；
- 固定 EWT 解析器前置核验完整通过；当前批准范围包括通过专用 Phase A 入口执行 Control/Treatment 抽取器训练、source-dev 评估和 target-unlabeled 目标无标签伪推理。本轮代码修复本身不启动这些运行。

## 本任务边界

- 不得从 `t5_absa_train.py` 直接绕过专用入口启动图训练；Phase A 允许的训练、评估和目标无标签伪推理必须保持成对 DANN 批次与逐阶段身份审计；
- 本轮代码修复期间不启动 Phase A GPU（图形处理器）实验、正式训练或新 target pseudo labels（目标伪标签）；
- generator（生成器）、augmentation（增强）、NLI/exact、final ASTE、Phase B（下游阶段）、target test（目标测试）及任何目标测试金标读取始终禁止；
- 修改 parser（解析器）、图层数、attention heads（注意力头数）、graph hidden size（图隐藏维度）、DANN（领域对抗网络）、伪标签权重、筛选阈值或现有生成器；
- fallback 到无图、删除失败行、参数网格、隐式改 tokenizer（分词器）或启动未经批准的 GPU（图形处理器）实验；不得绕过专用审计脚本进入图训练。

## 本轮策略修复结论

- 上一版全量只读预检已完成 1730/1730 行，字符覆盖率 100%、failed_rows=0、目标测试隔离为 false；其中 12 项 `alignment_policy_violation` 已确定属于合法的连续部分共享边界，包括缩写、`Registration/1st` 和 `WIth`，不是数据失败。
- 统一策略版本更新为 `overlap-contiguous-contained-sharing-v3`：共享子词要求解析词位置连续、解析词跨度连续、同句、子词跨度包含于解析词联合跨度且与每个记录词有非空重叠；`exact_union` 保留为诊断字段，并新增 `contained_in_parser_union` 和 `partial_contiguous_shared_subword`。
- `abx` 联合覆盖、跨空格、非连续解析词、跨句、越界和字符覆盖缺口仍被拒绝；错误信息现在分别标识 `shared_subword_outside_parser_union`、`non_contiguous_shared_subword`、`cross_space_shared_subword` 和 `cross_sentence_shared_subword`。旧版 BLOCKED 预检目录未删除或覆盖。
- 本轮新增 CPU 测试 2 项，覆盖 7 类部分共享及预检输出；M1 相关直接测试共 78 项全部通过，AST（抽象语法树）解析和 `git diff --check`（差异格式检查）通过。新版 GPU 全量预检、正式训练、伪标签生成和目标测试读取均未执行；M1 尚不能写成已通过，正式实验索引暂不更新。

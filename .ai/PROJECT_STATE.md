# CD-C3DA 项目状态

> 更新时间：2026-08-28（北京时间）

## 当前工程修复状态

`M1_DANN_AUDIT_RESUME_FIX_V1`（Phase A DANN 审计与恢复修复）已按复审要求完成代码实现和 CPU（中央处理器）验证，状态为 `APPROVED`（已批准实施，未批准运行实验）。修复只影响成对 DANN 的审计身份、原子落盘、检查点恢复和正式验证，不改变模型公式、图传播、损失、DANN=0.03、数据、优化器、调度器或训练参数。旧入口的采样顺序语义已恢复：`sampling_epoch=int(Trainer.state.epoch)` 仍只用于 `seed+sampling_epoch` 洗牌，独立的 `physical_traversal_index` 不参与洗牌。

- `PairedDomainBatchSampler`（成对领域批次采样器）现在独立记录物理 DataLoader（数据加载器）遍历序号、既有采样轮次、计划/issued（已发出）/processed（已确认）批次数、完整/部分遍历和 Trainer（训练器）global step（全局步数）身份；物理序号不参与洗牌。
- 审计协议升级为 schema 3（模式3）：每批只追加带完整行校验和哈希链的 journal（日志），遍历边界、检查点和正常结束时才原子压缩快照；崩溃后可从完整日志行恢复，且不会把 issued 冒充 processed。
- 成对检查点只接受 schema 3、身份与哈希一致且恢复语义有效的点；`resume_complete` 与 `training_terminal_partial` 分离，末尾部分遍历只有 Trainer global step==max_steps 且 issued==processed 时才可作为终端恢复点，未确认批次会硬拒绝。
- Phase A 验证器不再固定要求 `len(epochs)==num_train_epochs`；独立依据冻结配方和实际成对 DataLoader 长度计算 max_steps，并严格校验每次遍历的 optimizer step 区间、单调性、边界连续性、最终 global step 和 Control/Treatment（对照组/处理组）逐批对齐。
- 旧运行只能通过显式 `legacy_diagnostic_migration`（旧运行阻塞/迁移审计）写入 `legacy_diagnostic_migration.json`（旧版迁移报告）；该路径保留旧提交和产物哈希、拒绝训练续跑、不修改 `stage_status.json`，旧运行仍是方向性诊断而非正式通过证据。
- TDD（测试驱动开发）新增边界覆盖重复采样轮次、906/16/25→1400 采样哈希、追加日志规模、落盘后未确认崩溃、恢复重放、步区间、终止部分遍历、Control/Treatment 对齐和 legacy 不得正式 PASS（通过）。M1 六个测试文件共97项通过；完整项目回归以最终命令输出为准。

## 当前任务状态

`M1_SYNTACTIC_RGAT_PSEUDO_QUICK_ABLATION_V1`（laptop14 -> rest15）当前为 `APPROVED`（已批准），固定种子为 1000，入口前置 zero-update 审计为 15/15 PASS（通过），父代码身份为 `158654021fc5f26bf1cfb8e803d7d1b592bd8534`。本轮只实现 Phase A（上游阶段）快速消融入口；实际 GPU（图形处理器）运行、正式训练、正式伪标签实验和目标测试读取尚未执行，Phase B（下游阶段）仍未批准。

## 当前 Phase A 实现状态

- 新增专用 `m1_syntactic_rgat_pseudo_quick_ablation.py` 和固定配方，覆盖 Control（无图）/Treatment（句法 RGAT）四个上游调用点、Control 身份机器复核、A1-A4 门控、断点恢复、配置/代码/输入/模型文件哈希和中文/JSON（结构化）结果输出。
- 现有 `t5_absa_train.py` 的直接图训练入口继续硬停止；只有专用 Phase A API 可在入口完成配方、Git（版本管理）和数据边界校验后调用既有训练主体。图模块不进入生成器、增强、NLI（自然语言推断）、选择器、最终 ASTE（方面级情感三元组抽取）或 target_test（目标测试集）。
- 当前只完成 CPU（中央处理器）/静态实现验证，未运行 Phase A 实验，因此不能写成 Phase A 或 M1 已通过；正式实验索引暂不更新，等待用户实际运行结果。
- 本轮最终整体验收修复已完成：成对 DANN 生成损失按有效源域权重归一化；Control/Treatment 初始化隔离并输出 `phase_a_initialization_audit.json`；完整检查点原子保存采样器状态、已完成轮次审计和身份清单；恢复校验支持损坏点安全回退；配方、真实输入、T5 文件、DANN 审计和 A4 零分母均硬校验。
- 本轮修复不改变句法图传播公式、图结构、DANN=0.03、Gate、伪标签规则、Phase B 或最终 ASTE（方面级情感三元组抽取）研究范围。

## 当前任务边界

- 本轮只修复 Phase A 最终验收中的成对 DANN 生成损失语义、Control/Treatment 初始化、完整检查点恢复、配方/真实输入冻结和 DANN 审计真实性；不修改句法图公式、图结构、DANN 系数、Gate、伪标签规则、实验研究范围或数据增强逻辑。
- Phase A 允许通过专用入口执行 Control/Treatment（对照组/实验组）抽取器训练、source-dev（源域开发集）评估、目标无标签 DANN 和目标伪推理；本轮代码修复不启动 GPU（图形处理器）实验、正式训练、正式伪标签或目标测试。
- 不使用 `nan_to_num`、梯度裁剪、关闭图模块或改成 FP32 掩盖异常；Phase B（下游阶段）、生成器、增强、NLI（自然语言推断）、最终 ASTE（方面级情感三元组抽取）和 target test（目标测试）始终禁止。
- 本轮复审只修复 Phase A 的成对 DANN 批次和逐阶段产物身份恢复：每个逻辑批次严格为 source=1/target=1，目标行仅承担 DANN 损失；Control/Treatment 抽取器训练、source-dev 评估和 target-unlabeled 伪推理仍只能经专用入口执行。A4 六个伪标签产物、训练模型与 DANN 审计报告均逐项记录和重算校验；外部 Control 缺少报告或报告/批次身份不一致时硬失败。
- PairedDomainBatchSampler（成对领域批次采样器）在正式 Trainer（训练器）入口复用旧的 `int(state.epoch)` 采样轮次 provider（提供器），并以独立 physical traversal ID（物理遍历 ID）审计；额外 DataLoader（数据加载器）迭代不会覆盖身份，恢复后批次顺序按既有 `seed+sampling_epoch` 保持一致。

## 本轮接口修复状态

- `generate_texts` 新增可选 `graph_cache_identity_rows`：完整 `target_unlabeled` 行只进入 `load_graph_cache_directory` 的 manifest 身份验证，`graph_rows` 仍只负责当前子集的 `GraphCache.get` 和批次整理；未提供新参数时回退到原有 `graph_rows` 行为。
- 审计入口已传入完整 `target_rows` 和当前 `target_sample`，仍只保留一条内存诊断结果；缓存 manifest 哈希校验、子集行 ID/文本/输入哈希硬拒绝和 `target_test` 禁止访问均未放宽。
- 本轮 M1 相关 CPU 直接测试共 124 项通过，覆盖成对生成损失等价、真实微型 T5 初始化哈希、阶段产物篡改、外部 Control、DANN 报告真实性、采样器恢复、配方突变、A4 零分母和 target_test（目标测试集）硬停止；AST（抽象语法树）检查与 `git diff --check` 通过。未运行 GPU、正式训练、正式伪标签或目标测试；Phase B 仍未批准。

## 上一任务只读对齐预检状态

- 上一版全量只读预检完成 1730/1730 行，字符覆盖率 100%、failed_rows=0、目标测试隔离为 false；12 项 `alignment_policy_violation` 已确定属于合法的连续部分共享边界，包括缩写、`Registration/1st` 和 `WIth`。
- 新版 `overlap-contiguous-contained-sharing-v3` 由 `syntactic_graph.py` 的公开 `validate_alignment_policy` 统一提供，正式图缓存与预检共同使用；`exact_union` 保留，同时记录 `contained_in_parser_union` 和 `partial_contiguous_shared_subword`。`GRAPH_SCHEMA_VERSION`（图模式版本）保持不变，旧缓存因策略版本变化不可复用。
- 预检仍只扫描 source_train、source_dev 和 target_unlabeled，不请求 target_test；`abx`、跨空格、非连续解析词、跨句、越界和字符缺口仍拒绝。旧版 BLOCKED 预检目录未删除或覆盖。
- 可疑清单包含合法连续共享和所有异常类型；机器可读汇总记录句/词/子词统计、分布、成功/失败行、覆盖率、截断和 PASS/BLOCKED 门控。
- 本轮新增 CPU 测试 2 项，覆盖 7 类部分共享及预检输出；M1 相关直接测试共 78 项通过，未更新正式实验索引，等待新版实际预检结果后统一记录。

## 上一轮参数协议修复证据

- 审计启动在读取数据、加载模型和探测 CUDA 前硬校验配方与命令行：source/target 数据集、seed=1000、lambda_domain_adv=0.03、fp16、gradient_checkpointing、训练/评估/DANN/伪推理批次以及 max_source_length=128、max_target_length=96；错误项立即生成 BLOCKED（阻塞）报告。
- 源域抽取训练固定使用 train batch=1，source-dev 评估固定使用 eval batch=2；target-unlabeled DANN 明确记录 source_batch_size=1、target_batch_size=1 和 total_batch_size，不再使用含义模糊的统一 batch_size。
- JSON（结构化）报告记录 actual、expected、recipe 和逐项 matches；当前冻结配方 SHA256 为 `774b3ca39b4bac29fa63d7a78c3c91f62b8d55695f83867298fad91827f78a3c`。
- 上一轮已通过复验的报告组装、图训练直达硬拦截、四个正式调用点和实际身份哈希审计保持不变；图结构、DANN 系数和研究方案未改动。
- CPU 新增 11 项直接测试、AST（抽象语法树）解析和 `git diff --check` 通过；pytest（Python 测试框架）未安装，未启动 GPU zero-update（零更新）审计、正式训练、伪标签生成或目标测试读取。

## 当前目标

六个跨域 ASTE（方面级情感三元组抽取）方向分别超过 BGCA（双向生成跨域方法）。不能用单方向、平均值或 fixed F1（修正后 F1）替代六方向正式 raw F1（原始严格 F1）比较。

## 六方向正式最佳与 BGCA 差距

| 方向 | CD-C3DA raw F1 | BGCA F1 | 差距 |
|---|---:|---:|---:|
| rest14 -> laptop14 | 52.54 | 53.64 | -1.10 |
| rest15 -> laptop14 | 45.27 | 45.69 | -0.42 |
| rest16 -> laptop14 | 48.93 | 47.28 | +1.65 |
| laptop14 -> rest14 | 56.94 | 65.27 | -8.33 |
| laptop14 -> rest15 | 54.01 | 58.95 | -4.94 |
| laptop14 -> rest16 | 61.55 | 64.00 | -2.45 |

rest16 -> laptop14 的 48.93 是唯一超过 BGCA 的保护基线，不在独立验收现场继续试验。

## 当前研究基线

laptop14 -> rest15 的正式基线为 raw P/R/F1（精确率/召回率/F1）= 56.98/51.34/54.01，fixed F1 = 55.53；负面 F1 = 49.46，多三元组 F1/召回 = 50.21/43.27%。伪标签有效权重为 0.75，DANN（领域对抗网络）为 0.03。

固定生成器 FGSM（快速梯度符号法）完整运行结果为 raw/fixed F1 = 53.76/54.65，已输出 NO_FGSM_GENERATOR_ROUTE；FGSM 的参数、损失比例、作用范围和结构定向搜索均永久关闭。实现、CUDA（统一计算设备架构）修复和零更新入口审计仅作为负结果基础设施保留。

## 主要瓶颈

- 目标域方面直接覆盖只有 2.0%–11.1%，多三元组贡献大量 FN（假阴性）。
- 锚句隔离的留出证据显示遗漏主要属于 element_absent（元素缺失）；纯 relation_unbound（关系未绑定）为 0。
- M2 可以形成符号层完整组合计划，但 M3 当前接口不能稳定把完整计划实现为目标域文本。
- 扩大候选或下游小损失会同时放大 FP（假阳性）或重新分配错误，不能代替上游结构供给。

## 当前研究主线

当前没有新的训练变量获准。M1（目标知识获取）冻结为有界辅助；M2/M3 没有形成合格文本监督；M4（精确验证）保留为硬门槛；M5 只保留联合配额 MILP（混合整数线性规划）基础设施；M6 的 DANN=0.03 按正式历史配置冻结。下一研究变量必须先由 Chat Sol（研究负责人）提出并经用户明确批准。

## 已关闭路线

已关闭：FGSM 参数和作用范围搜索、第三版图计划压缩与长度放宽、当前图计划训练路线、k=2、双生成器、教师—学生、伪标签调权和梯度补偿、回放、双呈现、普通质量/结构权重、NLL（负对数似然）选择或训练加权、EOS（序列终止）损失和解码限制、ECAL（元素覆盖辅助损失）、当前组合软提示、FP-LCR（冻结计划词汇约束）、语义换供体/阈值搜索、局部子句插入变体、M5 gap/coverage/novelty（缺口/覆盖/新颖性）次目标、DANN 开关或系数搜索，以及供给不变时新增 pairing（配对）、计数或结构损失。

## 保留部件与安全边界

保留单生成器、T5-base（T5 基础模型）、k=1、目标真实句锚定、硬契约、双向 NLI（自然语言推断）、extractor exact（抽取器精确回抽）、冲突拒绝、联合配额选择器、增强总有效质量 30、source-dev（源域开发集）选模和无金标 uptake（吸收）门控。目标测试金标只能在完成训练后用于最终报告，不能用于选模、调参、候选、阈值或门控。

正式实验默认由用户在 CMD（命令提示符）中运行。未获用户明确代跑授权，不启动 GPU、不删除文件、不合并或推送分支。所有实验遵守 RTX 3070 8GB（8 GB 显存）、版本兼容、进度条、断点恢复、干净工作树和阶段复用规则。

## 协作入口

先读取仓库事实，但不得自行猜测关键需求。普通实现由 Codex Luna（默认执行器）处理；复杂工程问题升级 Codex Sol（高级工程模型）；大量 JSON、CSV、日志和多运行分析交给 Work Luna（重文件分析模型）；新的研究决策交给 Chat Sol，并由用户最终批准。

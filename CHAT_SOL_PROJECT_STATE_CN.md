# CD-C3DA 项目状态

> 更新时间：2026-08-27 17:04（北京时间）

## 当前任务状态

`M1_SYNTACTIC_RGAT_FP16_NUMERICAL_TRACE_V1`（laptop14 -> rest15）当前为 `APPROVED`（已批准），V3 数值追踪已 PASS（通过），确认此前问题是基础 T5 检查点加载时新增自定义图适配器参数未初始化，而不是 FP16 图传播公式。完整 zero-update 入口审计当前 14/15 门控通过，唯一阻塞是 target pseudo inference 中完整 target_unlabeled 缓存身份与单条推理子集混用。本轮只修复接口职责分离和回归测试；新版 GPU 入口审计尚未运行，正式训练、正式伪标签生成和目标测试读取仍未批准/执行，M1 不能记为通过。

## 当前任务边界

- 不修改模型公式、训练逻辑、实验参数、图结构、DANN 系数或数据增强逻辑。
- 不使用 `nan_to_num`、梯度裁剪、关闭图模块或改成 FP32 掩盖异常；不创建优化器、更新调度器、保存模型或读取 target test；只允许记录目标无标签推理异常，不保存新的目标伪标签。
- 逐阶段报告 FP32 与 CUDA autocast FP16 的张量统计、首个非有限阶段、首个异常行/边、关系类型、入边数量、target pseudo inference 异常类型和消息；CPU 测试验证有限路径、人工溢出定位、参数不变和异常不静默吞掉。

## 本轮接口修复状态

- `generate_texts` 新增可选 `graph_cache_identity_rows`：完整 `target_unlabeled` 行只进入 `load_graph_cache_directory` 的 manifest 身份验证，`graph_rows` 仍只负责当前子集的 `GraphCache.get` 和批次整理；未提供新参数时回退到原有 `graph_rows` 行为。
- 审计入口已传入完整 `target_rows` 和当前 `target_sample`，仍只保留一条内存诊断结果；缓存 manifest 哈希校验、子集行 ID/文本/输入哈希硬拒绝和 `target_test` 禁止访问均未放宽。
- 本轮 M1 CPU 直接测试共 75 项通过，新增部分图检查点硬拒绝回归；未运行 GPU、正式训练、正式伪标签或目标测试，正式训练仍未批准。

## 上一任务只读对齐预检状态

- 上一版全量只读预检完成 1730/1730 行，字符覆盖率 100%、failed_rows=0、目标测试隔离为 false；12 项 `alignment_policy_violation` 已确定属于合法的连续部分共享边界，包括缩写、`Registration/1st` 和 `WIth`。
- 新版 `overlap-contiguous-contained-sharing-v3` 由 `syntactic_graph.py` 的公开 `validate_alignment_policy` 统一提供，正式图缓存与预检共同使用；`exact_union` 保留，同时记录 `contained_in_parser_union` 和 `partial_contiguous_shared_subword`。`GRAPH_SCHEMA_VERSION`（图模式版本）保持不变，旧缓存因策略版本变化不可复用。
- 预检仍只扫描 source_train、source_dev 和 target_unlabeled，不请求 target_test；`abx`、跨空格、非连续解析词、跨句、越界和字符缺口仍拒绝。旧版 BLOCKED 预检目录未删除或覆盖。
- 可疑清单包含合法连续共享和所有异常类型；机器可读汇总记录句/词/子词统计、分布、成功/失败行、覆盖率、截断和 PASS/BLOCKED 门控。
- 本轮新增 CPU 测试 2 项，覆盖 7 类部分共享及预检输出；M1 相关直接测试共 78 项通过，未更新正式实验索引，等待新版实际预检结果后统一记录。

## 上一轮参数协议修复证据

- 审计启动在读取数据、加载模型和探测 CUDA 前硬校验配方与命令行：source/target 数据集、seed=1000、lambda_domain_adv=0.03、fp16、gradient_checkpointing、训练/评估/DANN/伪推理批次以及 max_source_length=128、max_target_length=96；错误项立即生成 BLOCKED（阻塞）报告。
- 源域抽取训练固定使用 train batch=1，source-dev 评估固定使用 eval batch=2；target-unlabeled DANN 明确记录 source_batch_size=1、target_batch_size=1 和 total_batch_size，不再使用含义模糊的统一 batch_size。
- JSON（结构化）报告记录 actual、expected、recipe 和逐项 matches；配方 SHA256 已按新内容更新为 `e7c27b2a918eff11ae62bbb2ebc6042d80b457dfaaa21907ae9a0408115dece7`。
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

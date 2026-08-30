# 协作决策日志

> 更新时间：2026-08-30 16:10（北京时间）

## 2026-08-30：元素感知多三元组 RGAT 工程入口

变量限定为单一标量显著性头、消息源显著性注意力偏置、平衡元素聚焦损失和多元素覆盖损失；DANN（领域对抗网络）为0，固定权重0.05。服务器GPU（图形处理器）不可用时硬阻塞；GPU恢复后只运行入口审计并返回Chat Sol（研究负责人），不自行训练。

## 2026-08-29：V8 Phase A 已启动并进入 Treatment 训练

- 用户已从新目录启动 V8；Control（对照组）没有重训，运行器受控复用 V6 已审计 Control，`reuse_depth=1`。
- V8 当前从头训练唯一新变量 Treatment（带句法 RGAT 的伪标签抽取器），总计 1400 步；后续只执行源域开发集评估、目标无标签伪标签推理及 A1–A4 无金标 Gate（门控）。
- 当前日志数值有限且损失下降，只能说明运行正常，不能说明研究有效。V8 完成前不新增变量、不运行目标测试、不更新正式最佳。

## 2026-08-29：采用独立子进程并严格复用 V6 Control

- 同一 Python 进程清理 Control 后仍存在对象可达性，继续追加清理补丁不再作为方案；Control 与 Treatment 必须由不同操作系统子进程执行。
- V6 的唯一 `issued-but-unprocessed`（已签发但未处理）批次经 journal、优化步、梯度累积和终端检查点联合审计，判定为 `terminal_lookahead_not_consumed`（终端预取未消费）；V6 整次运行不翻案，但其 Control 允许受控复用一次。
- 新训练主动阻止相同预取；Windows 输入换行身份改为稳定 LF 写入与语义恢复校验。模型公式、图传播、DANN、参数、数据和 Gate 均未改变。
- 全项目 CPU 测试 425 项通过。下一运行必须使用新目录、外部 V6 Control、reuse_depth=1；不得恢复 V6/V7，不得再训练 Control，不得读取目标测试。

## 2026-08-28 20:40：完成 M1 显存归因诊断工具实现

- 新增独立只读 `m1_vram_attribution_audit.py` 和 CPU 合成测试；诊断固定使用真实 T5-base/图缓存、source=1/target=1、FP16、梯度检查点和 DANN=0.03，但只在用户手动运行 GPU 命令后执行。
- 记录模型加载、优化器、批次搬运、T5 编码器、词池化、节点/边投影、注意力、关系消息/聚合、图融合、解码、反向和清理后的显存与张量元数据，并比较三次 zero-update 的 Control/Treatment 峰值。
- 未改变模型、图传播、损失、数据、配方、训练参数或实验范围；GPU 诊断、正式训练、正式伪标签、Phase B、最终 ASTE 和 target_test 均未运行。V4 仍为 `INCOMPLETE_VRAM_THRASHING`，不得恢复或用于实验结论。

## 2026-08-28 20:17：批准 M1 显存归因专项诊断

- 任务 `M1_SYNTACTIC_RGAT_VRAM_ATTRIBUTION_AUDIT_V1` 已由用户批准，方向为 `laptop14 -> rest15`，父代码身份为 `8f165cf50ac30bcdee1a4173af54813087194f6c`。
- V4 运行因 Treatment（实验组）显存压力/抖动在约 134/1400 处不完整，状态标记为 `INCOMPLETE_VRAM_THRASHING`；不得恢复、删除或用于任何实验结论。
- 本轮只实现独立显存归因与入口审计：同一 source=1/target=1 批次、真实 T5-base、真实图缓存、FP16、梯度检查点和 DANN=0.03；记录 Control/Treatment 各调用点显存和张量信息，并执行至少三次 zero-update（零更新）诊断。
- 只运行 CPU（中央处理器）测试和静态检查；不运行 GPU（图形处理器）诊断、完整训练、正式伪标签、Phase B、最终 ASTE 或 target_test，不实施优化方案，不改变模型、图传播、损失、配方、训练参数或研究范围。

## 2026-08-28 17:16：V4 修复 live replay 污染（等待最终复审）

- v3 运行已确认被 `[-0:]` 的余数为零错误和 live sampler 状态旁路污染；已标记为 `INVALID_REPLAY_CONTAMINATED`，不得恢复或用于实验结论，未经再次授权不得删除。
- V4 只修复检查点序列化状态隔离、真实累积余数、非整除轮末尾部提交和 journal replay 门控；不改变句法图、DANN=0.03、数据、训练参数或研究范围。
- fresh run 必须 `batch_replayed=0`；显式 checkpoint（检查点）恢复才允许重放，且事件必须绑定检查点路径、哈希和恢复批次身份。
- CPU 回归已通过 141 项；未运行 GPU、正式训练、伪标签、生成器、增强、Phase B、最终 ASTE 或 target_test。等待 Codex Sol 最终只读验收后，再由用户决定是否从 v4 新目录启动。

## 2026-08-28：V3 复审修复（验收仍 BLOCKED）

- 不得把 `63fd1a6` 称为验收通过。复审确认的终止多签发已修复：采样器在准备下一批前读取 Trainer 状态及累积预算，终止边界不再追加 issued；`complete` 必须同时满足 issued=processed=planned，审计加载器和正式验证器均强制该条件。
- 恢复不再使用 `max(processed, issued)`；processed-but-unfinished accumulation（已处理但梯度尚未完成的累积）通过梯度快照/偏移恢复，issued-but-unprocessed 批次从 processed 位置重新进入真实 training_step。缺失或哈希不一致的梯度身份硬拒绝。
- Phase A 入口在长训练前预检 manifest、relation_vocab、source_train/source_dev/target_unlabeled 三个缓存文件及输入/文件哈希；缓存缺失或身份不符立即失败。正式命令必须使用已验证的 `J:\nlp\CD-C3DA\runs\diagnostics\laptop14_to_rest15_m1_syntactic_rgat_entry_audit_v4\graph_cache_resume`。
- 保持既有 `int(Trainer.state.epoch)` 洗牌语义；physical traversal ID 仅审计用途，不改变采样顺序。未运行 GPU、正式实验、伪标签推理或 target_test。

## 2026-08-28：完成 Phase A DANN 物理遍历审计与恢复修复（CPU 验证，未启动实验）

- 根因确认：旧采样器把浮点 `state.epoch`（轮次状态）取整后作为审计编号，并在生成器完全耗尽后才追加报告；梯度累积和训练器立即停止造成重复编号覆盖、缺失轮次和末尾报告丢失。旧运行 `control/dann_batch_audit.json` 保留为 legacy（旧版）方向性诊断，不伪造缺失观测、不把门槛降为21。
- P0 修复：物理 DataLoader（数据加载器）遍历序号与既有采样轮次分离；Trainer（训练器）仍通过显式 provider（提供器）使用修复前的 `int(state.epoch)` 洗牌语义。906/906、梯度累积16、25轮/1400步边界模拟已核对轮次标签和批次哈希，未改变采样顺序。
- P1 修复：schema 3（模式3）采用带哈希链的追加式 JSONL journal（日志），每批只追加小记录；epoch/checkpoint/正常结束再写原子快照。`issued_batches` 与 `processed_batches` 分离，Trainer 的成功 `training_step` 后才 acknowledge（确认）；恢复严格校验身份、单调性、步数区间、物理遍历完成度和 Control/Treatment（对照组/处理组）逐批对应。
- 终止与恢复：区分 `resume_complete` 与 `training_terminal_partial`；最终达到 max_steps（最大步数）但物理遍历未完整时，仅在 issued==processed 且身份有效时允许终端恢复，未确认或身份缺口硬拒绝。旧 schema 与旧运行不静默续跑。
- 兼容边界：新增显式 `--legacy_diagnostic_migration`（旧版阻塞/迁移审计）路径，并保留旧参数别名；只写迁移报告，保留源提交/产物哈希，不修改 `stage_status.json`，返回 BLOCKED（阻塞），只能在新目录正式重跑。
- 测试结果：六个 M1 测试文件 97 项 CPU 测试通过；全项目 CPU 测试 366 通过、7 项失败均为基线已有的 Windows 编码/工作流 Skill 断言问题，与本修复无关；AST（抽象语法树）编译检查和 `git diff --check` 在最终提交前复核。未启动 GPU（图形处理器）训练、正式实验、伪标签推理或 target_test（目标测试集）。
- 实验逻辑：模型公式、图传播、DANN=0.03、数据、batch（批大小）、优化器、调度器和研究范围不变；旧采样语义已由 provider 与边界哈希测试保持。最早失效阶段是 Phase A target-unlabeled DANN 审计及其真实下游，旧运行不晋级，修复后新目录从头运行。

## 2026-08-27 10:53：完成全量词—子词对齐只读预检入口，等待实际 GPU 预检

- 新增独立 `m1_alignment_preflight.py` 和 8 项 CPU 测试；按 source_train、source_dev、target_unlabeled 逐行扫描，行级失败写入可疑清单后继续，不请求 target_test。
- 预检复用 Stanza English EWT、T5 fast tokenizer（T5 快速分词器）和 `overlap-contiguous-sharing-v2`；检测未对齐、字符覆盖、非连续/跨空格/跨句共享、越界、合法连续共享、异常多子词、子词多词和截断未覆盖。
- 输出目录固定为 summary JSON、suspicious JSONL 和中文 Markdown（标记文档）三文件；进度按数据划分显示，恢复硬校验 Git commit、输入 SHA256、解析器/分词器身份、策略版本和 `max_source_length=128`。
- 新增测试 8 项、M1 图相关回归 56 项、AST 解析和 `git diff --check` 通过；本轮不运行 GPU 预检、正式训练、伪标签或目标测试，不更新正式实验索引，待实际预检结果后统一记录。

## 2026-08-26 22:59：补齐审计配方参数硬校验与显式批次协议，等待最终验收

- 审计启动先读取 recipe 并逐项比较实际参数、协议预期和 recipe 值；source_dataset、target_dataset、seed、lambda_domain_adv、fp16、gradient_checkpointing、全部批次参数以及 max_source_length/max_target_length 任一不匹配都会在数据、模型和 CUDA 之前抛出 `AuditConfigurationError`，主入口输出 BLOCKED JSON。
- 固定协议为 seed=1000、lambda_domain_adv=0.03、fp16=true、gradient_checkpointing=true、extractor train batch=1、source-dev eval batch=2、DANN source/target batch=1/1、target pseudo batch=1、max source/target length=128/96；DANN JSON 额外记录 source、target 和 total 组成。
- 新增错误 seed、错误批次、错误长度及“拒绝数据/CUDA 前继续”的 CPU 测试；当前 11 项审计测试、AST 解析和 `git diff --check` 通过。pytest 未安装，不安装依赖。
- 本轮不运行 GPU 审计、不启动正式训练、不生成伪标签、不读取目标测试；图结构参数和 DANN 系数保持冻结。下一步等待 Codex Sol 最后一次只读验收。

## 2026-08-26 22:34：按 Codex Sol 复审意见完成四项入口修复，等待再次验收

- 修复 `run_audit` 报告组装阶段的局部变量引用错误；报告现在从 `model_measurements["measurements"]` 读取梯度检查点状态，并由 CPU 测试覆盖该组装函数。
- `t5_absa_train.py` 的图训练入口在参数解析后硬停，只有专用 `m1_syntactic_graph_entry_audit.py` 可进入图零更新审计；无图流程不受影响。
- 审计四个调用点改走正式 `WeightedSeq2SeqTrainer.compute_loss`、`prediction_step`、混合批次 `compute_loss` 和 `t5_aste_pipeline.generate_texts`；不调用 `train`、优化器/调度器更新或模型保存。
- 审计实际重算并比较 Git、配方、T5 配置/权重/分词器、source train/dev 与 target unlabeled 输入、三类图缓存/关系词表/manifest、Stanza 六文件及控制组/处理组参数前后哈希，并将实际值写入 JSON；身份不一致按对应门控失败。
- CPU 新增测试、正式训练器烟测和相关旧回归通过；未启动 GPU zero-update（零更新）审计、正式训练、伪标签生成或目标测试读取。当前状态仍为 `RUNNING`，等待 Codex Sol 再次只读验收。

## 2026-08-26 21:53：M1句法图工程修复完成，等待 Codex Sol 验收

- 保留固定 Stanza English EWT（Stanza 英语 EWT）解析身份、六文件 SHA256（哈希）硬校验、字符偏移对齐、依存正反向边、POS（词性）邻接双向边、自环、1 层 4 头 256 维 RGAT（关系图注意力）、零初始化 `W_o` 和门控残差；未修改图结构参数。
- 目标无标签 DANN 只沿既有训练接口和审计路径接入，系数固定 `0.03`；目标行 `domain_label=1`，ASTE 标签全为 `-100`，不参与生成损失、结构损失或伪标签监督。
- 新增 `m1_syntactic_graph_entry_audit.py`：只读取 source train/dev 与 target train 无标签文本，覆盖源抽取训练、源开发评估、目标 DANN、目标伪推理四个调用点；输出 15 项逐项 `PASS/FAIL`、优化器/调度器/参数更新计数、参数前后哈希、梯度路径、fp16（半精度）、3070 8GB 显存和 JSON 报告。审计脚本不启动正式训练、生成器、增强、NLI、最终 ASTE，也不读取目标测试。
- 缓存改为 partial/progress（部分文件/进度文件）逐行追加、fsync（落盘同步）、身份校验和确定性复建；正式运行器新增 `--syntactic_graph_entry_audit_only`，图开关未带专用审计标志时硬停，普通无图流程保持原路径。
- 验证结果：语法编译通过；M1 图、DANN、入口边界和恢复测试通过；旧回归直接调用 122 项通过；`git diff --check` 已通过。pytest（Python 测试框架）未安装，使用当前环境 Python 直接调用测试函数完成验证；尚未运行 GPU zero-update 审计。
- 当前仍为 `RUNNING`：等待 Codex Sol 只读工程验收；通过后再等待用户明确“你来跑”才运行唯一 zero-update 审计命令，正式训练仍需单独批准。

## 2026-08-26 14:54：M1句法图 EWT 解析器与模型哈希前置核验完成

- 用户 CMD 已确认 `cuda_available=True`，固定 Stanza English EWT 模型全部加载完成，包映射为 `tokenize=ewt`、`mwt=ewt`、`pos=ewt_charlm`、`lemma=combined_nocharlm`、`depparse=ewt_charlm`。
- 单句验证输出 1 个句子、5 个词，依存关系为 `det`、`nsubj`、`cop`、`root`、`punct`；这证明当前 parser（解析器）和模型路径可用。
- 用户 CMD 已记录六个文件 SHA256（哈希）：`resources.json=4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c`、`tokenize/ewt.pt=fc2fed0cd74dbaef1620bd3e776141ae76c4e28eb5aeff369b2715c31cc73cba`、`mwt/ewt.pt=73411a30da7638bbda2ebd9490e017d78feb4e029e90c9f5c9f37e5433292eb0`、`pos/ewt_charlm.pt=f89696d286c29aff173061fbd4b581c73525257ce38015804be047a5e40f9614`、`lemma/combined_nocharlm.pt=e3cb21e3c97a514d102fcc95e78fbc2ab838bc7b306a48029022f35caba1aa2c`、`depparse/ewt_charlm.pt=7386666c2054363f6c4eae702f84ef7d4a11aa4708c2907b82b105e56925d897`。
- 固定 EWT 前置核验完整通过，状态从 `BLOCKED` 更新为 `APPROVED`；允许进入已批准的 Python 实现入口与 zero-update 审计，正式训练仍未批准。

## 2026-08-26 14:05：M1句法图伪标签接口实现入口阻塞

- 用户批准 `M1_SYNTACTIC_RGAT_PSEUDO_INTERFACE_V1` 进入 implementation entry + zero-update audit（实现入口与零更新审计），方向为 `laptop14 -> rest15`、seed 1000；正式训练仍未批准。
- 用户随后补齐 `udtools==0.2.8`、`udapi==0.5.2`、`emoji`、`platformdirs`、`colorama` 和 `termcolor`，核心导入已通过；但验证命令使用不存在的 `J:\nlp\models\stanza\_resources`，实际 `J:\nlp\models\stanza_resources` 缺少 `en\tokenize\ewt.pt`，且资源身份文件仍不可读，English EWT 模型核验未完成。
- 按任务边界不自动安装、不下载、不改 parser（解析器）、不替换为其他 UD（通用依存）解析器，因此状态为 `BLOCKED`，未进入代码实现、测试或 zero-update 审计。

## 2026-08-26：固化精简协作架构

### 决策

采用四角色协作：Chat Sol（研究负责人）负责研究决策；Codex Luna（默认执行器）负责普通实现；Codex Sol（高级工程模型）只处理复杂工程问题；Work Luna（重文件分析模型）负责大量实验文件分析。

### 事实与边界

- 能从仓库可靠确认的事实不重复询问；存在不理解、信息冲突或关键歧义时必须先向用户确认，不得自行猜测后执行。
- ChatGPT 建议不等于用户批准；新的研究变量必须经用户明确批准。
- 用户未明确授权代跑时，Codex 只修改代码或文档、做必要测试并给出命令。
- 目标测试金标保持隔离，实验复用、3070 8GB 约束、删除审批和双目录同步规则不变。

### 文档决定

- 以 .ai/PROJECT_STATE.md 保存长期事实。
- 以 .ai/CURRENT_TASK.md 保存唯一当前任务和受限状态。
- 以 .ai/DECISION_LOG.md 保存重要路线、基线、Gate、禁止路线和下一动作决定。
- 实验记录与模型索引_CN.md 继续承担 RUN_INDEX 职责，不创建重复索引。
- 00_CODEX新模型交接_CN.md 和 02_CD-C3DA实验工作流Skill_CN.md 保留，但退出日常读取和维护。

### 唯一下一动作

本次架构调整完成后不启动实验；等待用户或 Chat Sol 明确提出并批准新的研究任务。

## 2026-08-26 13:16：修正协作入口与可见性边界

- 新对话入口固定为：AGENTS.md → .ai/PROJECT_STATE.md → .ai/CURRENT_TASK.md → .ai/DECISION_LOG.md → 项目 Skill；实验索引、03 和 07 按任务需要读取，00 和 02 不作为日常入口。
- Chat Sol 不能直接读取 J 盘，只能读取已推送 GitHub 的文件或用户上传文件；本地未推送状态由 Codex 使用 RUN / CODE / CHANGE / REUSE / RESULT / BOUNDARY 精简格式传递。
- Skill 和 07 不重复维护完整角色列表，角色、升级和通信规则统一以 AGENTS.md 为准；Skill 保留实验协议，07 保留研究事实和同步边界。

# 当前任务

> 更新时间：2026-08-27 19:30（北京时间）

- 任务编号：M1_SYNTACTIC_RGAT_PSEUDO_QUICK_ABLATION_V1
- 任务类型：QUICK ABLATION（快速消融）
- 方向：laptop14 -> rest15
- 随机种子：1000
- 入口身份：M1 句法 RGAT zero-update（零更新）入口审计已达到 15/15 PASS（通过）；固定代码身份为 `158654021fc5f26bf1cfb8e803d7d1b592bd8534`
- 状态：APPROVED（已批准）
- 当前实现范围：仅实现 Phase A（上游阶段）可复现运行入口、Control/Treatment（对照组/实验组）身份审计、四项门控、断点恢复和硬停止
- Phase B（下游阶段）执行状态：NOT APPROVED（未批准）；本轮不实现或运行 Phase B

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

## 冻结参数与输出

- 冻结 T5-base、seed=1000、optimizer（优化器）、LR（学习率）、epoch（轮数）、batch（批大小）、checkpoint selection（检查点选择）、pseudo decoding/filtering（伪标签解码/过滤）、pseudo weight=0.75、DANN=0.03、generator/augmentation 配方、k=1 和 final ASTE 架构；不做参数搜索。
- Phase A 输出 `phase_a_summary.json`、`phase_a_result_CN.md`、`stage_status.json`、`control_identity_audit.json`、配置快照、Git 身份、父运行身份和文件哈希，并支持 `--resume` 与长阶段进度条。
- 本轮仅进行 DANN（领域对抗）成对批次和阶段身份恢复代码修复，运行 CPU（中央处理器）测试与静态检查；不启动 GPU（图形处理器）实验、正式训练、正式伪标签或目标测试。Phase A 的 Control/Treatment（对照组/实验组）抽取器训练、源域开发评估和目标无标签伪推理只允许由专用入口在用户运行实验时执行。

## 当前实现完成状态

- Phase A 专用入口、固定配方、Control 身份审计、A1-A4 门控、断点恢复和硬停止已实现；实际 Phase A 运行尚未启动。
- 本轮复审修复已补齐 Phase A 成对 DANN（领域对抗）批次：每个逻辑批次严格为 source=1/target=1，目标行标签为 `-100`、生成权重为 0，并记录逐 epoch（轮次）组成；同时为六个训练/评估/伪推理阶段写入可重算的命令、输入、配方、模型/输出、解析模型路径和 producer commit（产物提交）身份，外部 Control 路径与模型树哈希也纳入恢复校验。
- RED（失败先行）证据：新增阶段身份测试在实现前因 API（接口）不存在而失败；GREEN（修复后）证据：本轮新增 Phase A/DANN/resume 测试 9 项通过，全部 M1 相关 CPU 测试共 94 项通过，AST（抽象语法树）检查和 `git diff --check`（差异格式检查）通过。
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

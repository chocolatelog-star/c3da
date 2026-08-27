# 当前任务

> 更新时间：2026-08-27 17:04（北京时间）

- 任务编号：M1_SYNTACTIC_RGAT_FP16_NUMERICAL_TRACE_V1
- 任务类型：READ-ONLY DIAGNOSTIC IMPLEMENTATION（只读诊断实现）
- 方向：laptop14 -> rest15
- 状态：APPROVED（已批准）
- 用户批准范围：同一固定样本的 FP32（单精度）/FP16（半精度）逐层数值追踪、首个非有限张量定位、target pseudo inference（目标伪标签推理）异常记录、CPU（中央处理器）合成测试、机器可读 JSON（结构化报告）和中文报告；本轮只修复完整缓存身份与目标推理子集的接口分离
- 正式训练状态：NOT APPROVED（未批准）

## 已知阻塞事实

- M1 V3 数值追踪已通过（PASS）：此前的 trace-only（仅追踪）dtype mismatch（数据类型不一致）和基础检查点图参数未初始化问题均已修复，未改变 FP16 图传播公式。
- 完整 zero-update（零更新）入口审计当前为 14/15 门控通过，唯一失败为 `target_pseudo_inference`（目标伪标签推理）中的完整 `target_unlabeled` 缓存身份与单条推理子集混用。
- 本轮修复要求完整 `target_rows` 仅用于 manifest（清单）身份校验，`target_sample` 仅用于缓存读取和批次推理；不放宽身份校验，不生成正式伪标签，不读取 target test（目标测试集）。
- 正式训练仍为 NOT APPROVED（未批准），M1 尚未最终通过；新版 GPU（图形处理器）入口审计尚未运行。

## 本轮实现状态

- V3 数值追踪已 PASS；报告确认首个异常根因为基础 T5 检查点加载时新增自定义图适配器参数未初始化，而不是 FP16 图传播公式。
- 本轮仅修改 `t5_aste_pipeline.generate_texts` 的可选 `graph_cache_identity_rows` 接口和审计入口调用：完整身份清单与实际推理子集职责分离，默认旧调用保持原行为。
- 已补充目标子集外部/改 ID/改文本/长度不一致拒绝、完整清单加单条推理、审计接线、部分图检查点拒绝和 target test 隔离回归测试；M1 CPU 直接测试共 75 项通过。
- 新版 GPU 入口审计尚未运行；正式训练、正式伪标签生成和目标测试读取均未执行，M1 仍未最终通过。

## 批准范围

- 新增独立数值追踪入口，对同一固定 source train、source dev 和 target unlabeled（目标无标签）样本分别执行 FP32 完整前向与 CUDA autocast FP16（自动混合精度半精度）完整前向。
- 逐阶段记录编码器、词池化、投影、边注意力、logits、softmax、关系消息、图融合、解码器 logits、损失、反向梯度和 DANN domain loss/gradient（领域损失/梯度）。
- 报告必须给出 `first_nonfinite_stage`、FP32/FP16 首个非有限阶段、首个异常行、边及关系类型、入边数量和 target pseudo inference 异常类型与消息。
- 只读诊断必须输出机器可读 JSON 和简短中文报告；不得创建优化器、更新调度器、保存新模型或修改模型参数。

## 验收要求

- 每个张量记录 dtype、shape、finite/nan/正无穷/负无穷计数、min/max、max_abs、mean/std，以及首个非有限位置。
- CPU 合成测试覆盖全阶段有限、人工溢出首阶段定位、参数不变和异常不静默吞掉。
- 报告记录 `optimizer_updates=0`、`scheduler_steps=0`，并明确区分 FP32 与 FP16 结果；不得用 `nan_to_num`、梯度裁剪、关闭图模块或改成 FP32 掩盖异常。

## 禁止事项

- 不修改模型公式、训练逻辑、实验参数、DANN 系数、图层数、头数或隐藏维度。
- 不运行正式训练，不读取 target test，不保存新的目标伪标签，不由执行器启动 GPU（图形处理器）诊断；GPU 命令仅由用户运行。
- 不修改数据增强、生成器、NLI（自然语言推断）、最终 ASTE 或实验索引。

## 研究边界

本任务仅限 FP32/FP16 数值追踪、CPU 合成测试和只读报告，不改变既有句法图、DANN（领域对抗网络）或训练方案。target pseudo inference（目标伪标签推理）只允许记录异常，不得产生新的实验伪标签。

## 前置核验状态

- 指定正式环境的 `stanza==1.14.0`、`torch==2.2.2+cu121` 及其运行依赖已通过导入验证，CUDA（英伟达 GPU 加速）可用；
- 用户 CMD（命令提示符）验证已确认 `resources.json`、EWT 模型文件和 Stanza English EWT（Stanza 英语 EWT）解析管线均可用，实际设备为 CUDA；
- 固定包映射已确认：`tokenize=ewt`、`mwt=ewt`、`pos=ewt_charlm`、`lemma=combined_nocharlm`、`depparse=ewt_charlm`；单句输出为 1 个句子、5 个词，依存关系完整；
- 任务协议要求的模型文件 SHA256（哈希）已由用户在 CMD 中记录：`resources.json=4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c`、`tokenize/ewt.pt=fc2fed0cd74dbaef1620bd3e776141ae76c4e28eb5aeff369b2715c31cc73cba`、`mwt/ewt.pt=73411a30da7638bbda2ebd9490e017d78feb4e029e90c9f5c9f37e5433292eb0`、`pos/ewt_charlm.pt=f89696d286c29aff173061fbd4b581c73525257ce38015804be047a5e40f9614`、`lemma/combined_nocharlm.pt=e3cb21e3c97a514d102fcc95e78fbc2ab838bc7b306a48029022f35caba1aa2c`、`depparse/ewt_charlm.pt=7386666c2054363f6c4eae702f84ef7d4a11aa4708c2907b82b105e56925d897`；
- 固定 EWT 解析器前置核验完整通过，允许进入已批准的 Python 实现入口与 zero-update（零更新）审计；正式训练、优化器更新、新目标伪标签和下游实验仍未批准。

## 本任务禁止

- 正式 extractor（抽取器）训练、optimizer step（优化器更新）、参数更新和新 target pseudo labels（目标伪标签）；
- generator（生成器）、augmentation（增强）、NLI/exact、final ASTE、target test（目标测试）及任何目标测试金标读取；
- 修改 parser（解析器）、图层数、attention heads（注意力头数）、graph hidden size（图隐藏维度）、DANN（领域对抗网络）、伪标签权重、筛选阈值或现有生成器；
- fallback 到无图、删除失败行、参数网格、隐式改 tokenizer（分词器）或启动未经批准的 GPU（图形处理器）实验；不得绕过专用审计脚本进入图训练。

## 本轮策略修复结论

- 上一版全量只读预检已完成 1730/1730 行，字符覆盖率 100%、failed_rows=0、目标测试隔离为 false；其中 12 项 `alignment_policy_violation` 已确定属于合法的连续部分共享边界，包括缩写、`Registration/1st` 和 `WIth`，不是数据失败。
- 统一策略版本更新为 `overlap-contiguous-contained-sharing-v3`：共享子词要求解析词位置连续、解析词跨度连续、同句、子词跨度包含于解析词联合跨度且与每个记录词有非空重叠；`exact_union` 保留为诊断字段，并新增 `contained_in_parser_union` 和 `partial_contiguous_shared_subword`。
- `abx` 联合覆盖、跨空格、非连续解析词、跨句、越界和字符覆盖缺口仍被拒绝；错误信息现在分别标识 `shared_subword_outside_parser_union`、`non_contiguous_shared_subword`、`cross_space_shared_subword` 和 `cross_sentence_shared_subword`。旧版 BLOCKED 预检目录未删除或覆盖。
- 本轮新增 CPU 测试 2 项，覆盖 7 类部分共享及预检输出；M1 相关直接测试共 78 项全部通过，AST（抽象语法树）解析和 `git diff --check`（差异格式检查）通过。新版 GPU 全量预检、正式训练、伪标签生成和目标测试读取均未执行；M1 尚不能写成已通过，正式实验索引暂不更新。

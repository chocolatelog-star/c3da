# 当前任务

> 更新时间：2026-08-26 22:59（北京时间）

- 任务编号：M1_SYNTACTIC_RGAT_PSEUDO_INTERFACE_V1
- 任务类型：DIAGNOSTIC + IMPLEMENTATION ENTRY（诊断与实现入口）
- 方向：laptop14 -> rest15
- 随机种子：1000
- 状态：RUNNING（已完成最后参数协议修复，等待 Codex Sol 最后一次只读验收）
- 用户批准状态：APPROVED_FOR_IMPLEMENTATION_ENTRY（已批准实现入口）
- 正式训练状态：NOT APPROVED（未批准）

## 研究边界

本任务只批准外部 Stanza English EWT（Stanza 英语 EWT）句法图接口、word-subword（词到子词）对齐、固定关系图注意力适配器、正式抽取器调用点、配置与清单接入、单元测试、zero-update（零更新）入口审计和审计报告。目标无标签 DANN（领域对抗网络）只沿既有训练接口和审计路径接通，系数固定为 `0.03`；目标行只提供领域标签，ASTE 标签全部屏蔽为 `-100`。

Control（控制组）与 Treatment（处理组）只允许在源域抽取器到目标域伪标签接口引入句法图变量；不得接入生成器、增强、NLI/exact（自然语言推断/精确回抽）、候选选择、最终 ASTE（方面级情感三元组抽取）或目标测试评估。

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

## 当前结论

本轮固定 EWT 解析器前置核验完整通过；已完成最后参数协议修复：审计入口硬校验 recipe（配方）参数，固定 seed=1000、lambda_domain_adv=0.03、fp16、gradient_checkpointing、source train batch=1、source-dev eval batch=2、DANN source/target 组成和 128/96 长度，并将 actual/expected/matches 写入 JSON。图开关未带 `--syntactic_graph_entry_audit_only` 时继续硬停，普通无图流程保持旧路径；目标行 ASTE 标签为全 `-100`。CPU 11 项直接测试、AST 解析和差异格式检查通过；尚未启动 GPU zero-update（零更新）审计、正式训练、伪标签生成或目标测试读取。下一步等待 Codex Sol 最后一次只读验收。

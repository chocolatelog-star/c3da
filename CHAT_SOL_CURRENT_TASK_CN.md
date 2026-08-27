# 当前任务

> 更新时间：2026-08-27 12:46（北京时间）

- 任务编号：M1_ALIGNMENT_PREFLIGHT_V1
- 任务类型：READ-ONLY DIAGNOSTIC + IMPLEMENTATION（只读诊断与实现）
- 方向：laptop14 -> rest15
- 扫描划分：source_train、source_dev、target_unlabeled
- 状态：RUNNING（本轮策略修复与 CPU 验证完成，等待 Codex Sol 最终只读验收；新版 GPU 全量预检尚未运行）
- 用户批准状态：APPROVED_FOR_READ_ONLY_PREFLIGHT（已批准只读预检）
- 正式训练状态：NOT APPROVED（未批准）

## 本任务交付边界

- 新增独立 `m1_alignment_preflight.py`，复用 `syntactic_graph.py` 的公开 `validate_alignment_policy`、Stanza EWT、fast offset mapping（快速偏移映射）和 `overlap-contiguous-contained-sharing-v3` 对齐策略；不修改模型、训练、损失、数据、标签或正式实验逻辑。
- 对三份实际输入逐行扫描；遇到行级异常只记录 `row_failure` 并继续后续行。合法连续共享也写入可疑清单，异常类型、覆盖率、分布和截断信息写入 JSON（结构化）汇总。
- 输出目录只允许 `alignment_preflight_summary.json`、`alignment_suspicious_rows.jsonl` 和 `alignment_preflight_CN.md` 三个文件；身份完成校验后才初始化输出，恢复时硬校验 Git commit、输入 SHA256、解析器、分词器、图模式、策略版本和 `max_source_length=128`，并只截断未提交尾部。
- 目标测试隔离为硬约束；代码只加载 source train/dev 和 target train 作为 target_unlabeled，不请求 target_test。

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

## 本轮策略修复结论

- 上一版全量只读预检已完成 1730/1730 行，字符覆盖率 100%、failed_rows=0、目标测试隔离为 false；其中 12 项 `alignment_policy_violation` 已确定属于合法的连续部分共享边界，包括缩写、`Registration/1st` 和 `WIth`，不是数据失败。
- 统一策略版本更新为 `overlap-contiguous-contained-sharing-v3`：共享子词要求解析词位置连续、解析词跨度连续、同句、子词跨度包含于解析词联合跨度且与每个记录词有非空重叠；`exact_union` 保留为诊断字段，并新增 `contained_in_parser_union` 和 `partial_contiguous_shared_subword`。
- `abx` 联合覆盖、跨空格、非连续解析词、跨句、越界和字符覆盖缺口仍被拒绝；错误信息现在分别标识 `shared_subword_outside_parser_union`、`non_contiguous_shared_subword`、`cross_space_shared_subword` 和 `cross_sentence_shared_subword`。旧版 BLOCKED 预检目录未删除或覆盖。
- 本轮新增 CPU 测试 2 项，覆盖 7 类部分共享及预检输出；M1 相关直接测试共 78 项全部通过，AST（抽象语法树）解析和 `git diff --check`（差异格式检查）通过。新版 GPU 全量预检、正式训练、伪标签生成和目标测试读取均未执行；M1 尚不能写成已通过，正式实验索引暂不更新。

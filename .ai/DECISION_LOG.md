# 协作决策日志

> 更新时间：2026-08-26 22:34（北京时间）

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

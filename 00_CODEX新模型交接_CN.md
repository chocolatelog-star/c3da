# CD-C3DA 新对话移交快照

> 更新时间：2026-08-29 16:50（北京时间）
> 用途：切换 Codex（代码执行模型）对话时快速恢复当前有效状态。长期事实仍以 `AGENTS.md`、`.ai/`、实验索引和 Git（版本管理）为准。

## 1. 首次读取顺序

1. `AGENTS.md`
2. `.ai/PROJECT_STATE.md`
3. `.ai/CURRENT_TASK.md`
4. `.ai/DECISION_LOG.md`
5. `docs/skills/c3da-experiment-workflow/SKILL.md`
6. 按任务需要读取 `实验记录与模型索引_CN.md`、`03_CD-C3DA下一阶段改进计划_CN.md` 和 `07_CD-C3DA六组跨域实验详细分析与GPT交接_CN.md`

事实冲突时依次服从：用户当前明确指令、`AGENTS.md`、项目 Skill（技能）、`.ai` 当前状态、正式实验索引、Git 与运行产物。不得凭旧聊天记忆覆盖本地事实。

## 2. 项目目录与 Git 身份

- 正式仓库：`J:\nlp\CD-C3DA`
- 当前功能工作树：`J:\nlp\CD-C3DA\.worktrees\m1-syntactic-rgat-entry-audit-v1`
- 只读原论文代码：`J:\nlp\C3DA-main`
- 最佳流程验收现场：`J:\nlp\CD-C3DA-native-best-rc-v1`
- 当前功能分支：`codex/m1-syntactic-rgat-entry-audit-v1`
- 当前 V8 代码提交：`33a7e3d3fa9846de2ddac52484365b4bf3c649c4`
- 正式文档分支：`docs/account-migration-handoff-v1`

不得在原论文副本或最佳验收现场继续开发。V3/V4/V6/V7 历史失败运行未经用户许可不得删除，也不得作为新的训练父运行；V6 仅有经审计的 Control（对照组）允许被 V8 复用一次。

## 3. 当前正式结果

| 方向 | CD-C3DA raw F1 | BGCA F1 | 差距 |
|---|---:|---:|---:|
| rest14 → laptop14 | 52.54 | 53.64 | -1.10 |
| rest15 → laptop14 | 45.27 | 45.69 | -0.42 |
| rest16 → laptop14 | 48.93 | 47.28 | +1.65 |
| laptop14 → rest14 | 56.94 | 65.27 | -8.33 |
| laptop14 → rest15 | 54.01 | 58.95 | -4.94 |
| laptop14 → rest16 | 61.55 | 64.00 | -2.45 |

- 当前研究基线：`laptop14 -> rest15`，raw P/R/F1 = 56.98/51.34/54.01，fixed F1 = 55.53。
- 当前保护基线：`rest16 -> laptop14` raw F1 = 48.93；它是唯一超过 BGCA（双向生成跨域方法）的正式方向，不在保护现场继续试验。
- 核心瓶颈：目标域元素和复杂结构供给不足，多三元组完整生成与召回不足；中性类别只作辅助。

## 4. 当前唯一实验：V8 M1 Phase A

- 任务：`M1_SYNTACTIC_RGAT_PSEUDO_QUICK_ABLATION_V8`
- 类型：`QUICK_ABLATION`（快速消融）/ `PHASE A`（阶段A）
- 方向与随机种子：`laptop14 -> rest15`，seed 1000
- 运行目录：`J:\nlp\CD-C3DA\runs\reproducible\laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_v8\laptop14-rest15-m1-syntactic-rgat-pseudo-phase-a-seed1000-v8`
- 状态：`RUNNING`（运行中）
- Control：不重训；以 `reuse_depth=1` 复用 V6 审计通过的 Control 模型、DANN（领域对抗网络）报告及身份哈希。
- Treatment：带句法 RGAT（关系图注意力网络）的抽取器从头训练 1400 步。用户最近观测约为 `172/1400`。
- `104/110` 等短进度条是 source-dev（源域开发集）评估，不是第二次训练。
- 当前训练损失、开发损失和梯度范数均为有限值且总体下降；Accelerate（加速训练库）提示只是弃用警告，不是错误。

## 5. M1 模型边界

第一版只在上游伪标签抽取器中加入：外部 Stanza English EWT（英语依存解析器）拓扑、T5 Encoder（编码器）词节点语义、一层四头 RGAT 和门控残差融合。DANN 系数固定 0.03，目标无标签数据仅提供领域损失。

Phase A 不修改生成器、增强、NLI（自然语言推断）、候选选择、最终 ASTE（方面级情感三元组抽取）或目标测试；最终 ASTE 当前仍完全不带图。只有 A1–A4 无金标 Gate（门控）全部通过，才可向 Chat Sol（研究负责人）申请 Phase B（阶段B）。

## 6. 当前必须保持的边界

- V8 运行中不得修改代码、配方、数据、门槛或运行产物。
- 不读取 target_test（目标测试集），不提前运行目标 F1，不启动 Phase B。
- 不把训练 loss（损失）或阶段性进度当成研究成功证据。
- 不启动双生成器、教师—学生、`k=2`、中性强加权或已关闭路线。
- 未经用户明确要求，不由模型代跑实验；默认改好后提供完整单行 CMD（命令提示符）命令。
- 删除任何文件、模型、检查点或运行目录必须获得用户明确许可。

## 7. V8 完成后的必做事项

1. 读取 `stage_status.json` 和 Phase A 最终机器可读报告。
2. 对比 Control/Treatment 的 source-dev 指标、目标无标签伪标签数量、置信度、稳定性和结构分布。
3. 逐项给出 A1–A4 `PASS/FAIL`（通过/失败），说明证据强度和是否存在数据泄漏。
4. 给出实验结论、与 54.01 基线及 BGCA 差距的关系、唯一下一步建议。
5. 更新 `.ai` 三份状态文档、三份 Chat Sol 上传镜像、实验索引、`03`；重大研究结论变化时同时更新 `07` 和本移交快照。
6. 只有用户明确批准后，才执行下一阶段代码或实验。

## 8. 新对话首次回复要求

完成只读核验后，首次回复只需说明：实际工作树、分支/提交、V8 当前阶段、是否发现冲突、当前允许做什么和禁止做什么。不要重新设计研究路线，也不要启动实验。

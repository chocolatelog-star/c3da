# M1 句法 RGAT 伪标签快速消融实现计划

> 任务：`M1_SYNTACTIC_RGAT_PSEUDO_QUICK_ABLATION_V1`
> 方向：`laptop14 -> rest15`；种子：`1000`

## 目标

实现一个只覆盖 Phase A 的专用、可恢复、可审计运行入口。Control 使用既有无图 T5 抽取器，Treatment 只在源域抽取训练、源域开发集评估、目标无标签 DANN 和目标伪标签推理四个既定调用点启用句法 RGAT。Phase A 结束后硬停止，不进入生成器、增强、NLI、选择器或最终 ASTE。

## 实施步骤

1. 先在 `test_m1_syntactic_rgat_pseudo_quick_ablation.py` 增加门控、作用域、Control 身份、恢复、冻结配置和 Phase B 硬停止的失败测试，并记录 RED。
2. 新增 `m1_syntactic_rgat_pseudo_quick_ablation.py`，集中实现冻结配方、机器可验证身份哈希、Control 复用判定、阶段状态/原子写入/恢复、A1-A4 门控和 JSON/中文报告。
3. 将已有 `t5_absa_train.py` 的训练主体提取为可调用入口；保持脚本直接运行图训练的硬拦截，专用 Phase A 入口才可显式调用图训练 API。不得改动图传播公式、损失公式或训练参数。
4. 复用既有 `t5_aste_pipeline.py` 的 source-dev 评估、target-unlabeled 生成和伪标签质量/筛选函数；使用新运行目录和固定配置，不调用下游 Phase B。
5. 实现 A1（严格三元组 F1）、A2（多三元组句召回）、A3（元素缺失率）和 A4（合格伪标签总量/多三元组量）的机器可读结果与硬停止行为。
6. 先运行新增 CPU 测试取得 GREEN，再运行全部 M1 相关 CPU 回归、AST 语法检查和 `git diff --check`；确认未读取 target-test、未启动 GPU/正式训练/正式伪标签。
7. 更新正式仓库与当前工作树的 `.ai` 状态文档及 Chat Sol 镜像，记录本轮只实现入口、尚未运行实验；提交功能分支和正式文档分支，保持两边工作树干净，不推送。

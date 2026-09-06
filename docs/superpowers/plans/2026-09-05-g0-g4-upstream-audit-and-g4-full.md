# C3DA G0-G4 上游审计与 G4 完整实验计划

**目标：** 保留正在运行的 4×4 完整实验，同时准备 Chat 任务卡要求的 16×2 G0-G3 上游统一审计和 G4 完整实验。

**范围：** G0-G3 不重训，只复用现有 16×2 checkpoint；G4 使用 16×2、DANN=0、组合关系且关闭 Focus/Coverage，完整跑到最终 Raw/Fixed F1。

## 任务 1：确认服务器运行状态和可复用输入

- [ ] 确认 4×4 G0-G3 当前进程，不修改其输出目录。
- [ ] 确认 G0-G3 `run2` checkpoint、输入文件、seed、batch、梯度累积和上游 DANN=0。
- [ ] 记录三份输入文件 SHA256（哈希）。
- [ ] 确认 G0-G3 只生成新审计产物，不启动 Final ASTE。

## 任务 2：审计入口与四阶段统计

**服务器文件：** `/root/CD-C3DA` 下的审计脚本和测试文件。

- [ ] 先写失败测试，锁定 G0-G3 使用已有 checkpoint、统一 generation/filtering（生成/筛选）配置和四阶段输出字段。
- [ ] 实现小型审计入口：读取 checkpoint，统一执行 source-dev、target-unlabeled 推理和四层伪标签筛选。
- [ ] 对每层输出 rows、triplets、single/multi/3plus、Precision、Recall、Raw F1。
- [ ] 只允许 retrospective hidden-gold（事后隐藏金标）审计，不参与训练、选择或参数决定。

## 任务 3：G0-G3 比较产物

- [ ] 生成统一 `G0_G1_G2_G3_G4_upstream_comparison.csv` 的 G0-G3 部分。
- [ ] 生成 G0-G3 两两 overlap（重叠）和 G0-vs-G3 深度比较摘要。
- [ ] 生成中文 `UPSTREAM_AUDIT_RESULT_CARD.md`，不报告“下一步最佳模型”结论。
- [ ] 使用 gzip（压缩）保存必要 JSONL，避免复制相同阶段文件。

## 任务 4：G4 入口

- [ ] 新增/确认 G4 变体：Compositional Relation（组合关系），Focus=OFF，Coverage=OFF，DANN=0。
- [ ] 复用 G1/G2/G3 已验证的图关系实现和 matched recipe（匹配配方）。
- [ ] G4 使用 batch=16、gradient accumulation=2、seed=1000。
- [ ] 完整运行伪标签、增强、最终训练和目标测试评估，保存 Raw/Fixed P/R/F1。
- [ ] 保留一个最终 checkpoint，删除/限制中间 checkpoint。

## 任务 5：验证与启动顺序

- [ ] 服务器 Conda（康达）环境运行语法检查和最小 dry-run（试运行）。
- [ ] 4×4 完成前不启动 Chat 审计或 G4。
- [ ] 4×4 完成后先运行 G0-G3 审计；确认显存和磁盘后再运行 G4。
- [ ] 完成后报告每组伪标签 F1、G4 最终 F1、输入哈希、磁盘空间和文件清单。

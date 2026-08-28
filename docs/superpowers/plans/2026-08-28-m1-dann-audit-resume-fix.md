# M1 DANN 轮次审计与恢复修复实施计划

**目标：** 修复 Phase A 成对 DANN 的物理遍历身份、部分遍历持久化、检查点恢复和正式验证逻辑，使 Control/Treatment 的逐批审计可重放、可对齐且不会把旧缺口冒充为正式证据。

**架构：** `PairedDomainBatchSampler` 维护独立于训练器浮点进度的物理遍历序号、采样轮次和当前部分报告；每次实际产生批次后原子落盘，正常结束时才将其标记为完整。训练器在显式遍历边界保存完整检查点身份，恢复时校验单调性、完成度、global step 区间和 Control/Treatment 批次对应关系。Phase A 验证器按实际 `max_steps`、已消费批次及完整/部分遍历判断，不再把 `num_train_epochs` 直接当成完整审计份数。

**技术栈：** Python、PyTorch、Transformers Trainer、现有 JSON 原子写入、CPU 单元/集成测试。

## 任务一：先写并验证 RED 测试

- [x] 在 `test_m1_syntactic_rgat_pseudo_quick_ablation.py` 增加重复浮点进度不会覆盖物理遍历的测试。
- [x] 增加最后部分遍历会保留 `partial` 审计、消费批次数和 global step 区间的测试。
- [x] 增加生成器中断或尾部清理未执行时审计仍已落盘的测试。
- [x] 增加恢复后物理遍历序号、采样轮次和报告序列单调的测试。
- [x] 增加 Control/Treatment 部分与完整遍历对齐要求的测试。
- [x] 增加 legacy 诊断报告不能被正式验证器判为 PASS 的测试。
- [x] 仅运行这些新增测试，确认它们因新接口/行为缺失而失败；不得启动 GPU、正式实验、伪标签推理或 target_test。

## 任务二：实现采样器和原子审计持久化

- [x] 修改 `t5_absa_train.py`：为采样器加入独立物理遍历计数、采样 epoch、计划/消费批次、完整性、global step 区间和报告 schema 版本。
- [x] 删除基于 `int(self.state.epoch)` 的轮次来源，改由 sampler 在每次新 DataLoader 物理遍历开始时分配独立身份，并由 Trainer 只提供整数步数元数据；不得改变 seed、洗牌公式、样本顺序或批次大小。
- [x] 每次实际产生批次后原子写入当前审计；遍历自然结束时标记完整；异常/中断留下部分审计而不覆盖已有完整遍历。
- [x] 在 sampler `state_dict/load_state_dict` 和审计加载中严格校验身份、单调性、完整/部分状态以及恢复位置。

## 任务三：实现 Trainer 检查点和恢复语义

- [x] 修改 `t5_absa_train.py`：在检查点 manifest 中记录审计 schema、物理遍历序号、采样 epoch、计划/消费批次数、完整度和 optimizer/global step 区间。
- [x] 只有模型、Trainer state、sampler state、审计和 manifest 全部原子完成且哈希一致的检查点才可恢复；最新损坏点安全回退到最新完整合法点。
- [x] 恢复时拒绝身份改变、报告倒退、重复物理遍历、Control/Treatment 不对应或部分报告被误认为完整。
- [x] 保持非成对 DANN、损失公式、图传播、参数和采样顺序不变。

## 任务四：修正 Phase A 验证器与 legacy 诊断路径

- [x] 修改 `m1_syntactic_rgat_pseudo_quick_ablation.py`：验证器使用 Trainer 实际 `max_steps`、global step、计划/消费批次和完整度计算有效覆盖。
- [x] 不再固定要求 `len(epochs) == num_train_epochs`；缺失物理遍历、部分遍历或旧 schema 均输出诊断/阻塞状态，不能正式 PASS。
- [x] 为旧运行增加显式 legacy diagnostic resume 路径，写入迁移/兼容报告、原提交和原产物哈希，并禁止静默改写 `stage_status.json` 或冒充当前提交。
- [x] 明确旧运行只能作为方向性 QUICK_ABLATION_DIAGNOSTIC；若无法保持训练同一性则硬拒绝并要求新目录重跑。

## 任务五：GREEN、整体复审和文档同步

- [x] 先运行新增测试，再运行所有相关 M1 CPU 测试、AST 解析和 `git diff --check`；不安装依赖、不运行 GPU。
- [x] 复审恢复身份、原子持久化、最后部分遍历、旧证据降级、Control/Treatment 对齐和目标测试隔离。
- [x] 同步 `.ai/PROJECT_STATE.md`、`.ai/CURRENT_TASK.md`、`.ai/DECISION_LOG.md` 及三个 `CHAT_SOL_*_CN.md` 镜像；CURRENT_TASK 标记为本修复任务并保持 `APPROVED`，不得写成实验已通过。
- [x] 检查正式仓库 `J:/nlp/CD-C3DA` 的对应文档同步条件；保留用户已有改动，不覆盖或删除任何旧运行产物。
- [x] 提交功能分支，不推送；报告提交哈希、工作树状态、测试结果、实验逻辑是否变化，以及两条不实际执行的一行 CMD 命令。

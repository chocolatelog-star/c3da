# 当前任务

> 更新时间：2026-08-27 13:16（北京时间）

- 任务编号：M1_SYNTACTIC_RGAT_FP16_NUMERICAL_TRACE_V1
- 任务类型：READ-ONLY DIAGNOSTIC IMPLEMENTATION（只读诊断实现）
- 方向：laptop14 -> rest15
- 状态：APPROVED（已批准）
- 用户批准范围：同一固定样本的 FP32（单精度）/FP16（半精度）逐层数值追踪、首个非有限张量定位、target pseudo inference（目标伪标签推理）异常记录、CPU（中央处理器）合成测试、机器可读 JSON（结构化报告）和中文报告
- 正式训练状态：NOT APPROVED（未批准）

## 已知阻塞事实

- M1 句法图 zero-update（零更新）入口审计当前为 BLOCKED（阻塞），代码提交为 `3877838c0e0b0e5079bd4b0797ec7014d301ab30`。
- 审计目录为 `J:\\nlp\\CD-C3DA\\runs\\diagnostics\\laptop14_to_rest15_m1_syntactic_rgat_entry_audit_v2`；图缓存、身份、边合法性、断点恢复、8GB 显存和零更新门控已通过。
- control_loss（控制组损失）有限；treatment_loss（处理组损失）、重复前向、source training/dev、DANN（领域对抗网络）损失、treatment encoder/logit 差异和 ASTE/DANN 梯度为 NaN（非数值）；参数未更新，target test（目标测试集）未访问。

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
- 不运行正式训练，不读取 target test，不生成新的目标伪标签，不启动 GPU（图形处理器）诊断；GPU 命令仅由用户运行。
- 不修改数据增强、生成器、NLI（自然语言推断）、最终 ASTE 或实验索引。


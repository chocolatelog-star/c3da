# G0/G3 内容差异与增强失败审计计划

任务：C3DA_G0_G3_CONTENT_DIFFERENCE_AUDIT_V1

## 阶段
- [x] 读取实际 pseudo、complete_multi、augmentation、final_train
- [x] 计算 shared/G0-only/G3-only 重叠及隐藏金标质量
- [x] 分析样本复杂度、增强来源和下游放大
- [x] 保存小型审计产物并形成 RESULT_CARD

增强失败审计结论：主要为 multi-triplet structure preservation 失败，次要为 opinion channel 可靠性不足。

## 约束
- 只读审计，不训练、不改生成逻辑、不跑 G4
- 不使用 target-test gold；hidden gold 仅作 target-train 事后审计

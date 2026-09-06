# G0-G3 下游反转审计计划

**目标：** 只读取现有 G0-G3 运行产物，解释上游伪标签排序与最终 Raw F1 排序相反的原因。

**约束：** 不训练、不改模型、不运行 G4、不使用 target-test gold（目标测试金标）做分析或决策；只生成小型汇总。

## 审计步骤

- [x] 定位 G0-G3 的最终运行目录、配置、manifest 和日志（基于已核实历史记录）。
- [x] 核对最终 ASTE（方面级情感三元组抽取）训练配方及 DANN（领域对抗）设置（G0 已核实，四组 recipe 名称一致）。
- [ ] 统计 pseudo、complete-multi、augmentation 和 final-train 组成及保留率（G1-G3 原始文件缺失，SSH 读取被拒绝）。
- [ ] 读取已有 target-unlabeled / source-dev 预测摘要，不生成新的 checkpoint（本轮未访问远程，避免伪造结果）。
- [x] 输出证据、主因、次因和是否需要交叉复现实验的判断；结果标记为部分完成。

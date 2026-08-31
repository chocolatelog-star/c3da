# 当前任务

> 更新时间：2026-08-31 00:20（北京时间）

## 任务身份

```text
TASK：M1_ELEMENT_AWARE_COMPONENT_ATTRIBUTION_V1
状态：RUNNING
方向：laptop14 -> rest15
seed：1000
实验类型：QUICK_DIAGNOSTIC
```

项目级截止：2026年9月16日提交ICASSP 2027常规4+1页论文；本任务必须在9月3日前完成研究判定，不得演变为损失系数或图结构网格。

## 当前目标

在普通RGAT（关系图注意力网络）、完整Treatment（实验组）之间增加两个严格组件对照：

- Focus-only（仅聚焦）：`lambda_focus=0.05`，`lambda_coverage=0`；
- Coverage-only（仅覆盖）：`lambda_focus=0`，`lambda_coverage=0.05`。

只判断多三元组召回和元素缺失改善来自哪个组件，不搜索系数。

## 当前代码身份

```text
分支：codex/m1-element-aware-multi-triplet-rgat-v1
已推送HEAD：135c8be
本地实现HEAD：bee4498
服务器可见提交：135c8be
```

本地已完成G1/G2权重校验、V9e（第9e版）固定批次入口、同目录自动恢复、运行身份与目标测试隔离及Phase A结果卡聚合。结果卡包含source-dev严格F1、多三元组召回、元素缺失率、合格伪标签总量/multi/3+和Focus/Coverage机制观测字段；`test_element_aware_component_ablation.py`与句法图相关回归共75项通过，Python（编程语言）语法检查和Git diff（差异格式检查）通过。

Codex Sol（高级工程模型）复审结论为`PASS_WITH_NOTES`：无硬阻塞；备注为评估/伪标签参数哈希、DANN=0配对字段显式化和训练遥测可后续增强。本地最终提交`7e39287`尚未推送，服务器实例当前SSH（安全外壳）不可达，未启动新的GPU（图形处理器）实验。

## 固定项

固定数据、T5-base、图缓存、解析器、图层1、隐藏维256、头数4、DANN=0、训练参数、伪标签规则、source-dev（源域开发集）选模和目标无金标边界。

## 禁止事项

不得从完整Treatment检查点热启动；不得运行Phase B（阶段B）、增强、最终ASTE（方面级情感三元组抽取）或目标测试；不得改变batch、损失系数、阈值、注意力头、DANN或伪标签配额；不得用目标测试选择组件。

## 完成条件

两组均输出source-dev总体/单/多结构指标、元素absence（缺失）、合格伪标签总量与multi/3+数量、身份哈希和结果卡。两组完成后返回Chat Sol（研究负责人）作`KEEP_FOCUS / KEEP_COVERAGE / INTERACTION_ONLY / STOP_GRAPH_TUNING`判定。

## 当前唯一下一步

Codex Sol复审已通过（`PASS_WITH_NOTES`），双目录文档已完成最终哈希核验。功能分支已推送；后续由用户决定是否直接启动G1/G2两个独立正式诊断目录。

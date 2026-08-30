# 当前任务

> 更新时间：2026-08-30 23:17（北京时间）

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
当前未跟踪测试：test_element_aware_component_ablation.py
```

未跟踪测试必须先完成只读复审、CPU测试、提交和推送，才能提供服务器GPU（图形处理器）运行命令。

## 固定项

固定数据、T5-base、图缓存、解析器、图层1、隐藏维256、头数4、DANN=0、训练参数、伪标签规则、source-dev（源域开发集）选模和目标无金标边界。

## 禁止事项

不得从完整Treatment检查点热启动；不得运行Phase B（阶段B）、增强、最终ASTE（方面级情感三元组抽取）或目标测试；不得改变batch、损失系数、阈值、注意力头、DANN或伪标签配额；不得用目标测试选择组件。

## 完成条件

两组均输出source-dev总体/单/多结构指标、元素absence（缺失）、合格伪标签总量与multi/3+数量、身份哈希和结果卡。两组完成后返回Chat Sol（研究负责人）作`KEEP_FOCUS / KEEP_COVERAGE / INTERACTION_ONLY / STOP_GRAPH_TUNING`判定。

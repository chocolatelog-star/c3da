# CD-C3DA 项目状态

> 更新时间：2026-08-30 23:10（北京时间）

## 当前有效状态

CD-C3DA是单生成器T5-base（T5基础模型）跨域ASTE（方面级情感三元组抽取）系统。当前研究方向为`laptop14 -> rest15`，正式最佳raw/fixed F1=54.01/55.53。保护方向`rest16 -> laptop14`正式最佳为48.93/50.21，并超过BGCA（双向生成跨域方法）论文值1.65个百分点。

当前主要瓶颈是目标域元素供给不足和多三元组完整召回，不是单三元组能力。中性类别仅作辅助。

## 六方向最佳

| 方向 | CD-C3DA | BGCA | 差距 |
|---|---:|---:|---:|
| rest14 -> laptop14 | 52.54 | 53.64 | -1.10 |
| rest15 -> laptop14 | 45.27 | 45.69 | -0.42 |
| rest16 -> laptop14 | 48.93 | 47.28 | +1.65 |
| laptop14 -> rest14 | 56.94 | 65.27 | -8.33 |
| laptop14 -> rest15 | 54.01 | 58.95 | -4.94 |
| laptop14 -> rest16 | 61.55 | 64.00 | -2.45 |

## 当前图结构路线

目标伪标签形成阶段加入Dependency/POS Graph（依存/词性图）、四头RGAT（关系图注意力网络）、Element Salience（元素显著性）和Multi-Element Coverage（多元素覆盖）。最终ASTE模型尚未批准使用图结构。

服务器Treatment-only（仅实验组）batch=1/4/8/16已完成。batch16：source-dev strict F1=56.58%，multi recall=48.11%，overall absence=69.77%，qualified total/multi=552/209。相对普通RGAT，multi recall提高2.36个百分点、absence下降3.49个百分点；相对无图T5没有同时通过F1和召回门槛。因此状态为`QUICK_DIAGNOSTIC`（快速诊断），不进入Graph Freeze（图模块冻结）。

当前唯一任务是Focus-only（仅聚焦）和Coverage-only（仅覆盖）组件归因，详见`.ai/CURRENT_TASK.md`。

## 当前代码与服务器

```text
正式文档分支：docs/account-migration-handoff-v1
元素感知分支：codex/m1-element-aware-multi-triplet-rgat-v1
已推送元素感知HEAD：135c8be
服务器项目：/root/CD-C3DA
服务器输出：/root/autodl-tmp/CD-C3DA-runs/
```

当前元素感知工作树有未跟踪组件测试，不得删除或启动正式GPU（图形处理器）实验，直至测试提交、推送和复审完成。

## 保留部件

单生成器、`k=1`、目标真实句锚定、完整标签条件、NLI（自然语言推断）/exact（精确回抽）硬验证、联合配额选择、source-dev（源域开发集）选模、无金标多结构门控、中性辅助。

## 已关闭路线

双生成器、教师—学生、`k=2`、中性强加权、普通质量/结构调权、原锚回放、复杂样本重复呈现、NLL（负对数似然）排序/加权、EOS（终止符）系列、ECAL（元素覆盖辅助损失）系列、FGSM（快速梯度符号法）参数搜索、DANN（领域对抗网络）开关/调权，以及无上游供给证据的配对/计数损失。

## 实验边界

快速诊断不能成为新父运行；目标测试金标不能用于组件、阈值、配额、损失或检查点选择；正式新最佳和六方向扩展必须完整从头运行。用户未明确要求代跑时，Codex只实现、测试并提供服务器单行命令，不自行启动GPU实验。

## 文档入口

日常读取顺序：`AGENTS.md` → 本文件 → `.ai/CURRENT_TASK.md` → `.ai/DECISION_LOG.md` → `实验记录与模型索引_CN.md` → `03_CD-C3DA下一阶段改进计划_CN.md`。

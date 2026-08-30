# 元素感知 RGAT 快速诊断结论

## 实验范围

- 方向：laptop14 -> rest15
- 模块：Dependency/POS Graph + 4-head RGAT + Element Salience + Multi-Element Coverage
- DANN（域对抗）：0
- 四组实验均为 Treatment-only（仅实验组快速诊断），未启动 Phase B（阶段二）和 target test（目标测试）。

## 已知结果

| 配置 | strict F1（严格 F1） | multi recall（多三元组召回率） | overall absence（整体缺失率） | qualified total（合格伪标签总量） | qualified multi（合格多三元组） |
|---|---:|---:|---:|---:|---:|
| Plain Control（原始无图 T5） | 57.84% | 47.64% | 69.77% | 508 | 185 |
| Graph Reference（普通 RGAT+DANN=0） | 56.57% | 45.75% | 73.26% | 557 | 209 |
| Treatment batch=8（批大小 8） | 55.07% | 47.17% | 70.93% | 545 | 224 |
| Treatment batch=16（批大小 16） | 56.58% | 48.11% | 69.77% | 552 | 209 |

batch=1（批大小 1）的完整结果以服务器保留的指标 JSON 为准，不在本文档重复推测未完全记录的数字。

## 研究判断

Element Salience（元素显著性）和 Coverage Loss（覆盖损失）对多三元组覆盖有正向信号：batch=16 比普通 RGAT 的 multi recall 高 2.36 个百分点，overall absence 降至与 Plain Control 持平。但相对正式 Gate（门控），batch=16 的 F1 和 multi recall 仍未同时达标，因此当前只能记为快速诊断，不宣布 Graph Freeze（图模块冻结）。

## 服务器保留清单

- 保留四组实验的指标 JSON、target pseudo 分析、日志和 `treatment_only_entry.json`。
- 仅保留 batch=16 的 `best` 模型，用于后续复核。
- 已删除 batch=1/4/8 以及旧 treatment-only 的模型副本，未删除指标和分析文件。
- 服务器数据盘清理后约使用 5.9 GB / 50 GB，可用约 45 GB。

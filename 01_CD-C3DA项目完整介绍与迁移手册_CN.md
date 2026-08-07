# CD-C3DA 项目完整介绍与迁移手册

## 1. 项目定位

本项目最初基于论文《A Contrastive Cross-Channel Data Augmentation Framework for Aspect-Based Sentiment Analysis》的 C3DA 代码。原论文任务主要是给定方面词后的方面级情感分类。本项目已扩展为跨域 ASTE：输入一个目标域句子，直接生成一个或多个“情感、方面词、观点词”三元组。

统一标签格式：

```text
<pos> aspect <opinion> opinion
<neg> aspect <opinion> opinion
<neu> aspect <opinion> opinion
```

多三元组使用分号分隔：

```text
<pos> food <opinion> delicious ; <neg> service <opinion> slow
```

项目研究目标是在目标域训练标签不可见的严格跨域设置下，利用源域金标、目标域无标签文本、伪标签和 C3DA 双通道增强，提高目标域测试集三元组抽取性能。

## 2. 数据与跨域组合

数据来自：

```text
J:\nlp\BGCA-master\data\aste\cross_domain
```

数据集：

- `laptop14`：2014笔记本评论。
- `rest14`：2014餐馆评论。
- `rest15`：2015餐馆评论。
- `rest16`：2016餐馆评论。

主要六组跨域组合：

```text
rest14 -> laptop14
rest15 -> laptop14
rest16 -> laptop14
laptop14 -> rest14
laptop14 -> rest15
laptop14 -> rest16
```

目标域训练标签仅写入 `target_train_gold_analysis.jsonl` 用于离线诊断，禁止参与训练、阈值选择、增强配额或模型选择。目标测试集只能在最终评估阶段使用。

## 3. 项目目录

```text
J:\nlp\CD-C3DA
├─ configs\recipes                 正式可复现配方
├─ dataset                         原始 C3DA 数据兼容目录
├─ docs\skills                    项目 Skill 正式位置
├─ docs\superpowers               保留的正式设计与实施记录
├─ models                          项目内旧模型目录，正式基础模型优先使用 J:\nlp\models
├─ runs                            实验产物、日志、模型和清单
├─ test_fixtures                   自动化测试夹具
├─ .worktrees                      Git 隔离工作树
├─ t5_absa_train.py                提取器、生成器和最终模型训练器
├─ t5_aste_pipeline.py             数据准备、伪标签、增强、最终数据和评估主逻辑
├─ t5_aste_augment.py              双通道增强、候选和编辑请求构造
├─ t5_aste_data.py                 ASTE 数据解析和结构工具
├─ t5_aste_postprocess.py          预测修正与规范化
├─ run_reproducible_pipeline.py    十阶段可复现实验调度器
├─ reproducibility.py              清单、哈希、恢复和运行身份保护
├─ run_best_reproducible_pipeline.ps1  当前最佳正式入口
└─ 实验记录与模型索引_CN.md         当前实验总览
```

## 4. 三个 C3DA 目录与权威边界

| 目录 | 身份 | Git | 用途 | 是否继续开发 |
|---|---|---|---|---|
| `J:\nlp\C3DA-main` | 原始论文代码副本 | 无 | 原始实现和早期资料 | 否 |
| `J:\nlp\CD-C3DA` | 当前正式主仓库 | GitHub `chocolatelog-star/c3da` | 当前代码、实验、分支与文档 | 是 |
| `J:\nlp\CD-C3DA-native-best-rc-v1` | 原生最佳验收仓库 | 本地远程指向主仓库 | 保存十阶段精确复现现场 | 否，只审计 |

权威顺序：当前主仓库 `master` 和正式配方最高；验收仓库用于核对；原始副本仅用于理解论文最初实现。

## 5. 当前正式最佳

当前最佳配方：

```text
configs\recipes\rest16_to_laptop14_best_v1.json
```

当前主线：

```text
master = d2f2a35
正式 GPU 验收代码 = 558e4de
```

最新再次复现运行：

```text
J:\nlp\CD-C3DA\runs\reproducible\rest16_to_laptop14_best_v1\master-best-check-v1
```

结果：

| 指标 | 数值 |
|---|---:|
| raw precision | 58.31 |
| raw recall | 42.14 |
| raw F1 | 48.93 |
| fixed precision | 59.85 |
| fixed recall | 43.25 |
| fixed F1 | 50.21 |
| raw TP/FP/FN | 228/163/313 |
| fixed TP/FP/FN | 234/157/307 |

分情感 raw F1：正面51.54、负面55.05、中性3.03。单三元组 raw F1 为53.05，多三元组 raw F1 为46.13。主要短板是中性召回和多三元组召回。

## 6. 当前最佳十阶段流程

### 阶段1：prepare

读取源域训练/开发集和目标域无标签/测试集，生成提取器训练数据和标签到文本生成器训练数据。目标域训练金标只另存为分析文件。

### 阶段2：extractor

使用源域金标训练 T5 提取器25轮，检查点选择 `last`。该模型用于目标域伪标签和增强回抽过滤。

### 阶段3：pseudo

对目标域906条无标签文本生成伪标签，使用单束搜索。历史最佳观察到421条高精度基础伪标签，但未来方向不得为了匹配421而人为裁剪。

### 阶段4：generator

使用源域金标的标签到文本数据训练 T5 生成器8轮，检查点选择 `best`。这与后续轮次扫描实验中的 `last` 不同，不能混淆。

### 阶段5：augment

运行 `masked_mutual` 双通道增强：方面通道替换方面，观点/情感通道替换观点和情感；经过一致性、NLI 和提取器过滤，按配方最多选择150条，训练权重0.20。

### 阶段6：prepare_final

将原始数据和伪标签转换为下游最终训练目录结构。

### 阶段7：complete_multi2

在高精度单三元组伪标签基础上补充严格完整双三元组样本，额外权重0.25。最佳运行从421条基础候选扩展为494条完整训练伪标签。

### 阶段8：build_final_train

组合源域857条、目标伪标签492条有效训练记录和增强150条，形成1499条最终训练数据。黄金停止条件使用训练语义哈希，不按无关审计字段判断。

### 阶段9：final_train

训练最终 T5 模型5轮。保留领域对抗损失 `lambda_domain_adv=0.03`、源域类平衡情感对比损失 `lambda_sentiment_contrastive=0.01`、伪标签权重0.65和增强权重0.20。

### 阶段10：evaluate

在目标测试集上生成预测，输出 raw、fixed、分情感、单/多三元组和错误分析指标。

## 7. 当前最佳参数

| 项目 | 参数 |
|---|---|
| 随机种子 | 1000 |
| 复现模式 | `historical_seed_only` |
| 训练批大小 | 1 |
| 评估批大小 | 2 |
| 梯度累积 | 16 |
| 学习率 | 0.0003 |
| 半精度 | fp16 |
| 梯度检查点 | 开启 |
| 提取器 | 25轮，`last` |
| 生成器 | 8轮，`best` |
| 最终模型 | 5轮，`best` |
| 伪标签解码 | 1束，最多128个新 token |
| 增强预算 | 150 |
| 增强权重 | 0.20 |
| 伪标签权重 | 0.65 |
| 双三元组额外权重 | 0.25 |
| 领域对抗 | 0.03 |
| 情感对比 | 0.01，仅源域、类平衡 |

## 8. 运行环境

验证环境：

```text
Windows 10/11
Python 3.10.20
Conda 环境 J:\conda\envs\c3da
NVIDIA GeForce RTX 3070 8GB
PyTorch 2.2.2+cu121
CUDA runtime 12.1
cuDNN 8801
Transformers 4.39.3
Accelerate 0.28.0
NumPy 1.26.4
SentencePiece 0.2.0
```

模型：

```text
J:\nlp\models\t5-base-py
J:\nlp\models\nli-deberta-v3-base-mnli-fever-anli
```

禁止重新下载其他格式模型覆盖这些已验证文件。模型哈希写在正式配方和运行 `environment.json` 中。

## 9. 正式运行命令

从当前最佳主线建立干净工作树后运行。已有验收运行的命令示例：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA\.worktrees\master-best-resume-v1 && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_best_reproducible_pipeline.ps1 -RunId master-best-check-v1 -OutputRoot J:\nlp\CD-C3DA\runs\reproducible -Cuda 0"
```

新实验必须使用新的 `RunId`。断电或手动停止后，只有 Git、配方和输入哈希都未改变时，才能重复同一命令恢复。不得复制其他运行的模型、伪标签或增强文件来绕过阶段。

## 10. 每次运行必须保存的证据

```text
manifest.json
run_command.cmd
commands.jsonl
environment.json
stage_status.json
RUN_RECORD_CN.md
logs\<stage>.log
```

`manifest.json` 保存 Git 提交、分支、配方、阶段输入输出和哈希；`stage_status.json` 保存阶段状态；`commands.jsonl` 保存实际展开命令；`environment.json` 保存 Python、Conda、PyTorch、CUDA、GPU、依赖和模型哈希。

## 11. Git 版本管理

### 11.1 核心规则

- `master` 永远只保存完整 GPU 实验验证后的当前最佳版本。
- 所有修改从 `master` 创建语义明确的新分支。
- 分支完成单元测试、语法检查、配方检查、干运行和完整 GPU 实验后，先向用户报告证据并获得许可，再合并。
- 实验运行期间不得修改工作树；正式运行要求干净 Git 状态。
- 不得把实验分支的结果自动合并到 `master`。
- 不得用强制重置删除用户改动。

### 11.2 分支类型

```text
feature/*       新实验或新功能
fix/*           缺陷修复
docs/*          文档和迁移资料
historical/*    历史代码边界，只用于审计
master          当前最佳正式版本
```

### 11.3 工作树

隔离实验位于：

```text
J:\nlp\CD-C3DA\.worktrees
```

工作树不是历史产物复用许可。正式实验只能运行工作树自身代码，并把全部本次产物写入独立 `run_id`。

## 12. 历史版本在哪里

### 12.1 Git 历史分支

重要历史分支：

```text
historical/best-upstream-9e78904
historical/reproduce-best-8c7f6b4
historical/reproduce-best-c0b2730
historical/reproduce-best-0332aee
feature/native-best-reproduction-v1
fix/native-best-training-semantic-v2
feature/complete-multitriplet-ablation
feature/opinion-soft-filter-v1
feature/dual-channel-hard-constraints-v2
feature/dual-generator-strict-pseudo-v1
feature/dual-generator-sentiment-aligned-v2
feature/single-generator-epoch-last-sweep-v1
feature/historical-best-generator-epoch-last-sweep-v2
```

### 12.2 历史工作树

当前可见工作树由 `git worktree list` 给出，主要位于 `.worktrees`。历史工作树只用于读代码、比较和审计，不能作为正式训练输入。

### 12.3 历史实验产物

```text
J:\nlp\CD-C3DA\runs
J:\nlp\CD-C3DA-native-best-rc-v1\runs
```

正式当前最佳、六组基线、软过滤、双生成器、生成器轮次扫描和失败诊断都在 `runs` 下。具体运行目录、指标和清理状态见 `实验记录与模型索引_CN.md`。

## 13. 已验证结论

- 完整双三元组补充有效，权重0.25优于更高权重。
- 简单放宽伪标签会引入噪声。
- 强制确定性会改变历史轨迹，但不是全部性能差异来源。
- 增强内容是性能差异的重要来源。
- 单纯增加生成器轮次并使用 `last` 不能替代8轮 `best` 主线。
- 双生成器和三任务混合生成器在现有训练设计下没有超过主线。
- 第一轮观点软过滤提高通过率但降低最终 F1，原因是编辑目标与最终标签没有闭环。
- 中性和多三元组召回是主要模型短板。

## 14. 下一阶段

先修复双通道增强正确性，再增加数量或模型结构：

1. 控制符硬门禁。
2. 方面、观点、情感和未编辑三元组保持契约。
3. 生成器训练任务与增强编辑任务对齐。
4. 提取器从一票否决改为支持信号。
5. 150条隔离验证后再测试250条分层配额。
6. 数据增强有效后才测试分层权重、辅助头和轻量注意力适配器。

详细内容见根目录 `03`、`04`、`05` 文档。

## 15. 新账号交接检查清单

- [ ] 已阅读根目录 `00` 到 `06` 文档。
- [ ] 已读取 `AGENTS.md` 和正式 Skill。
- [ ] 已运行 Git 只读检查。
- [ ] 已确认当前主线和实验分支身份。
- [ ] 已确认用户当前请求是否允许改代码或只需要分析。
- [ ] 已说明改动计划并等待确认。
- [ ] 已确认8GB显存参数。
- [ ] 已确认不读取其他运行产物。
- [ ] 已为新实验选择独立分支和 `RunId`。
- [ ] 已准备完整单行 CMD 命令和断点恢复说明。
- [ ] 已在实验后更新根目录索引、指标、命令、提交和清理状态。

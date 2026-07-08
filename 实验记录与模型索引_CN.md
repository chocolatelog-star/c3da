# CD-C3DA 实验记录与模型索引

本文档用于配合 git（版本控制）管理本项目的实验。以后每次改代码、训练模型、跑评估，都需要在这里记录：代码版本、实验目录、模型路径、命令、数据组成、指标和分析结论。

## 记录原则

1. 代码、脚本、数据处理逻辑、中文实验记录进入 git（版本控制）和 GitHub（代码托管平台）。
2. 模型权重、训练输出、临时分析输出不进入 git（版本控制），只保存在本地目录，并在本文档记录路径。
3. 每次修改代码后先 commit（提交），再跑实验；实验结果对应到具体 commit（提交）。
4. 如果新实验效果低于当前最好结果，需要先询问是否删除对应实验输出和模型目录。
5. 每次实验至少记录 raw F1（原始 F1）和 fixed F1（修正 F1）。论文对比默认看 raw F1（原始 F1）。

## 文件保存规则

### git 管理的内容

- Python 代码：`*.py`
- 中文文档：`*.md`
- 复现实验脚本和批量运行脚本
- 小规模数据集文件：`dataset/`
- 测试脚本：`test_*.py`

### git 忽略的内容

这些内容保留在本地，但不上传 GitHub（代码托管平台）：

- `runs/`：训练输出、最终模型、生成数据、评估结果
- `models/`：本地模型文件
- `analysis_outputs/`：分析输出
- `__pycache__/`：Python 缓存
- `C3DA/state_dict/`：旧模型权重
- `*.bin`、`*.safetensors`、`*.pt`、`*.pth`、`checkpoint-*` 等模型权重或检查点

## 命名规则

### run_dir（实验目录）

建议格式：

```txt
runs\<任务名>_<源域>_to_<目标域>_<方法名>_<日期或版本>
```

示例：

```txt
runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1
runs\bgca_aste_stage1_baseline
```

### model_dir（模型目录）

建议格式：

```txt
<run_dir>\models\<阶段>_<方法>_<关键参数>_ep<轮数>
```

示例：

```txt
runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1\models\extractor_ep25_plain_last
runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1\models\final_dann_l0.03_strict_aug150_w020_ep5
```

注意：当前训练脚本中最终模型通常保存在模型目录下的 `best` 子目录；如果使用 `--checkpoint_selection last`，目录名仍可能叫 `best`，但含义是 last checkpoint（最后检查点）。

## 当前代码基线

| 项目 | 内容 |
|---|---|
| git commit（提交） | `770d004 Initial project baseline` |
| GitHub（代码托管平台） | https://github.com/chocolatelog-star/c3da.git |
| 当前分支 | `master` |
| 初始管理时间 | 2026-07-08 |

## 当前最好实验

### rest16 -> laptop14：DANN + 严格掩码增强

| 项目 | 内容 |
|---|---|
| 任务 | ASTE（方面-情感-观点三元组抽取）跨域：`rest16 -> laptop14` |
| run_dir（实验目录） | `runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1` |
| 最终模型 | `runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1\models\final_dann_l0.03_strict_aug150_w020_ep5\best` |
| 提取器 | `runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1\models\extractor_ep25_plain_last\best` |
| 生成器 | `runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1\models\generator_masked_aspect_ep8\best` |
| 伪标签来源 | high_precision（高精度伪标签），388 条 |
| 数据增强 | masked_mutual（互相掩码）严格过滤，150 条，权重 0.20 |
| DANN（领域对抗） | `lambda_domain_adv=0.03`，并使用 `domain_adv_exclude_augment` 排除增强样本参与领域对抗 |
| 解码 | `--no_constrained_decoding` |
| raw F1（原始 F1） | `0.4667349028` |
| fixed F1（修正 F1） | `0.4871531346` |
| raw precision（原始精确率） | `0.5229357798` |
| raw recall（原始召回率） | `0.4214417745` |
| fixed precision（修正精确率） | `0.5486111111` |
| fixed recall（修正召回率） | `0.4380776340` |

评估命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python t5_aste_pipeline.py evaluate --run_dir runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1 --model_path runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1\models\final_dann_l0.03_strict_aug150_w020_ep5\best --batch_size 2 --num_beams 4 --max_new_tokens 96 --cuda 0 --no_task_prefix --no_constrained_decoding"
```

实验分析：

- 这是当前单任务 `rest16 -> laptop14` 的阶段性最好结果，可以作为一阶段基线。
- 提升主要来自三个部分：严格伪标签、严格过滤后的掩码增强、DANN（领域对抗）但排除增强样本。
- `--no_constrained_decoding` 对召回率提升明显，但也需要和约束解码做消融，确认提升不是评估设置造成的。
- fixed F1（修正 F1）只作为辅助分析；论文对比优先看 raw F1（原始 F1）。

## 已知重要对照

| 实验 | 模型路径 | raw F1（原始 F1） | fixed F1（修正 F1） | 结论 |
|---|---|---:|---:|---|
| 旧复现最好 masked aspect | `models\final_repro_385_masked_aspect_ep5\best` | `0.3754` | `0.3787` | 旧 38 左右基线，后续 DANN 和解码设置明显超过它 |
| noaug + noDANN | `models\final_noaug_nodann_ep5\best` | 待补充 | 待补充 | 用于判断增强和 DANN 的独立贡献 |
| noaug + DANN l0.03 | `models\final_noaug_dann_l0.03_ep5\best` | 约 `0.4609` | 待补充 | DANN 在不加增强时有效 |
| DANN l0.03 exclude augment | `models\final_dann_l0.03_exclude_aug_ep5\best` | `0.4255` | `0.4390` | 排除增强参与 DANN 后比污染版更合理，但不如严格增强版本 |
| label_to_text 新生成器 | `runs\stage1_gen_label_to_text_test\rest16_to_laptop14` | 约 `0.4614` | 约 `0.4769` | 新标签到文本生成器可用，但暂未超过当前最好 |
| masked_mutual 新生成器 | `runs\stage1_gen_masked_mutual_test\rest16_to_laptop14` | 约 `0.4409` | 约 `0.4578` | 新训练掩码生成器低于当前最好，可能受提取器伪标签质量影响 |

## 伪标签记录

| run_dir（实验目录） | 提取器 | high_precision rows（高精度行数） | high_precision F1（高精度伪标签 F1） | 备注 |
|---|---|---:|---:|---|
| `runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1` | `extractor_ep25_plain_last\best` | `388` | `0.5162011173` | 当前最好流程使用的伪标签基础 |
| `runs\stage1_gen_label_to_text_test\rest16_to_laptop14` | 新训练提取器 | `421` | `0.5035` 左右 | 与当前最好有差距 |
| `runs\stage1_gen_masked_mutual_test\rest16_to_laptop14` | 新训练提取器 | `421` | `0.5035` 左右 | 与 label_to_text 对比时提取器基本一致 |

## 后续跨域实验

BGCA（论文方法）ASTE 跨域对比采用 4 个数据集、6 个跨域方向：

1. `rest14 -> laptop14`
2. `rest15 -> laptop14`
3. `rest16 -> laptop14`
4. `laptop14 -> rest14`
5. `laptop14 -> rest15`
6. `laptop14 -> rest16`

批量脚本：

```txt
run_bgca_aste_stage1_pairs.py
```

默认结果输出：

```txt
runs\bgca_aste_stage1_baseline
```

恢复运行命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_baseline --pairs all --generator_prompt_style label_to_text --cuda 0 --seed 1000 --eval_batch_size 2"
```

注意：

- 脚本通过 `stage_status.json` 记录阶段状态，中断后重新运行同一命令会跳过已完成阶段。
- 如果要强制全部重跑，才加 `--rerun`。

## 每次实验记录模板

### 实验编号：EXP-YYYYMMDD-编号

| 项目 | 内容 |
|---|---|
| 日期 |  |
| 目的 |  |
| git commit（提交） |  |
| 分支 |  |
| source（源域） |  |
| target（目标域） |  |
| run_dir（实验目录） |  |
| extractor（提取器） |  |
| generator（生成器） |  |
| final model（最终模型） |  |
| 数据组成 | source gold / target pseudo / augment |
| 关键参数 |  |
| 训练命令 |  |
| 评估命令 |  |
| raw precision（原始精确率） |  |
| raw recall（原始召回率） |  |
| raw F1（原始 F1） |  |
| fixed precision（修正精确率） |  |
| fixed recall（修正召回率） |  |
| fixed F1（修正 F1） |  |
| 是否超过当前最好 | 是 / 否 |
| 是否保留模型 | 是 / 否 / 待确认 |

分析：

- 

下一步：

- 

## git 操作规范

每次改代码后：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && git status && git add . && git commit -m \"说明这次改动\" && git push"
```

查看历史：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && git log --oneline"
```

查看某次提交内容：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && git show 提交ID"
```

只查看当前代码状态：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && git status"
```

## 当前待补充事项

- 补齐 `final_noaug_nodann_ep5`、`final_noaug_dann_l0.03_ep5` 的完整 raw/fixed 指标。
- 跨域 6 组实验跑完后，将结果汇总到本文档。
- 对当前最好模型补充完整训练命令和增强命令。
- 如果后续接入 GitHub release（发布）或外部网盘保存模型，需要在这里记录模型下载地址和校验信息。

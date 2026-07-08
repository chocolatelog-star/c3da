# CD-C3DA 实验记录与模型索引

本文档是项目实验总台账，配合 git（版本控制）和 GitHub（代码托管平台）一起管理代码版本、实验输出、模型路径和实验结论。

以后维护原则：优先整体修改表格和结论，不在文档末尾无限追加；每个实验都要能追溯到 git commit（提交版本）、run_dir（实验目录）、model_path（模型路径）和评估命令。

## 0. 当前总览

| 项目 | 内容 |
|---|---|
| GitHub（代码托管平台） | https://github.com/chocolatelog-star/c3da.git |
| 当前实验代码版本 | `4906184 Add experiment tracking log` |
| 当前分支 | `master` |
| 主实验目录 | `runs\bgca_aste_stage1_baseline` |
| 论文对比指标 | raw F1（原始 F1） |
| 辅助分析指标 | fixed F1（修正 F1） |
| 当前已完成跨域 | `rest14 -> laptop14` |

## 1. BGCA 论文 ASTE 六组结果

来源：BGCA 论文 Table 4（表 4），任务为 ASTE（方面情感三元组抽取），结果是 5 次运行的平均 F1 分数。这里用于论文基线对比，默认看 `BGCAlabel-to-text`（标签到文本）这一行。

| 跨域方向 | BGCA text-to-label（文本到标签）F1 | BGCA label-to-text（标签到文本）F1 | 论文最好对比方法 | 论文最好对比 F1 |
|---|---:|---:|---|---:|
| R14 -> L14 | 52.55 | 53.64 | GAS | 49.57 |
| R15 -> L14 | 45.85 | 45.69 | GAS | 43.78 |
| R16 -> L14 | 46.86 | 47.28 | GAS | 45.24 |
| L14 -> R14 | 61.52 | 65.27 | GAS | 64.40 |
| L14 -> R15 | 55.43 | 58.95 | GAS | 56.26 |
| L14 -> R16 | 61.15 | 64.00 | GAS | 63.14 |
| 平均 | 53.89 | 55.80 | GAS | 53.73 |

说明：

- R14/R15/R16 分别表示 restaurant14（餐厅 2014）、restaurant15（餐厅 2015）、restaurant16（餐厅 2016）。
- L14 表示 laptop14（笔记本 2014）。
- BGCA 论文中 ASTE（方面情感三元组抽取）的最终主结果是 `BGCAlabel-to-text`（标签到文本），平均 F1 为 55.80。
- 我们和论文比较时优先使用 raw F1（原始 F1），不要用 fixed F1（修正 F1）去直接对论文表格。

## 2. 我们的 BGCA 六组跨域实验进度

运行脚本：

```txt
run_bgca_aste_stage1_pairs.py
```

结果文件：

```txt
runs\bgca_aste_stage1_baseline\results_bgca_aste_stage1.csv
runs\bgca_aste_stage1_baseline\results_bgca_aste_stage1_CN.md
```

| 跨域方向 | 状态 | generator（生成器）方式 | pseudo high_precision F1（高精度伪标签 F1） | augment（增强条数） | final train（最终训练条数） | raw P | raw R | raw F1 | fixed F1 | 对 BGCA label-to-text 差值 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rest14 -> laptop14 | 已完成 | label_to_text（标签到文本） | 54.38 | 150 | 1835 | 60.19 | 45.84 | 52.05 | 53.36 | -1.59 |
| rest15 -> laptop14 | 待跑 | label_to_text（标签到文本） | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| rest16 -> laptop14 | 待跑 | label_to_text（标签到文本） | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| laptop14 -> rest14 | 待跑 | label_to_text（标签到文本） | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| laptop14 -> rest15 | 待跑 | label_to_text（标签到文本） | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| laptop14 -> rest16 | 待跑 | label_to_text（标签到文本） | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

当前已完成结果分析：

- `rest14 -> laptop14` 的 raw F1（原始 F1）为 52.05，低于 BGCA label-to-text（标签到文本）的 53.64，差值为 -1.59。
- 该组 fixed F1（修正 F1）为 53.36，接近论文 BGCA 的 53.64，但论文对比仍应以 raw F1（原始 F1）为准。
- 该组 precision（精确率）60.19 较高，recall（召回率）45.84 是主要短板。
- 伪标签 high_precision F1（高精度伪标签 F1）为 54.38，说明伪标签质量可用；最终模型没有超过 BGCA，后续重点看其他跨域是否也存在召回不足。

## 3. 已完成实验详情

### EXP-20260708-01：rest14 -> laptop14

| 项目 | 内容 |
|---|---|
| 任务 | ASTE（方面情感三元组抽取）跨域：`rest14 -> laptop14` |
| git commit（提交版本） | `4906184 Add experiment tracking log` |
| run_dir（实验目录） | `runs\bgca_aste_stage1_baseline\rest14_to_laptop14` |
| extractor（提取器） | `runs\bgca_aste_stage1_baseline\rest14_to_laptop14\models\extractor_ep25_plain_last\best` |
| generator（生成器） | `runs\bgca_aste_stage1_baseline\rest14_to_laptop14\models\generator_label_to_text_gen_ep8\best` |
| final model（最终模型） | `runs\bgca_aste_stage1_baseline\rest14_to_laptop14\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_ep5\best` |
| 源域人工标注 | 1266 条 |
| high_precision pseudo（高精度伪标签） | 421 条 |
| augment（增强数据） | 150 条 |
| final train（最终训练数据） | 1835 条 |
| 关键参数 | `lambda_domain_adv=0.03`，`domain_adv_exclude_augment`，严格增强 150 条，增强权重 0.20，`--no_constrained_decoding` |
| raw precision（原始精确率） | 0.6019417476 |
| raw recall（原始召回率） | 0.4584103512 |
| raw F1（原始 F1） | 0.5204616999 |
| fixed precision（修正精确率） | 0.6180048662 |
| fixed recall（修正召回率） | 0.4695009242 |
| fixed F1（修正 F1） | 0.5336134454 |
| 是否超过 BGCA label-to-text（标签到文本） | 否，低 1.59 F1 |
| 是否保留模型 | 暂时保留，作为六组跨域对比中的已完成结果 |

评估命令：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python t5_aste_pipeline.py evaluate --run_dir runs\bgca_aste_stage1_baseline\rest14_to_laptop14 --model_path runs\bgca_aste_stage1_baseline\rest14_to_laptop14\models\final_dann_l0.03_strict_aug150_w020_label_to_text_gen_ep5\best --batch_size 2 --num_beams 4 --max_new_tokens 96 --cuda 0 --no_task_prefix --no_constrained_decoding"
```

## 4. 当前单任务最好历史结果

这部分记录我们在 `rest16 -> laptop14` 上早期探索得到的阶段性最好结果，用于内部方法演进参考，不直接替代六组 BGCA 对比。

| 项目 | 内容 |
|---|---|
| 任务 | ASTE（方面情感三元组抽取）跨域：`rest16 -> laptop14` |
| run_dir（实验目录） | `runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1` |
| 最终模型 | `runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1\models\final_dann_l0.03_strict_aug150_w020_ep5\best` |
| 提取器 | `runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1\models\extractor_ep25_plain_last\best` |
| 生成器 | `runs\t5_c3da_aste_rest16_to_laptop14_retrain_extractor_plain_v1\models\generator_masked_aspect_ep8\best` |
| high_precision pseudo（高精度伪标签） | 388 条，F1 为 0.5162011173 |
| 数据增强 | masked_mutual（互相掩码）严格过滤，150 条，权重 0.20 |
| DANN（领域对抗） | `lambda_domain_adv=0.03`，`domain_adv_exclude_augment` |
| 解码 | `--no_constrained_decoding` |
| raw F1（原始 F1） | 0.4667349028 |
| fixed F1（修正 F1） | 0.4871531346 |

## 5. 实验输出和模型保存规则

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

### 目录命名规则

run_dir（实验目录）：

```txt
runs\<任务名>_<源域>_to_<目标域>_<方法名>_<日期或版本>
```

model_dir（模型目录）：

```txt
<run_dir>\models\<阶段>_<方法>_<关键参数>_ep<轮数>
```

注意：当前训练脚本中最终模型通常保存在模型目录下的 `best` 子目录；如果使用 `--checkpoint_selection last`，目录名仍可能叫 `best`，但含义是 last checkpoint（最后检查点）。

## 6. 后续运行命令

恢复或继续六组跨域实验：

```cmd
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_baseline --pairs all --generator_prompt_style label_to_text --cuda 0 --seed 1000 --eval_batch_size 2"
```

说明：

- 脚本通过 `stage_status.json` 记录阶段状态，中断后重新运行同一命令会跳过已完成阶段。
- 如果要强制全部重跑，才加 `--rerun`。

## 7. 每次实验记录模板

| 项目 | 内容 |
|---|---|
| 实验编号 | EXP-YYYYMMDD-编号 |
| 日期 |  |
| 目的 |  |
| git commit（提交版本） |  |
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

## 8. git 操作规范

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

## 9. 当前待补充事项

- 跑完剩余 5 组 BGCA ASTE 跨域实验，并更新第 2 节总表。
- 补齐每组的 pseudo high_precision F1（高精度伪标签 F1）、augment（增强条数）、final train（最终训练条数）。
- 六组完成后计算我们自己的平均 raw F1（原始 F1），并和 BGCA label-to-text（标签到文本）平均 55.80 对比。
- 如果后续模型效果低于当前最好结果，先询问是否删除对应实验输出和模型目录。

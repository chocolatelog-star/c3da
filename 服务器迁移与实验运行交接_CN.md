# CD-C3DA 服务器迁移与实验运行交接

本文档用于向后续对话交接 CD-C3DA 在 AutoDL Linux 服务器上的部署、运行和结果分析信息。研究状态仍以 `.ai/PROJECT_STATE.md`、`.ai/CURRENT_TASK.md`、`.ai/DECISION_LOG.md`、实验记录与模型索引_CN.md 和 03_CD-C3DA下一阶段改进计划_CN.md 为准。

## 当前服务器

```text
SSH 主机：connect.nmb1.seetacloud.com
SSH 端口：10336
SSH 用户：root
项目目录：/root/CD-C3DA
数据盘：/root/autodl-tmp
GPU：NVIDIA GeForce RTX 4090，24564 MiB
驱动：570.124.04，驱动报告 CUDA 12.8
```

不要在文档、脚本或命令中保存 SSH 密码。

## Linux 环境

```text
Conda：/root/miniconda3
环境：c3da
Python：3.10
PyTorch：2.2.2+cu121
Transformers：4.39.3
Accelerate：0.28.0
Stanza：1.14.0
```

## 输出目录约定

所有新实验的输出、日志、检查点和临时文件必须使用数据盘：

```text
输出根目录：/root/autodl-tmp/CD-C3DA-runs/
临时目录：/root/autodl-tmp/tmp/
```

启动命令应设置 `TMPDIR=/root/autodl-tmp/tmp`，不要把新的实验输出写入 `/root/CD-C3DA/runs`。

## 已完成的清理

已清空服务器回收站，并删除两个确认失败的旧运行目录：

```text
/root/CD-C3DA/runs/reproducible/rest16_to_laptop14_best_server
/root/CD-C3DA/runs/reproducible/rest16_to_laptop14_best_4090
```

项目、模型和 Conda 环境未删除。

## 入口脚本改动

服务器文件：`/root/CD-C3DA/run_bgca_aste_stage1_pairs.py`。

已新增命令行开关 `--disable_gradient_checkpointing`。默认行为仍保持开启梯度检查点；只有显式传入该参数时才关闭。修改后已在服务器使用 Python 语法检查通过。

## 当前最佳流程

```text
方向：rest16 -> laptop14
运行目录：/root/autodl-tmp/CD-C3DA-runs/reproducible/rest16_to_laptop14_best_4090_b100
日志：/root/autodl-tmp/CD-C3DA-runs/reproducible/rest16_to_laptop14_best_4090_b100.log
训练 batch：100
评估 batch：128
梯度累积：1
梯度检查点：关闭
seed：1000
```

抽取器和生成器训练已完成，最近观测流程处于 `augment`（增强数据生成/筛选）阶段，进度约 428/624。该阶段主要由 CPU、NLI 和数据筛选驱动，GPU 利用率下降是正常现象；应等待最终 ASTE 训练和评估自然完成，不要因短时 GPU 利用率低而中断。

## 运行监控命令

```bash
watch -n 1 nvidia-smi
ps -eo pid,etime,pcpu,pmem,cmd | grep -E 'run_bgca|t5_absa|t5_aste_pipeline|python' | grep -v grep
tail -f /root/autodl-tmp/CD-C3DA-runs/reproducible/rest16_to_laptop14_best_4090_b100.log
```

## 后续流程与分析边界

1. 当前 b100 流程结束前不要启动 M1 图结构流程。
2. 不得使用 target test 金标进行候选选择、调参、阈值选择或检查点选择；只能在训练完成后用于最终报告。
3. 流程结束后先确认最终状态，再读取最终指标、结果卡和阶段状态。
4. 恢复中断流程前先检查检查点和 `stage_status.json`，不要直接覆盖目录。
5. 新实验必须使用数据盘下的新输出目录，并记录完整命令、参数、主机、GPU、seed 和代码版本。
6. 最佳流程完成并报告后，才由用户决定是否启动最新 M1 句法 RGAT 流程。

## 后续对话接手

接手时先读取本文件和项目控制文档，再通过 SSH 只读检查 `df -h /`、`nvidia-smi`、当前进程、日志和阶段状态。不要因显存没有占满就自动改变研究参数；参数调整必须由用户明确批准。

补充：当前SSH（安全外壳）端口为48759；旧rest16 -> laptop14 b100运行及元素感知batch=1/4/8模型副本已删除，关键指标、日志、伪标签分析和batch=16 best模型保留。

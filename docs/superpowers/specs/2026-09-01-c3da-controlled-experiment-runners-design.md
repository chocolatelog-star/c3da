# C3DA 可控实验编排器设计

## 目标

在不改变历史最佳训练语义的前提下，补齐 2026-08-31 实验计划中尚缺的正式运行入口，使两台服务器可以并行、可恢复地完成以下实验：

1. upstream-only batch（仅上游批次）矩阵及 source-dev F1（源域开发集 F1）汇总；
2. 固定同一上游产物的下游 batch（批次）矩阵；
3. Batch16 单步梯度、参数更新及真实有效样本归一化审计；
4. 相同公共 recipe（配置方案）下的 Graph OFF / Graph ON（关闭图 / 开启图）完整 A/B。

历史最佳入口 `run_reproducible_pipeline.py` 保持为权威执行器。新功能通过独立编排器组合现有命令和产物，不把实验矩阵逻辑写入核心训练循环。

## 共同约束

- 所有正式训练使用 GPU，并显式记录物理 GPU、CUDA、PyTorch、Transformers 和 Python 版本。
- 每个实验单元使用独立输出目录，禁止不同配置写入同一目录。
- 所有训练设置 `save_total_limit=1`，只保留恢复所需检查点和最终 best/last 模型。
- hash（哈希）比较写入结果清单，但默认不阻塞后续阶段；只有显式 strict（严格）模式才阻塞。
- 恢复前验证配置、输入和关键产物身份；不完整 checkpoint（检查点）不得自动恢复。
- 不使用 target-test gold（目标测试集标注）选择 batch、归一化策略或图参数。
- 编排器输出机器可读 JSON 和中文 Markdown 汇总。
- 子进程退出码非零时记录失败阶段、命令和日志路径；其他独立单元可以继续运行。
- 默认串行；指定多个 GPU 时允许不同实验单元并行，但同一单元的阶段保持顺序执行。

## 组件一：上游 batch 矩阵

新增 `run_upstream_batch_matrix.py`。

### 输入

- 基础 recipe；
- 输出根目录；
- batch/accumulation 组合，默认：`1x16、8x2、16x1、16x2、32x1`；
- GPU 列表；
- seed，默认 1000。

### 执行流程

每个组合调用历史最佳执行器并停止在 pseudo（伪标签）阶段：

```text
prepare -> extractor -> source-dev evaluate -> target pseudo
```

source-dev evaluate 是独立评估步骤，使用训练完成的 extractor 和 `extract_dev.jsonl`，输出 Raw/Fixed F1，不以 Trainer 的 eval loss 代替 F1。

### 输出

每组输出独立目录和 `upstream_result.json`，矩阵根目录生成：

- `upstream_batch_matrix.json`；
- `upstream_batch_matrix.md`；
- 训练时间、optimizer steps、峰值显存、模型 hash；
- source-dev Raw/Fixed F1；
- full/high-precision pseudo 行数、情感分布和语义 hash。

## 组件二：固定上游的下游 batch 矩阵

新增 `run_fixed_upstream_downstream_batch_matrix.py`。

### 输入

- 一个已完成且通过身份检查的公共 upstream（上游）目录；
- 基础 recipe；
- final-train batch/accumulation 组合；
- GPU 列表；
- 输出根目录。

### 公平性约束

所有组必须引用完全相同的：

- extractor 模型；
- target pseudo 文件；
- generator 模型；
- selected augmentation 文件；
- final train/dev 文件。

编排器在启动前计算上述产物 hash，并把公共身份写入 `shared_upstream_manifest.json`。任一组若解析到不同输入立即失败。

### 执行流程与输出

每组只运行 final_train 和 evaluate，禁止重新生成上游产物。输出每组 Raw/Fixed F1、训练时间、optimizer steps、峰值显存和相对基准差值，最终生成 JSON 与中文 Markdown 矩阵。

## 组件三：梯度、参数更新与归一化审计

新增 `batch_gradient_parameter_audit.py`。

### 审计数据

- 从指定训练 JSONL 中按固定 seed 选择同一批 16 条真实样本；
- 保存样本 ID、triplet 数、token 数和 sample weight；
- 所有比较从同一模型 state dict（状态字典）、随机状态和优化器配置开始。

### 比较组

- `batch=1, accumulation=16`；
- `batch=4, accumulation=4`；
- `batch=8, accumulation=2`；
- `batch=16, accumulation=1`。

每组分别运行：

1. 当前 batch mean（批次均值）归约；
2. 按真实有效样本权重总和归一化的归约。

### 输出

- generation、focus、coverage 和 joint loss；
- 全局梯度范数及逐参数梯度摘要；
- optimizer step 后参数更新范数；
- 相对 `1x16` 的最大/平均参数差异；
- 尾 batch 行为；
- 是否满足数值等价容差的结论。

审计脚本默认只执行一个 optimizer step，不访问 target-test。

## 组件四：Graph OFF / Graph ON 完整 A/B

新增 `run_graph_control_ab.py`，替代当前只运行 Graph ON 的单边入口作为正式比较入口。

### 输入与流程

- 相同 source/target、seed、batch、accumulation 和历史最佳下游参数；
- Control OFF 与 Graph ON 各自独立 Phase A 目录；
- 允许验证并复用已完成 Phase A，禁止因下游失败重训已完成的 Phase A。

两组分别执行：

```text
Phase A identity check
-> pseudo filtering
-> generator
-> augmentation
-> final train
-> target evaluation
```

唯一允许的结构差异是 `graph_enabled=false/true`。编排器生成 `graph_control_identity.json`，逐项证明除图开关和图参数外的配置一致。

### 输出

- Control 和 Graph 的 Raw/Fixed F1；
- Graph-Control delta；
- single/multi-triplet 指标；
- pseudo 数量和情感分布；
- 两组公共配置、输入 hash 和阶段状态；
- `graph_control_ab.json` 与中文 Markdown 报告。

## 恢复与并行模型

每个实验单元维护 `status.json`：

```text
pending -> running -> complete | failed
```

阶段完成时原子写入输出身份。重新启动时：

1. 配置与输入身份一致且产物完整：跳过；
2. 身份一致但 checkpoint 不完整：忽略不完整 checkpoint，从该阶段重跑；
3. 身份不一致：拒绝混用，要求新的输出目录；
4. 独立实验单元失败：不阻塞其他 GPU 上的单元。

## 测试设计

所有生产代码遵循测试先行：

- 命令生成测试：确保 batch、accumulation、stop stage、DANN 和图开关正确；
- 身份测试：固定上游矩阵拒绝任何输入 hash 不一致；
- 恢复测试：完整阶段跳过、不完整 checkpoint 重跑；
- 汇总测试：缺失或失败单元仍生成明确状态；
- 审计数值测试：等价归约在合成数据上满足容差，非等价归约能被检测；
- Graph A/B 测试：两组除图字段外完全一致；
- 全量回归：现有 251 个测试必须继续通过。

## 非目标

- 本次不选择最佳 batch；
- 不搜索学习率、DANN 权重、伪标签权重或图超参数；
- 不改变历史最佳 recipe 的默认数值；
- 不删除已有实验产物；
- 不在实现阶段启动正式长时间 GPU 实验，只做单元测试、dry-run（空运行）和轻量 smoke test（冒烟测试）。

## 完成判据

- 四个编排/审计入口均有 `--help`、dry-run 和中文使用说明；
- 每个入口可从中断状态恢复；
- 关键单变量约束由自动化测试验证；
- 全量测试通过；
- 同一提交同步到两台服务器后，路径预检和 dry-run 均通过。

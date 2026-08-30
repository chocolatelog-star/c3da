# Element-Aware RGAT 组件消融设计

> 更新时间：2026-08-30（北京时间）
> 状态：已获用户批准，等待实现计划

## 1. 任务

- 任务编号：`M1_ELEMENT_AWARE_RGAT_COMPONENT_ABLATION_V1`
- 方向：`laptop14 -> rest15`
- 随机种子：`1000`
- 目标：只完成 G1（仅元素聚焦）和 G2（仅多元素覆盖）的正式可比快速消融。
- G3（元素聚焦与多元素覆盖）已有服务器诊断结果，不重跑。

本任务不增加模型结构、图通道、损失函数或可搜索超参数。Element Salience Gate（元素显著性门控）、Dependency/POS Graph（依存/词性图）、4-head RGAT（四头关系图注意力网络）和 Gated Fusion（门控融合）保持不变。

## 2. 冻结训练配方

按用户在原 P0 文本之后的修订，G1 和 G2 复用 V9e Graph+DANN0 的原始训练配方，而不是服务器 batch=16 诊断配置：

```text
micro batch = 1
gradient accumulation = 16
effective batch = 16
eval batch = 2
seed = 1000
DANN = 0
fp16 = true
gradient checkpointing = true
```

运行入口必须拒绝其他训练微批大小和梯度累积值，防止正式消融发生配置漂移。

## 3. 组件配置

### G1：仅元素聚焦

```text
element-aware attention = enabled
focus loss = enabled
coverage loss = disabled
focus weight = 0.05
effective coverage weight = 0
```

总损失：

```text
L_total = L_ASTE + 0.05 * L_focus
```

### G2：仅多元素覆盖

```text
element-aware attention = enabled
focus loss = disabled
coverage loss = enabled
effective focus weight = 0
coverage weight = 0.05
```

总损失：

```text
L_total = L_ASTE + 0.05 * L_coverage
```

单三元组源域样本的覆盖损失保持为0；多三元组源域样本按现有实现计算。关闭的损失不得执行，也不得向总损失贡献梯度。

## 4. 实现边界

### 训练入口

调整 `t5_absa_train.py` 的元素感知配置校验：

- G1 只接受 `focus=0.05, coverage=0`；
- G2 只接受 `focus=0, coverage=0.05`；
- G3 继续接受 `focus=0.05, coverage=0.05`；
- 未启用元素感知时不改变旧实验行为；
- 元素感知实验继续强制 `DANN=0`。

### 消融运行器

调整 `m1_element_aware_rgat_treatment_only.py`：

- `--focus_only` 与 `--coverage_only` 保持互斥；
- 不向子入口传递空字符串参数；
- 精确传递对应损失开关及权重；
- 固定并校验 V9e 的 `micro=1, accumulation=16, eval=2`；
- 训练从基础 T5 和确定性图初始化开始；
- 仅允许从同一运行目录的检查点自动恢复，不读取 G3、V9e 或其他运行的检查点；
- 运行身份文件记录组件版本、实际/有效权重、批次参数、DANN、代码身份和数据边界。

## 5. 数据与阶段边界

- 允许读取 source train、source dev 和 target unlabeled。
- 禁止读取 target test 及其金标。
- 禁止启动 augmentation（数据增强）、Phase B（阶段B）和 final ASTE（最终方面情感三元组抽取）。
- G1/G2 必须使用不同的新运行目录，禁止混用产物。
- G3 只作为既有训练配置诊断参考；由于其服务器运行使用 micro batch=16，不能冒充与 V9e 配方严格一致的正式对照。

## 6. 恢复与进度

- 训练、评估和目标无标签推理继续显示现有进度条。
- 中断后只允许用原命令和原运行目录恢复。
- 恢复前必须核验实验版本、批次、损失开关、输入和代码身份；身份不一致时停止。

## 7. 测试

先写失败测试，再做最小实现。至少覆盖：

1. G1 只启用聚焦损失，权重为 `0.05/0`；
2. G2 只启用覆盖损失，权重为 `0/0.05`；
3. G3 的 `0.05/0.05` 兼容性；
4. 非 V9e 批次参数被拒绝；
5. `DANN != 0` 被拒绝；
6. 命令参数中不存在空字符串；
7. 同目录自动恢复，不允许外部检查点热启动；
8. 运行身份记录准确；
9. target test 未访问；
10. 现有元素感知、图和训练入口回归；
11. Python 语法检查和 `git diff --check`。

## 8. 交付与同步

本轮先在本地元素感知工作树修改、测试并提交。服务器当前实验结束前，不切换服务器分支、不更新服务器工作目录。确认服务器无训练进程后，再将本地提交同步到服务器元素感知分支并执行 CPU 测试；正式 GPU 实验仍由用户手动启动。

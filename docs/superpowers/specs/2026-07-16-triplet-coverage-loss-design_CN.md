# 三元组覆盖辅助学习设计

## 1. 背景与目标

当前 `rest16 -> laptop14` 最佳模型的 raw F1（原始综合指标）为 46.82%，其中多三元组子集 raw recall（原始召回率）为 33.33%。编码器方面词-观点词配对损失把多三元组精确率从 59.20% 提高到 60.64%，但召回率降至 31.93%，说明模型的主要结构问题已经从“配错”收敛为“少生成”。

本实验在当前最佳最终训练流程上增加独立的 triplet coverage loss（三元组覆盖损失），使 T5（文本到文本转换器）编码器显式学习一句话应包含的三元组数量。实验只验证覆盖目标本身，不同时修改伪标签、增强数据、DANN（领域对抗神经网络）、情感对比学习或解码参数。

## 2. 实验边界

保持不变：

- 复用 `final_train_strict_aug150_w020_label_to_text_gen.jsonl` 和对应验证集。
- 使用 `label_to_text`（标签到文本）生成器和 `masked_mutual`（互相掩码）增强。
- 使用 hp1（每句最多一个三元组）高精度伪标签、150 条严格增强和增强权重 0.20。
- DANN（领域对抗神经网络）权重保持 0.03。
- 最终阶段源域类别平衡情感对比学习权重保持 0.01。
- 最终模型训练 5 轮，评估使用 batch size（批大小）2、beam search（束搜索）4 和无约束解码。
- `lambda_pairing_loss`（配对损失权重）固定为 0。

本轮不实现：

- 不改变提取器、伪标签或生成器。
- 不加入数量控制特殊标记。
- 不改变目标文本格式和三元组解析协议。
- 不加入解码阶段数量强制或召回重排。
- 不处理新的中性样本构造。

## 3. 数据与标签

最终训练文件中的 857 条 source gold（源域人工标注）分布为：

| 覆盖类别 | 三元组数量 | 行数 |
|---|---:|---:|
| 0 | 1 | 505 |
| 1 | 2 | 229 |
| 2 | 3 个及以上 | 123 |

覆盖标签由训练目标中的规范三元组列表直接计算：

```text
coverage_label = min(max(triplet_count, 1), 3) - 1
```

只允许 source gold（源域人工标注）参与覆盖损失。target pseudo（目标域伪标签）被 hp1 规则截断，无法提供真实数量监督；masked augmentation（掩码增强）经过筛选但结构可靠性仍不足，也不参与覆盖损失。无效样本使用 `-100` 作为忽略标签。

## 4. 模型结构

新增 `TripletCoverageHead`（三元组覆盖分类头）：

```text
encoder_last_hidden_state
  -> attention-mask mean pooling（注意力掩码均值池化）
  -> dropout(0.1)（随机失活）
  -> linear(hidden_size, 3)（三分类线性层）
  -> coverage logits（覆盖分类分数）
```

均值池化复用现有 `mean_pool_encoder_hidden`，不增加第二次编码器前向传播。分类头参数量相对 T5-base（基础版文本到文本转换器）很小，RTX 3070 8GB（8GB显存）的原 batch size（批大小）1和 gradient accumulation（梯度累积）16保持不变。

## 5. 损失函数

总损失为：

```text
L_total = L_generation
        + lambda_domain * L_domain
        + lambda_sentiment * L_sentiment
        + lambda_coverage * L_coverage
```

其中：

```text
lambda_domain = 0.03
lambda_sentiment = 0.01
lambda_coverage = 0.01
lambda_pairing = 0
```

`L_coverage` 使用带类别权重的 cross entropy（交叉熵）：

```text
w_c = (1 / sqrt(n_c)) / mean_c(1 / sqrt(n_c))
```

权重根据实际 source gold（源域人工标注）训练分布自动计算并记录，不在代码中硬编码。覆盖损失只对 `coverage_label != -100` 的行求平均；没有有效覆盖标签的微批次返回与编码器相连的零损失，避免设备或混合精度问题。

## 6. 参数与路由

`t5_absa_train.py` 新增：

```text
--lambda_triplet_coverage 0.01
--triplet_coverage_source_only
--triplet_coverage_class_balanced
```

`run_bgca_aste_stage1_pairs.py` 暴露同名参数，并在最终模型名、评估标签和汇总文件中加入：

```text
coverage_encoder_l001_source_balanced
```

当覆盖损失为 0 时，行为和旧版完全一致，不创建覆盖分类头，不要求覆盖专属评估文件。覆盖实验只重跑最终 5 轮训练与评估，继续使用 `--resume_from_checkpoint auto`（自动从检查点恢复）。

## 7. 训练日志

每个训练日志周期输出：

- `triplet_coverage_loss`（三元组覆盖损失）。
- `triplet_coverage_accuracy`（总体覆盖分类准确率）。
- `triplet_coverage_count1_accuracy`（单三元组准确率）。
- `triplet_coverage_count2_accuracy`（双三元组准确率）。
- `triplet_coverage_count3plus_accuracy`（3个及以上准确率）。
- `triplet_coverage_active_rows`（有效源域行数）。

准确率统计必须只除以对应有效行数，不能像旧配对日志一样把无效微批次按 0 纳入平均。有效行数按日志周期求和，其余指标按有效样本加权聚合。

## 8. 评估与输出

在现有总体、分情感、单/多三元组评估基础上新增数量覆盖分析：

```text
aste_metrics_by_triplet_count_<tag>.json
```

包含：

- 金标数量类别与预测数量类别的 3x3 混淆矩阵。
- 数量完全匹配率。
- `under_generated_rows`（少生成行数）。
- `exact_count_rows`（数量正确行数）。
- `over_generated_rows`（多生成行数）。
- 1、2、3个及以上金标子集各自的 raw/fixed（原始/修正）精确率、召回率和 F1（综合指标）。

评估数量直接从解析后的金标和预测三元组列表计算，不依赖训练分类头；因此即使使用普通 `T5ForConditionalGeneration`（T5条件生成模型）加载检查点，也能完成覆盖分析。

## 9. 验收标准

与当前最佳模型在相同测试集和解码参数下比较：

1. 首要指标：raw F1（原始综合指标）超过 46.82%。
2. 多三元组 raw recall（原始召回率）超过 33.33%。
3. FP（错误预测三元组）不能出现抵消召回收益的大幅增长。
4. 数量混淆矩阵中，多三元组被预测为单三元组的比例应下降。
5. 中性 F1（综合指标）只记录，不作为本轮通过条件。

若 raw F1 未超过最佳，但多三元组召回显著提高，则保留指标和代码作为后续解码召回机制的依据，不直接进入最佳流程。

## 10. 测试与验证

新增或扩展测试覆盖：

- 覆盖标签 1、2、3个及以上映射正确。
- 伪标签和增强样本在 source-only（仅源域）模式下标签为 `-100`。
- 类别权重由训练集分布自动计算且归一化正确。
- 覆盖头前向形状和 CPU/GPU（中央处理器/图形处理器）设备一致。
- 混合精度下无有效标签的微批次返回可反向传播的零损失。
- 覆盖日志只按有效样本聚合。
- 运行脚本正确生成独立模型名和结果标签。
- 数量混淆矩阵、少生成/正确/多生成统计正确。
- 当 `lambda_triplet_coverage=0` 时旧流程回归测试通过。

最后执行静态编译、相关单元测试、CUDA（显卡计算平台）最小前向/反向测试和 `--dry_run`（预演）完整命令检查；不在代码验证阶段启动完整训练。

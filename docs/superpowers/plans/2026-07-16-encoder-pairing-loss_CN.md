# 编码器上下文方面词-观点词配对损失实施计划

> **For agentic workers（面向代理执行者）：** REQUIRED SUB-SKILL（必需子技能）：使用 `superpowers:subagent-driven-development`（子代理驱动开发，推荐）或 `superpowers:executing-plans`（执行计划），按任务逐项实施。所有步骤使用复选框跟踪。

**目标：** 在当前最佳最终训练中加入只使用源域人工标注多三元组句子的编码器上下文多正例双向配对损失，并输出配对训练日志和多三元组子集指标。

**架构：** 数据集从原始输入句子定位方面词和观点词跨度；训练器在 T5 编码器隐藏状态上计算同句多正例双向对比损失；批处理脚本复用现有最佳最终训练文件，只生成独立最终模型和评估结果。默认配对系数为 0，保证历史流程兼容。

**技术栈：** Python（编程语言）3.10、PyTorch（深度学习框架）、Transformers（预训练模型库）、T5-base（基础文本到文本模型）、`unittest`（单元测试框架）、Git（版本管理）。

---

## 文件结构

- 修改 `t5_absa_train.py`：编码器输入跨度、源域限定、多正例双向损失、统计日志和命令参数。
- 修改 `run_bgca_aste_stage1_pairs.py`：配对参数路由、独立模型/指标/汇总标签和旧最终训练集复用。
- 修改 `t5_aste_pipeline.py`：评估输出单三元组与多三元组子集指标。
- 修改 `test_dynamic_sample_weight.py`：数据跨度、多正例损失、共享跨度和无负例测试。
- 修改 `test_domain_adversarial_train.py`：训练器配对损失和日志测试。
- 修改 `test_run_bgca_stage1_pairs.py`：最终训练单变量预演与输出隔离测试。
- 修改 `test_evaluate_output_isolation.py`：结构子集指标输出隔离测试。
- 修改 `实验记录与模型索引_CN.md`、`CD_C3DA_BGCA超越目标路线图_CN.md`：版本、命令、模型位置和验收标准。

### 任务 1：编码器输入配对跨度

**文件：**

- 修改：`test_dynamic_sample_weight.py`
- 修改：`t5_absa_train.py`

- [ ] **步骤 1：添加失败测试**

测试多词方面词和观点词从 `input`（输入句子）而非 `target`（目标标签）定位：

```python
rows = [{
    "input": "The battery life is very long but the screen is dark.",
    "target": "<pos> battery life <opinion> very long ; <neg> screen <opinion> dark",
    "sample_weight": 1.0,
}]
dataset = JsonlSeq2SeqDataset(
    rows, TinyTokenizer(), 96, 96, 1.0, 0.5, 0.2,
    max_pairing_triplets=4, pairing_source_only=True,
)
item = dataset[0]
assert len(item["pairing_aspect_spans"]) == 2
assert len(item["pairing_opinion_spans"]) == 2
```

增加伪标签、增强数据和单三元组均返回空配对特征的测试。

- [ ] **步骤 2：运行失败测试**

```text
J:\conda\envs\c3da\python.exe -c "import test_dynamic_sample_weight as m; m.test_dataset_returns_encoder_pairing_spans_for_source_multi_triplets(); m.test_pairing_source_only_excludes_pseudo_augment_and_single_triplets()"
```

预期：因 `pairing_source_only` 参数或输入跨度功能不存在而失败。

- [ ] **步骤 3：实现输入跨度和源域限定**

给数据集增加：

```python
pairing_source_only: bool = False
```

将调用改为：

```python
model_inputs.update(self.pairing_features(row, model_inputs["input_ids"]))
```

把观点词定位函数泛化为：

```python
def find_fragment_span_in_input(tokenizer, text, input_ids, fragment):
    # 保留现有大小写兼容候选逻辑，并在截断后的 input_ids 中查找。
```

`pairing_features()` 对源域人工标注多三元组逐个定位方面词和观点词，无法完整定位的三元组跳过；若最终少于 2 个有效三元组则返回空特征。

- [ ] **步骤 4：运行数据集回归测试**

```text
J:\conda\envs\c3da\python.exe -c "import test_dynamic_sample_weight as m; tests=[getattr(m,n) for n in sorted(dir(m)) if n.startswith('test_')]; [t() for t in tests]; print(f'{len(tests)} tests passed')"
```

预期：全部通过。

- [ ] **步骤 5：提交**

```text
git add t5_absa_train.py test_dynamic_sample_weight.py && git commit -m "Extract source encoder pairing spans"
```

### 任务 2：多正例双向配对损失与日志

**文件：**

- 修改：`test_dynamic_sample_weight.py`
- 修改：`test_domain_adversarial_train.py`
- 修改：`t5_absa_train.py`

- [ ] **步骤 1：添加损失失败测试**

新增 `encoder_pairing_contrastive_loss()` 测试：

```python
loss, stats = encoder_pairing_contrastive_loss(
    hidden, aspect_spans, opinion_spans, mask,
    temperature=0.1, return_stats=True,
)
assert good_loss < swapped_loss
assert stats["pairing_aspect_accuracy"] == 1.0
assert stats["pairing_opinion_accuracy"] == 1.0
```

共享方面词时，具有相同方面词跨度的两个观点词都必须进入正例集合；共享观点词时反向同理。全部候选均为正例、没有真实负例时，损失必须是有限的 0。

- [ ] **步骤 2：运行失败测试**

```text
J:\conda\envs\c3da\python.exe -c "import test_dynamic_sample_weight as m; m.test_encoder_pairing_loss_prefers_correct_pairs(); m.test_encoder_pairing_loss_supports_multiple_positives(); m.test_encoder_pairing_loss_is_zero_without_real_negatives()"
```

预期：因新函数不存在而失败。

- [ ] **步骤 3：实现多正例方向损失**

实现内部方向函数：

```python
direction_loss = logsumexp(all_candidate_logits) - logsumexp(positive_logits)
```

方面词到观点词的正例掩码由相同方面词跨度产生；观点词到方面词由相同观点词跨度产生。仅对同时含正例和负例的锚点计损失，两个方向取平均。

返回统计：

```python
{
    "pairing_aspect_accuracy": ...,
    "pairing_opinion_accuracy": ...,
    "pairing_active_rows": ...,
    "pairing_active_pairs": ...,
}
```

- [ ] **步骤 4：接入训练器**

新增：

```text
--pairing_temperature 0.1
--pairing_source_only
```

训练器直接使用 `outputs.encoder_last_hidden_state`，不再因配对损失请求解码器全部隐藏状态。训练时记录 `pairing_loss` 和全部配对统计。

- [ ] **步骤 5：运行训练相关测试**

```text
J:\conda\envs\c3da\python.exe -c "import test_dynamic_sample_weight as a, test_domain_adversarial_train as b; tests=[getattr(m,n) for m in (a,b) for n in sorted(dir(m)) if n.startswith('test_')]; [t() for t in tests]; print(f'{len(tests)} tests passed')"
```

预期：配对和既有 DANN（领域对抗神经网络）、情感对比学习测试全部通过。

- [ ] **步骤 6：提交**

```text
git add t5_absa_train.py test_dynamic_sample_weight.py test_domain_adversarial_train.py && git commit -m "Add multi-positive encoder pairing loss"
```

### 任务 3：结构子集评估

**文件：**

- 修改：`test_evaluate_output_isolation.py`
- 修改：`t5_aste_pipeline.py`

- [ ] **步骤 1：添加失败测试**

构造一条单三元组和一条多三元组样本，断言评估结果新增：

```python
structure_scores = {
    "single_triplet_rows": {"rows": 1, "raw": ..., "fixed": ...},
    "multi_triplet_rows": {"rows": 1, "raw": ..., "fixed": ...},
}
```

并写入带 `output_tag`（输出标签）的独立 `aste_metrics_by_structure_*.json` 文件。

- [ ] **步骤 2：运行失败测试**

```text
J:\conda\envs\c3da\python.exe -m unittest test_evaluate_output_isolation.py -v
```

预期：结构指标文件不存在或结果缺少 `structure_scores`。

- [ ] **步骤 3：实现结构分组评分**

按金标三元组数量将行分为 1 个和至少 2 个两组，分别聚合 raw/fixed（原始/修正）预测并计算 micro F1（微平均 F1）。零样本组返回行数 0 和零指标。

- [ ] **步骤 4：运行评估回归测试**

```text
J:\conda\envs\c3da\python.exe -m unittest test_evaluate_output_isolation.py test_fixed_pseudo_analysis.py -v
```

预期：全部通过，默认和带标签输出互不覆盖。

- [ ] **步骤 5：提交**

```text
git add t5_aste_pipeline.py test_evaluate_output_isolation.py && git commit -m "Report single and multi-triplet evaluation"
```

### 任务 4：最终训练单变量路由与输出隔离

**文件：**

- 修改：`test_run_bgca_stage1_pairs.py`
- 修改：`run_bgca_aste_stage1_pairs.py`

- [ ] **步骤 1：添加失败测试**

在当前最佳目录运行 `dry_run`（仅打印不执行），传入：

```text
--lambda_pairing_loss 0.01 --pairing_temperature 0.1 --pairing_source_only
```

断言只打印最终训练和评估两个命令，不打印准备、提取器、伪标签、生成器或增强命令；训练命令使用现有：

```text
final_train_strict_aug150_w020_label_to_text_gen.jsonl
```

并包含独立标签：

```text
pairing_encoder_l001_source_only
```

- [ ] **步骤 2：运行失败测试**

```text
J:\conda\envs\c3da\python.exe -m unittest test_run_bgca_stage1_pairs.py -v
```

预期：命令解析器不认识配对参数或输出标签未隔离。

- [ ] **步骤 3：实现参数和标签路由**

批处理参数新增：

```text
--lambda_pairing_loss
--pairing_temperature
--pairing_source_only
```

最终训练命令传递参数；`result_tag` 和 `summary_tag` 在配对系数大于 0 时加入 `pairing_encoder_l001_source_only`。历史默认系数 0 时输出完全不变。

- [ ] **步骤 4：运行批处理回归测试**

```text
J:\conda\envs\c3da\python.exe -m unittest test_run_bgca_stage1_pairs.py -v
```

预期：全部通过。

- [ ] **步骤 5：提交**

```text
git add run_bgca_aste_stage1_pairs.py test_run_bgca_stage1_pairs.py && git commit -m "Route isolated encoder pairing ablation"
```

### 任务 5：整体验证与中文记录

**文件：**

- 修改：`实验记录与模型索引_CN.md`
- 修改：`CD_C3DA_BGCA超越目标路线图_CN.md`

- [ ] **步骤 1：统计真实有效配对覆盖率**

对当前最佳最终训练文件只读运行数据集特征构造，输出源域多三元组行数、成功定位行数、有效配对数和覆盖率。若行覆盖率低于 95%，停止训练并修复定位。

- [ ] **步骤 2：运行相关完整测试与语法检查**

```text
J:\conda\envs\c3da\python.exe -c "import test_dynamic_sample_weight as a, test_domain_adversarial_train as b; tests=[getattr(m,n) for m in (a,b) for n in sorted(dir(m)) if n.startswith('test_')]; [t() for t in tests]; print(f'{len(tests)} function tests passed')" && J:\conda\envs\c3da\python.exe -m unittest test_run_bgca_stage1_pairs.py test_evaluate_output_isolation.py test_fixed_pseudo_analysis.py -v && J:\conda\envs\c3da\python.exe -m py_compile t5_absa_train.py t5_aste_pipeline.py run_bgca_aste_stage1_pairs.py
```

预期：全部通过且语法检查无输出。

- [ ] **步骤 3：运行最终预演并自动断言**

从 CMD（命令提示符）调用批处理脚本，确认只输出最终训练和评估两个阶段，训练命令包含 8GB 显存参数、自动续训、配对参数、DANN（领域对抗神经网络）和情感对比学习参数。

- [ ] **步骤 4：整体更新中文文档**

记录设计、代码提交、旧训练文件、独立模型目录、完整命令、覆盖率、日志字段和 46.82 验收线，不在文档末尾重复追加。

- [ ] **步骤 5：提交并推送**

```text
git add 实验记录与模型索引_CN.md CD_C3DA_BGCA超越目标路线图_CN.md && git commit -m "Document encoder pairing ablation workflow" && git push
```

## 完成标准

- 编码器输入跨度覆盖率至少 95%。
- 多正例双向损失正确处理共享方面词和共享观点词。
- 配对关闭时历史行为不变。
- 配对损失、双向准确率和有效计数进入训练日志。
- 单三元组/多三元组结构指标独立保存。
- 最终预演只重跑当前最佳训练集的最终 5 轮模型和评估。
- 模型、预测、指标和汇总均使用独立标签，并支持相同命令断点续训。

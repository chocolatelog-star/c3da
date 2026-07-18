# 动态多三元组训练与增强实施计划

> **面向智能体执行者：** 必须使用 `superpowers:subagent-driven-development`（子智能体驱动开发，推荐）或 `superpowers:executing-plans`（执行计划）逐项实施。所有步骤使用复选框跟踪。

**目标：** 将固定1至3个三元组的筛选流程改为源域完整结构训练、目标域逐三元组动态筛选和完整集合掩码增强，并保持当前48.93最佳实验可复现。

**架构：** `t5_aste_data.py` 提供数量分组与评估基础函数；`t5_absa_train.py` 负责生成式验证指标和检查点选择；`t5_aste_pipeline.py` 负责源域结构权重、动态伪标签及严格增强过滤；`t5_aste_augment.py` 负责完整集合编辑样本；`run_bgca_aste_stage1_pairs.py` 负责独立实验路由、阶段状态与断点续跑。旧固定数量模式保留兼容，只有显式开启动态模式才执行新流程。

**技术栈：** Python 3.10、PyTorch（张量计算框架）、Transformers（预训练模型库）、T5、unittest（单元测试框架）、Git（版本管理）、RTX 3070 8GB。

---

## 文件职责

- 修改 `t5_aste_data.py`：数量分组、按数量统计微平均指标和检查点选择分数。
- 修改 `t5_absa_train.py`：生成式验证、结构分组日志和按抽取F1选择提取器检查点。
- 修改 `t5_aste_pipeline.py`：源域结构权重、动态伪标签筛选、权重诊断和多三元组严格回抽过滤。
- 修改 `t5_aste_augment.py`：互相掩码训练时保留整句完整标签集合。
- 修改 `run_bgca_aste_stage1_pairs.py`：动态模式参数、阶段命名、独立产物和自动续跑。
- 创建 `test_dynamic_multitriplet.py`：数量统计、权重和动态筛选测试。
- 修改 `test_dynamic_sample_weight.py`：源域结构权重进入数据集和损失的测试。
- 修改 `test_masked_mutual_augment.py`：完整集合方面词与观点词编辑测试。
- 修改 `test_model_filter.py`：多三元组完整覆盖过滤测试。
- 修改 `test_run_bgca_stage1_pairs.py`：命令路由、产物隔离和续跑测试。
- 整体更新 `实验记录与模型索引_CN.md` 和 `CD_C3DA_BGCA超越目标路线图_CN.md`：代码版本、实验命令、产物位置和待运行状态。

### 任务1：数量分组和生成式验证指标

**文件：**
- 修改：`t5_aste_data.py`
- 创建：`test_dynamic_multitriplet.py`

- [ ] **步骤1：编写数量分组失败测试**

在 `test_dynamic_multitriplet.py` 写入：

```python
import unittest

from t5_aste_data import micro_f1_by_triplet_count, triplet_count_bucket


class DynamicMultitripletTest(unittest.TestCase):
    def test_triplet_count_bucket_keeps_three_and_four_plus_separate(self):
        self.assertEqual(triplet_count_bucket(1), "count1")
        self.assertEqual(triplet_count_bucket(2), "count2")
        self.assertEqual(triplet_count_bucket(3), "count3")
        self.assertEqual(triplet_count_bucket(4), "count4plus")
        self.assertEqual(triplet_count_bucket(7), "count4plus")

    def test_micro_f1_by_triplet_count_groups_by_gold_structure(self):
        golds = [
            "<pos> a <opinion> good",
            "<pos> a <opinion> good ; <neg> b <opinion> bad ; <neu> c <opinion> average",
        ]
        predictions = [golds[0], "<pos> a <opinion> good ; <neg> b <opinion> bad"]
        result = micro_f1_by_triplet_count(predictions, golds)
        self.assertEqual(result["count1"]["micro_f1"], 1.0)
        self.assertEqual(result["count3"]["tp"], 2)
        self.assertEqual(result["count3"]["fn"], 1)
```

- [ ] **步骤2：运行测试并确认失败**

运行：`J:\conda\envs\c3da\python.exe -m unittest test_dynamic_multitriplet.py -v`

预期：因 `triplet_count_bucket` 尚未定义而失败。

- [ ] **步骤3：实现数量分组与指标**

在 `t5_aste_data.py` 增加：

```python
def triplet_count_bucket(count: int) -> str:
    if count <= 1:
        return "count1"
    if count == 2:
        return "count2"
    if count == 3:
        return "count3"
    return "count4plus"


def micro_f1_by_triplet_count(predictions: list[str], golds: list[str]) -> dict[str, dict[str, float]]:
    grouped = {"count1": ([], []), "count2": ([], []), "count3": ([], []), "count4plus": ([], [])}
    for prediction, gold in zip(predictions, golds):
        bucket = triplet_count_bucket(len(parse_triplet_text_list(gold)))
        grouped[bucket][0].append(prediction)
        grouped[bucket][1].append(gold)
    return {
        bucket: {"rows": len(bucket_golds), **micro_f1(bucket_predictions, bucket_golds)}
        for bucket, (bucket_predictions, bucket_golds) in grouped.items()
    }
```

- [ ] **步骤4：增加数量误差测试并实现**

测试要求返回 `exact_count_rows`、`under_generated_rows` 和 `over_generated_rows`。实现 `triplet_count_diagnostics(predictions, golds)`，逐行比较解析后的预测数量与金标数量，并返回总行数及三个计数。

- [ ] **步骤5：运行测试并提交**

运行：`J:\conda\envs\c3da\python.exe -m unittest test_dynamic_multitriplet.py -v`

预期：全部通过。

提交：`git add t5_aste_data.py test_dynamic_multitriplet.py`，然后执行 `git commit -m "Add multi-triplet evaluation metrics"`。

### 任务2：源域多三元组结构权重

**文件：**
- 修改：`t5_aste_pipeline.py`
- 修改：`run_bgca_aste_stage1_pairs.py`
- 修改：`test_dynamic_multitriplet.py`
- 修改：`test_run_bgca_stage1_pairs.py`

- [ ] **步骤1：编写结构权重失败测试**

```python
from t5_aste_pipeline import assign_source_triplet_count_weights


def test_source_triplet_count_weights_only_change_source_gold_rows():
    rows = [
        {"id": "s1", "label": "<pos> a <opinion> good", "augmentation": "source_gold"},
        {"id": "s3", "label": "<pos> a <opinion> good ; <neg> b <opinion> bad ; <neu> c <opinion> average", "augmentation": "source_gold"},
        {"id": "p3", "label": "<pos> a <opinion> good ; <neg> b <opinion> bad ; <neu> c <opinion> average", "augmentation": "target_pseudo", "sample_weight": 0.65},
    ]
    weighted, stats = assign_source_triplet_count_weights(rows, 1.0, 1.15, 1.25, 1.30)
    assert weighted[0]["sample_weight"] == 1.0
    assert weighted[1]["sample_weight"] == 1.25
    assert weighted[2]["sample_weight"] == 0.65
    assert stats["count3"]["rows"] == 1
```

- [ ] **步骤2：运行测试并确认失败**

运行：`J:\conda\envs\c3da\python.exe -m unittest test_dynamic_multitriplet.DynamicMultitripletTest.test_source_triplet_count_weights_only_change_source_gold_rows -v`

预期：导入失败。

- [ ] **步骤3：实现结构权重函数**

在 `t5_aste_pipeline.py` 实现 `assign_source_triplet_count_weights`。函数复制输入行，根据 `parse_triplet_text_list(label)` 的数量选择权重，只修改 `augmentation == "source_gold"` 或没有增强来源的源域行，并写入：

```python
{
    "sample_weight": selected_weight,
    "source_triplet_count": triplet_count,
    "source_triplet_count_bucket": bucket,
    "source_triplet_count_weight": selected_weight,
}
```

统计结果按 `count1`、`count2`、`count3`、`count4plus` 返回行数、平均权重、最小值和最大值。

- [ ] **步骤4：把权重应用到提取器训练文件**

在准备 `extract_train.jsonl` 前调用结构权重函数。新增命令参数：

```text
--source_count1_weight 1.0
--source_count2_weight 1.15
--source_count3_weight 1.25
--source_count4plus_weight 1.30
```

只有 `--dynamic_multitriplet` 开启时应用；旧实验输出保持逐字兼容。将统计写入 `extract_train_multitriplet_weight_analysis.json`。

- [ ] **步骤5：增加批处理命令测试**

在 `test_run_bgca_stage1_pairs.py` 断言动态命令包含四个权重参数，并且阶段名含 `dynamic_multitriplet`；未开启时不得出现这些参数。

- [ ] **步骤6：运行测试并提交**

运行：`J:\conda\envs\c3da\python.exe -m unittest test_dynamic_multitriplet.py test_run_bgca_stage1_pairs.py -v`

预期：全部通过。

提交：`git add t5_aste_pipeline.py run_bgca_aste_stage1_pairs.py test_dynamic_multitriplet.py test_run_bgca_stage1_pairs.py`，然后执行 `git commit -m "Weight source multi-triplet extraction rows"`。

### 任务3：按抽取F1选择提取器检查点

**文件：**
- 修改：`t5_absa_train.py`
- 修改：`test_dynamic_sample_weight.py`

- [ ] **步骤1：编写指标构造失败测试**

使用一个只实现 `batch_decode` 的假分词器，验证 `build_aste_compute_metrics` 返回：

```python
{
    "micro_f1": 1.0,
    "multi_micro_f1": 1.0,
    "count3_micro_f1": 1.0,
    "exact_count_accuracy": 1.0,
    "selection_score": 1.001,
}
```

选择分数固定定义为：

```python
selection_score = overall_micro_f1 + 0.001 * multi_micro_f1
```

0.001仅用于总体F1相同或极接近时打破平局，不允许多三元组指标掩盖总体退化。

- [ ] **步骤2：运行测试并确认失败**

运行：`J:\conda\envs\c3da\python.exe -c "import test_dynamic_sample_weight as m; m.test_aste_compute_metrics_reports_structure_groups()"`

预期：因 `build_aste_compute_metrics` 不存在而失败。

- [ ] **步骤3：实现生成式指标函数**

在 `t5_absa_train.py` 从 `t5_aste_data.py` 导入 `micro_f1`、`micro_f1_by_triplet_count` 和 `triplet_count_diagnostics`。实现 `build_aste_compute_metrics(tokenizer)`：

1. 将标签中的 `-100` 替换为分词器填充编号。
2. 批量解码预测和标签。
3. 计算总体及数量分组指标。
4. 合并2、3、4个及以上组计算 `multi_micro_f1`。
5. 返回浮点指标，供训练器记录和选择检查点。

- [ ] **步骤4：增加新的检查点选择模式**

把 `--checkpoint_selection` 选项扩展为 `last`、`best`、`aste_f1`。当值为 `aste_f1` 时设置：

```python
predict_with_generate=True
generation_num_beams=1
generation_max_length=128
load_best_model_at_end=True
metric_for_best_model="eval_selection_score"
greater_is_better=True
compute_metrics=build_aste_compute_metrics(tokenizer)
```

`best` 仍按最低 `eval_loss` 选择，保证生成器和历史最终模型不受影响。

- [ ] **步骤5：验证3070显存参数没有变化**

测试断言训练批量大小仍由现有命令控制、生成式验证使用贪心解码、没有增加训练批量或关闭梯度检查点。

- [ ] **步骤6：运行测试并提交**

运行：`J:\conda\envs\c3da\python.exe -c "import test_dynamic_sample_weight as m; m.test_aste_compute_metrics_reports_structure_groups()"`

预期：全部通过。

提交：`git add t5_absa_train.py test_dynamic_sample_weight.py`，然后执行 `git commit -m "Select extractor checkpoints by ASTE F1"`。

### 任务4：动态逐三元组伪标签筛选

**文件：**
- 修改：`t5_aste_pipeline.py`
- 修改：`test_dynamic_multitriplet.py`

- [ ] **步骤1：编写无数量上限测试**

构造一条含4个合法三元组的伪标签，调用 `select_dynamic_high_precision_pseudo_rows`，断言该行保留4个三元组且没有 `too_many_triplets`。

- [ ] **步骤2：编写部分删除测试**

构造一条含3个三元组的伪标签，其中两个距离合法、一个距离超限。断言输出行保留两个合法三元组，并写入：

```python
{
    "dynamic_triplet_count_before": 3,
    "dynamic_triplet_count_after": 2,
    "dynamic_removed_triplets": 1,
    "dynamic_retention_ratio": 2 / 3,
}
```

- [ ] **步骤3：运行测试并确认失败**

运行：`J:\conda\envs\c3da\python.exe -m unittest test_dynamic_multitriplet.py -v`

预期：动态筛选函数不存在。

- [ ] **步骤4：实现动态筛选**

新增 `select_dynamic_high_precision_pseudo_rows(rows, min_weight=0.65, max_token_distance=5)`：

1. 执行空标签、低基础权重、词项修正和整句词项覆盖检查，但不传入 `max_triplets`。
2. 对每个三元组调用现有 `high_precision_triplet_reject_reason`。
3. 至少保留一个三元组才输出该行。
4. 最终权重定义为：

```python
base_weight = min(0.65, float(row.get("sample_weight", 0.65)))
retention_ratio = kept_count / original_count
change_factor = 1.0 if kept_count == original_count else 0.85
sample_weight = round(max(0.25, base_weight * retention_ratio * change_factor), 6)
```

5. 统计输入、输出、部分删除、完全删除、各删除原因、各数量分组和权重摘要。

- [ ] **步骤5：明确动态模式不复用双三元组专用函数**

动态模式不得调用 `build_complete_multitriplet_pseudo_rows`。旧函数保留用于复现48.93实验；新产物写入：

```text
pseudo_variants\dynamic_dist5\target_pseudo_high_precision.jsonl
pseudo_variants\dynamic_dist5\target_pseudo_high_precision_analysis.json
```

- [ ] **步骤6：运行测试并提交**

运行：`J:\conda\envs\c3da\python.exe -m unittest test_dynamic_multitriplet.py test_complete_multitriplet_pseudo.py -v`

运行：`J:\conda\envs\c3da\python.exe -c "import test_fixed_pseudo_analysis as m; m.test_select_high_confidence_pseudo_rows_keeps_clean_raw_fixed_agreement(); m.test_select_high_precision_pseudo_rows_filters_triplets_by_distance()"`

预期：新旧筛选测试全部通过。

提交：`git add t5_aste_pipeline.py test_dynamic_multitriplet.py`，然后执行 `git commit -m "Add dynamic pseudo-triplet filtering"`。

### 任务5：完整集合互相掩码训练

**文件：**
- 修改：`t5_aste_augment.py`
- 修改：`test_masked_mutual_augment.py`

- [ ] **步骤1：编写方面词通道失败测试**

输入包含三个三元组的源句，生成方面词编辑训练样本。断言每条训练标签仍有三个三元组，只有被选中三元组的方面词变化，其余两个三元组保持不变。

- [ ] **步骤2：编写观点词-情感通道失败测试**

对同一源句生成观点词-情感编辑样本，断言完整标签有三个三元组，只有被选中索引的观点词和情感发生配对变化。

- [ ] **步骤3：运行测试并确认失败**

运行：`J:\conda\envs\c3da\python.exe -c "import test_masked_mutual_augment as m; m.test_masked_aspect_training_preserves_full_multitriplet_label(); m.test_masked_opinion_training_preserves_full_multitriplet_label()"`

预期：当前训练标签只有一个三元组，断言失败。

- [ ] **步骤4：修改训练样本构建逻辑**

在 `build_generator_training_rows` 的互相掩码分支中，先保存：

```python
all_triplets = list(parse_triplet_text_list(label))
```

遍历时使用索引，复制完整集合并替换选中项：

```python
new_triplets = list(all_triplets)
new_triplets[index] = new_triplet
new_label = canonicalize_triplet_text(triplets_to_text(new_triplets))
```

训练行写入 `label=new_label`、`new_triplets`、`edited_triplet_index`、`preserved_triplet_count`。生成目标文本仍只替换选中词项。

- [ ] **步骤5：保证单三元组行为兼容**

现有单三元组测试必须保持原输入、目标文本和标签不变。混合生成器每个源句每通道最多一行的选择规则不变。

- [ ] **步骤6：运行测试并提交**

运行：`J:\conda\envs\c3da\python.exe -c "import test_masked_mutual_augment as a; a.test_masked_aspect_training_preserves_full_multitriplet_label(); a.test_masked_opinion_training_preserves_full_multitriplet_label(); a.test_masked_mutual_generator_training_uses_source_masked_prompts()"`

运行：`J:\conda\envs\c3da\python.exe -m unittest test_generator_label_to_text.py -v`

预期：全部通过。

提交：`git add t5_aste_augment.py test_masked_mutual_augment.py`，然后执行 `git commit -m "Preserve full labels in masked editing"`。

### 任务6：多三元组增强完整覆盖过滤

**文件：**
- 修改：`t5_aste_pipeline.py`
- 修改：`test_model_filter.py`

- [ ] **步骤1：编写完整覆盖失败测试**

构造请求标签含3个三元组、回抽结果只命中2个的增强行。即使通道感知过滤能命中被编辑三元组，也必须返回：

```python
{
    "model_filter_passed": False,
    "model_filter_reason": "multitriplet_partial_coverage",
    "model_filter_requested_triplets": 3,
    "model_filter_matched_triplets": 2,
}
```

- [ ] **步骤2：编写额外生成失败测试**

请求3个、回抽4个时返回 `multitriplet_extra_generation`；请求集合与回抽集合完全一致时返回 `multitriplet_exact`。

- [ ] **步骤3：运行测试并确认失败**

运行：`J:\conda\envs\c3da\python.exe -c "import test_model_filter as m; m.test_multitriplet_filter_rejects_partial_coverage(); m.test_multitriplet_filter_rejects_extra_generation(); m.test_multitriplet_filter_keeps_exact_set()"`

预期：部分覆盖当前可能由通道感知分支接受，测试失败。

- [ ] **步骤4：实现多三元组严格覆盖优先规则**

在通道特定放宽逻辑之前判断请求标签数量。请求数量至少2时，比较规范化后的完整集合：

```python
requested = _triplet_set_for_filter(row["label"])
predicted = _triplet_set_for_filter(pred_fixed if mode == "fixed" else pred_raw)
```

只有 `requested == predicted` 才保留。记录完整覆盖、部分覆盖、额外生成和完全失败计数。单三元组继续使用现有通道感知规则。

- [ ] **步骤5：把统计写入增强分析**

`c3da_model_filter_analysis_<标签>.json` 增加：

```text
multitriplet_input_rows
multitriplet_exact_rows
multitriplet_partial_rows
multitriplet_extra_rows
multitriplet_failed_rows
```

- [ ] **步骤6：运行测试并提交**

运行：`J:\conda\envs\c3da\python.exe -c "import test_model_filter as m, test_augment_quality_filters as q; m.test_multitriplet_filter_rejects_partial_coverage(); m.test_multitriplet_filter_rejects_extra_generation(); m.test_multitriplet_filter_keeps_exact_set(); q.test_prompt_leak_filter_rejects_generated_template_fragments(); q.test_prompt_leak_filter_keeps_natural_sentence()"`

预期：全部通过。

提交：`git add t5_aste_pipeline.py test_model_filter.py`，然后执行 `git commit -m "Require complete multi-triplet augmentation coverage"`。

### 任务7：批处理路由、断点续跑和实验隔离

**文件：**
- 修改：`run_bgca_aste_stage1_pairs.py`
- 修改：`test_run_bgca_stage1_pairs.py`

- [ ] **步骤1：编写完整干运行失败测试**

动态模式命令必须包含：

```text
--dynamic_multitriplet
--checkpoint_selection aste_f1
--source_count2_weight 1.15
--source_count3_weight 1.25
--source_count4plus_weight 1.30
--pseudo_train_file ...\pseudo_variants\dynamic_dist5\target_pseudo_high_precision.jsonl
```

并且不得包含 `--complete_multi_extra_weight` 或 `complete_multi2`。

- [ ] **步骤2：实现独立阶段和标签**

新增批处理参数 `--dynamic_multitriplet`。动态模式使用独立阶段：

```text
train_extractor_dynamic_multitriplet
select_pseudo_dynamic_dist5
train_generator_masked_fullset
augment_masked_fullset
train_final_dynamic_multitriplet
evaluate_dynamic_multitriplet
```

输出根目录建议为：

```text
runs\bgca_aste_stage1_dynamic_multitriplet_v1
```

- [ ] **步骤3：实现阶段指纹防误复用**

阶段状态写入四个结构权重、动态距离阈值、完整集合增强开关、输入文件大小与修改时间。参数或输入指纹变化时不得跳过阶段。存在最新有效检查点时继续传递 `--resume_from_checkpoint auto`。

- [ ] **步骤4：保留3070参数**

干运行断言提取器和生成器继续使用训练批量1、验证批量2、梯度累积16、半精度和梯度检查点；最终模型沿用当前最佳的8GB显存参数。

- [ ] **步骤5：运行测试并提交**

运行：`J:\conda\envs\c3da\python.exe -m unittest test_run_bgca_stage1_pairs.py test_evaluate_output_isolation.py -v`

预期：全部通过。

提交：`git add run_bgca_aste_stage1_pairs.py test_run_bgca_stage1_pairs.py`，然后执行 `git commit -m "Route dynamic multi-triplet experiments"`。

### 任务8：动态结构组合的独立D阶段

**文件：**
- 修改：`t5_aste_augment.py`
- 修改：`t5_aste_pipeline.py`
- 修改：`run_bgca_aste_stage1_pairs.py`
- 修改：`test_masked_mutual_augment.py`
- 修改：`test_run_bgca_stage1_pairs.py`

- [ ] **步骤1：编写按源域分布采样测试**

给定源域数量分布 `{1: 5, 2: 3, 3: 1, 4: 1}` 和固定随机种子，`sample_composition_triplet_count` 只能从实际出现的数量中采样，并按经验频率产生确定性序列。函数不得使用固定2或3上限。

- [ ] **步骤2：实现经验分布采样**

新增：

```python
def sample_composition_triplet_count(rng: random.Random, source_rows: list[dict]) -> int:
    counts = [len(parse_triplet_text_list(row.get("label", ""))) for row in source_rows]
    counts = [count for count in counts if count >= 2]
    if not counts:
        raise ValueError("multi-triplet source rows are required for composition")
    return rng.choice(counts)
```

实际组合受可用兼容候选数限制；候选不足时放弃该请求，不回退到伪造固定数量。

- [ ] **步骤3：构建动态标签组合请求**

在现有标签组合兼容检查基础上循环选择互不重复方面词的三元组，直到达到本次采样数量。请求记录 `requested_triplet_count` 和完整 `new_triplets`。

- [ ] **步骤4：隔离D阶段参数与产物**

新增 `--dynamic_composition_max_rows`，默认0表示关闭。D阶段首次实验使用50条上限，输出标签包含 `dynamic_composition50`，不得覆盖C阶段。

- [ ] **步骤5：运行测试并提交**

运行：`J:\conda\envs\c3da\python.exe -c "import test_masked_mutual_augment as m; m.test_dynamic_composition_samples_empirical_source_counts(); m.test_dynamic_composition_builds_requested_full_label_set()"`

预期：全部通过。

提交：`git add t5_aste_augment.py t5_aste_pipeline.py run_bgca_aste_stage1_pairs.py test_masked_mutual_augment.py test_run_bgca_stage1_pairs.py`，然后执行 `git commit -m "Add dynamic multi-triplet composition stage"`。

### 任务9：全量验证、文档和实验命令

**文件：**
- 修改：`实验记录与模型索引_CN.md`
- 修改：`CD_C3DA_BGCA超越目标路线图_CN.md`

- [ ] **步骤1：运行相关测试集**

运行：`J:\conda\envs\c3da\python.exe -m unittest discover -v`

预期：所有 `unittest（单元测试框架）`测试通过。

运行：`J:\conda\envs\c3da\python.exe -c "import importlib,inspect; modules=[importlib.import_module(name) for name in ['test_dynamic_sample_weight','test_masked_mutual_augment','test_model_filter','test_augment_quality_filters','test_fixed_pseudo_analysis']]; tests=[fn for module in modules for name,fn in vars(module).items() if name.startswith('test_') and callable(fn) and not inspect.signature(fn).parameters]; [fn() for fn in tests]; print('zero-argument function tests passed:', len(tests))"`

预期：所有普通函数测试执行完成并输出实际数量；任一断言失败都会返回非零退出码。

- [ ] **步骤2：运行语法检查**

运行：`J:\conda\envs\c3da\python.exe -m py_compile t5_aste_data.py t5_absa_train.py t5_aste_pipeline.py t5_aste_augment.py run_bgca_aste_stage1_pairs.py test_dynamic_multitriplet.py`

预期：退出码0且无输出。

- [ ] **步骤3：运行B阶段干运行**

运行：`J:\conda\envs\c3da\python.exe run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_dynamic_multitriplet_v1 --pairs rest16:laptop14 --extractor_model_path J:\nlp\models\t5-base-py --generator_model_path J:\nlp\models\t5-base-py --generator_prompt_style label_to_text --augment_prompt_style masked_mutual --domain_prefix_style text --dynamic_multitriplet --source_count1_weight 1.0 --source_count2_weight 1.15 --source_count3_weight 1.25 --source_count4plus_weight 1.30 --high_precision_max_token_distance 5 --extractor_epochs 25 --generator_epochs 8 --final_epochs 5 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --learning_rate 0.0003 --eval_batch_size 2 --cuda 0 --seed 1000 --dry_run`

预期：显示完整B、C阶段命令、独立路径、进度与自动续跑参数，不执行训练。

- [ ] **步骤4：整体更新中文文档**

在文档开头保留当前48.93最佳结果，新增“动态多三元组待验证”表格，记录：设计提交、实现提交、分支、完整命令、输出根目录、模型命名、复用边界和验收指标。修改既有章节，不在文件末尾无限追加重复内容。

- [ ] **步骤5：检查差异并提交**

运行：`git diff --check`

预期：退出码0且无输出。

提交：`git add 实验记录与模型索引_CN.md CD_C3DA_BGCA超越目标路线图_CN.md`，然后执行 `git commit -m "Document dynamic multi-triplet experiments"`。

- [ ] **步骤6：最终状态检查与推送**

运行：`git status --short --branch`

预期：工作区干净。

运行：`git push origin feature/complete-multitriplet-ablation`

预期：远端功能分支更新成功。

## 实施顺序约束

1. 先完成任务1至7和任务9，得到B、C阶段可运行代码。
2. 不等待真实训练即可完成单元测试、语法检查、干运行和文档提交。
3. 先运行B；B的动态伪标签无标签质量代理没有明显退化后再运行C。
4. 任务8的D阶段代码可以最后实现；只有C不低于B且多三元组指标改善时，才执行真实D实验。
5. 任何删除旧模型、检查点或失败实验目录的操作都必须另行取得用户许可。

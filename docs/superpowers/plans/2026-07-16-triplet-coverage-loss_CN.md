# 三元组覆盖辅助学习实施计划

> **For agentic workers（面向代理执行者）:** REQUIRED SUB-SKILL（必需子流程）: Use superpowers:subagent-driven-development（子代理驱动开发，推荐） or superpowers:executing-plans（执行计划） to implement this plan task-by-task（逐任务实施本计划）. Steps use checkbox（复选框） `- [ ]` syntax for tracking（用于跟踪）。

**Goal（目标）:** 在当前 raw F1（原始综合指标）46.82%的最佳最终训练流程上增加仅由源域人工标注监督的1/2/3+三元组覆盖辅助目标，提高多三元组召回率，同时保持旧流程兼容和断点续训能力。

**Architecture（架构）:** 数据集从规范目标文本派生覆盖类别，T5（文本到文本转换器）编码器隐藏状态经过注意力掩码均值池化后送入三分类覆盖头。训练器以0.01权重加入类别平衡交叉熵，并按有效源域样本累计日志；生成评估独立计算预测数量混淆矩阵，不依赖覆盖头加载。

**Tech Stack（技术栈）:** Python 3.8、PyTorch（张量计算框架）、Transformers（预训练模型库）、T5-base（基础版文本到文本转换器）、unittest（单元测试框架）、CUDA（显卡计算平台）、Git（版本管理）。

---

## 文件边界

- 修改 `t5_absa_train.py`：覆盖标签、类别统计、覆盖头、损失、有效样本日志和命令行参数。
- 修改 `t5_aste_pipeline.py`：数量覆盖评估及独立 JSON（结构化数据）输出。
- 修改 `run_bgca_aste_stage1_pairs.py`：单变量参数路由、模型名、阶段完成条件和汇总标签。
- 修改 `test_domain_adversarial_train.py`：数据标签、类别权重、覆盖头、损失和日志测试。
- 修改 `test_evaluate_output_isolation.py`：数量混淆矩阵和隔离输出测试。
- 修改 `test_run_bgca_stage1_pairs.py`：最终5轮复用和独立命名预演测试。
- 修改 `实验记录与模型索引_CN.md`、`CD_C3DA_BGCA超越目标路线图_CN.md`：代码版本、实验目的和完整命令。

## Task 1：覆盖标签与类别权重

**Files（文件）:**
- Modify（修改）: `t5_absa_train.py:45-230`
- Test（测试）: `test_domain_adversarial_train.py`

- [ ] **Step 1：先写覆盖标签失败测试**

在 `test_domain_adversarial_train.py` 增加：

```python
def test_dataset_assigns_source_only_triplet_coverage_labels():
    rows = [
        {"input": "one", "target": "<pos> a <opinion> x"},
        {"input": "two", "target": "<pos> a <opinion> x ; <neg> b <opinion> y"},
        {"input": "three", "target": "<pos> a <opinion> x ; <neg> b <opinion> y ; <neu> c <opinion> z"},
        {"input": "pseudo", "target": "<pos> d <opinion> q", "augmentation": "target_pseudo"},
        {"input": "augment", "target": "<pos> e <opinion> r", "augmentation": "masked_aspect_channel"},
    ]
    dataset = JsonlSeq2SeqDataset(
        rows, TinyTokenizer(), 128, 96, 1.0, 0.5, 0.2,
        triplet_coverage_source_only=True,
    )

    assert [dataset[i]["triplet_coverage_label"] for i in range(5)] == [0, 1, 2, -100, -100]
```

- [ ] **Step 2：运行测试并确认失败**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m unittest test_domain_adversarial_train.py -v
```

Expected（预期）: FAIL（失败），提示 `triplet_coverage_source_only` 或 `triplet_coverage_label` 尚不存在。

- [ ] **Step 3：实现最小覆盖标签派生**

给 `JsonlSeq2SeqDataset.__init__` 增加 `triplet_coverage_source_only: bool = False`，保存为实例字段；在 `__getitem__` 增加：

```python
model_inputs["triplet_coverage_label"] = self.triplet_coverage_label(row)
```

新增方法：

```python
def triplet_coverage_label(self, row: dict) -> int:
    augmentation = row.get("augmentation")
    if self.triplet_coverage_source_only and (
        augmentation == "target_pseudo" or augmentation in CSA_AUGMENT_CHANNELS
    ):
        return -100
    triplet_count = len(parse_triplet_text_list(row.get("target", "")))
    if triplet_count <= 0:
        return -100
    return min(triplet_count, 3) - 1
```

- [ ] **Step 4：写类别统计与权重失败测试**

```python
def test_triplet_coverage_class_weights_upweight_rare_multi_triplet_classes():
    summary = summarize_triplet_coverage_rows([
        *[{"target": "<pos> a <opinion> x"} for _ in range(9)],
        *[{"target": "<pos> a <opinion> x ; <neg> b <opinion> y"} for _ in range(4)],
        {"target": "<pos> a <opinion> x ; <neg> b <opinion> y ; <neu> c <opinion> z"},
        {"target": "<pos> p <opinion> q", "augmentation": "target_pseudo"},
    ], source_only=True)
    weights = build_triplet_coverage_class_weights(summary)

    assert summary["count1"] == 9
    assert summary["count2"] == 4
    assert summary["count3plus"] == 1
    assert weights[2] > weights[1] > weights[0]
    assert abs(sum(weights) / 3.0 - 1.0) < 1e-6
```

- [ ] **Step 5：实现自动类别统计和平方根倒频率权重**

新增纯函数：

```python
def summarize_triplet_coverage_rows(rows: list[dict], source_only: bool) -> dict[str, int]:
    counts = [0, 0, 0]
    for row in rows:
        augmentation = row.get("augmentation")
        if source_only and (augmentation == "target_pseudo" or augmentation in CSA_AUGMENT_CHANNELS):
            continue
        triplet_count = len(parse_triplet_text_list(row.get("target", "")))
        if triplet_count > 0:
            counts[min(triplet_count, 3) - 1] += 1
    return {"count1": counts[0], "count2": counts[1], "count3plus": counts[2], "total": sum(counts)}


def build_triplet_coverage_class_weights(summary: dict[str, int]) -> list[float]:
    counts = [summary["count1"], summary["count2"], summary["count3plus"]]
    if any(count <= 0 for count in counts):
        raise ValueError("triplet coverage class balancing requires all three classes")
    inverse_sqrt = [1.0 / math.sqrt(count) for count in counts]
    mean_weight = sum(inverse_sqrt) / len(inverse_sqrt)
    return [weight / mean_weight for weight in inverse_sqrt]
```

- [ ] **Step 6：运行测试确认通过并提交**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m unittest test_domain_adversarial_train.py -v
```

Expected（预期）: PASS（通过）。

Commit（提交）:

```text
git add t5_absa_train.py test_domain_adversarial_train.py && git commit -m "Add source triplet coverage labels"
```

## Task 2：覆盖分类头、损失与有效样本日志

**Files（文件）:**
- Modify（修改）: `t5_absa_train.py:360-900,1020-1250`
- Test（测试）: `test_domain_adversarial_train.py`

- [ ] **Step 1：写覆盖头和损失失败测试**

```python
def test_triplet_coverage_head_and_loss_use_only_active_labels():
    head = TripletCoverageHead(hidden_size=4)
    pooled = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    logits = head(pooled)
    labels = torch.tensor([1, -100])
    loss, stats = triplet_coverage_loss(
        logits, labels, class_weights=torch.tensor([0.7, 1.0, 1.3]), return_stats=True
    )

    assert list(logits.shape) == [2, 3]
    assert torch.isfinite(loss)
    assert stats["active_rows"] == 1
    assert stats["class_total"][1] == 1
    loss.backward()
```

- [ ] **Step 2：运行并确认失败**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m unittest test_domain_adversarial_train.py -v
```

Expected（预期）: FAIL（失败），缺少覆盖头和损失函数。

- [ ] **Step 3：实现覆盖头与损失函数**

```python
class TripletCoverageHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(pooled_hidden))


def triplet_coverage_loss(logits, labels, class_weights=None, return_stats=False):
    active_mask = labels.ne(-100)
    if not active_mask.any():
        zero = logits.sum() * 0.0
        stats = {"correct": 0, "active_rows": 0, "class_correct": [0, 0, 0], "class_total": [0, 0, 0]}
        return (zero, stats) if return_stats else zero
    active_logits = logits[active_mask]
    active_labels = labels[active_mask]
    loss = F.cross_entropy(active_logits, active_labels, weight=class_weights)
    predictions = active_logits.argmax(dim=-1)
    class_correct = [int(((predictions == idx) & (active_labels == idx)).sum().item()) for idx in range(3)]
    class_total = [int((active_labels == idx).sum().item()) for idx in range(3)]
    stats = {
        "correct": int((predictions == active_labels).sum().item()),
        "active_rows": int(active_labels.numel()),
        "class_correct": class_correct,
        "class_total": class_total,
    }
    return (loss, stats) if return_stats else loss
```

- [ ] **Step 4：写训练器集成与有效样本聚合失败测试**

扩展现有 `FakeModel`（伪模型），挂载 `triplet_coverage_head`，输入 `triplet_coverage_label=torch.tensor([1])`，断言：

```python
assert "triplet_coverage_loss" in trainer._component_sums
assert trainer._coverage_correct >= 0
assert trainer._coverage_total == 1
```

再构造一个 `-100` 微批次，断言 `_coverage_total` 不增加，保证无效微批次不稀释准确率。

- [ ] **Step 5：在训练器中加入覆盖损失和计数器**

给 `WeightedSeq2SeqTrainer` 增加：

```python
lambda_triplet_coverage: float = 0.0
triplet_coverage_class_weights: list[float] | None = None
```

`compute_loss` 弹出 `triplet_coverage_label`，复用 `outputs.encoder_last_hidden_state`：

```python
if self.lambda_triplet_coverage > 0 and coverage_labels is not None and hasattr(model, "triplet_coverage_head"):
    pooled = mean_pool_encoder_hidden(outputs.encoder_last_hidden_state, attention_mask)
    coverage_logits = model.triplet_coverage_head(pooled)
    coverage_loss, coverage_stats = triplet_coverage_loss(
        coverage_logits,
        coverage_labels,
        class_weights=(
            torch.tensor(self.triplet_coverage_class_weights, device=coverage_logits.device, dtype=coverage_logits.dtype)
            if self.triplet_coverage_class_weights else None
        ),
        return_stats=True,
    )
    loss = loss + self.lambda_triplet_coverage * coverage_loss
```

日志内部保存 `correct/total` 和各类别 `correct/total` 的整数和；`log` 时计算准确率后清零。`triplet_coverage_loss` 仍使用现有组件均值记录。

- [ ] **Step 6：加入命令行、模型挂载和检查点兼容**

新增参数：

```python
parser.add_argument("--lambda_triplet_coverage", type=float, default=0.0)
parser.add_argument("--triplet_coverage_source_only", action="store_true")
parser.add_argument("--triplet_coverage_class_balanced", action="store_true")
```

当权重大于0时，在 `trainer.train`（训练器训练）之前挂载：

```python
model.triplet_coverage_head = TripletCoverageHead(hidden_size=hidden_size)
```

类别统计和权重从 `train_rows` 自动构建并打印。训练器恢复检查点时模型已经挂载覆盖头，因此 `--resume_from_checkpoint auto`（自动从检查点恢复）可恢复覆盖头参数；增加一次保存后加载 state dict（状态字典）的测试确认 `triplet_coverage_head.classifier.weight` 存在。

- [ ] **Step 7：运行训练模块测试并提交**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m unittest test_domain_adversarial_train.py -v
```

Expected（预期）: PASS（通过），覆盖日志只按有效源域行统计。

Commit（提交）:

```text
git add t5_absa_train.py test_domain_adversarial_train.py && git commit -m "Train encoder triplet coverage head"
```

## Task 3：预测数量覆盖评估

**Files（文件）:**
- Modify（修改）: `t5_aste_pipeline.py:2580-2685`
- Test（测试）: `test_evaluate_output_isolation.py`

- [ ] **Step 1：写数量评估失败测试**

构造三行数据：金标数量分别为1、2、3，预测数量分别为1、1、3。评估后断言：

```python
path = run_dir / "aste_metrics_by_triplet_count_coverage.json"
metrics = json.loads(path.read_text(encoding="utf-8"))
assert metrics["raw"]["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
assert metrics["raw"]["under_generated_rows"] == 1
assert metrics["raw"]["exact_count_rows"] == 2
assert metrics["raw"]["over_generated_rows"] == 0
assert metrics["raw"]["exact_count_accuracy"] == 2 / 3
```

- [ ] **Step 2：运行并确认失败**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m unittest test_evaluate_output_isolation.py -v
```

Expected（预期）: FAIL（失败），覆盖指标文件不存在。

- [ ] **Step 3：实现纯函数数量分析**

新增：

```python
def triplet_count_class(triplets: list[tuple[str, str, str]]) -> int:
    return min(max(len(triplets), 1), 3) - 1


def build_triplet_count_metrics(prediction_rows: list[dict], prediction_key: str) -> dict:
    matrix = [[0, 0, 0] for _ in range(3)]
    under = exact = over = 0
    for row in prediction_rows:
        gold_count = len(row["gold_triplets"])
        pred_count = len(row[prediction_key])
        matrix[triplet_count_class(row["gold_triplets"])][triplet_count_class(row[prediction_key])] += 1
        under += int(pred_count < gold_count)
        exact += int(pred_count == gold_count)
        over += int(pred_count > gold_count)
    return {
        "rows": len(prediction_rows),
        "confusion_matrix": matrix,
        "under_generated_rows": under,
        "exact_count_rows": exact,
        "over_generated_rows": over,
        "exact_count_accuracy": exact / max(1, len(prediction_rows)),
    }
```

为1、2、3+金标子集复用 `micro_f1`（微平均综合指标），输出 raw/fixed（原始/修正）指标。

- [ ] **Step 4：写入隔离文件并打印摘要**

`evaluate` 同时生成：

```python
triplet_count_metrics = {
    "raw": build_triplet_count_metrics(result["predictions"], "raw_triplets"),
    "fixed": build_triplet_count_metrics(result["predictions"], "fixed_triplets"),
    "gold_count_groups": count_group_metrics,
}
dump_json(tagged_output_path(run_dir, "aste_metrics_by_triplet_count.json", output_tag), triplet_count_metrics)
```

- [ ] **Step 5：运行评估测试并提交**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m unittest test_evaluate_output_isolation.py -v
```

Expected（预期）: PASS（通过），默认无标签文件不被覆盖。

Commit（提交）:

```text
git add t5_aste_pipeline.py test_evaluate_output_isolation.py && git commit -m "Report triplet count coverage metrics"
```

## Task 4：批处理脚本单变量路由与续跑

**Files（文件）:**
- Modify（修改）: `run_bgca_aste_stage1_pairs.py:480-620,850-940`
- Test（测试）: `test_run_bgca_stage1_pairs.py`

- [ ] **Step 1：写严格上游复用失败测试**

仿照配对消融测试准备已完成上游文件，执行：

```python
"--lambda_triplet_coverage", "0.01",
"--triplet_coverage_source_only",
"--triplet_coverage_class_balanced",
```

断言只有训练和评估两条命令，并包含：

```python
self.assertIn("--lambda_triplet_coverage 0.01", output)
self.assertIn("--triplet_coverage_source_only", output)
self.assertIn("--triplet_coverage_class_balanced", output)
self.assertIn("coverage_encoder_l001_source_balanced", output)
self.assertNotIn("--lambda_pairing_loss 0.01", output)
self.assertIn("aste_metrics_by_triplet_count", output)
```

- [ ] **Step 2：运行并确认失败**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m unittest test_run_bgca_stage1_pairs.py -v
```

Expected（预期）: FAIL（失败），参数尚未注册。

- [ ] **Step 3：实现参数、标签和训练命令路由**

新增参数并构造：

```python
if args.lambda_triplet_coverage > 0:
    coverage_lambda_tag = str(args.lambda_triplet_coverage).replace(".", "")
    result_tag += f"_coverage_encoder_l{coverage_lambda_tag}"
    if args.triplet_coverage_source_only:
        result_tag += "_source"
    if args.triplet_coverage_class_balanced:
        result_tag += "_balanced"
```

将三个参数传给 `t5_absa_train.py`。legacy（旧版）结果复用条件增加 `args.lambda_triplet_coverage == 0`，覆盖实验的阶段完成条件要求存在 `aste_metrics_by_triplet_count_<tag>.json`。

- [ ] **Step 4：验证相同命令可中断续跑**

测试预先创建覆盖模型 `config.json` 和覆盖指标文件时阶段跳过；只有检查点存在但最佳模型不存在时，训练命令仍出现并保留 `--resume_from_checkpoint auto`（自动从检查点恢复）。

- [ ] **Step 5：运行路由测试并提交**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m unittest test_run_bgca_stage1_pairs.py -v
```

Expected（预期）: PASS（通过），预演只重跑最终训练和评估。

Commit（提交）:

```text
git add run_bgca_aste_stage1_pairs.py test_run_bgca_stage1_pairs.py && git commit -m "Route triplet coverage ablation"
```

## Task 5：完整回归、CUDA最小验证与文档

**Files（文件）:**
- Modify（修改）: `实验记录与模型索引_CN.md`
- Modify（修改）: `CD_C3DA_BGCA超越目标路线图_CN.md`

- [ ] **Step 1：运行静态编译**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m py_compile t5_absa_train.py t5_aste_pipeline.py run_bgca_aste_stage1_pairs.py
```

Expected（预期）: exit code（退出码）0。

- [ ] **Step 2：运行相关测试**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m unittest test_domain_adversarial_train.py test_evaluate_output_isolation.py test_run_bgca_stage1_pairs.py -v
```

Expected（预期）: 全部 PASS（通过）。

- [ ] **Step 3：运行项目完整测试集**

Run（运行）:

```text
J:\conda\envs\c3da\python.exe -m unittest discover -p "test*.py" -v
```

Expected（预期）: 全部 PASS（通过），无旧流程回归。

- [ ] **Step 4：运行CUDA最小前向与反向验证**

在 `test_domain_adversarial_train.py` 增加真实本地模型测试。文件顶部补充 `os`、`unittest`、`Path`、`AutoModelForSeq2SeqLM` 和 `AutoTokenizer` 导入，然后加入：

```python
@unittest.skipUnless(
    os.environ.get("RUN_CUDA_SMOKE") == "1"
    and torch.cuda.is_available()
    and Path(r"J:\nlp\models\t5-base-py").exists(),
    "requires RUN_CUDA_SMOKE=1, CUDA, and the local t5-base model",
)
class TripletCoverageCudaSmokeTest(unittest.TestCase):
    def test_t5_encoder_coverage_head_backward_on_cuda(self):
        model_path = r"J:\nlp\models\t5-base-py"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda:0")
        hidden_size = int(model.config.d_model)
        model.triplet_coverage_head = TripletCoverageHead(hidden_size).to("cuda:0")
        encoded = tokenizer(
            ["The screen is bright.", "The screen is bright but the keyboard is stiff."],
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        labels = torch.tensor([0, 1], device="cuda:0")

        with torch.cuda.amp.autocast():
            hidden = model.get_encoder()(**encoded, return_dict=True).last_hidden_state
            pooled = mean_pool_encoder_hidden(hidden, encoded["attention_mask"])
            logits = model.triplet_coverage_head(pooled)
            loss = triplet_coverage_loss(logits, labels)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.triplet_coverage_head.classifier.weight.grad)
        self.assertTrue(torch.isfinite(model.triplet_coverage_head.classifier.weight.grad).all())
        self.assertLess(torch.cuda.max_memory_allocated(), 8 * 1024**3)
```

该测试默认跳过，只有显式设置环境变量时才加载大模型。命令必须从 `J:\nlp\CD-C3DA` 执行：

```text
cmd /c "set RUN_CUDA_SMOKE=1 && J:\conda\envs\c3da\python.exe -m unittest test_domain_adversarial_train.TripletCoverageCudaSmokeTest -v"
```

Expected（预期）: PASS（通过）；显存峰值低于8GB。

- [ ] **Step 5：执行完整流程预演**

Run（运行，整行）:

```text
J:\conda\envs\c3da\python.exe run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_domain_prompt_text_v1 --pairs rest16:laptop14 --extractor_model_path J:\nlp\models\t5-base-py --generator_model_path J:\nlp\models\t5-base-py --generator_prompt_style label_to_text --augment_prompt_style masked_mutual --domain_prefix_style text --final_epochs 5 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --lambda_triplet_coverage 0.01 --triplet_coverage_source_only --triplet_coverage_class_balanced --lambda_pairing_loss 0 --learning_rate 0.0003 --eval_batch_size 2 --cuda 0 --seed 1000 --dry_run
```

Expected（预期）: 只打印最终5轮训练与评估；模型和指标标签均为 `coverage_encoder_l001_source_balanced`；训练参数保持 batch size（批大小）1、梯度累积16、FP16（半精度）和梯度检查点。

- [ ] **Step 6：整体更新中文文档**

在两个中文文档顶部更新：设计提交、实现提交、作用数据、覆盖类别分布、模型输出路径、结果文件、完整 CMD（命令提示符）单行命令、续跑说明和验收标准。不要在文档末尾追加重复流水账。

- [ ] **Step 7：最终自查并提交**

Run（运行）:

```text
git diff --check
```

Expected（预期）: 无空白错误。

Commit（提交）:

```text
git add t5_absa_train.py t5_aste_pipeline.py run_bgca_aste_stage1_pairs.py test_domain_adversarial_train.py test_evaluate_output_isolation.py test_run_bgca_stage1_pairs.py "实验记录与模型索引_CN.md" "CD_C3DA_BGCA超越目标路线图_CN.md" && git commit -m "Document triplet coverage experiment workflow"
```

## Task 6：交付实验命令

- [ ] **Step 1：确认工作区和版本**

Run（运行）:

```text
git status --short && git log -1 --oneline
```

Expected（预期）: 工作区干净，显示最终实现提交。

- [ ] **Step 2：向用户提供从CMD开始的完整单行命令**

```text
cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_domain_prompt_text_v1 --pairs rest16:laptop14 --extractor_model_path J:\nlp\models\t5-base-py --generator_model_path J:\nlp\models\t5-base-py --generator_prompt_style label_to_text --augment_prompt_style masked_mutual --domain_prefix_style text --final_epochs 5 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --lambda_triplet_coverage 0.01 --triplet_coverage_source_only --triplet_coverage_class_balanced --lambda_pairing_loss 0 --learning_rate 0.0003 --eval_batch_size 2 --cuda 0 --seed 1000"
```

说明同一命令中断后可再次执行：已完成上游阶段会跳过，最终训练从最新检查点继续，评估只有在覆盖专属指标齐全后才判定完成。

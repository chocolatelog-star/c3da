# 目标域方面候选发现 v1 实施计划

> **For agentic workers（面向智能体执行者）：** REQUIRED SUB-SKILL（必需子技能）为 `superpowers:executing-plans`（执行计划）；本计划由当前智能体在本会话内逐项执行，不启动多个智能体。开始执行时先使用 `using-git-worktrees`（使用 Git 工作树）技能，从提交 `753bbb5` 创建隔离分支和工作树。

**目标：** 在不生成新句子、不使用目标测试标签筛选、保持单生成器流程的前提下，为 `rest14 -> laptop14` 建立可恢复的目标域方面候选发现诊断阶段。

**架构：** 从原始目标域句子进行抽取器多候选解码，聚合同一三元组的多序列支持；使用原始语料中的跨度、文档频率、方面—观点距离和现有单个标签到文本生成器对“候选标签生成原句”的归一化损失做独立验证。第一版只输出诊断候选和分析文件，不把候选加入最终训练集。

**技术栈：** Python（编程语言）、PyTorch（深度学习框架）、Transformers（模型库）、T5-base（文本到文本基础模型）、现有 JSONL（逐行结构化数据）与可复现运行框架；RTX 3070 8 GB，批次1/评估批次2、fp16（半精度）、梯度检查点、CUDA（英伟达计算平台）0。

**实施状态（2026-08-08）：** 代码已在 `feature/target-aspect-discovery-v1` 完成。提交为 `07865a7`、`91b8f6c`、`3a7b766`、`506b36b`；62项相关回归测试、新配方5阶段干运行和 RTX 3070 两行显卡冒烟测试均已通过，候选解码与重构评分均验证了逐批完成行断点。现有 PowerShell（微软命令行脚本）入口本身已支持任意配方，因此无需修改。正式单种子诊断尚未启动。

**正式结果（2026-08-08）：** 单种子诊断五阶段全部完成，从1889条候选保留237条，后验 precision/recall/F1（精确率/召回率/F1）为54.85%/8.93%/15.36%，低于现有421条高精度伪标签61.995%的精确率，因此未通过进入 A2 训练注入的门禁。后续只允许使用源域开发集金标校准联合门禁，不能依据目标隐藏金标分层结果直接调参。短流程汇总兼容修复为 `e4aff0c`，阶段哈希复用修复为 `408b1fb`，67项相关回归测试通过；原实验科学计算身份保持 `506b36b`。

**v2 正式结果（2026-08-09）：** `b16fab0` 的五阶段单种子诊断已完成。源域搜索选择支持度2、损失分位点1.0和损失上限4.529653，源域340条候选精度65.59%；冻结应用到目标域后保留702条，TP/FP/FN为338/364/1118，precision/recall/F1（精确率/召回率/F1）为48.15%/23.21%/31.33%。该精度低于现有高精度伪标签61.995%，不允许进入训练注入。根因是“达到65%后最大化覆盖”选择了最宽松的1.0分位点，使生成器损失只拒绝4条目标候选；2路支持候选精度仅39.38%，新增候选精度仅45.59%。实验标记为诊断失败；用户许可后已删除约11.64 GB模型与检查点，分析证据完整保留。

---

## 1. 版本基点与不可变边界

- 新分支必须从 `753bbb5` 创建。该提交以 `master=d2f2a35` 为祖先，提供 `rest14 -> laptop14` 的52.60/55.09同方向对照，且不包含历史双生成器实现。
- 不从 `d08a272` 或 `e49bfb6` 继续开发，避免继承已经失败的动态比例、直接观点编辑和无上限增强逻辑。
- 保持一个8轮、最佳检查点的标签到文本生成器；生成器同时为既有两个增强通道服务，但本阶段不修改增强阶段。
- 不实现 Teacher–Student（教师—学生）、EMA（指数移动平均）、PCGrad（梯度冲突投影）、多候选最终解码、中性最小对比样本。
- 不读取任何历史 `runs` 作为正式输入；诊断运行从原始数据重新训练抽取器与生成器。
- 目标隐藏标签只允许在候选已经冻结后生成分析指标，不能参与阈值、排序和保留决策。

## 2. 文件结构

- 新建 `target_aspect_discovery.py`：纯数据逻辑，包括候选解析、跨度检查、文档频率、跨序列一致性、距离计算、生成器分数门禁和诊断统计。
- 新建 `test_target_aspect_discovery.py`：覆盖候选聚合、标签冲突、跨度门禁、频率门禁、源域校准和目标标签隔离。
- 新建 `test_single_generator_guard.py`：锁定单生成器阶段、参数和模型目录，拒绝任何方面/观点双生成器参数。
- 修改 `t5_aste_pipeline.py`：增加多候选生成、标签到原句损失评分、`discover_target_aspects` 子命令和断点状态。
- 修改 `run_reproducible_pipeline.py`：增加可选 `target_aspect_discovery` 配方段、诊断阶段和 `execution.stop_after_stage` 门禁；历史配方没有该字段时命令图必须保持不变。
- 修改 `run_recipe_reproducible_pipeline.ps1`：透传诊断配方；不增加第二生成器参数。
- 新建 `configs/recipes/experiments/rest14_to_laptop14_target_aspect_discovery_diag_v1.json`：单种子1000诊断配方，执行到方面发现阶段后停止。
- 修改 `实验记录与模型索引_CN.md`：记录单生成器审计、诊断配方、验收/停止条件和运行命令位置。

## 3. 任务一：锁定单生成器和历史命令图

**测试先行：** 在 `test_single_generator_guard.py` 中构建历史最佳配方、`rest14 -> laptop14` 对照配方和新诊断配方的阶段图，并断言：

```python
generator_stages = [stage for stage in stages if stage.name == "generator"]
assert len(generator_stages) == 1
assert "generator_aspect" not in {stage.name for stage in stages}
assert "generator_opinion" not in {stage.name for stage in stages}

augment_argv = list(next(stage for stage in stages if stage.name == "augment").argv)
assert "--aspect_generator_model_path" not in augment_argv
assert "--opinion_generator_model_path" not in augment_argv
assert augment_argv.count("--model_path") == 1
```

同时保留历史命令图哈希：

```python
assert historical_command_graph_sha256 == (
    "205a94fb99c92a8c26310884c66e3994c8cb7355885f6a5042a6cd6dda7480ec"
)
```

运行：

```cmd
cmd /c "conda activate c3da && python -m unittest test_single_generator_guard.py test_native_best_runner.py"
```

预期：测试在实现前因缺少新测试文件失败；实现后全部通过。任何历史配方被无条件加入新参数都必须失败。

提交：

```cmd
cmd /c "git add test_single_generator_guard.py && git commit -m ^"test: lock single-generator command graph^""
```

## 4. 任务二：实现纯候选聚合与验证

在 `target_aspect_discovery.py` 定义以下不可变数据结构：

```python
@dataclass(frozen=True)
class AspectTripletCandidate:
    row_id: int
    text: str
    aspect: str
    opinion: str
    sentiment: str
    sequence_support: int
    sequence_total: int
    aspect_document_frequency: int
    token_distance: int
    generator_nll: float | None = None

@dataclass(frozen=True)
class AspectDiscoveryThresholds:
    min_sequence_support: int = 2
    min_document_frequency: int = 2
    max_token_distance: int = 8
    generator_nll_max: float | None = None
```

实现以下函数：

```python
def collect_triplet_candidates(
    rows: Sequence[dict], predictions_by_row: Sequence[Sequence[str]]
) -> list[AspectTripletCandidate]: ...

def calibrate_generator_nll(
    correct_source_dev_nll: Sequence[float], quantile: float = 0.75
) -> float: ...

def validate_candidate(
    candidate: AspectTripletCandidate,
    thresholds: AspectDiscoveryThresholds,
) -> tuple[bool, str]: ...

def build_discovery_analysis(
    candidates: Sequence[AspectTripletCandidate],
    accepted: Sequence[AspectTripletCandidate],
    existing_high_precision_labels: Sequence[str],
) -> dict: ...
```

硬门禁：

- 方面和观点必须是原句中的连续跨度，大小写和规范化空格允许等价。
- 情感必须为 `pos/neg/neu` 三者之一。
- 同一完整三元组至少获得4个候选序列中的2个支持。
- 同一方面—观点出现情感冲突时，最高情感支持数必须严格大于第二名，否则拒绝。
- 方面目标语料文档频率至少2；若序列支持达到3，可允许频率1。
- 方面与观点最大词元距离8。
- 生成器归一化负对数似然不得高于源域开发集正确三元组分布的75%分位阈值。
- 已经存在于高精度伪标签中的三元组标记为 `existing`，不计作新增候选。

测试至少包括：有效多序列候选、单序列拒绝、情感平票拒绝、非原文跨度拒绝、低频拒绝、距离拒绝、生成器分数拒绝、已有伪标签去重、目标标签字段不参与选择。

运行：

```cmd
cmd /c "conda activate c3da && python -m unittest test_target_aspect_discovery.py"
```

提交：

```cmd
cmd /c "git add target_aspect_discovery.py test_target_aspect_discovery.py && git commit -m ^"feat: add target aspect candidate validation^""
```

## 5. 任务三：增加多候选解码与原句重构评分

在 `t5_aste_pipeline.py` 中新增独立函数，不修改现有单序列 `batched_generate` 的返回格式：

```python
def batched_generate_candidates(
    model_path: Path,
    texts: Sequence[str],
    batch_size: int,
    num_beams: int,
    num_return_sequences: int,
    max_new_tokens: int,
    cuda: int,
    use_task_prefix: bool,
) -> list[list[str]]: ...

def score_label_to_text_reconstruction(
    model_path: Path,
    labels: Sequence[str],
    texts: Sequence[str],
    domain_name: str,
    batch_size: int,
    cuda: int,
) -> list[float]: ...
```

要求：

- `num_beams=4`、`num_return_sequences=4`。
- 返回结构严格为每个输入对应4条候选，不能打乱行号。
- 重构输入格式保持现有生成器训练格式：`target domain: [laptop14] ; generate aste sentence: <label>`。
- 重构分数是忽略填充位置后的每目标词元平均负对数似然。
- 抽取器和生成器顺序加载、评分后释放并清空 CUDA 缓存，不能同时常驻8 GB显存。
- 两个循环均显示进度条，并按已完成行号保存状态文件。

测试使用小型假模型和假 tokenizer（分词器），验证候选行顺序、候选数、填充掩码、归一化损失以及中断恢复不重复处理已完成行。

运行：

```cmd
cmd /c "conda activate c3da && python -m unittest test_target_aspect_discovery.py test_native_best_runner.py"
```

提交：

```cmd
cmd /c "git add t5_aste_pipeline.py test_target_aspect_discovery.py && git commit -m ^"feat: add multi-view aspect discovery inference^""
```

## 6. 任务四：增加可恢复诊断子命令

为 `t5_aste_pipeline.py` 新增：

```cmd
discover_target_aspects
```

输入必须全部来自本次 `run_id`：

- `target_unlabeled.jsonl`
- `source_dev.jsonl`
- `target_pseudo_high_precision.jsonl`
- 本次抽取器 `model.safetensors`
- 本次单生成器 `model.safetensors`

输出：

- `target_aspect_candidates.jsonl`
- `target_aspect_candidates_validated.jsonl`
- `target_aspect_discovery_analysis.json`
- `target_aspect_discovery_calibration.json`
- `target_aspect_discovery_state.json`

状态文件记录输入哈希、模型哈希、已完成行号和参数。任何哈希或参数变化时拒绝恢复，不能静默重算。

诊断分析至少包含：总候选、有效候选、新增候选、已有候选、各拒绝原因、序列支持分布、方面文档频率分布、情感分布、单双三元组句子分布、生成器负对数似然分布。若运行目录提供隐藏金标，只在选择结束后追加 precision/recall/F1（精确率/召回率/F1），选择函数不得接收金标参数。

运行聚焦测试：

```cmd
cmd /c "conda activate c3da && python -m unittest test_target_aspect_discovery.py test_reproducibility_provenance.py"
```

提交：

```cmd
cmd /c "git add t5_aste_pipeline.py target_aspect_discovery.py test_target_aspect_discovery.py && git commit -m ^"feat: add resumable aspect discovery diagnostic^""
```

## 7. 任务五：接入可复现配方但保持历史配方不变

在 `run_reproducible_pipeline.py` 中仅当配方包含 `target_aspect_discovery.enabled=true` 时插入 `target_aspect_discovery` 阶段。历史配方不得出现任何新增参数。

新配方关键字段：

```json
{
  "target_aspect_discovery": {
    "enabled": true,
    "num_beams": 4,
    "num_return_sequences": 4,
    "min_sequence_support": 2,
    "min_document_frequency": 2,
    "max_token_distance": 8,
    "generator_nll_quantile": 0.75,
    "use_for_training": false
  },
  "execution": {
    "stop_after_stage": "target_aspect_discovery"
  }
}
```

配方验证必须拒绝：候选数大于 beam、支持数小于2、目标标签参与筛选、`use_for_training=true`、未知停止阶段和双生成器字段。

测试断言：

- 新配方阶段顺序为 `prepare/extractor/pseudo/generator/target_aspect_discovery`，随后停止。
- 历史最佳仍为原十阶段且命令图哈希完全不变。
- 新阶段输入全部位于同一 `run_id`。
- 断点恢复验证配方和输入哈希。

运行：

```cmd
cmd /c "conda activate c3da && python -m unittest test_single_generator_guard.py test_target_aspect_discovery.py test_native_best_runner.py test_reproducibility_provenance.py"
```

提交：

```cmd
cmd /c "git add run_reproducible_pipeline.py run_recipe_reproducible_pipeline.ps1 configs/recipes/experiments/rest14_to_laptop14_target_aspect_discovery_diag_v1.json test_single_generator_guard.py test_target_aspect_discovery.py && git commit -m ^"feat: add reproducible aspect discovery recipe^""
```

## 8. 任务六：必要验证与文档维护

只运行必要验证：

1. 新增候选发现测试。
2. 单生成器与历史命令图回归测试。
3. 配方和来源隔离测试。
4. Python 语法检查。
5. 十阶段/诊断阶段命令干运行。
6. RTX 3070上2行抽取器候选生成和2条生成器重构评分冒烟测试。

不在此阶段运行完整最终训练。诊断配方通过后提供一条从 CMD（命令提示符）开始的单行命令，由用户运行到 `target_aspect_discovery` 阶段。

诊断进入 A2 训练实验的门禁：

- 新增有效候选数量大于0。
- 有效候选不能主要由单序列或情感冲突产生。
- 候选后验精确率不低于现有421条高精度伪标签的对应精确率下限。
- 新增方面覆盖明显增加。
- 选择过程与目标隐藏标签完全隔离。

若不通过，只保留分析文件并停止；若通过，再单独设计 A2“候选加入训练”配方，不在本计划提前实现。

最终提交：

```cmd
cmd /c "git add 实验记录与模型索引_CN.md docs/superpowers/plans/2026-08-08-target-aspect-discovery-v1_CN.md && git commit -m ^"docs: record target aspect discovery diagnostic^""
```

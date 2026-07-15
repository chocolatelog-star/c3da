# 混合任务生成器训练实施计划

> **For agentic workers（面向代理执行者）：** REQUIRED SUB-SKILL（必需子技能）：使用 `superpowers:subagent-driven-development`（子代理驱动开发，推荐）或 `superpowers:executing-plans`（执行计划），按任务逐项实施。所有步骤使用复选框跟踪。

**目标：** 新增一个 1:1:1 的混合生成器训练模式，让同一个全新 T5（文本到文本转换模型）学习标签到文本、方面词掩码编辑、观点词-情感掩码编辑，同时保持最终互相掩码增强流程和当前最佳配置不变。

**架构：** `t5_aste_augment.py` 负责构造和按源句平衡三类训练样本；`t5_aste_pipeline.py` 负责写入训练文件并记录通道统计；`run_bgca_aste_stage1_pairs.py` 负责命令路由、独立命名、阶段状态和完整实验流程。旧 `label_to_text`（标签到文本）与 `masked_mutual`（互相掩码）模式保持兼容。

**技术栈：** Python（编程语言）3.10、PyTorch（深度学习框架）、Transformers（预训练模型库）、T5-base（基础文本到文本模型）、`unittest`（单元测试框架）、Git（版本管理）。

---

## 文件结构

- 修改 `t5_aste_augment.py`：新增 `mixed`（混合）训练风格、源样本关联元数据和三通道按源句选择逻辑。
- 修改 `t5_aste_pipeline.py`：允许准备命令接收混合风格，并在清单中写入各训练/验证通道计数。
- 修改 `run_bgca_aste_stage1_pairs.py`：增加混合风格命名、命令选项、独立训练文件与阶段状态。
- 修改 `test_masked_mutual_augment.py`：覆盖混合数据构造、比例、固定随机种子、无效通道和领域前缀。
- 修改 `test_generator_label_to_text.py`：覆盖准备阶段文件与通道统计。
- 修改 `test_run_bgca_stage1_pairs.py`：覆盖完整 `dry_run`（仅打印不执行）、独立命名和自动续训参数。
- 修改 `实验记录与模型索引_CN.md`：登记代码版本、实验命令、输出位置和待跑状态。
- 修改 `CD_C3DA_BGCA超越目标路线图_CN.md`：同步第四项实现状态与验收标准。

### 任务 1：锁定混合训练数据契约

**文件：**

- 修改：`test_masked_mutual_augment.py`
- 测试：`test_masked_mutual_augment.py`

- [ ] **步骤 1：添加单句三通道失败测试**

添加测试，使用至少两个可替换方面词和同情感观点词的源域样本，固定种子后断言每个有效源句最多产生以下三个通道：

```python
rows = [
    {"id": "s1", "text": "The battery is great.", "label": "<pos> battery <opinion> great"},
    {"id": "s2", "text": "The screen is bright.", "label": "<pos> screen <opinion> bright"},
]
mixed_rows = build_generator_training_rows(
    rows,
    seed=13,
    prompt_style="mixed",
    channel_mode="all",
    domain_name="rest16",
    domain_prefix_style="text",
)
counts = Counter(row["channel"] for row in mixed_rows)
self.assertEqual(counts["label_to_text_generator"], 2)
self.assertEqual(counts["masked_aspect_editor"], 2)
self.assertEqual(counts["masked_opinion_sentiment_editor"], 2)
self.assertEqual(len(mixed_rows), 6)
```

同时断言：

```python
self.assertTrue(all(row["source_index"] in {0, 1} for row in mixed_rows))
self.assertTrue(all(row["domain_name"] == "rest16" for row in mixed_rows))
self.assertTrue(all(row["domain_prefix_style"] == "text" for row in mixed_rows))
```

- [ ] **步骤 2：添加多三元组、固定种子和无效通道失败测试**

测试必须确认同一源句即使含多个三元组，每个通道也最多一行；相同种子得到完全相同的输出；无法形成有效替换的通道可以缺失但不会复制其他通道补齐：

```python
first = build_generator_training_rows(rows, seed=1000, prompt_style="mixed")
second = build_generator_training_rows(rows, seed=1000, prompt_style="mixed")
self.assertEqual(first, second)
for source_index in {row["source_index"] for row in first}:
    source_rows = [row for row in first if row["source_index"] == source_index]
    self.assertEqual(len(source_rows), len({row["channel"] for row in source_rows}))
```

- [ ] **步骤 3：运行测试并确认因模式未实现而失败**

运行一行命令：

```text
J:\conda\envs\c3da\python.exe -m unittest test_masked_mutual_augment.py -v
```

预期：新增测试因 `mixed` 不在 `PROMPT_STYLES`（提示风格集合）中而失败；既有测试继续通过。

- [ ] **步骤 4：提交测试契约**

```text
git add test_masked_mutual_augment.py && git commit -m "Test mixed generator training rows"
```

### 任务 2：实现混合训练数据构造

**文件：**

- 修改：`t5_aste_augment.py`
- 测试：`test_masked_mutual_augment.py`

- [ ] **步骤 1：注册明确的混合风格常量**

在 `PROMPT_STYLES`（提示风格集合）中加入：

```python
"mixed",
```

把错误信息改为从集合生成，避免以后新增风格后提示遗漏：

```python
if prompt_style not in PROMPT_STYLES:
    raise ValueError(f"prompt_style must be one of {', '.join(sorted(PROMPT_STYLES))}")
```

- [ ] **步骤 2：为递归构造增加仅内部使用的源句索引**

混合分支先复制输入行并增加内部索引：

```python
indexed_rows = [dict(row, _generator_source_index=index) for index, row in enumerate(rows)]
```

现有训练行构造完成时，仅当输入包含该内部字段才输出公开的 `source_index`：

```python
if "_generator_source_index" in row:
    train_row["source_index"] = int(row["_generator_source_index"])
```

该字段只用于混合模式分组，不改变旧模式必需字段和训练读取逻辑。

- [ ] **步骤 3：构造三类候选并按源句最多选一行**

在 `build_generator_training_rows()` 开头为混合模式增加专用分支：

```python
if prompt_style == "mixed":
    indexed_rows = [dict(row, _generator_source_index=index) for index, row in enumerate(rows)]
    label_rows = build_generator_training_rows(
        indexed_rows, seed, "label_to_text", "all", domain_name, domain_prefix_style
    )
    aspect_rows = build_generator_training_rows(
        indexed_rows, seed + 1, "masked_mutual", "aspect", domain_name, domain_prefix_style
    )
    opinion_rows = build_generator_training_rows(
        indexed_rows, seed + 2, "masked_mutual", "opinion", domain_name, domain_prefix_style
    )
    candidates = {
        "label_to_text_generator": label_rows,
        "masked_aspect_editor": aspect_rows,
        "masked_opinion_sentiment_editor": opinion_rows,
    }
    selected = []
    selector = random.Random(seed + 3)
    for source_index in range(len(rows)):
        for channel in (
            "label_to_text_generator",
            "masked_aspect_editor",
            "masked_opinion_sentiment_editor",
        ):
            matches = [row for row in candidates[channel] if row.get("source_index") == source_index]
            if matches:
                selected.append(selector.choice(matches))
    return selected
```

这里使用三个不同但确定的种子，防止两个掩码通道共享随机状态产生隐式耦合；每个源句每通道最多一行，实际无法构造时自然缺失。

- [ ] **步骤 4：运行混合与旧模式回归测试**

```text
J:\conda\envs\c3da\python.exe -m unittest test_masked_mutual_augment.py test_generator_label_to_text.py -v
```

预期：新增混合测试通过；既有标签到文本与互相掩码测试全部通过。

- [ ] **步骤 5：提交数据构造实现**

```text
git add t5_aste_augment.py test_masked_mutual_augment.py && git commit -m "Add balanced mixed generator training rows"
```

### 任务 3：准备阶段统计与数据文件验证

**文件：**

- 修改：`t5_aste_pipeline.py`
- 修改：`test_generator_label_to_text.py`
- 测试：`test_generator_label_to_text.py`

- [ ] **步骤 1：添加准备阶段失败测试**

新增 `prepare`（准备）测试，传入：

```python
augment_prompt_style="mixed"
generator_output_tag="mixed_l2t_masked_aspect_masked_opinion"
```

断言四个独立文件存在：

```python
self.assertTrue((run_dir / "generator_train_mixed_l2t_masked_aspect_masked_opinion.jsonl").exists())
self.assertTrue((run_dir / "generator_dev_mixed_l2t_masked_aspect_masked_opinion.jsonl").exists())
self.assertTrue((run_dir / "c3da_generator_train_mixed_l2t_masked_aspect_masked_opinion.jsonl").exists())
self.assertTrue((run_dir / "c3da_generator_dev_mixed_l2t_masked_aspect_masked_opinion.jsonl").exists())
```

读取 `manifest.json`（清单文件），断言包含：

```python
self.assertEqual(manifest["augment_prompt_style"], "mixed")
self.assertEqual(sum(manifest["generator_train_channel_counts"].values()), manifest["generator_train"])
self.assertEqual(sum(manifest["generator_dev_channel_counts"].values()), manifest["generator_dev"])
```

- [ ] **步骤 2：运行测试并确认参数选项或统计字段失败**

```text
J:\conda\envs\c3da\python.exe -m unittest test_generator_label_to_text.py -v
```

预期：新增测试因清单缺少通道计数而失败。

- [ ] **步骤 3：实现通道统计并允许命令行混合模式**

导入并使用 `Counter`（计数器）：

```python
generator_train_channel_counts = dict(Counter(row.get("channel", "unknown") for row in generator_train_rows))
generator_dev_channel_counts = dict(Counter(row.get("channel", "unknown") for row in generator_dev_rows))
```

将两个字段写入清单：

```python
"generator_train_channel_counts": generator_train_channel_counts,
"generator_dev_channel_counts": generator_dev_channel_counts,
```

在 `prepare`（准备）子命令的 `--augment_prompt_style` 选项中加入 `mixed`。

- [ ] **步骤 4：运行准备阶段与生成器数据测试**

```text
J:\conda\envs\c3da\python.exe -m unittest test_generator_label_to_text.py test_masked_mutual_augment.py -v
```

预期：所有测试通过，清单计数之和与总行数一致。

- [ ] **步骤 5：提交准备阶段实现**

```text
git add t5_aste_pipeline.py test_generator_label_to_text.py && git commit -m "Report mixed generator channel counts"
```

### 任务 4：接入完整流程、隔离命名和断点续跑

**文件：**

- 修改：`run_bgca_aste_stage1_pairs.py`
- 修改：`test_run_bgca_stage1_pairs.py`
- 测试：`test_run_bgca_stage1_pairs.py`

- [ ] **步骤 1：添加混合流程失败测试**

扩展测试辅助函数允许覆盖生成器风格，使用以下参数运行 `dry_run`（仅打印不执行）：

```text
--output_root runs\bgca_aste_stage1_mixed_generator_v1 --pairs rest16:laptop14 --generator_prompt_style mixed --augment_prompt_style masked_mutual --domain_prefix_style text --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --dry_run
```

断言输出包含：

```python
self.assertIn("--augment_prompt_style mixed", output)
self.assertIn("c3da_generator_train_mixed_l2t_masked_aspect_masked_opinion.jsonl", output)
self.assertIn("generator_mixed_l2t_masked_aspect_masked_opinion_ep8", output)
self.assertIn("--resume_from_checkpoint auto", output)
self.assertIn("--augment_prompt_style masked_mutual", output)
self.assertIn("strict_aug150_w020_mixed_l2t_masked_aspect_masked_opinion", output)
self.assertIn("--per_device_train_batch_size 1", output)
self.assertIn("--per_device_eval_batch_size 2", output)
self.assertIn("--gradient_accumulation_steps 16", output)
self.assertIn("--fp16", output)
self.assertIn("--gradient_checkpointing", output)
```

- [ ] **步骤 2：运行测试并确认参数选项失败**

```text
J:\conda\envs\c3da\python.exe -m unittest test_run_bgca_stage1_pairs.py -v
```

预期：命令解析器拒绝 `mixed`。

- [ ] **步骤 3：实现独立生成器标签和命令选项**

扩展标签函数：

```python
def generator_tag(prompt_style: str) -> str:
    if prompt_style == "label_to_text":
        return "label_to_text_gen"
    if prompt_style == "masked_mutual":
        return "masked_mutual_gen"
    if prompt_style == "mixed":
        return "mixed_l2t_masked_aspect_masked_opinion"
    raise ValueError(f"unsupported generator prompt style: {prompt_style}")
```

扩展命令选项：

```python
parser.add_argument(
    "--generator_prompt_style",
    choices=["label_to_text", "masked_mutual", "mixed"],
    default="label_to_text",
)
```

现有 `gen_tag`（生成器标签）会自动隔离以下产物：

```text
prepare_mixed_l2t_masked_aspect_masked_opinion
train_generator_mixed_l2t_masked_aspect_masked_opinion
c3da_generator_train_mixed_l2t_masked_aspect_masked_opinion.jsonl
models\generator_mixed_l2t_masked_aspect_masked_opinion_ep8
strict_aug150_w020_mixed_l2t_masked_aspect_masked_opinion
```

不为混合模式添加旧阶段别名，防止错误复用旧生成器或旧增强结果。

- [ ] **步骤 4：运行批处理和状态回归测试**

```text
J:\conda\envs\c3da\python.exe -m unittest test_run_bgca_stage1_pairs.py -v
```

预期：混合模式测试和全部既有状态测试通过。

- [ ] **步骤 5：提交完整流程接入**

```text
git add run_bgca_aste_stage1_pairs.py test_run_bgca_stage1_pairs.py && git commit -m "Route mixed generator stage1 experiment"
```

### 任务 5：整体验证、中文记录与实验命令

**文件：**

- 修改：`实验记录与模型索引_CN.md`
- 修改：`CD_C3DA_BGCA超越目标路线图_CN.md`
- 测试：全部相关测试文件

- [ ] **步骤 1：运行相关单元测试**

```text
J:\conda\envs\c3da\python.exe -m unittest test_masked_mutual_augment.py test_generator_label_to_text.py test_run_bgca_stage1_pairs.py test_evaluate_output_isolation.py -v
```

预期：全部通过，无失败和错误。

- [ ] **步骤 2：运行 Python（编程语言）语法检查**

```text
J:\conda\envs\c3da\python.exe -m py_compile t5_aste_augment.py t5_aste_pipeline.py run_bgca_aste_stage1_pairs.py test_masked_mutual_augment.py test_generator_label_to_text.py test_run_bgca_stage1_pairs.py
```

预期：退出码为 0，无输出。

- [ ] **步骤 3：运行单方向完整流程预演**

```text
J:\conda\envs\c3da\python.exe run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_mixed_generator_v1 --pairs rest16:laptop14 --extractor_model_path J:\nlp\models\t5-base-py --generator_model_path J:\nlp\models\t5-base-py --generator_prompt_style mixed --augment_prompt_style masked_mutual --domain_prefix_style text --extractor_epochs 25 --generator_epochs 8 --final_epochs 5 --high_precision_max_triplets 1 --high_precision_max_token_distance 5 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --learning_rate 0.0003 --eval_batch_size 2 --cuda 0 --seed 1000 --dry_run
```

预期：完整打印准备、提取器、伪标签、全新混合生成器、互相掩码增强、最终训练和评估命令；训练命令包含自动续训和 8GB 显存参数。

- [ ] **步骤 4：整体审查代码和差异**

```text
git diff --check && git status --short && git diff --stat
```

预期：没有空白错误；只包含计划内代码、测试和中文文档。

- [ ] **步骤 5：整体更新中文实验文档**

在 `实验记录与模型索引_CN.md` 开头当前阶段表中加入：

```text
第四项代码完成：混合生成器按标签到文本、方面词掩码、观点词-情感掩码三任务训练；实际增强仍为互相掩码；待跑 rest16 -> laptop14 单变量实验。
```

记录最终 Git（版本管理）提交号、完整输出目录、生成器模型目录、最终模型目录、完整命令和验收标准。同步修改路线图第四项状态，但不追加重复历史长段落。

- [ ] **步骤 6：提交实现与文档收尾**

```text
git add 实验记录与模型索引_CN.md CD_C3DA_BGCA超越目标路线图_CN.md && git commit -m "Document mixed generator experiment workflow"
```

- [ ] **步骤 7：推送 GitHub（代码托管平台）**

```text
git push
```

预期：本地 `master`（主分支）与 `origin/master`（远程主分支）同步。若网络失败，保留本地提交并明确报告，不重复修改提交历史。

## 完成标准

- `mixed`（混合）模式每个有效源句每通道最多一行，固定种子可复现。
- 生成器训练与验证清单输出三通道实际计数。
- 旧模式测试全部通过。
- 新实验目录、训练文件、模型、阶段状态和结果标签均与旧实验隔离。
- 完整命令支持相同命令重跑、阶段跳过和训练检查点续跑。
- 用户获得从 CMD（命令提示符）开始的一行完整实验命令及每阶段说明。

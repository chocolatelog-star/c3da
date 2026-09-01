# C3DA 可控实验编排器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐上游 batch、固定上游下游 batch、梯度归一化审计和 Graph OFF/ON A/B 四类实验入口，并保证可恢复、可汇总和可同步。

**Architecture:** 新增共享的命令构造、状态记录和结果汇总小模块；四个 CLI 编排器只负责参数解析、子进程编排和产物校验，复用现有训练/评估脚本。每个单元独立目录，阶段完成后写原子状态 JSON，默认 hash 只记录不阻塞。

**Tech Stack:** Python 3.10、PyTorch、Transformers、pytest、现有 `run_reproducible_pipeline.py` 和 `run_bgca_aste_stage1_pairs.py`。

---

### Task 1: 建立共享实验编排工具

**Files:**
- Create: `experiment_runner_common.py`
- Test: `test_experiment_runner_common.py`

- [ ] **Step 1: Write the failing test**

```python
def test_matrix_unit_paths_are_isolated(tmp_path):
    from experiment_runner_common import matrix_unit_dir
    assert matrix_unit_dir(tmp_path, 8, 2).name == "batch8_accum2"
    assert matrix_unit_dir(tmp_path, 16, 1) != matrix_unit_dir(tmp_path, 8, 2)

def test_atomic_json_round_trip(tmp_path):
    from experiment_runner_common import atomic_write_json, read_json
    path = tmp_path / "status.json"
    atomic_write_json(path, {"status": "complete"})
    assert read_json(path)["status"] == "complete"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q test_experiment_runner_common.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiment_runner_common'`.

- [ ] **Step 3: Write minimal implementation**

Implement `matrix_unit_dir(root, batch_size, accumulation)`, `atomic_write_json(path, value)`, `read_json(path)`, `run_command(command, log_path, env)`, `sha256_file(path)`, `write_status(path, status, **fields)` and `completed_file(path)` using temporary files followed by `os.replace`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q test_experiment_runner_common.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiment_runner_common.py test_experiment_runner_common.py
git commit -m "feat: add shared experiment runner utilities"
```

### Task 2: 上游 batch 矩阵入口

**Files:**
- Create: `run_upstream_batch_matrix.py`
- Test: `test_upstream_batch_matrix.py`

- [ ] **Step 1: Write the failing test**

```python
def test_upstream_commands_stop_at_pseudo(tmp_path):
    from run_upstream_batch_matrix import build_upstream_command
    command = build_upstream_command(
        project_root=tmp_path, recipe="recipe.json", output_root=tmp_path / "out",
        run_id="b8", train_batch_size=8, gradient_accumulation_steps=2, cuda="0"
    )
    assert "--stop_after_stage" in command
    assert command[command.index("--stop_after_stage") + 1] == "pseudo"
    assert "--final_train" not in command
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q test_upstream_batch_matrix.py`
Expected: FAIL because the module and command builder do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement CLI defaults `1x16,8x2,16x1,16x2,32x1`, per-unit output directories, subprocess logs, `--gpus`, `--resume`, and JSON/Markdown aggregation. Call `run_reproducible_pipeline.py` with `--stop_after_stage pseudo`, batch/accumulation overrides, `--allow_dirty`, and never append generator/final stages.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q test_upstream_batch_matrix.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add run_upstream_batch_matrix.py test_upstream_batch_matrix.py
git commit -m "feat: add upstream batch matrix runner"
```

### Task 3: 固定上游下游 batch 矩阵入口

**Files:**
- Create: `run_fixed_upstream_downstream_batch_matrix.py`
- Test: `test_fixed_upstream_downstream_batch_matrix.py`

- [ ] **Step 1: Write the failing test**

```python
def test_downstream_command_reuses_inputs(tmp_path):
    from run_fixed_upstream_downstream_batch_matrix import build_downstream_command
    command = build_downstream_command(tmp_path, 16, 1, "0")
    assert "--reuse_upstream_run_dir" in command
    assert command[command.index("--train_batch_size") + 1] == "16"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q test_fixed_upstream_downstream_batch_matrix.py`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Write minimal implementation**

Validate and hash extractor, pseudo, generator, selected augmentation, final train and dev before launching. Generate one command per batch pair using `run_bgca_aste_stage1_pairs.py --reuse_upstream_run_dir`; write a shared manifest and aggregate only final F1 and runtime.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q test_fixed_upstream_downstream_batch_matrix.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add run_fixed_upstream_downstream_batch_matrix.py test_fixed_upstream_downstream_batch_matrix.py
git commit -m "feat: add fixed-upstream downstream batch runner"
```

### Task 4: 梯度与真实样本归一化审计

**Files:**
- Create: `batch_gradient_parameter_audit.py`
- Test: `test_batch_gradient_parameter_audit.py`

- [ ] **Step 1: Write the failing test**

```python
def test_weighted_mean_uses_effective_sample_weight():
    from batch_gradient_parameter_audit import weighted_mean
    assert weighted_mean([2.0, 4.0], [1.0, 3.0]) == 3.5

def test_audit_matrix_has_four_comparison_groups():
    from batch_gradient_parameter_audit import default_audit_groups
    assert default_audit_groups() == [(1, 16), (4, 4), (8, 2), (16, 1)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q test_batch_gradient_parameter_audit.py`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement deterministic 16-row selection, token/sample-weight reporting, batch-mean and effective-weight reductions, one optimizer update per group, gradient/update norms, parameter deltas, and JSON/Markdown output. Never load target-test data.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q test_batch_gradient_parameter_audit.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add batch_gradient_parameter_audit.py test_batch_gradient_parameter_audit.py
git commit -m "feat: add batch gradient and normalization audit"
```

### Task 5: Graph OFF/ON 完整 A/B 入口

**Files:**
- Create: `run_graph_control_ab.py`
- Test: `test_graph_control_ab.py`
- Modify: `run_plan_a_graph_best.py`

- [ ] **Step 1: Write the failing test**

```python
def test_ab_commands_differ_only_by_graph_switch(tmp_path):
    from run_graph_control_ab import build_ab_specs
    specs = build_ab_specs(tmp_path, train_batch_size=16, accumulation=1)
    assert {spec["name"] for spec in specs} == {"control", "graph"}
    assert specs[0]["graph_enabled"] is False
    assert specs[1]["graph_enabled"] is True
    assert specs[0]["train_batch_size"] == specs[1]["train_batch_size"] == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q test_graph_control_ab.py`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement two independent Phase A/full-output directories, identity comparison excluding only graph fields, reuse/resume of complete stages, and final JSON/Markdown with Control F1, Graph F1, delta and single/multi-triplet metrics. Update `run_plan_a_graph_best.py` to call the shared downstream builder and preserve `--dry_run`.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q test_graph_control_ab.py test_plan_a_graph_best.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add run_graph_control_ab.py test_graph_control_ab.py run_plan_a_graph_best.py
git commit -m "feat: add graph control treatment A/B runner"
```

### Task 6: 全量验证、文档和服务器同步

**Files:**
- Modify: `docs/实验计划_2026-08-31.md`
- Modify: `README.md`

- [ ] **Step 1: Run all tests**

Run: `J:\conda\envs\c3da\python.exe -m pytest -q J:\nlp\CD-C3DA`
Expected: all existing and new tests pass.

- [ ] **Step 2: Run CLI and dry-run checks**

Run: `python run_upstream_batch_matrix.py --help`, `python run_fixed_upstream_downstream_batch_matrix.py --help`, `python batch_gradient_parameter_audit.py --help`, `python run_graph_control_ab.py --help`.
Expected: all commands exit 0 and show GPU/output/resume arguments.

- [ ] **Step 3: Compile modules**

Run: `python -m compileall -q .`
Expected: no syntax errors.

- [ ] **Step 4: Update Chinese documentation**

Document exact one-line server commands, expected outputs, disk retention rules, and the distinction between upstream-only and fixed-upstream downstream experiments.

- [ ] **Step 5: Commit and synchronize**

```bash
git add docs/实验计划_2026-08-31.md README.md
git commit -m "docs: document controlled experiment runners"
```

Upload changed files to both servers, run the same focused tests and CLI smoke checks remotely, then report commit and verification output.

# Element-Aware RGAT 组件消融实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复并冻结 G1（仅元素聚焦）与 G2（仅多元素覆盖）正式消融入口，使两组只改变 Focus/Coverage 开关并严格复用 V9e 的训练配方。

**Architecture:** 将元素感知参数合法性集中为训练入口的纯校验函数；将消融版本解析、冻结批次校验、训练命令构造和运行身份集中在专用 Treatment-only（仅实验组）运行器。模型、图传播和损失计算保持原样，运行器只调用已有训练、source-dev 评估和 target-unlabeled 伪标签阶段。

**Tech Stack:** Python 3.10、PyTorch、Transformers、unittest/pytest 风格函数测试、Git。

---

## 文件结构

- 修改 `t5_absa_train.py`：增加纯参数校验函数并替换 `main` 中写死的双0.05校验。
- 修改 `m1_element_aware_rgat_treatment_only.py`：修复组件开关、冻结 V9e 批次、自动恢复、运行身份和最终边界记录。
- 新建 `test_element_aware_component_ablation.py`：覆盖训练入口与运行器配置，不加载真实模型或启动训练。
- 修改 `.ai/CURRENT_TASK.md` 与 `CHAT_SOL_CURRENT_TASK_CN.md`：记录本地实现和服务器同步边界；正式入口同步同内容。

### Task 1：训练入口组件权重校验

**Files:**
- Modify: `t5_absa_train.py:3379-3386`
- Create: `test_element_aware_component_ablation.py`

- [ ] **Step 1: 写失败测试**

在 `test_element_aware_component_ablation.py` 中直接测试纯校验接口：

```python
import pytest

from t5_absa_train import validate_element_aware_training_configuration


def test_focus_only_accepts_only_focus_weight():
    validate_element_aware_training_configuration(
        element_aware_attention=True,
        use_syntactic_graph_adapter=True,
        focus_enabled=True,
        coverage_enabled=False,
        focus_weight=0.05,
        coverage_weight=0.0,
        lambda_domain_adv=0.0,
    )


def test_coverage_only_accepts_only_coverage_weight():
    validate_element_aware_training_configuration(
        element_aware_attention=True,
        use_syntactic_graph_adapter=True,
        focus_enabled=False,
        coverage_enabled=True,
        focus_weight=0.0,
        coverage_weight=0.05,
        lambda_domain_adv=0.0,
    )


def test_enabled_and_disabled_loss_weights_must_match_flags():
    with pytest.raises(ValueError, match="weights must match enabled losses"):
        validate_element_aware_training_configuration(
            element_aware_attention=True,
            use_syntactic_graph_adapter=True,
            focus_enabled=True,
            coverage_enabled=False,
            focus_weight=0.05,
            coverage_weight=0.05,
            lambda_domain_adv=0.0,
        )
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run:

```text
J:\conda\envs\c3da\python.exe -m pytest test_element_aware_component_ablation.py -q
```

Expected: collection error，提示无法从 `t5_absa_train` 导入 `validate_element_aware_training_configuration`。

- [ ] **Step 3: 增加最小校验函数**

在参数解析入口之前增加：

```python
def validate_element_aware_training_configuration(
    *,
    element_aware_attention: bool,
    use_syntactic_graph_adapter: bool,
    focus_enabled: bool,
    coverage_enabled: bool,
    focus_weight: float,
    coverage_weight: float,
    lambda_domain_adv: float,
) -> None:
    if (focus_enabled or coverage_enabled) and not element_aware_attention:
        raise ValueError("element auxiliary losses require --element_aware_attention")
    if element_aware_attention and not use_syntactic_graph_adapter:
        raise ValueError("element-aware attention requires the syntactic graph adapter")
    if element_aware_attention and lambda_domain_adv != 0.0:
        raise ValueError("the approved element-aware M1 configuration requires lambda_domain_adv=0")
    if not element_aware_attention:
        return
    expected_focus_weight = 0.05 if focus_enabled else 0.0
    expected_coverage_weight = 0.05 if coverage_enabled else 0.0
    if focus_weight != expected_focus_weight or coverage_weight != expected_coverage_weight:
        raise ValueError("element-aware weights must match enabled losses")
```

在 `main` 中用一次函数调用替换现有四段内联校验，传入解析后的实际参数。

- [ ] **Step 4: 补齐 G3、DANN 和旧无图边界测试**

增加测试：G3 接受 `0.05/0.05`；元素感知且 `DANN != 0` 拒绝；无图旧流程保留默认权重时不受元素感知校验影响。

- [ ] **Step 5: 运行测试并确认通过**

Run:

```text
J:\conda\envs\c3da\python.exe -m pytest test_element_aware_component_ablation.py -q
```

Expected: Task 1 全部测试 PASS。

- [ ] **Step 6: 提交**

```text
git add t5_absa_train.py test_element_aware_component_ablation.py
git commit -m "fix:validate-element-ablation-weights"
```

### Task 2：构造严格的 G1/G2 冻结训练命令

**Files:**
- Modify: `m1_element_aware_rgat_treatment_only.py`
- Test: `test_element_aware_component_ablation.py`

- [ ] **Step 1: 写运行器失败测试**

为以下公开测试辅助接口写断言：

```python
from argparse import Namespace
from pathlib import Path

from m1_element_aware_rgat_treatment_only import build_train_args, resolve_variant


def _args(**overrides):
    values = {
        "model_path": "models/t5-base-py",
        "graph_cache_dir": "graph_cache/graph_cache_resume",
        "parser_dir": "models/stanza_resources",
        "cuda": "0",
        "train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "focus_only": False,
        "coverage_only": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_focus_only_train_args_are_exact_v9e_component_ablation(tmp_path):
    args = _args(focus_only=True)
    variant = resolve_variant(args)
    command, config = build_train_args(args, Path(tmp_path), variant)
    assert "--element_focus_loss" in command
    assert "--multi_element_coverage_loss" not in command
    assert command[command.index("--element_focus_weight") + 1] == "0.05"
    assert command[command.index("--element_coverage_weight") + 1] == "0"
    assert command[command.index("--per_device_train_batch_size") + 1] == "1"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "16"
    assert command[command.index("--per_device_eval_batch_size") + 1] == "2"
    assert command[command.index("--resume_from_checkpoint") + 1] == "auto"
    assert "" not in command
    assert config["variant"] == "focus_only"
```

增加对称的 G2 测试，并验证默认 G3 仍同时传入两个开关。

- [ ] **Step 2: 运行测试并确认按预期失败**

Run:

```text
J:\conda\envs\c3da\python.exe -m pytest test_element_aware_component_ablation.py -q
```

Expected: import error，当前运行器尚无 `build_train_args` 和 `resolve_variant`。

- [ ] **Step 3: 实现版本解析与冻结批次校验**

在运行器中加入：

```python
TASK_ID = "M1_ELEMENT_AWARE_RGAT_COMPONENT_ABLATION_V1"
FROZEN_TRAIN_BATCH_SIZE = 1
FROZEN_GRADIENT_ACCUMULATION_STEPS = 16
FROZEN_EVAL_BATCH_SIZE = 2


def resolve_variant(args: argparse.Namespace) -> dict:
    if args.focus_only:
        return {"name": "focus_only", "focus_enabled": True, "coverage_enabled": False}
    if args.coverage_only:
        return {"name": "coverage_only", "focus_enabled": False, "coverage_enabled": True}
    return {"name": "focus_plus_coverage", "focus_enabled": True, "coverage_enabled": True}


def validate_frozen_training_recipe(args: argparse.Namespace) -> None:
    if args.train_batch_size != FROZEN_TRAIN_BATCH_SIZE:
        raise ValueError("formal component ablations require V9e train_batch_size=1")
    if args.gradient_accumulation_steps != FROZEN_GRADIENT_ACCUMULATION_STEPS:
        raise ValueError("formal component ablations require V9e gradient_accumulation_steps=16")
```

- [ ] **Step 4: 实现无空参数的命令构造**

将现有内联列表移入 `build_train_args`。基础参数固定使用 `micro=1`、`eval=2`、`accumulation=16`、`DANN=0` 和 `resume_from_checkpoint=auto`，再独立追加：

```python
if variant["focus_enabled"]:
    train_args.append("--element_focus_loss")
if variant["coverage_enabled"]:
    train_args.append("--multi_element_coverage_loss")
train_args.extend(
    [
        "--element_focus_weight",
        "0.05" if variant["focus_enabled"] else "0",
        "--element_coverage_weight",
        "0.05" if variant["coverage_enabled"] else "0",
    ]
)
```

返回命令和包含实际/有效权重、批次及 DANN 的配置字典。

- [ ] **Step 5: 运行测试并确认通过**

Run:

```text
J:\conda\envs\c3da\python.exe -m pytest test_element_aware_component_ablation.py -q
```

Expected: G1、G2、G3、空参数和冻结批次测试全部 PASS。

- [ ] **Step 6: 提交**

```text
git add m1_element_aware_rgat_treatment_only.py test_element_aware_component_ablation.py
git commit -m "fix:freeze-v9e-component-ablation-entry"
```

### Task 3：运行身份、同目录恢复与边界记录

**Files:**
- Modify: `m1_element_aware_rgat_treatment_only.py`
- Test: `test_element_aware_component_ablation.py`

- [ ] **Step 1: 写身份失败测试**

测试同一身份可恢复、版本或批次变化被拒绝：

```python
import json

import pytest

from m1_element_aware_rgat_treatment_only import ensure_run_identity


def test_run_identity_allows_exact_resume_and_rejects_variant_change(tmp_path):
    path = tmp_path / "component_ablation_identity.json"
    identity = {
        "task_id": "M1_ELEMENT_AWARE_RGAT_COMPONENT_ABLATION_V1",
        "variant": "focus_only",
        "train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "eval_batch_size": 2,
        "dann": 0.0,
        "git_commit": "abc",
    }
    ensure_run_identity(path, identity)
    ensure_run_identity(path, identity)
    changed = dict(identity, variant="coverage_only")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        ensure_run_identity(path, changed)
    assert json.loads(path.read_text(encoding="utf-8")) == identity
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Expected: import error，尚无 `ensure_run_identity`。

- [ ] **Step 3: 实现原子身份写入与严格比较**

复用 `reproducibility.write_json_atomic`。首次运行写入身份；文件存在时按完整结构比较，不一致立即抛出 `RuntimeError`，不得覆盖旧身份。

身份至少包含：任务编号、版本、开关、实际/有效权重、`micro/accumulation/effective/eval`、DANN、seed、模型路径、三份输入哈希、图缓存身份路径、解析器路径和 Git 提交。

- [ ] **Step 4: 更新最终结果边界**

将 `treatment_only_entry.json` 的任务编号改为组件消融任务，并写入：

```python
{
    "task": TASK_ID,
    "variant": variant["name"],
    "training": frozen_config,
    "target_test_accessed": False,
    "target_test_gold": False,
    "augmentation_started": False,
    "phase_b_started": False,
}
```

- [ ] **Step 5: 测试 target test 与阶段边界**

静态断言运行器命令中不包含 `target_test`、augmentation 或 Phase B；结果身份字段全部为 false。

- [ ] **Step 6: 运行组件测试并提交**

Run:

```text
J:\conda\envs\c3da\python.exe -m pytest test_element_aware_component_ablation.py -q
```

Expected: 全部 PASS。

Commit:

```text
git add m1_element_aware_rgat_treatment_only.py test_element_aware_component_ablation.py
git commit -m "feat:audit-component-ablation-resume-identity"
```

### Task 4：完整本地回归与协作文档

**Files:**
- Modify: `.ai/CURRENT_TASK.md`
- Modify: `CHAT_SOL_CURRENT_TASK_CN.md`
- Modify in formal root: `J:\nlp\CD-C3DA\.ai\CURRENT_TASK.md`
- Modify in formal root: `J:\nlp\CD-C3DA\CHAT_SOL_CURRENT_TASK_CN.md`

- [ ] **Step 1: 运行直接回归**

Run:

```text
J:\conda\envs\c3da\python.exe -m pytest test_element_aware_component_ablation.py test_element_aware_rgat.py -q
```

Expected: 两个测试文件全部 PASS。

- [ ] **Step 2: 运行语法和差异检查**

```text
J:\conda\envs\c3da\python.exe -m py_compile t5_absa_train.py m1_element_aware_rgat_treatment_only.py test_element_aware_component_ablation.py
git diff --check
```

Expected: 两条命令退出码均为0。

- [ ] **Step 3: 更新当前任务文档和上传镜像**

记录：方案 A、V9e `1×16` 配方、G1/G2 入口实现状态、测试证据、服务器尚未同步、GPU 实验未启动、唯一下一步为等待服务器当前实验结束后同步并运行 CPU 验证。

工作树中的 `.ai/CURRENT_TASK.md` 与 `CHAT_SOL_CURRENT_TASK_CN.md` 必须规范化文本一致；正式根目录同步相同文本，保留两边既有用户修改。

- [ ] **Step 4: 提交工作树文档与最终代码状态**

```text
git add .ai/CURRENT_TASK.md CHAT_SOL_CURRENT_TASK_CN.md
git commit -m "docs:record-component-ablation-entry"
git status --short --branch
```

- [ ] **Step 5: 保持服务器不变**

只读检查服务器训练进程。只要当前实验仍运行，就不得切换服务器分支、推送到服务器当前检出分支或修改 `/root/CD-C3DA` 工作目录。向用户报告本地提交身份和待同步状态。

# 当前代码原生最佳流程复现实施计划

> **For agentic workers（供代理执行者）：** REQUIRED SUB-SKILL（必须使用的子技能）：使用 `subagent-driven-development`（子代理驱动开发，推荐）或 `executing-plans`（执行计划），逐项实现并勾选步骤。

**Goal（目标）：** 在当前分支代码中建立不依赖历史工作树、不跨运行复用产物、自动保存完整命令与血缘信息的最佳流程正式入口，并从头复现 `rest16 -> laptop14` 的 raw F1（原始 F1）48.93 / fixed F1（修正 F1）50.21。

**Architecture（架构）：** 使用版本化 JSON（结构化数据）配方定义十阶段命令图；Python（编程语言）模块负责路径隔离、清单、命令、环境、哈希和恢复；PowerShell（微软命令行环境）仅提供 Windows 一行入口。历史提交和历史运行目录只用于审计测试，正式模式只能消费本次 `run_id`（运行编号）目录中的阶段产物。

**Tech Stack（技术栈）：** Python 3.10、PowerShell、PyTorch、Transformers、JSON/JSONL（结构化数据）、unittest（单元测试）、Git（版本管理）、CUDA（并行计算平台）。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `configs/recipes/rest16_to_laptop14_best_v1.json` | 不可变最佳配方、真实参数、黄金观察值和哈希 |
| `reproducibility.py` | 路径隔离、文件哈希、清单、环境快照、命令记录和恢复校验 |
| `run_reproducible_pipeline.py` | 读取配方、构建十阶段命令、执行阶段、校验产物和生成中文运行记录 |
| `run_best_reproducible_pipeline.ps1` | 用户从 CMD（命令提示符）启动的正式入口 |
| `t5_aste_augment.py` | 增加显式历史最佳兼容配置，不改变其他增强模式默认行为 |
| `t5_aste_pipeline.py` | 将兼容配置透传到增强请求构建，不增加历史路径回退 |
| `test_reproducible_recipe.py` | 配方结构、数量语义和不可变黄金值测试 |
| `test_reproducibility_provenance.py` | 来源隔离、清单、命令和断点恢复测试 |
| `test_native_best_runner.py` | 当前代码十阶段命令图与正式入口测试 |
| `test_historical_best_compatibility.py` | 当前代码兼容配置与历史行为审计测试 |
| `AGENTS.md` | 项目级强制入口，要求未来代理读取项目 Skill（技能） |
| `docs/skills/c3da-experiment-workflow/SKILL.md` | 分支、产物血缘、命令归档和主分支晋级规则 |
| `实验记录与模型索引_CN.md` | 当前最佳、运行索引、差距和下一步决策 |

## Task 1：建立不可变最佳配方

**Files（文件）：**
- Create（新建）：`configs/recipes/rest16_to_laptop14_best_v1.json`
- Create（新建）：`test_reproducible_recipe.py`

- [ ] **Step 1：先写配方失败测试**

```python
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RECIPE = ROOT / "configs" / "recipes" / "rest16_to_laptop14_best_v1.json"


class ReproducibleRecipeTest(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads(RECIPE.read_text(encoding="utf-8"))

    def test_observed_counts_never_become_selection_limits(self):
        golden = self.recipe["golden"]
        self.assertEqual(golden["base_pseudo"]["observed_golden_rows"], 421)
        self.assertNotIn("selection_limit", golden["base_pseudo"])
        self.assertEqual(golden["augment"]["selection_limit"], 150)
        self.assertEqual(golden["complete_pseudo"]["observed_golden_rows"], 494)
        self.assertNotIn("selection_limit", golden["complete_pseudo"])
        self.assertEqual(golden["final_train"]["observed_golden_rows"], 1499)

    def test_recipe_uses_only_raw_data_and_declared_models_as_external_inputs(self):
        text = RECIPE.read_text(encoding="utf-8").lower()
        self.assertNotIn(".worktrees", text)
        self.assertNotIn("reuse_upstream", text)
        self.assertNotIn("runs\\", text)
        self.assertEqual(self.recipe["source_dataset"], "rest16")
        self.assertEqual(self.recipe["target_dataset"], "laptop14")

    def test_golden_hashes_are_complete(self):
        required = {
            "extractor", "base_pseudo", "generator", "augment",
            "complete_pseudo", "final_train", "final_model", "predictions",
        }
        self.assertEqual(set(self.recipe["golden"]), required)
        for item in required - {"augment"}:
            self.assertRegex(self.recipe["golden"][item]["sha256"], r"^[A-F0-9]{64}$")
        self.assertRegex(self.recipe["golden"]["augment"]["semantic_sha256"], r"^[A-F0-9]{64}$")
```

- [ ] **Step 2：运行测试并确认因配方不存在而失败**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_reproducible_recipe.py"`

Expected（预期）：`FileNotFoundError`，指出 `rest16_to_laptop14_best_v1.json` 不存在。

- [ ] **Step 3：创建完整配方**

配方必须包含下面的确定内容；阶段参数以当前成功复现入口的十条命令为准，不允许引用历史目录：

```json
{
  "schema_version": 1,
  "recipe_id": "rest16_to_laptop14_best_v1",
  "source_dataset": "rest16",
  "target_dataset": "laptop14",
  "seed": 1000,
  "reproducibility_mode": "historical_seed_only",
  "models": {
    "t5_base": "J:\\nlp\\models\\t5-base-py",
    "nli": "J:\\nlp\\models\\nli-deberta-v3-base-mnli-fever-anli"
  },
  "external_inputs": {
    "rest16_train": {"path": "J:\\nlp\\BGCA-master\\data\\aste\\cross_domain\\rest16\\train.txt", "sha256": "6CBBD94268889E751422776F3558212650E633E3C39366CC33B5A5785FD310A4"},
    "rest16_dev": {"path": "J:\\nlp\\BGCA-master\\data\\aste\\cross_domain\\rest16\\dev.txt", "sha256": "C3A6BF12A879C947842964725A4E387BA85619BFC8E5ED8A889EBE87E9784594"},
    "rest16_test": {"path": "J:\\nlp\\BGCA-master\\data\\aste\\cross_domain\\rest16\\test.txt", "sha256": "DE842428DC8BC11A4EEE87B0BA382B0A59EF37B973A92829759CAA8664208630"},
    "laptop14_train": {"path": "J:\\nlp\\BGCA-master\\data\\aste\\cross_domain\\laptop14\\train.txt", "sha256": "512EA143794332967B346D76791BBA10E0C10C5D20D8E1FA8F6534C2069F9B7C"},
    "laptop14_dev": {"path": "J:\\nlp\\BGCA-master\\data\\aste\\cross_domain\\laptop14\\dev.txt", "sha256": "0ADBF20FC75284D5D49523F56EC368D94A9DA305E87411E0C627D12CD4510BC0"},
    "laptop14_test": {"path": "J:\\nlp\\BGCA-master\\data\\aste\\cross_domain\\laptop14\\test.txt", "sha256": "413A3F655409AF25BCB03A9499709925349FFF3EDBC3A7CA95FC6CDF788EEB92"},
    "t5_config": {"path": "J:\\nlp\\models\\t5-base-py\\config.json", "sha256": "46DD7CB62D29C81FB551E0EF1EA274C24A46BA441EEB948897706252933DF033"},
    "t5_weights": {"path": "J:\\nlp\\models\\t5-base-py\\pytorch_model.bin", "sha256": "AB97165968EDC4AACD30554D18D7BECA7F18B3A83E1A47ABBAD29792D984651F"},
    "t5_tokenizer": {"path": "J:\\nlp\\models\\t5-base-py\\spiece.model", "sha256": "D60ACB128CF7B7F2536E8F38A5B18A05535C9E14C7A355904270E15B0945EA86"},
    "nli_config": {"path": "J:\\nlp\\models\\nli-deberta-v3-base-mnli-fever-anli\\config.json", "sha256": "A6C616D6DABEACF90FD0E776C741D3F0F30A05533CCF0BD3B5B62E94CFAA8D57"},
    "nli_weights": {"path": "J:\\nlp\\models\\nli-deberta-v3-base-mnli-fever-anli\\model.safetensors", "sha256": "06D6FD89EDD4F97816831626DAAFBDB0B029CF63BAE8EDC0BCCAB1D64E2E7707"}
  },
  "training": {
    "train_batch_size": 1,
    "eval_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "learning_rate": 0.0003,
    "fp16": true,
    "gradient_checkpointing": true,
    "extractor_epochs": 25,
    "extractor_checkpoint_selection": "last",
    "generator_epochs": 8,
    "generator_checkpoint_selection": "best",
    "final_epochs": 5,
    "final_checkpoint_selection": "best"
  },
  "pseudo": {
    "num_beams": 1,
    "max_new_tokens": 128,
    "high_precision_max_triplets": 1,
    "high_precision_max_token_distance": 5
  },
  "augment": {
    "compatibility_profile": "historical_best_v1",
    "prompt_style": "masked_mutual",
    "channel_mode": "all",
    "domain_prefix_style": "text",
    "selection_limit": 150,
    "max_per_base": 1,
    "sample_weight": 0.2,
    "require_raw_exact": true,
    "require_model_filter_passed": true
  },
  "complete_multi": {"max_triplets": 2, "max_token_distance": 5, "extra_weight": 0.25},
  "final": {
    "pseudo_weight": 0.65,
    "augment_weight": 0.2,
    "lambda_domain_adv": 0.03,
    "lambda_sentiment_contrastive": 0.01,
    "sentiment_contrastive_source_only": true,
    "sentiment_contrastive_class_balanced": true
  },
  "golden": {
    "extractor": {"sha256": "6AD985A7D61274B6553C65B305BE18BBA8618B25B98742F0594C5336A3925F3E"},
    "base_pseudo": {"observed_golden_rows": 421, "sha256": "0536D99840054EE928B5FB746EC60326640C9A23C8A676A2A8D25DF3D8C15C84"},
    "generator": {"sha256": "0C93F7660E136862428AC23797339D0196047F8C2A1FADE8C99B7635F68CB1CE"},
    "augment": {"selection_limit": 150, "observed_golden_rows": 150, "semantic_sha256": "5A5B87707BFA6C2D6416AF7962C390207CF1FAC9AFEDD5B7B4799A4C4570B2FF"},
    "complete_pseudo": {"observed_golden_rows": 494, "sha256": "F3C6E0CF841FA84DD3F522248B3C0214B9FD1CC469A991FE853E7AFDE58AB710"},
    "final_train": {"observed_golden_rows": 1499, "sha256": "4876753D495A284FCAA454004CB441099421415B0DEFC353C5EDDA2E2FF36A88"},
    "final_model": {"sha256": "FC8BC8A4736E5CF4A0575C6C52A9349E34363E01556CC5D3397FDF0029AFAB1F"},
    "predictions": {"observed_golden_rows": 328, "sha256": "66E34B17512690C94425E0D64626AF5E101158CB8F5F4DAA705C59D1E5B115A9"}
  },
  "metrics": {
    "raw": {"precision": 0.5831202046035806, "recall": 0.4214417744916821, "micro_f1": 0.4892703862660945, "tp": 228, "fp": 163, "fn": 313},
    "fixed": {"precision": 0.59846547314578, "recall": 0.43253234750462105, "micro_f1": 0.502145922746781, "tp": 234, "fp": 157, "fn": 307}
  }
}
```

同时保存六个原始数据文件以及 T5/NLI 模型关键文件哈希；这些哈希放在 `external_inputs`，不允许指向 `runs`。

- [ ] **Step 4：运行配方测试并确认通过**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_reproducible_recipe.py"`

Expected（预期）：3 项测试全部 `OK`。

- [ ] **Step 5：提交配方与测试**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && git add configs\recipes\rest16_to_laptop14_best_v1.json test_reproducible_recipe.py && git commit -m \"Add immutable best experiment recipe\""`

## Task 2：实现路径隔离和运行清单

**Files（文件）：**
- Create（新建）：`reproducibility.py`
- Create（新建）：`test_reproducibility_provenance.py`

- [ ] **Step 1：编写路径和恢复失败测试**

```python
import json
import tempfile
import unittest
from pathlib import Path

from reproducibility import ReproducibilityError, RunContext


class ProvenanceTest(unittest.TestCase):
    def test_rejects_artifact_outside_run_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            context = RunContext.create(root, "run-001", "recipe-v1", "abc123", "feature/test")
            with self.assertRaisesRegex(ReproducibilityError, "outside current run root"):
                context.require_internal_artifact(Path(temp) / "other" / "pseudo.jsonl")

    def test_existing_directory_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            with self.assertRaisesRegex(ReproducibilityError, "manifest.json"):
                RunContext.open_or_create(root, "run-001", "recipe-v1", "abc123", "feature/test")

    def test_matching_manifest_resumes_and_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            first = RunContext.open_or_create(root, "run-001", "recipe-v1", "abc123", "feature/test")
            resumed = RunContext.open_or_create(root, "run-001", "recipe-v1", "abc123", "feature/test")
            self.assertEqual(resumed.manifest["resume_count"], 1)
            with self.assertRaisesRegex(ReproducibilityError, "git_commit"):
                RunContext.open_or_create(root, "run-001", "recipe-v1", "different", "feature/test")
```

- [ ] **Step 2：运行并确认导入失败**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_reproducibility_provenance.py"`

Expected（预期）：`ModuleNotFoundError: reproducibility`。

- [ ] **Step 3：实现最小清单接口**

`reproducibility.py` 定义以下稳定接口和身份校验逻辑：

```python
class ReproducibilityError(RuntimeError):
    pass


@dataclass
class RunContext:
    run_root: Path
    manifest_path: Path
    manifest: dict

    @classmethod
    def open_or_create(
        cls, run_root: Path, run_id: str, recipe_id: str, git_commit: str, git_branch: str
    ) -> "RunContext":
        identity = {
            "run_id": run_id,
            "recipe_id": recipe_id,
            "git_commit": git_commit,
            "git_branch": git_branch,
        }
        manifest_path = run_root / "manifest.json"
        if run_root.exists() and not manifest_path.exists() and any(run_root.iterdir()):
            raise ReproducibilityError(f"existing run directory has no manifest.json: {run_root}")
        run_root.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, expected in identity.items():
                if manifest.get(key) != expected:
                    raise ReproducibilityError(f"{key} mismatch: {manifest.get(key)} != {expected}")
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        else:
            manifest = {**identity, "resume_count": 0, "stages": {}, "artifacts": {}}
        write_json_atomic(manifest_path, manifest)
        return cls(run_root=run_root.resolve(), manifest_path=manifest_path, manifest=manifest)

    def require_internal_artifact(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.run_root):
            raise ReproducibilityError(f"artifact is outside current run root: {resolved}")
        return resolved

    def record_artifact(
        self, stage: str, path: Path, input_hashes: dict[str, str], semantic_hash: str = ""
    ) -> dict:
        resolved = self.require_internal_artifact(path)
        record = {
            "producer_stage": stage,
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "input_hashes": dict(input_hashes),
            "semantic_sha256": semantic_hash,
        }
        self.manifest["artifacts"][str(resolved)] = record
        write_json_atomic(self.manifest_path, self.manifest)
        return record
```

`mark_stage_complete` 对每个输出调用 `record_artifact`，再把状态写为 `completed`；`validate_completed_stage` 重新计算已记录输出的 SHA256，全部一致才返回 `True`，缺失或不一致直接抛出 `ReproducibilityError`。初始清单使用临时文件加 `os.replace()` 原子写入。

- [ ] **Step 4：增加损坏产物不能跳过的测试并实现 SHA256**

```python
def test_completed_stage_with_changed_hash_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "run"
        context = RunContext.open_or_create(root, "run-001", "recipe-v1", "abc123", "feature/test")
        output = root / "target_pseudo.jsonl"
        output.write_text("first", encoding="utf-8")
        context.mark_stage_complete("pseudo", [output])
        output.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ReproducibilityError, "hash mismatch"):
            context.validate_completed_stage("pseudo", [output])
```

实现 `sha256_file(path: Path) -> str`、`count_jsonl_rows(path: Path) -> int` 和原子 JSON 写入。

- [ ] **Step 5：运行测试并提交**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_reproducibility_provenance.py && git add reproducibility.py test_reproducibility_provenance.py && git commit -m \"Add strict run provenance manifests\""`

## Task 3：实现命令和环境自动归档

**Files（文件）：**
- Modify（修改）：`reproducibility.py`
- Modify（修改）：`test_reproducibility_provenance.py`

- [ ] **Step 1：写命令失败也必须保留的测试**

```python
def test_failed_command_is_recorded_before_and_after_execution(self):
    with tempfile.TemporaryDirectory() as temp:
        context = RunContext.open_or_create(
            Path(temp) / "run", "run-001", "recipe-v1", "abc123", "feature/test"
        )
        with self.assertRaises(subprocess.CalledProcessError):
            context.run_command("failing", [sys.executable, "-c", "raise SystemExit(7)"])
        records = [
            json.loads(line)
            for line in context.commands_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(records[-1]["stage"], "failing")
        self.assertEqual(records[-1]["exit_code"], 7)
        self.assertTrue(records[-1]["started_at"])
        self.assertTrue(records[-1]["finished_at"])
```

- [ ] **Step 2：运行并确认 `run_command` 不存在**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_reproducibility_provenance.ProvenanceTest.test_failed_command_is_recorded_before_and_after_execution"`

- [ ] **Step 3：实现命令日志和环境快照**

新增 `run_command(stage, argv, cwd=None, dry_run=False)`、`write_user_command(command)`、`capture_environment(python_executable, model_paths)` 和 `render_run_record_cn()`。每个命令先写 `status="running"` 记录，完成后追加同一 `command_id` 的 `status="completed"` 或 `status="failed"` 记录。标准输出通过逐行读取 `subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)` 实时显示并同步写入 `logs/<stage>.log`。环境快照记录 Python、Conda、PyTorch、Transformers、Accelerate、CUDA、cuDNN、GPU、驱动、模型哈希、随机环境变量和 `pip freeze`。

文件名固定为：用户命令写入 `run_command.cmd`，阶段命令事件写入 `commands.jsonl`，环境写入 `environment.json`，阶段完成和失败状态写入 `stage_status.json`，中文记录写入 `RUN_RECORD_CN.md`。这些文件都位于本次 `run_root`，不得写入仓库根目录或其他运行目录。

- [ ] **Step 4：测试 `run_command.cmd` 与中文运行记录包含完整命令**

```python
def test_user_and_stage_commands_are_persisted(self):
    with tempfile.TemporaryDirectory() as temp:
        context = RunContext.open_or_create(
            Path(temp) / "run", "run-001", "recipe-v1", "abc123", "feature/test"
        )
        context.write_user_command('cmd /c "python run.py --seed 1000"')
        context.render_run_record_cn()
        self.assertIn("--seed 1000", context.run_command_path.read_text(encoding="utf-8"))
        self.assertIn("完整运行命令", context.run_record_path.read_text(encoding="utf-8"))
```

- [ ] **Step 5：运行测试并提交**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_reproducibility_provenance.py && git add reproducibility.py test_reproducibility_provenance.py && git commit -m \"Archive commands and training environments\""`

## Task 4：构建当前代码十阶段命令图

**Files（文件）：**
- Create（新建）：`run_reproducible_pipeline.py`
- Create（新建）：`test_native_best_runner.py`

- [ ] **Step 1：写正式入口不含历史依赖的失败测试**

```python
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"J:\conda\envs\c3da\python.exe")


class NativeBestRunnerTest(unittest.TestCase):
    def test_dry_run_uses_current_repository_for_all_ten_stages(self):
        result = subprocess.run(
            [str(PYTHON), "run_reproducible_pipeline.py", "--recipe", "configs/recipes/rest16_to_laptop14_best_v1.json", "--run_id", "dry-run-test", "--output_root", "runs/reproducible_test", "--cuda", "0", "--dry_run"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        output = result.stdout.lower()
        self.assertEqual(output.count("[native-repro] start"), 10)
        self.assertNotIn(".worktrees", output)
        self.assertNotIn("reuse_upstream", output)
        self.assertNotIn("9e78904", output)
        self.assertNotIn("8c7f6b4", output)
        self.assertIn(str(ROOT / "t5_aste_pipeline.py").lower(), output)
        self.assertIn(str(ROOT / "t5_absa_train.py").lower(), output)
```

- [ ] **Step 2：运行并确认脚本不存在**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_native_best_runner.py"`

- [ ] **Step 3：实现配方加载和十阶段命令**

定义：

```python
@dataclass(frozen=True)
class Stage:
    name: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...]
    golden_key: str = ""


def load_recipe(path: Path) -> dict:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "recipe_id", "source_dataset", "target_dataset", "models", "training"}
    missing = sorted(required - set(recipe))
    if missing:
        raise ValueError(f"recipe missing required keys: {missing}")
    return recipe


def validate_git_state(project_root: Path, allow_dirty: bool) -> tuple[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=project_root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True).strip()
    if dirty and not allow_dirty:
        raise ReproducibilityError("formal run requires a clean git worktree")
    return commit, branch
```

`build_best_v1_stages` 必须显式构造并返回十个 `Stage`：`prepare`、`extractor`、`pseudo`、`generator`、`augment`、`prepare_final`、`complete_multi2`、`build_final_train`、`final_train`、`evaluate`。每个 `argv` 逐项复制已验证历史入口的参数，只把脚本和输出路径替换为当前项目根目录和本次 `run_root`。所有脚本路径取 `Path(__file__).resolve().parent`，所有中间文件位于同一个 `run_root`；`prepare_final` 再次调用当前 `prepare`，输出放在本次目录的 `final_data` 子目录。

- [ ] **Step 4：加入参数精确测试**

测试三个训练命令只传 `--seed 1000`，不传 `--deterministic` 或 `--legacy_stochastic`；检查提取器 25 轮 last、生成器 8 轮 best、增强 150 上限、完整双三元组 0.25、最终伪标签 0.65、DANN 0.03、情感对比 0.01 和最终训练 5 轮。

- [ ] **Step 5：运行测试并提交**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_native_best_runner.py test_reproducible_recipe.py test_reproducibility_provenance.py && git add run_reproducible_pipeline.py test_native_best_runner.py && git commit -m \"Build native best pipeline command graph\""`

## Task 5：加入黄金观察校验且禁止影响筛选

**Files（文件）：**
- Modify（修改）：`reproducibility.py`
- Modify（修改）：`run_reproducible_pipeline.py`
- Modify（修改）：`test_reproducibility_provenance.py`
- Modify（修改）：`test_native_best_runner.py`

- [ ] **Step 1：写“只比较、不裁剪”的失败测试**

```python
def test_observed_golden_rows_never_changes_actual_rows(self):
    rows = [{"id": str(index)} for index in range(3)]
    result = compare_observed_rows("pseudo", rows, observed_golden_rows=2)
    self.assertEqual(len(rows), 3)
    self.assertEqual(result["actual_rows"], 3)
    self.assertFalse(result["matched"])

def test_selection_limit_is_applied_only_when_recipe_declares_it(self):
    self.assertEqual(apply_selection_limit([1, 2, 3], 2), [1, 2])
    self.assertEqual(apply_selection_limit([1, 2, 3], None), [1, 2, 3])
```

- [ ] **Step 2：实现哈希和指标校验器**

```python
def compare_observed_rows(stage: str, rows: Sequence[dict], observed_golden_rows: int) -> dict:
    actual_rows = len(rows)
    return {
        "stage": stage,
        "actual_rows": actual_rows,
        "observed_golden_rows": observed_golden_rows,
        "matched": actual_rows == observed_golden_rows,
    }


def semantic_text_label_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for row in read_jsonl(path):
        payload = json.dumps(
            {"label": row["label"], "text": row["text"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update((payload + "\n").encode("utf-8"))
    return digest.hexdigest().upper()


def validate_metrics(actual: dict, expected: dict, tolerance: float = 1e-12) -> None:
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            if abs(float(actual_value) - expected_value) > tolerance:
                raise GoldenMismatchError(f"metric mismatch for {key}: {actual_value} != {expected_value}")
        elif actual_value != expected_value:
            raise GoldenMismatchError(f"metric mismatch for {key}: {actual_value} != {expected_value}")
```

`validate_golden_artifact` 根据 `Stage.golden_key` 选择普通文件哈希、JSONL 行数、增强语义哈希或最终指标校验。增强规范化对象固定为 `{"label": "<label>", "text": "<text>"}` 的两个字段。发现偏差时先写入清单与中文记录，再抛出 `GoldenMismatchError`；绝不修改源文件。

- [ ] **Step 3：加入其他数据集不继承黄金数量测试**

构造没有 `golden` 字段的临时配方，确认阶段使用全部实际伪标签且不会访问 421、494、1499。

- [ ] **Step 4：运行测试并提交**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_reproducibility_provenance.py test_native_best_runner.py && git add reproducibility.py run_reproducible_pipeline.py test_reproducibility_provenance.py test_native_best_runner.py && git commit -m \"Validate golden outputs without forcing counts\""`

## Task 6：移植增强历史最佳兼容行为

**Files（文件）：**
- Modify（修改）：`t5_aste_augment.py`
- Modify（修改）：`t5_aste_pipeline.py`
- Create（新建）：`test_historical_best_compatibility.py`
- Create（新建）：`test_fixtures/historical_best_augment_cases.jsonl`

- [ ] **Step 1：从黄金运行提取最小审计夹具**

夹具只保存能够覆盖旧增强与当前默认增强分歧的输入行、伪标签记忆、随机种子和预期请求，不保存模型或作为训练输入使用。测试必须明确标记 `audit_fixture_only=true`。

- [ ] **Step 2：写默认行为不变、兼容行为精确的失败测试**

```python
def test_historical_profile_matches_audit_fixture_without_changing_default():
    fixture = load_fixture()
    default_rows = build_augmentation_requests(**fixture["inputs"])
    compat_rows = build_augmentation_requests(
        **fixture["inputs"], compatibility_profile="historical_best_v1"
    )
    self.assertEqual(project_requests(compat_rows), fixture["expected_historical_requests"])
    self.assertEqual(project_requests(default_rows), fixture["expected_current_default_requests"])
    self.assertNotEqual(project_requests(default_rows), project_requests(compat_rows))
```

- [ ] **Step 3：实现显式兼容配置**

给 `build_augmentation_requests` 增加默认值为空的 `compatibility_profile`。仅当值为 `historical_best_v1` 时，恢复 `9e78904` 中经过审计的候选顺序、随机调用顺序和旧字段生成顺序；其他模式继续执行当前逻辑。禁止导入历史文件、动态执行 `git show` 或复制整个旧模块。

在 `t5_aste_pipeline.py augment` 增加：

```python
p.add_argument("--compatibility_profile", choices=["", "historical_best_v1"], default="")
```

并把参数直接传给 `build_augmentation_requests`。正式最佳配方显式传 `historical_best_v1`，普通实验不传。

- [ ] **Step 4：运行增强相关回归测试**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_historical_best_compatibility.py test_masked_mutual_augment.py test_augment_quality_filters.py test_model_filter.py"`

Expected（预期）：兼容夹具通过，现有增强测试无回归。

- [ ] **Step 5：提交兼容配置**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && git add t5_aste_augment.py t5_aste_pipeline.py test_historical_best_compatibility.py test_fixtures\historical_best_augment_cases.jsonl && git commit -m \"Add explicit historical best augmentation profile\""`

## Task 7：实现正式 PowerShell 入口

**Files（文件）：**
- Create（新建）：`run_best_reproducible_pipeline.ps1`
- Modify（修改）：`test_native_best_runner.py`

- [ ] **Step 1：写入口失败测试**

测试脚本只调用当前 `run_reproducible_pipeline.py`，支持 `-RunId`、`-OutputRoot`、`-Cuda`、`-DryRun`、`-AllowDirtyDiagnostic`，默认模型参数适配 RTX 3070 8GB；脚本文本不得包含 `.worktrees`、历史提交号或复用参数。

- [ ] **Step 2：实现薄包装脚本**

核心调用必须等价于：

```powershell
& $Python $Runner --recipe $Recipe --run_id $RunId --output_root $OutputRoot --cuda $Cuda @OptionalArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

脚本在控制台显示当前阶段、日志路径和断点恢复状态；真实进度条继续由训练器和生成器输出。

- [ ] **Step 3：试运行并检查十阶段命令**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_best_reproducible_pipeline.ps1 -RunId native-best-dry-run -OutputRoot J:\nlp\CD-C3DA\runs\reproducible -Cuda 0 -DryRun"`

Expected（预期）：打印十个当前代码阶段；生成命令记录和清单；不启动训练；不出现历史工作树路径。

- [ ] **Step 4：提交入口**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && git add run_best_reproducible_pipeline.ps1 test_native_best_runner.py && git commit -m \"Add native reproducible pipeline entrypoint\""`

## Task 8：把规则写入项目 Skill 和 AGENTS.md

**Files（文件）：**
- Modify（修改）：`docs/skills/c3da-experiment-workflow/SKILL.md`
- Create（新建）：`AGENTS.md`
- Create（新建）：`test_project_reproducibility_policy.py`

- [ ] **Step 1：写项目规则失败测试**

```python
class ProjectPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        cls.skill = (ROOT / "docs/skills/c3da-experiment-workflow/SKILL.md").read_text(encoding="utf-8")

    def test_agents_requires_project_skill(self):
        self.assertIn("docs/skills/c3da-experiment-workflow/SKILL.md", self.agents)

    def test_skill_requires_new_branch_before_changes(self):
        self.assertIn("修改前创建新分支", self.skill)

    def test_skill_forbids_cross_run_artifact_reuse(self):
        self.assertIn("禁止跨运行复用或混合产物", self.skill)

    def test_skill_requires_full_command_and_hash_records(self):
        self.assertIn("完整训练命令", self.skill)
        self.assertIn("SHA256", self.skill)

    def test_skill_keeps_master_as_verified_best_only(self):
        self.assertIn("master", self.skill)
        self.assertIn("当前最佳", self.skill)
```

- [ ] **Step 2：运行并确认 `AGENTS.md` 不存在或规则不足**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_project_reproducibility_policy.py"`

- [ ] **Step 3：更新 Skill 并新增项目入口**

`AGENTS.md` 要求任何代理在本项目修改代码、给训练命令、恢复实验或分析结果前读取 `docs/skills/c3da-experiment-workflow/SKILL.md`。Skill 增加：新分支、主分支晋级门槛、正式运行禁止跨目录复用、命令和环境归档、黄金观察值不参与筛选、断点恢复只能使用同一 `run_id`、删除许可、中文文档整体维护。

- [ ] **Step 4：运行规则测试并提交**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_project_reproducibility_policy.py && git add AGENTS.md docs\skills\c3da-experiment-workflow\SKILL.md test_project_reproducibility_policy.py && git commit -m \"Enforce reproducible experiment project policy\""`

## Task 9：更新中文总览和本次黄金运行记录

**Files（文件）：**
- Modify（修改）：`实验记录与模型索引_CN.md`
- Create（新建）：`runs/historical_best_two_stage_v1/rest16_to_laptop14/RUN_RECORD_CN.md`（若 `runs` 被 Git 忽略，则只保存在实验目录，不提交）

- [ ] **Step 1：整体更新首页当前状态**

将本次结果标记为“历史代码边界下从头精确复现成功”，记录 421/150/494/1499 是黄金观察值而非固定配额，加入全部关键 SHA256、版本链、manifest 路径和完整命令记录路径。

- [ ] **Step 2：更新待改进项**

把原 P0 复现任务标记完成；新增当前 P0：完成当前代码原生迁移并通过同样哈希。保留中性召回、多三元组召回作为迁移完成后的模型改进方向。

- [ ] **Step 3：生成历史成功运行中文记录**

记录原始 CMD 单行命令、十阶段展开命令、环境、关键哈希、指标、开始结束时间和“未复用历史产物”的证据。

- [ ] **Step 4：检查文档并提交可跟踪文件**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && git diff --check && git add 实验记录与模型索引_CN.md && git commit -m \"Record exact full historical best reproduction\""`

## Task 10：全量自动化验证和分支推送

**Files（文件）：**
- Verify（验证）：全部上述文件

- [ ] **Step 1：运行新增测试**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest -v test_reproducible_recipe.py test_reproducibility_provenance.py test_native_best_runner.py test_historical_best_compatibility.py test_project_reproducibility_policy.py"`

- [ ] **Step 2：运行全项目测试**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && python -m unittest discover -v"`

- [ ] **Step 3：运行正式入口试运行**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_best_reproducible_pipeline.ps1 -RunId native-best-dry-run-v1 -OutputRoot J:\nlp\CD-C3DA\runs\reproducible -Cuda 0 -DryRun"`

- [ ] **Step 4：确认分支干净并推送候选分支**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && git status --short --branch && git push -u origin feature/native-best-reproduction-v1"`

Expected（预期）：所有测试通过，试运行十阶段正确，工作树干净，候选分支成功推送；此时仍不得合并 `master`。

## Task 11：RTX 3070 全流程验收

**Files（文件）：**
- Generate（生成）：`runs/reproducible/rest16_to_laptop14_best_v1/native-best-v1/**`
- Modify after result（结果出来后修改）：`实验记录与模型索引_CN.md`

- [ ] **Step 1：从 CMD 启动全新正式运行**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && conda activate c3da && powershell -NoProfile -ExecutionPolicy Bypass -File run_best_reproducible_pipeline.ps1 -RunId native-best-v1 -OutputRoot J:\nlp\CD-C3DA\runs\reproducible -Cuda 0"`

该命令必须实时显示进度。断电后执行同一条命令，只能恢复 `native-best-v1` 自己的检查点和清单。

- [ ] **Step 2：逐阶段验收**

每个阶段完成后自动核对哈希。421、494、1499 仅比较不干预；增强 150 按配方上限选择。首次偏差立即停止并报告，不运行后续昂贵阶段。

- [ ] **Step 3：最终验收**

必须同时满足：最终模型和预测哈希一致，raw 指标为 P=58.31/R=42.14/F1=48.93，fixed F1=50.21，TP/FP/FN 完全一致，运行记录包含所有命令、环境和输入输出哈希。

- [ ] **Step 4：结果更新与晋级决策**

通过时：整体更新中文总览，在候选分支提交结果文档并推送。随后向用户报告证据并请求明确合并许可；不得自动合并。

失败时：保持 `master` 不变，保存首个偏差点、命令、清单和日志；从候选分支创建 `fix/native-best-reproduction-vN`，不得在原分支直接堆叠未经设计的修复。

- [ ] **Step 5：获得用户许可后合并并打标签**

Run（运行）：`cmd /c "J: && cd /d J:\nlp\CD-C3DA && git switch master && git merge --no-ff feature/native-best-reproduction-v1 && git tag -a best-rest16-laptop14-v1 -m \"Reproducible rest16 to laptop14 best pipeline\" && git push origin master && git push origin best-rest16-laptop14-v1"`

只有用户确认、全流程通过并且中文文档已提交时才执行该命令。

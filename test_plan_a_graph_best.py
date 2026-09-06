import argparse
import json
import sys
import subprocess
from pathlib import Path

from run_plan_a_graph_best import build_adapter_manifest, required_adapter_paths


def test_adapter_manifest_maps_phase_a_treatment(tmp_path: Path):
    treatment = tmp_path / "phase_a" / "treatment"
    model = treatment / "models" / "extractor" / "best"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (treatment / "target_pseudo_selected.jsonl").write_text('{"text":"x","label":""}\n', encoding="utf-8")
    for name in ("source_train.jsonl", "source_dev.jsonl", "target_unlabeled.jsonl", "target_test.jsonl"):
        (treatment / name).write_text("{}\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    manifest = build_adapter_manifest(treatment, adapter, source="laptop14", target="rest15", seed=1000)
    assert manifest["target_test_access"] is False
    assert manifest["source"] == str(treatment.resolve())
    assert all(path.exists() for path in required_adapter_paths(adapter))
    assert all((adapter / name).is_file() for name in ("source_train.jsonl", "source_dev.jsonl", "target_unlabeled.jsonl", "target_test.jsonl"))
    state = json.loads((adapter / "target_pseudo_generation_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["target_test_access"] is False


def test_internal_dann_batch_sizes_are_forwarded_from_recipe(tmp_path: Path):
    from m1_syntactic_rgat_pseudo_quick_ablation import _read_json, _training_argv

    recipe_path = Path(__file__).parent / "configs" / "recipes" / "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann003_internal16_v1.json"
    recipe = _read_json(recipe_path)
    args = argparse.Namespace(
        recipe_data=recipe,
        model_path="model",
        cuda="0",
        graph_cache_dir=tmp_path / "cache",
        parser_dir=tmp_path / "parser",
    )
    argv = _training_argv(args, tmp_path / "treatment", graph_enabled=True)
    assert argv[argv.index("--dann_source_batch_size") + 1] == "16"
    assert argv[argv.index("--dann_target_batch_size") + 1] == "16"


def test_dann_zero_coverage_recipe_does_not_forward_domain_batching(tmp_path: Path):
    from m1_syntactic_rgat_pseudo_quick_ablation import _read_json, _training_argv

    recipe_path = Path(__file__).parent / "configs" / "recipes" / "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann0_16x2_coverage_v1.json"
    recipe = _read_json(recipe_path)
    args = argparse.Namespace(
        recipe_data=recipe,
        model_path="model",
        cuda="0",
        graph_cache_dir=tmp_path / "cache",
        parser_dir=tmp_path / "parser",
    )
    argv = _training_argv(args, tmp_path / "treatment", graph_enabled=True)
    assert "--paired_domain_batches" not in argv
    assert "--dann_source_batch_size" not in argv
    assert "--dann_target_batch_size" not in argv


def test_dann_zero_training_does_not_forward_batch_audit_path(tmp_path: Path):
    from m1_syntactic_rgat_pseudo_quick_ablation import _read_json, _training_argv

    recipe_path = Path(__file__).parent / "configs" / "recipes" / "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann0_16x2_coverage_v1.json"
    recipe = _read_json(recipe_path)
    args = argparse.Namespace(
        recipe_data=recipe,
        cuda="0",
        model_path=tmp_path / "model",
        graph_cache_dir=tmp_path / "graph-cache",
        parser_dir=tmp_path / "parser",
    )
    argv = _training_argv(args, tmp_path / "treatment", graph_enabled=True)
    assert "--dann_batch_audit_path" not in argv


def test_plan_a_dry_run_forwards_graph_variant(tmp_path: Path):
    from run_plan_a_graph_best import main

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "run_plan_a_graph_best.py"),
            "--phase_a_output", str(tmp_path / "phase_a"),
            "--full_output_root", str(tmp_path / "full"),
            "--recipe", str(Path(__file__).parent / "configs" / "recipes" / "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann0_v1.json"),
            "--graph_cache_dir", str(tmp_path / "cache"),
            "--model_path", "models/t5-base-py",
            "--variant", "G2",
            "--dry_run",
        ],
        cwd=Path(__file__).parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--variant G2" in result.stdout


def test_plan_a_dry_run_accepts_g4(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "run_plan_a_graph_best.py"),
            "--phase_a_output", str(tmp_path / "phase_a"),
            "--full_output_root", str(tmp_path / "full"),
            "--recipe", str(Path(__file__).parent / "configs" / "recipes" / "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_v1.json"),
            "--graph_cache_dir", str(tmp_path / "cache"),
            "--model_path", "models/t5-base-py",
            "--variant", "G4",
            "--train_batch_size", "4",
            "--gradient_accumulation_steps", "4",
            "--dry_run",
        ],
        cwd=Path(__file__).parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--variant G4" in result.stdout
    assert "--extractor_train_batch_size 4" in result.stdout
    assert "--gradient_accumulation_steps 4" in result.stdout


def test_full_command_defaults_to_laptop14_rest15_and_forwards_4x4(tmp_path: Path):
    from run_plan_a_graph_best import build_full_command

    command = build_full_command(
        project_root=Path(__file__).parent,
        adapter=tmp_path / "adapter",
        output_root=tmp_path / "full",
        model_path="models/t5-base-py",
        train_batch_size=4,
        accumulation=4,
        cuda="0",
        seed=1000,
    )
    assert "--pairs" in command
    assert command[command.index("--pairs") + 1] == "laptop14:rest15"
    assert command[command.index("--train_batch_size") + 1] == "4"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "4"
    assert command[command.index("--final_lambda_domain_adv") + 1] == "0.0"


def test_no_dann_phase_a_can_keep_explicit_final_dann():
    from run_plan_a_graph_best import resolve_final_domain_adv

    assert resolve_final_domain_adv(no_dann=True, explicit=0.03, recipe_value=0.03) == 0.03
    assert resolve_final_domain_adv(no_dann=True, explicit=None, recipe_value=0.03) == 0.0


def test_explicit_final_dann_sweep_values_are_forwarded():
    from run_plan_a_graph_best import resolve_final_domain_adv

    assert [resolve_final_domain_adv(no_dann=True, explicit=value, recipe_value=0.03) for value in (0.01, 0.02, 0.03)] == [0.01, 0.02, 0.03]

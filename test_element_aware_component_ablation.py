import json
import re
from argparse import Namespace
from pathlib import Path

import pytest

from m1_element_aware_rgat_treatment_only import (
    build_train_args,
    build_result_record,
    build_pseudo_args,
    build_serialized_input_hashes,
    ensure_run_identity,
    resolve_variant,
    validate_component_attribution_variant,
    validate_frozen_training_recipe,
    validate_base_model_path,
)
from t5_absa_train import validate_element_aware_training_configuration


def _validate(**overrides):
    values = {
        "element_aware_attention": True,
        "use_syntactic_graph_adapter": True,
        "focus_enabled": True,
        "coverage_enabled": True,
        "focus_weight": 0.05,
        "coverage_weight": 0.05,
        "lambda_domain_adv": 0.0,
    }
    values.update(overrides)
    validate_element_aware_training_configuration(**values)


def test_focus_only_accepts_only_focus_weight():
    _validate(
        focus_enabled=True,
        coverage_enabled=False,
        focus_weight=0.05,
        coverage_weight=0.0,
    )


def test_coverage_only_accepts_only_coverage_weight():
    _validate(
        focus_enabled=False,
        coverage_enabled=True,
        focus_weight=0.0,
        coverage_weight=0.05,
    )


def test_focus_plus_coverage_accepts_both_frozen_weights():
    _validate()


def test_enabled_and_disabled_loss_weights_must_match_flags():
    with pytest.raises(ValueError, match="weights must match enabled losses"):
        _validate(
            focus_enabled=True,
            coverage_enabled=False,
            focus_weight=0.05,
            coverage_weight=0.05,
        )


def test_element_aware_attention_rejects_nonzero_dann():
    with pytest.raises(ValueError, match="lambda_domain_adv=0"):
        _validate(lambda_domain_adv=0.03)


def test_auxiliary_losses_require_element_aware_attention():
    with pytest.raises(ValueError, match="require --element_aware_attention"):
        _validate(element_aware_attention=False)


def test_element_aware_attention_requires_graph_adapter():
    with pytest.raises(ValueError, match="requires the syntactic graph adapter"):
        _validate(use_syntactic_graph_adapter=False)


def test_legacy_non_element_aware_configuration_is_unchanged():
    _validate(
        element_aware_attention=False,
        use_syntactic_graph_adapter=False,
        focus_enabled=False,
        coverage_enabled=False,
        focus_weight=0.05,
        coverage_weight=0.05,
        lambda_domain_adv=0.03,
    )


def _runner_args(**overrides):
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


def _assert_frozen_train_args(command):
    assert command[command.index("--per_device_train_batch_size") + 1] == "1"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "16"
    assert command[command.index("--per_device_eval_batch_size") + 1] == "2"
    assert command[command.index("--resume_from_checkpoint") + 1] == "auto"
    assert command[command.index("--lambda_domain_adv") + 1] == "0"
    assert "" not in command


def test_focus_only_train_args_are_exact_v9e_component_ablation():
    args = _runner_args(focus_only=True)
    variant = resolve_variant(args)
    command, config = build_train_args(args, Path("run"), variant)
    _assert_frozen_train_args(command)
    assert "--element_focus_loss" in command
    assert "--multi_element_coverage_loss" not in command
    assert command[command.index("--element_focus_weight") + 1] == "0.05"
    assert command[command.index("--element_coverage_weight") + 1] == "0"
    assert config["variant"] == "focus_only"
    assert config["effective_batch_size"] == 16


def test_coverage_only_train_args_are_exact_v9e_component_ablation():
    args = _runner_args(coverage_only=True)
    variant = resolve_variant(args)
    command, config = build_train_args(args, Path("run"), variant)
    _assert_frozen_train_args(command)
    assert "--element_focus_loss" not in command
    assert "--multi_element_coverage_loss" in command
    assert command[command.index("--element_focus_weight") + 1] == "0"
    assert command[command.index("--element_coverage_weight") + 1] == "0.05"
    assert config["variant"] == "coverage_only"


def test_pseudo_args_preserve_v9e_base_weight():
    args = _runner_args(focus_only=True)
    command = build_pseudo_args(args, Path("run"), Path("run/models/extractor/best"))
    assert command[command.index("--pseudo_base_weight") + 1] == "0.75"


def test_focus_plus_coverage_train_args_enable_both_losses():
    args = _runner_args()
    variant = resolve_variant(args)
    command, config = build_train_args(args, Path("run"), variant)
    _assert_frozen_train_args(command)
    assert "--element_focus_loss" in command
    assert "--multi_element_coverage_loss" in command
    assert config["variant"] == "focus_plus_coverage"


def test_component_attribution_runner_rejects_implicit_g3():
    with pytest.raises(ValueError, match="requires --focus_only or --coverage_only"):
        validate_component_attribution_variant(_runner_args())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_batch_size": 16}, "train_batch_size=1"),
        ({"gradient_accumulation_steps": 1}, "gradient_accumulation_steps=16"),
    ],
)
def test_formal_component_ablation_rejects_non_v9e_batch_recipe(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_frozen_training_recipe(_runner_args(**overrides))


def test_run_identity_allows_exact_resume_and_rejects_variant_change(tmp_path):
    path = tmp_path / "component_ablation_identity.json"
    identity = {
        "task_id": "M1_ELEMENT_AWARE_COMPONENT_ATTRIBUTION_V1",
        "variant": "focus_only",
        "training": {
            "train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "effective_batch_size": 16,
            "eval_batch_size": 2,
            "dann": 0.0,
        },
        "git_commit": "abc",
    }
    ensure_run_identity(path, identity)
    ensure_run_identity(path, identity)
    changed = dict(identity, variant="coverage_only")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        ensure_run_identity(path, changed)
    assert json.loads(path.read_text(encoding="utf-8")) == identity


def test_run_identity_rejects_existing_artifacts_without_identity(tmp_path):
    path = tmp_path / "component_ablation_identity.json"
    checkpoint = tmp_path / "models" / "extractor" / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="existing artifacts without identity"):
        ensure_run_identity(path, {"task_id": "component"})
    assert not path.exists()


def test_result_record_preserves_phase_a_data_boundary():
    variant = resolve_variant(_runner_args(focus_only=True))
    result = build_result_record(
        root=Path("run"),
        model=Path("run/models/extractor/best"),
        variant=variant,
        frozen_config={"variant": "focus_only", "focus_weight": 0.05, "coverage_weight": 0.0},
        identity_sha256="ABC",
    )
    assert result["task"] == "M1_ELEMENT_AWARE_COMPONENT_ATTRIBUTION_V1"
    assert result["variant"] == "focus_only"
    assert result["target_test_accessed"] is False
    assert result["target_test_gold"] is False
    assert result["augmentation_started"] is False
    assert result["phase_b_started"] is False


def test_result_record_aggregates_phase_a_metrics_without_target_test(tmp_path):
    variant = resolve_variant(_runner_args(focus_only=True))
    prediction_path = tmp_path / "aste_predictions_raw_fixed_element_aware_source_dev.jsonl"
    prediction_path.write_text(
        json.dumps({"gold": "<pos> a <opinion> op", "pred_raw": "<pos> a <opinion> op"})
        + "\n"
        + json.dumps({"gold": "<pos> a <opinion> op ; <pos> b <opinion> good", "pred_raw": "<pos> a <opinion> op"})
        + "\n",
        encoding="utf-8",
    )
    pseudo_path = tmp_path / "target_pseudo_selected.jsonl"
    pseudo_path.write_text(
        json.dumps({"label": "<pos> a <opinion> op"})
        + "\n"
        + json.dumps({"label": "<pos> a <opinion> op ; <pos> b <opinion> good ; <neg> c <opinion> bad"})
        + "\n",
        encoding="utf-8",
    )
    result = build_result_record(
        root=tmp_path,
        model=tmp_path / "models" / "extractor" / "best",
        variant=variant,
        frozen_config={"variant": "focus_only", "focus_weight": 0.05, "coverage_weight": 0.0},
        identity_sha256="ABC",
    )
    assert result["metrics"]["source_dev"]["strict_triplet_f1"] == pytest.approx(0.8)
    assert result["metrics"]["source_dev"]["multi_triplet_sentence_recall"] == pytest.approx(0.5)
    assert set(result["metrics"]["source_dev"]["absence_rates"]) == {"overall", "aspect", "opinion"}
    assert result["metrics"]["target_unlabeled_pseudo"]["qualified_total_rows"] == 2
    assert result["metrics"]["target_unlabeled_pseudo"]["qualified_multi_rows"] == 1
    assert result["metrics"]["target_unlabeled_pseudo"]["qualified_3plus_rows"] == 1
    assert result["mechanism_diagnostics"]["component"] == "focus_only"
    assert result["mechanism_diagnostics"]["focus_weight"] == 0.05
    assert result["mechanism_diagnostics"]["coverage_weight"] == 0.0
    assert result["mechanism_diagnostics"]["observed"]["qualified_3plus_rows"] == 1
    assert result["target_test_accessed"] is False
    assert result["target_test_gold"] is False


def test_result_record_reports_unavailable_phase_a_metrics_without_reading_target_test(tmp_path):
    variant = resolve_variant(_runner_args(coverage_only=True))
    result = build_result_record(
        root=tmp_path,
        model=tmp_path / "models" / "extractor" / "best",
        variant=variant,
        frozen_config={"variant": "coverage_only", "focus_weight": 0.0, "coverage_weight": 0.05},
        identity_sha256="ABC",
    )
    assert result["metrics"]["status"] == "unavailable"
    assert result["metrics"]["source_dev"]["status"] == "missing"
    assert result["metrics"]["target_unlabeled_pseudo"]["status"] == "missing"
    assert "target_test" not in result["metrics"]
    assert result["mechanism_diagnostics"]["component"] == "coverage_only"


def test_base_model_path_rejects_treatment_checkpoints():
    validate_base_model_path("J:/nlp/models/t5-base-py")
    with pytest.raises(ValueError, match="T5-base base model"):
        validate_base_model_path("run/models/extractor/checkpoint-1600")


def test_active_recipe_has_no_windows_absolute_paths():
    recipe = Path("configs/recipes/laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann0_v1.json").read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:[\\/]", recipe)
    assert "BGCA-master" not in recipe


def test_shared_phase_a_defaults_are_platform_neutral():
    source = Path("m1_syntactic_rgat_pseudo_quick_ablation.py").read_text(encoding="utf-8")
    assert 'Path(r"J:\\nlp\\models\\t5-base-py")' not in source
    assert 'default=r"J:\\nlp\\models\\t5-base-py"' not in source


def test_input_identity_is_computed_before_run_files_exist(tmp_path):
    root = tmp_path / "not-created"
    rows = {
        "source_train": [{"id": "s1", "input": "a", "target": "b"}],
        "source_dev": [{"id": "d1", "input": "c", "target": "d"}],
        "target_unlabeled": [{"id": "t1", "input": "e", "target": ""}],
    }
    identities = build_serialized_input_hashes(root, rows)
    assert not root.exists()
    assert identities["source_train"]["rows"] == 1
    assert len(identities["source_train"]["sha256"]) == 64

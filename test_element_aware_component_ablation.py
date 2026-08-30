from argparse import Namespace
from pathlib import Path

import pytest

from m1_element_aware_rgat_treatment_only import (
    build_train_args,
    resolve_variant,
    validate_frozen_training_recipe,
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


def test_focus_plus_coverage_train_args_enable_both_losses():
    args = _runner_args()
    variant = resolve_variant(args)
    command, config = build_train_args(args, Path("run"), variant)
    _assert_frozen_train_args(command)
    assert "--element_focus_loss" in command
    assert "--multi_element_coverage_loss" in command
    assert config["variant"] == "focus_plus_coverage"


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

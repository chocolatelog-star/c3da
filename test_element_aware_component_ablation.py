import pytest

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

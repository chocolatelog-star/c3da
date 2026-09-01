import pytest


def test_weighted_mean_uses_effective_sample_weight():
    from batch_gradient_parameter_audit import weighted_mean

    assert weighted_mean([2.0, 4.0], [1.0, 3.0]) == pytest.approx(3.5)


def test_weighted_mean_rejects_zero_total_weight():
    from batch_gradient_parameter_audit import weighted_mean

    with pytest.raises(ValueError, match="positive"):
        weighted_mean([2.0], [0.0])


def test_audit_matrix_has_four_comparison_groups():
    from batch_gradient_parameter_audit import default_audit_groups

    assert default_audit_groups() == [(1, 16), (4, 4), (8, 2), (16, 1)]

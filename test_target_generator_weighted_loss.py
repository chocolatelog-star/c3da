import pytest

from t5_absa_train_target_weighted import combine_source_target_loss


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75])
def test_target_weighted_loss_formula(weight):
    source = 2.0
    target = 3.0
    assert combine_source_target_loss(source, target, weight) == pytest.approx(source + weight * target)


def test_zero_target_weight_is_source_only():
    assert combine_source_target_loss(1.234, 99.0, 0.0) == pytest.approx(1.234)

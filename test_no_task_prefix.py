from t5_aste_data import to_extract_rows
from t5_aste_pipeline import build_extract_inputs


def test_to_extract_rows_keeps_prefix_by_default():
    rows = [{"text": "battery life is long", "label": "<pos> battery life <opinion> long"}]

    converted = to_extract_rows(rows)

    assert converted[0]["input"] == "extract aste: battery life is long"


def test_to_extract_rows_can_disable_prefix():
    rows = [{"text": "battery life is long", "label": "<pos> battery life <opinion> long"}]

    converted = to_extract_rows(rows, use_task_prefix=False)

    assert converted[0]["input"] == "battery life is long"


def test_build_extract_inputs_can_disable_prefix():
    rows = [{"text": "battery life is long"}]

    assert build_extract_inputs(rows, use_task_prefix=True) == ["extract aste: battery life is long"]
    assert build_extract_inputs(rows, use_task_prefix=False) == ["battery life is long"]

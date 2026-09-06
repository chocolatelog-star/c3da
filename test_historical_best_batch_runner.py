from pathlib import Path


def test_extract_result_keeps_only_raw_f1_fields(tmp_path: Path):
    from run_historical_best_batch_matrix import extract_result

    (tmp_path / "target_pseudo_analysis.json").write_text(
        '{"raw_pseudo_micro_f1_against_hidden_gold": {"micro_f1": 0.41}}', encoding="utf-8"
    )
    (tmp_path / "final_data").mkdir()
    (tmp_path / "final_data" / "aste_metrics_raw_target_test_4beam96.json").write_text(
        '{"micro_f1": 0.49}', encoding="utf-8"
    )
    result = extract_result(tmp_path)
    assert result["pseudo_raw_f1"] == 0.41
    assert result["final_raw_f1"] == 0.49


def test_skip_validation_flag_is_available():
    from run_reproducible_pipeline import parse_args

    assert "--skip_validation" in parse_args.__code__.co_consts

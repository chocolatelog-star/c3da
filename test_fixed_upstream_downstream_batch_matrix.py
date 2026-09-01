from pathlib import Path


def test_downstream_command_reuses_inputs(tmp_path: Path):
    from run_fixed_upstream_downstream_batch_matrix import build_downstream_command

    command = build_downstream_command(
        project_root=tmp_path,
        shared_final_train=tmp_path / "final_train.jsonl",
        shared_final_dev=tmp_path / "final_dev.jsonl",
        output_dir=tmp_path / "out",
        train_batch_size=16,
        accumulation=1,
        cuda="0",
    )
    assert command[1].endswith("t5_absa_train.py")
    assert command[command.index("--train_file") + 1] == str(tmp_path / "final_train.jsonl")
    assert command[command.index("--dev_file") + 1] == str(tmp_path / "final_dev.jsonl")
    assert command[command.index("--per_device_train_batch_size") + 1] == "16"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "1"
    assert command[command.index("--save_total_limit") + 1] == "1"


def test_shared_manifest_rejects_missing_inputs(tmp_path: Path):
    from run_fixed_upstream_downstream_batch_matrix import build_shared_manifest

    try:
        build_shared_manifest(tmp_path / "missing")
    except FileNotFoundError as exc:
        assert "missing shared upstream artifacts" in str(exc)
    else:
        raise AssertionError("missing shared inputs must be rejected")

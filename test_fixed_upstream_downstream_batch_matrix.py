from pathlib import Path
from types import SimpleNamespace


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


def test_run_unit_writes_terminal_status_without_duplicate_status(tmp_path: Path, monkeypatch):
    import run_fixed_upstream_downstream_batch_matrix as runner

    root = tmp_path / "matrix"
    root.mkdir()
    (root / "shared_upstream_manifest.json").write_text("{}", encoding="utf-8")
    paths = {}
    for name in ("final_train", "final_dev", "target_test"):
        paths[name] = tmp_path / f"{name}.jsonl"
        paths[name].write_text("{}\n", encoding="utf-8")
    status_calls = []

    monkeypatch.setattr(runner, "run_command", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        runner,
        "write_status",
        lambda path, status, **fields: status_calls.append((status, fields)),
    )

    result = runner.run_unit(
        SimpleNamespace(resume=False, project_root=tmp_path),
        root,
        paths,
        {},
        batch=8,
        accumulation=2,
        gpu="0",
    )

    assert result["status"] == "complete"
    assert status_calls[-1][0] == "complete"
    assert "status" not in status_calls[-1][1]


def test_shared_final_data_prefers_completed_final_data_directory(tmp_path: Path):
    from run_fixed_upstream_downstream_batch_matrix import discover_shared_artifacts

    required_files = (
        "models/extractor_ep25_plain_last/best/config.json",
        "target_pseudo_high_precision.jsonl",
        "models/generator_label_to_text_gen_ep8/best/config.json",
        "c3da_two_channel_augmented_selected.jsonl",
        "final_data/target_test.jsonl",
        "final_train_intermediate.jsonl",
        "final_dev_intermediate.jsonl",
        "final_data/final_train_complete.jsonl",
        "final_data/final_dev_complete.jsonl",
    )
    for relative_path in required_files:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    artifacts = discover_shared_artifacts(tmp_path)

    assert artifacts["final_train"].parent.name == "final_data"
    assert artifacts["final_dev"].parent.name == "final_data"

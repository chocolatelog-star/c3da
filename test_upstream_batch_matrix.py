from pathlib import Path


def test_upstream_commands_stop_at_pseudo(tmp_path: Path):
    from run_upstream_batch_matrix import build_upstream_command

    command = build_upstream_command(
        project_root=tmp_path,
        recipe="recipe.json",
        output_root=tmp_path / "out",
        run_id="b8",
        train_batch_size=8,
        gradient_accumulation_steps=2,
        cuda="0",
    )
    assert command[command.index("--stop_after_stage") + 1] == "pseudo"
    assert command[command.index("--train_batch_size") + 1] == "8"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "2"


def test_source_dev_command_uses_extractor_and_dev_file(tmp_path: Path):
    from run_upstream_batch_matrix import build_source_dev_command

    command = build_source_dev_command(tmp_path, tmp_path / "run", "0")
    assert command[2] == "evaluate"
    assert command[command.index("--eval_file") + 1].endswith("extract_dev.jsonl")
    assert command[command.index("--output_tag") + 1] == "source_dev"


def test_pipeline_run_dir_includes_recipe_and_run_id(tmp_path: Path):
    from run_upstream_batch_matrix import pipeline_run_dir

    assert pipeline_run_dir(tmp_path, "recipe_a", "batch8") == tmp_path / "recipe_a" / "batch8"

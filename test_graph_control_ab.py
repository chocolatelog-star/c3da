from pathlib import Path


def test_ab_specs_differ_only_by_graph_switch(tmp_path: Path):
    from run_graph_control_ab import build_ab_specs

    specs = build_ab_specs(tmp_path, train_batch_size=16, accumulation=1)
    assert [spec["name"] for spec in specs] == ["control", "graph"]
    assert specs[0]["graph_enabled"] is False
    assert specs[1]["graph_enabled"] is True
    common_control = {k: v for k, v in specs[0].items() if k not in {"name", "graph_enabled", "output_dir"}}
    common_graph = {k: v for k, v in specs[1].items() if k not in {"name", "graph_enabled", "output_dir"}}
    assert common_control == common_graph


def test_identity_comparison_allows_only_graph_fields():
    from run_graph_control_ab import compare_ab_identity

    control = {"seed": 1000, "batch": 16, "graph_enabled": False}
    graph = {"seed": 1000, "batch": 16, "graph_enabled": True}
    assert compare_ab_identity(control, graph)["matched"] is True
    graph["batch"] = 8
    result = compare_ab_identity(control, graph)
    assert result["matched"] is False
    assert "batch" in result["mismatches"]


def test_phase_a_cli_overrides_batch_without_mutating_input():
    from m1_syntactic_rgat_pseudo_quick_ablation import apply_training_overrides

    recipe = {"training": {"extractor_train_batch_size": 1, "gradient_accumulation_steps": 16}}
    resolved = apply_training_overrides(recipe, extractor_train_batch_size=16, gradient_accumulation_steps=1)
    assert resolved["training"]["extractor_train_batch_size"] == 16
    assert resolved["training"]["gradient_accumulation_steps"] == 1
    assert recipe["training"]["extractor_train_batch_size"] == 1


def test_complete_phase_a_artifacts_allow_downstream_after_gate_stop(tmp_path: Path):
    from run_graph_control_ab import phase_a_artifacts_complete

    for variant in ("control", "treatment"):
        model = tmp_path / variant / "models" / "extractor" / "best"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / variant / "target_pseudo_selected.jsonl").write_text("{}\n", encoding="utf-8")
    assert phase_a_artifacts_complete(tmp_path) is True


def test_full_command_uses_supported_stage1_arguments(tmp_path: Path):
    from run_plan_a_graph_best import build_full_command

    command = build_full_command(
        project_root=tmp_path,
        adapter=tmp_path / "adapter",
        output_root=tmp_path / "output",
        model_path="model",
        train_batch_size=16,
        accumulation=1,
        cuda="0",
        seed=1000,
    )
    assert "--lambda_domain_adv" not in command


def test_phase_a_allows_explicit_batch_override_only():
    from m1_syntactic_rgat_pseudo_quick_ablation import (
        _read_json,
        _validate_recipe,
        apply_training_overrides,
    )

    recipe_path = Path(__file__).parent / "configs" / "recipes" / "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann0_v1.json"
    recipe = _read_json(recipe_path)
    overridden = apply_training_overrides(recipe, extractor_train_batch_size=16, gradient_accumulation_steps=1)
    _validate_recipe(overridden, allow_batch_overrides=True)

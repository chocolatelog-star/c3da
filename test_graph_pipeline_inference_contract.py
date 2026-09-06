import argparse
import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parent


COMPOSITIONAL_GRAPH_KEYS = {
    "graph_compositional_dependency_id",
    "graph_compositional_direction_id",
    "graph_compositional_src_pos_id",
    "graph_compositional_dst_pos_id",
}


def _class_method_node(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method_name:
                    return member
    raise AssertionError(f"{class_name}.{method_name} was not found in {path}")


def test_compositional_graph_fields_are_forwarded_during_training_and_inference():
    trainer_source = (ROOT / "t5_absa_train_graph.py").read_text(encoding="utf-8")
    for key in COMPOSITIONAL_GRAPH_KEYS:
        assert f'"{key}",' in trainer_source

    forward = _class_method_node(
        ROOT / "syntactic_graph_adapter.py",
        "SyntacticGraphT5ForConditionalGeneration",
        "forward",
    )
    parameter_names = {argument.arg for argument in forward.args.args}
    assert COMPOSITIONAL_GRAPH_KEYS <= parameter_names

    forward_source = ast.unparse(forward)
    for key in COMPOSITIONAL_GRAPH_KEYS:
        assert key in forward_source


def test_graph_pseudo_generation_fails_fast_when_all_predictions_are_empty():
    pipeline_source = (ROOT / "t5_aste_pipeline.py").read_text(encoding="utf-8")
    assert "graph generation produced zero usable pseudo rows" in pipeline_source


def test_graph_phase_a_pipeline_forwards_cache_and_split_contract(tmp_path: Path):
    from m1_syntactic_rgat_pseudo_quick_ablation import _pipeline_argv, _read_json

    recipe = _read_json(
        ROOT
        / "configs"
        / "recipes"
        / "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann003_internal16_v1.json"
    )
    args = argparse.Namespace(
        recipe_data=recipe,
        cuda="0",
        graph_cache_dir=tmp_path / "graph-cache",
        parser_dir=tmp_path / "parser",
        model_path=tmp_path / "model",
    )

    evaluate_argv = _pipeline_argv(args, tmp_path / "treatment", "evaluate", True, tmp_path / "best", "source_dev")
    pseudo_argv = _pipeline_argv(args, tmp_path / "treatment", "pseudo", True, tmp_path / "best")

    for argv, expected_split in ((evaluate_argv, "source_dev"), (pseudo_argv, "target_unlabeled")):
        assert "--use_syntactic_graph_adapter" in argv
        assert argv[argv.index("--syntactic_graph_cache_dir") + 1] == str(args.graph_cache_dir)
        assert argv[argv.index("--syntactic_graph_parser_dir") + 1] == str(args.parser_dir)
        assert argv[argv.index("--syntactic_graph_split") + 1] == expected_split


def test_graph_inference_task_prefix_matches_recipe_setting(tmp_path: Path):
    from m1_syntactic_rgat_pseudo_quick_ablation import _pipeline_argv, _read_json

    recipe = _read_json(
        ROOT
        / "configs"
        / "recipes"
        / "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann003_internal16_v1.json"
    )
    args = argparse.Namespace(
        recipe_data=recipe,
        cuda="0",
        graph_cache_dir=tmp_path / "graph-cache",
        parser_dir=tmp_path / "parser",
        model_path=tmp_path / "model",
    )
    recipe["pseudo"]["use_task_prefix"] = True
    argv = _pipeline_argv(args, tmp_path / "treatment", "pseudo", True, tmp_path / "best")
    assert "--no_task_prefix" not in argv

    recipe["pseudo"]["use_task_prefix"] = False
    argv = _pipeline_argv(args, tmp_path / "treatment", "pseudo", True, tmp_path / "best")
    assert "--no_task_prefix" in argv


def test_graph_inference_uses_checkpoint_tokenizer_for_aste_labels():
    source = (ROOT / "t5_aste_pipeline.py").read_text(encoding="utf-8")
    assert "cache_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path" in source
    assert "tokenizer = AutoTokenizer.from_pretrained(model_path" in source
    assert "build_tokenizer_identity(tokenizer_path, cache_tokenizer)" in source


def test_graph_split_is_a_supported_evaluate_and_pseudo_argument():
    for command in ("evaluate", "pseudo"):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "t5_aste_pipeline.py"), command, "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--syntactic_graph_split" in completed.stdout


def test_graph_inference_preserves_focus_setting_saved_in_checkpoint():
    import syntactic_graph_adapter as adapter

    class Config:
        d_model = 768
        graph_focus_enabled = True

    config = Config()
    with patch("transformers.AutoConfig.from_pretrained", return_value=config), patch.object(
        adapter.SyntacticGraphT5ForConditionalGeneration,
        "from_pretrained",
        return_value=object(),
    ) as from_pretrained:
        adapter.load_seq2seq_model(
            "checkpoint",
            use_syntactic_graph_adapter=True,
            relation_vocab_size=8,
        )

    assert from_pretrained.call_args.kwargs["config"].graph_focus_enabled is True

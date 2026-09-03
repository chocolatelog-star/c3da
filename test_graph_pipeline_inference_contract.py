import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parent


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

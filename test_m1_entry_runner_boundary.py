import subprocess
import sys
from pathlib import Path

from t5_absa_train import enforce_graph_training_boundary


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def test_graph_recipe_rejects_training_without_explicit_audit_only():
    result = subprocess.run(
        [
            PYTHON,
            "run_bgca_aste_stage1_pairs.py",
            "--pairs",
            "laptop14:rest15",
            "--output_root",
            "runs\\m1_boundary_test",
            "--use_syntactic_graph_adapter",
            "--dry_run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "syntactic_graph_entry_audit_only" in (result.stdout + result.stderr)


def test_graph_audit_only_dry_run_has_no_training_or_downstream_commands():
    result = subprocess.run(
        [
            PYTHON,
            "run_bgca_aste_stage1_pairs.py",
            "--pairs",
            "laptop14:rest15",
            "--output_root",
            "runs\\m1_boundary_test",
            "--use_syntactic_graph_adapter",
            "--syntactic_graph_entry_audit_only",
            "--dry_run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "m1_syntactic_graph_entry_audit.py" in result.stdout
    assert "t5_absa_train.py" not in result.stdout
    assert "target_test" not in result.stdout
    assert "t5_aste_pipeline.py pseudo" not in result.stdout


def test_direct_graph_training_entry_is_hard_blocked_but_no_graph_is_unchanged():
    try:
        enforce_graph_training_boundary(True)
    except RuntimeError as exc:
        assert "m1_syntactic_graph_entry_audit.py" in str(exc)
    else:
        raise AssertionError("graph training entry unexpectedly remained executable")
    assert enforce_graph_training_boundary(False) is None


def test_direct_graph_training_cli_stops_before_loading_training_files():
    result = subprocess.run(
        [
            PYTHON,
            "t5_absa_train.py",
            "--train_file",
            "missing-source.jsonl",
            "--dev_file",
            "missing-dev.jsonl",
            "--output_dir",
            "missing-output",
            "--use_syntactic_graph_adapter",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "m1_syntactic_graph_entry_audit.py" in output
    assert "FileNotFoundError" not in output

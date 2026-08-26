import subprocess
import sys
from pathlib import Path


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

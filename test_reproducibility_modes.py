from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from t5_absa_train import reproducibility_training_args


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"J:\conda\envs\c3da\python.exe")


class HistoricalSeedBehaviorTest(unittest.TestCase):
    def test_historical_seed_behavior_only_passes_trainer_seed(self) -> None:
        self.assertEqual(reproducibility_training_args(1000, "legacy"), {"seed": 1000})

    def test_stage1_seed_only_reaches_all_training_stages(self) -> None:
        result = subprocess.run(
            [
                str(PYTHON),
                "run_bgca_aste_stage1_pairs.py",
                "--output_root",
                "runs/reproducibility_mode_test",
                "--pairs",
                "rest16:laptop14",
                "--complete_multi_extra_weight",
                "0.25",
                "--seed",
                "1000",
                "--dry_run",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        training_commands = [
            line for line in result.stdout.splitlines() if "t5_absa_train.py" in line
        ]
        self.assertEqual(len(training_commands), 3)
        self.assertTrue(all("--seed 1000" in line for line in training_commands))
        self.assertTrue(all("--legacy_stochastic" not in line for line in training_commands))
        self.assertTrue(all("--deterministic" not in line for line in training_commands))
        self.assertTrue(all("data_seed" not in line for line in training_commands))
        self.assertTrue(all("full_determinism" not in line for line in training_commands))

    def test_full_pipeline_script_uses_seed_without_reproducibility_modes(self) -> None:
        script = PROJECT_ROOT / "run_full_pipeline_legacy_stochastic.ps1"
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-DryRun",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("bgca_aste_stage1_full_pipeline_historical_seed_v2", result.stdout)
        self.assertIn("--seed 1000", result.stdout)
        self.assertNotIn("--legacy_stochastic", result.stdout)
        self.assertNotIn("--deterministic", result.stdout)

    def test_seed_sweep_script_does_not_enable_deterministic_training(self) -> None:
        script = (PROJECT_ROOT / "run_full_pipeline_seed_sweep.ps1").read_text(encoding="utf-8")
        self.assertNotIn('"--deterministic"', script)
        self.assertNotIn('"--legacy_stochastic"', script)


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"J:\conda\envs\c3da\python.exe")
RECIPE = ROOT / "configs" / "recipes" / "rest16_to_laptop14_best_v1.json"


class NativeBestRunnerTest(unittest.TestCase):
    def test_dry_run_uses_current_repository_for_all_ten_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [
                    str(PYTHON),
                    "run_reproducible_pipeline.py",
                    "--recipe",
                    str(RECIPE),
                    "--run_id",
                    "dry-run-test",
                    "--output_root",
                    str(Path(temp) / "runs"),
                    "--cuda",
                    "0",
                    "--dry_run",
                    "--allow_dirty",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        output = result.stdout.lower()
        self.assertEqual(output.count("[native-repro] start"), 10)
        self.assertNotIn(".worktrees", output.replace(str(ROOT).lower(), "<root>"))
        self.assertNotIn("reuse_upstream", output)
        self.assertNotIn("9e78904", output)
        self.assertNotIn("8c7f6b4", output)
        self.assertIn(str(ROOT / "t5_aste_pipeline.py").lower(), output)
        self.assertIn(str(ROOT / "t5_absa_train.py").lower(), output)

    def test_command_graph_preserves_historical_best_parameters(self):
        from run_reproducible_pipeline import build_best_v1_stages, load_recipe

        recipe = load_recipe(RECIPE)
        with tempfile.TemporaryDirectory() as temp:
            stages = build_best_v1_stages(
                ROOT, Path(temp) / "run", recipe, PYTHON, "0"
            )

        self.assertEqual(
            [stage.name for stage in stages],
            [
                "prepare",
                "extractor",
                "pseudo",
                "generator",
                "augment",
                "prepare_final",
                "complete_multi2",
                "build_final_train",
                "final_train",
                "evaluate",
            ],
        )
        commands = {stage.name: list(stage.argv) for stage in stages}
        for name in ("extractor", "generator", "final_train"):
            self.assertIn("--seed", commands[name])
            self.assertIn("1000", commands[name])
            self.assertNotIn("--deterministic", commands[name])
            self.assertNotIn("--legacy_stochastic", commands[name])

        self.assert_command_value(commands["extractor"], "--num_train_epochs", "25")
        self.assert_command_value(commands["extractor"], "--checkpoint_selection", "last")
        self.assert_command_value(commands["generator"], "--num_train_epochs", "8")
        self.assert_command_value(commands["generator"], "--checkpoint_selection", "best")
        self.assert_command_value(commands["augment"], "--augment_select_max_rows", "150")
        self.assert_command_value(
            commands["complete_multi2"], "--complete_multi_extra_weight", "0.25"
        )
        self.assert_command_value(commands["final_train"], "--pseudo_weight", "0.65")
        self.assert_command_value(commands["final_train"], "--lambda_domain_adv", "0.03")
        self.assert_command_value(
            commands["final_train"], "--lambda_sentiment_contrastive", "0.01"
        )
        self.assert_command_value(commands["final_train"], "--num_train_epochs", "5")

    def test_recipe_file_is_valid_json(self):
        self.assertEqual(json.loads(RECIPE.read_text(encoding="utf-8"))["schema_version"], 1)

    def assert_command_value(self, command, option, expected):
        index = command.index(option)
        self.assertEqual(command[index + 1], expected)


if __name__ == "__main__":
    unittest.main()

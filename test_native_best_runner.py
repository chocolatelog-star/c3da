import json
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"J:\conda\envs\c3da\python.exe")
RECIPE = ROOT / "configs" / "recipes" / "rest16_to_laptop14_best_v1.json"
POWERSHELL_ENTRYPOINT = ROOT / "run_best_reproducible_pipeline.ps1"


class NativeBestRunnerTest(unittest.TestCase):
    def test_powershell_entrypoint_is_current_code_only_and_supports_resume_options(self):
        script = POWERSHELL_ENTRYPOINT.read_text(encoding="utf-8")
        lowered = script.lower()
        for parameter in (
            "RunId",
            "OutputRoot",
            "Cuda",
            "DryRun",
            "AllowDirtyDiagnostic",
        ):
            self.assertIn(parameter, script)
        self.assertIn("run_reproducible_pipeline.py", script)
        self.assertIn("--user_command", script)
        self.assertNotIn(".worktrees", lowered)
        self.assertNotIn("reuse_upstream", lowered)
        self.assertNotIn("9e78904", lowered)
        self.assertNotIn("8c7f6b4", lowered)

    def test_powershell_entrypoint_dry_run_preserves_user_command_as_one_argument(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(POWERSHELL_ENTRYPOINT),
                    "-RunId",
                    "powershell-dry-run",
                    "-OutputRoot",
                    str(Path(temp) / "runs"),
                    "-Cuda",
                    "0",
                    "-DryRun",
                    "-AllowDirtyDiagnostic",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.stdout.lower().count("[native-repro] start"), 10)

    def test_internal_input_hashes_include_files_but_exclude_outputs(self):
        from run_reproducible_pipeline import Stage, collect_internal_input_hashes

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            output = root / "output.jsonl"
            source.write_text("source", encoding="utf-8")
            output.write_text("old", encoding="utf-8")
            stage = Stage(
                "build",
                ("python", "script.py", "--input", str(source), "--output", str(output)),
                (output,),
            )
            hashes = collect_internal_input_hashes(stage, root)
            self.assertIn(str(source.resolve()), hashes)
            self.assertNotIn(str(output.resolve()), hashes)

    def test_completed_stage_is_skipped_only_after_hash_validation(self):
        from reproducibility import RunContext
        from run_reproducible_pipeline import Stage, execute_stages

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            output = root / "output.txt"
            context = RunContext.open_or_create(
                root, "run-001", "recipe-v1", "abc123", "feature/test"
            )
            stage = Stage(
                "write",
                (
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(output)!r}).write_text('once')",
                ),
                (output,),
            )
            execute_stages([stage], context, {}, ROOT, dry_run=False)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                execute_stages([stage], context, {}, ROOT, dry_run=False)
            self.assertIn("SKIP", buffer.getvalue())
            self.assertEqual(output.read_text(encoding="utf-8"), "once")

    def test_golden_validation_never_modifies_rows_on_mismatch(self):
        from reproducibility import GoldenMismatchError, sha256_file
        from run_reproducible_pipeline import Stage, validate_golden_artifact

        with tempfile.TemporaryDirectory() as temp:
            pseudo = Path(temp) / "pseudo.jsonl"
            original = "".join(
                json.dumps({"id": str(index), "text": "x", "label": "y"}) + "\n"
                for index in range(3)
            )
            pseudo.write_text(original, encoding="utf-8")
            stage = Stage("pseudo", ("python",), (pseudo,), "base_pseudo")
            recipe = {
                "golden": {
                    "base_pseudo": {
                        "observed_golden_rows": 2,
                        "sha256": sha256_file(pseudo),
                    }
                }
            }
            with self.assertRaisesRegex(GoldenMismatchError, "observed rows"):
                validate_golden_artifact(stage, recipe)
            self.assertEqual(pseudo.read_text(encoding="utf-8"), original)

    def test_recipe_without_golden_skips_validation(self):
        from run_reproducible_pipeline import Stage, validate_golden_artifact

        stage = Stage("pseudo", ("python",), (Path("missing.jsonl"),), "base_pseudo")
        self.assertIsNone(validate_golden_artifact(stage, {"recipe_id": "other-pair"}))

    def test_external_input_hash_mismatch_stops_before_training(self):
        from reproducibility import ReproducibilityError
        from run_reproducible_pipeline import validate_external_inputs

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "train.txt"
            source.write_text("source", encoding="utf-8")
            recipe = {
                "external_inputs": {
                    "source_train": {"path": str(source), "sha256": "0" * 64}
                }
            }
            with self.assertRaisesRegex(ReproducibilityError, "source_train"):
                validate_external_inputs(recipe)

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
            commands["augment"], "--compatibility_profile", "historical_best_v1"
        )
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

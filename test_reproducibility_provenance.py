import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from reproducibility import ReproducibilityError, RunContext, sha256_file


class ProvenanceTest(unittest.TestCase):
    def test_environment_snapshot_records_runtime_and_model_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            context = RunContext.open_or_create(
                root, "run-001", "recipe-v1", "abc123", "feature/test"
            )
            model_file = root / "model.bin"
            model_file.write_bytes(b"model")
            snapshot = context.capture_environment(sys.executable, [model_file])
            self.assertEqual(snapshot["python"]["executable"], sys.executable)
            self.assertIn("torch", snapshot["packages"])
            self.assertIn("cuda", snapshot)
            self.assertIn("PYTHONHASHSEED", snapshot["random_environment"])
            self.assertEqual(
                snapshot["models"][str(model_file.resolve())],
                sha256_file(model_file),
            )
            self.assertTrue(context.environment_path.is_file())

    def test_user_and_stage_commands_are_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            context = RunContext.open_or_create(
                Path(temp) / "run",
                "run-001",
                "recipe-v1",
                "abc123",
                "feature/test",
            )
            context.write_user_command('cmd /c "python run.py --seed 1000"')
            context.render_run_record_cn()
            self.assertIn(
                "--seed 1000",
                context.run_command_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "完整运行命令",
                context.run_record_path.read_text(encoding="utf-8"),
            )

    def test_failed_command_is_recorded_before_and_after_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            context = RunContext.open_or_create(
                Path(temp) / "run",
                "run-001",
                "recipe-v1",
                "abc123",
                "feature/test",
            )
            with self.assertRaises(subprocess.CalledProcessError):
                context.run_command(
                    "failing", [sys.executable, "-c", "raise SystemExit(7)"]
                )
            records = [
                json.loads(line)
                for line in context.commands_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(records[-1]["stage"], "failing")
            self.assertEqual(records[-1]["exit_code"], 7)
            self.assertTrue(records[-1]["started_at"])
            self.assertTrue(records[-1]["finished_at"])
            stage_status = json.loads(
                context.stage_status_path.read_text(encoding="utf-8")
            )
            self.assertEqual(stage_status["failing"]["status"], "failed")
            self.assertEqual(stage_status["failing"]["exit_code"], 7)

    def test_rejects_artifact_outside_run_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            context = RunContext.create(
                root, "run-001", "recipe-v1", "abc123", "feature/test"
            )
            with self.assertRaisesRegex(ReproducibilityError, "outside current run root"):
                context.require_internal_artifact(Path(temp) / "other" / "pseudo.jsonl")

    def test_existing_directory_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            (root / "orphan.txt").write_text("orphan", encoding="utf-8")
            with self.assertRaisesRegex(ReproducibilityError, "manifest.json"):
                RunContext.open_or_create(
                    root, "run-001", "recipe-v1", "abc123", "feature/test"
                )

    def test_matching_manifest_resumes_and_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            RunContext.open_or_create(
                root, "run-001", "recipe-v1", "abc123", "feature/test"
            )
            resumed = RunContext.open_or_create(
                root, "run-001", "recipe-v1", "abc123", "feature/test"
            )
            self.assertEqual(resumed.manifest["resume_count"], 1)
            with self.assertRaisesRegex(ReproducibilityError, "git_commit"):
                RunContext.open_or_create(
                    root, "run-001", "recipe-v1", "different", "feature/test"
                )

    def test_completed_stage_with_changed_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            context = RunContext.open_or_create(
                root, "run-001", "recipe-v1", "abc123", "feature/test"
            )
            output = root / "target_pseudo.jsonl"
            output.write_text("first", encoding="utf-8")
            context.mark_stage_complete("pseudo", [output])
            output.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ReproducibilityError, "hash mismatch"):
                context.validate_completed_stage("pseudo", [output])


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import reproducibility
from reproducibility import (
    GoldenMismatchError,
    ReproducibilityError,
    RunContext,
    apply_selection_limit,
    compare_observed_rows,
    semantic_text_label_sha256,
    sha256_file,
    console_safe_text,
    validate_metrics,
)


class ProvenanceTest(unittest.TestCase):
    def test_console_text_replaces_characters_unsupported_by_windows_encoding(self):
        self.assertEqual(console_safe_text("progress \ufffd", "gbk"), "progress ?")

    def test_observed_golden_rows_never_changes_actual_rows(self):
        rows = [{"id": str(index)} for index in range(3)]
        result = compare_observed_rows("pseudo", rows, observed_golden_rows=2)
        self.assertEqual(len(rows), 3)
        self.assertEqual(result["actual_rows"], 3)
        self.assertFalse(result["matched"])

    def test_selection_limit_is_applied_only_when_recipe_declares_it(self):
        self.assertEqual(apply_selection_limit([1, 2, 3], 2), [1, 2])
        self.assertEqual(apply_selection_limit([1, 2, 3], None), [1, 2, 3])

    def test_semantic_hash_uses_only_text_and_label(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first.jsonl"
            second = Path(temp) / "second.jsonl"
            first.write_text(
                json.dumps({"text": "great food", "label": "<pos> food", "score": 1})
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"label": "<pos> food", "text": "great food", "score": 0})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                semantic_text_label_sha256(first), semantic_text_label_sha256(second)
            )

    def test_training_semantic_hash_ignores_non_training_audit_fields(self):
        self.assertTrue(
            hasattr(reproducibility, "semantic_training_rows_sha256"),
            "training semantic hash function is missing",
        )
        hash_training_rows = reproducibility.semantic_training_rows_sha256
        with tempfile.TemporaryDirectory() as temp:
            historical = Path(temp) / "historical.jsonl"
            current = Path(temp) / "current.jsonl"
            training_row = {
                "id": 7,
                "base_id": 3,
                "input": "great food",
                "target": "<pos> food <opinion> great",
                "sample_weight": 0.2,
                "augmentation": "masked_aspect_channel",
            }
            historical.write_text(
                json.dumps(training_row) + "\n", encoding="utf-8"
            )
            current.write_text(
                json.dumps(
                    {
                        **training_row,
                        "model_filter_opinion_similarity": None,
                        "model_filter_opinion_polarity_score": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                hash_training_rows(historical), hash_training_rows(current)
            )

    def test_training_semantic_hash_covers_every_training_input_field(self):
        self.assertTrue(
            hasattr(reproducibility, "semantic_training_rows_sha256"),
            "training semantic hash function is missing",
        )
        hash_training_rows = reproducibility.semantic_training_rows_sha256
        base = {
            "id": 7,
            "base_id": 3,
            "input": "great food",
            "target": "<pos> food <opinion> great",
            "sample_weight": 0.2,
            "augmentation": "masked_aspect_channel",
        }
        replacements = {
            "id": 8,
            "base_id": 4,
            "input": "bad food",
            "target": "<neg> food <opinion> bad",
            "sample_weight": 0.3,
            "augmentation": "target_pseudo",
        }
        with tempfile.TemporaryDirectory() as temp:
            baseline = Path(temp) / "baseline.jsonl"
            changed = Path(temp) / "changed.jsonl"
            baseline.write_text(json.dumps(base) + "\n", encoding="utf-8")
            baseline_hash = hash_training_rows(baseline)
            for field, value in replacements.items():
                changed.write_text(
                    json.dumps({**base, field: value}) + "\n", encoding="utf-8"
                )
                self.assertNotEqual(
                    baseline_hash,
                    hash_training_rows(changed),
                    f"training field was not hashed: {field}",
                )

    def test_metric_mismatch_is_rejected(self):
        with self.assertRaisesRegex(GoldenMismatchError, "micro_f1"):
            validate_metrics({"micro_f1": 0.4}, {"micro_f1": 0.5})

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

    def test_completed_stage_with_changed_input_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            context = RunContext.open_or_create(
                root, "run-001", "recipe-v1", "abc123", "feature/test"
            )
            source = root / "source.jsonl"
            output = root / "target_pseudo.jsonl"
            source.write_text("first", encoding="utf-8")
            output.write_text("output", encoding="utf-8")
            context.mark_stage_complete(
                "pseudo", [output], {str(source): sha256_file(source)}
            )
            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ReproducibilityError, "input hash mismatch"):
                context.validate_completed_stage(
                    "pseudo", [output], {str(source): sha256_file(source)}
                )


if __name__ == "__main__":
    unittest.main()

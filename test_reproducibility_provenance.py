import tempfile
import unittest
from pathlib import Path

from reproducibility import ReproducibilityError, RunContext


class ProvenanceTest(unittest.TestCase):
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

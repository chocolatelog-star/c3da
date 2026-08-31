import unittest
import tempfile
from pathlib import Path

from run_final_only_batch import build_final_only_command, validate_upstream_files


class FinalOnlyRunnerTest(unittest.TestCase):
    def test_command_uses_fixed_upstream_and_requested_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            upstream = Path(temp) / "upstream"
            (upstream / "final_data").mkdir(parents=True)
            (upstream / "final_data" / "final_train.jsonl").write_text("{}\n")
            (upstream / "final_data" / "final_dev.jsonl").write_text("{}\n")
            command = build_final_only_command(
            python=Path("python"),
            project_root=Path("/repo"),
            upstream=upstream,
            output=Path("/runs/batch8"),
            model=Path("/repo/models/t5-base-py"),
            train_batch_size=8,
            gradient_accumulation_steps=2,
            epochs=5,
            )
        self.assertTrue(any(item.endswith("final_train.jsonl") for item in command))
        self.assertIn("--per_device_train_batch_size", command)
        self.assertIn("8", command)
        self.assertIn("--gradient_accumulation_steps", command)
        self.assertIn("2", command)
        self.assertIn("--source_weight", command)

    def test_missing_fixed_upstream_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            validate_upstream_files(Path("/missing/upstream"))


if __name__ == "__main__":
    unittest.main()

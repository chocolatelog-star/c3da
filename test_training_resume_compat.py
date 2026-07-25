import unittest
from pathlib import Path


TRAINER = Path(__file__).resolve().parent / "t5_absa_train.py"


class TrainingResumeCompatibilityTests(unittest.TestCase):
    def test_historical_trainer_can_resume_latest_checkpoint(self):
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--resume_from_checkpoint"', source)
        self.assertIn('output_dir.glob("checkpoint-*")', source)
        self.assertIn("trainer.train(resume_from_checkpoint=resume_from_checkpoint)", source)


if __name__ == "__main__":
    unittest.main()

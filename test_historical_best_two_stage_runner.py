import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "run_historical_best_two_stage.ps1"


class HistoricalBestTwoStageRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.read_text(encoding="utf-8")

    def test_pins_historical_code_boundaries(self):
        self.assertIn("9e789045b41df7af0dd73ccebc90f06a91d94f8e", self.script)
        self.assertIn("a7e7778869dce92fe778837715a814b5c6d2014b", self.script)
        self.assertIn("8c7f6b47b1b2b4ef9c11d7dffdf64758db7aace3", self.script)
        self.assertIn("a7d147364d4b7de37814e6ee12871a386394d5f5", self.script)
        self.assertIn("historical-best-upstream-9e78904", self.script)
        self.assertIn("reproduce-best-8c7f6b4", self.script)
        self.assertIn("merge-base", self.script)
        self.assertIn("historical_base_commit", self.script)
        self.assertIn("command_compat_commit", self.script)

    def test_preserves_historical_upstream_parameters(self):
        required = (
            '"--num_train_epochs", "25"',
            '"--checkpoint_selection", "last"',
            '"--num_train_epochs", "8"',
            '"--checkpoint_selection", "best"',
            '"--augment_prompt_style", "masked_mutual"',
            '"--augment_channel_mode", "all"',
            '"--domain_prefix_style", "text"',
            '"--augment_select_max_rows", "150"',
            '"--augment_select_require_raw_exact"',
            '"--augment_select_require_model_filter_passed"',
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.script)

        self.assertNotIn('"--deterministic"', self.script)
        self.assertNotIn('"--legacy_stochastic"', self.script)

    def test_complete_multitriplets_are_added_after_augmentation(self):
        augment = self.script.index('"upstream_augment"')
        complete = self.script.index('"downstream_complete_multi2"')
        build = self.script.index('"downstream_build_final_train"')
        self.assertLess(augment, complete)
        self.assertLess(complete, build)
        self.assertIn('"--complete_multi_extra_weight", "0.25"', self.script)
        self.assertIn('"--pseudo_weight", "0.65"', self.script)

    def test_final_training_keeps_best_configuration(self):
        required = (
            '"--num_train_epochs", "5"',
            '"--lambda_domain_adv", "0.03"',
            '"--lambda_sentiment_contrastive", "0.01"',
            '"--sentiment_contrastive_source_only"',
            '"--sentiment_contrastive_class_balanced"',
            '"--per_device_train_batch_size", "1"',
            '"--per_device_eval_batch_size", "2"',
            '"--gradient_accumulation_steps", "16"',
            '"--learning_rate", "0.0003"',
            '"--resume_from_checkpoint", "auto"',
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.script)
        self.assertGreaterEqual(self.script.count('"--resume_from_checkpoint", "auto"'), 3)

    def test_runner_has_resume_logs_hashes_manifest_and_dry_run(self):
        for value in (
            "Invoke-Stage",
            "stage_status.json",
            "Start-Transcript",
            "Get-FileHash",
            "manifest.json",
            "[switch]$DryRun",
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.script)

        stages = re.findall(r'Invoke-Stage\s+"([^"]+)"', self.script)
        self.assertEqual(len(stages), len(set(stages)))
        self.assertGreaterEqual(len(stages), 9)


if __name__ == "__main__":
    unittest.main()

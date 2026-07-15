from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import run_bgca_aste_stage1_pairs as stage1
from run_bgca_aste_stage1_pairs import stage_done


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT = PROJECT_ROOT / "run_bgca_aste_stage1_pairs.py"


class Stage1PairPseudoFilterTest(unittest.TestCase):
    def test_stage_done_accepts_completed_legacy_stage_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "config.json"
            output.write_text("{}", encoding="utf-8")

            self.assertTrue(
                stage_done(
                    {"train_extractor": True},
                    "train_extractor_ep25_plain_last",
                    [output],
                    rerun=False,
                    legacy_stages=("train_extractor",),
                )
            )

    def test_legacy_hp1_stage_names_cover_expensive_downstream_stages(self) -> None:
        aliases = stage1.legacy_hp1_stage_names("label_to_text_gen")

        self.assertEqual(aliases["augment"], ("augment_label_to_text_gen",))
        self.assertEqual(aliases["train_final"], ("train_final_label_to_text_gen",))
        self.assertEqual(aliases["evaluate"], ("evaluate_label_to_text_gen",))

    def run_dry(self, *extra_args: str) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            command = [
                sys.executable,
                str(SCRIPT),
                "--output_root",
                temp_dir,
                "--pairs",
                "rest16:laptop14",
                "--domain_prefix_style",
                "text",
                "--generator_prompt_style",
                "label_to_text",
                "--augment_prompt_style",
                "masked_mutual",
                "--lambda_sentiment_contrastive",
                "0.01",
                "--sentiment_contrastive_source_only",
                "--sentiment_contrastive_class_balanced",
                "--dry_run",
                *extra_args,
            ]
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        return result.stdout

    def test_default_filter_keeps_legacy_outputs(self) -> None:
        output = self.run_dry()

        self.assertNotIn("t5_aste_pipeline.py select_pseudo", output)
        self.assertNotIn("hp1_dist5", output)
        self.assertIn("strict_aug150_w020_label_to_text_gen", output)

    def test_hp2_creates_independent_pseudo_and_final_outputs(self) -> None:
        output = self.run_dry(
            "--high_precision_max_triplets",
            "2",
            "--high_precision_max_token_distance",
            "5",
        )

        self.assertIn("t5_aste_pipeline.py select_pseudo", output)
        self.assertIn("pseudo_variants\\hp2_dist5", output)
        self.assertIn("--high_precision_max_triplets 2", output)
        self.assertIn("--pseudo_train_file", output)
        self.assertIn("strict_aug150_w020_label_to_text_gen_hp2_dist5", output)
        self.assertIn("--output_tag strict_aug150_w020_label_to_text_gen_hp2_dist5", output)

    def test_variant_summary_paths_do_not_use_legacy_names(self) -> None:
        csv_path, md_path = stage1.summary_output_paths(Path("runs"), "hp2_dist5")

        self.assertEqual(csv_path.name, "results_bgca_aste_stage1_hp2_dist5.csv")
        self.assertEqual(md_path.name, "results_bgca_aste_stage1_hp2_dist5_CN.md")

    def test_neutral_weight_experiment_has_independent_model_and_train_args(self) -> None:
        output = self.run_dry(
            "--neutral_generation_loss_gain",
            "1.0",
            "--neutral_generation_max_effective_weight",
            "2.0",
        )

        self.assertIn("neutral_gain100_max200", output)
        self.assertIn("--neutral_generation_loss_gain 1.0", output)
        self.assertIn("--neutral_generation_max_effective_weight 2.0", output)


if __name__ == "__main__":
    unittest.main()

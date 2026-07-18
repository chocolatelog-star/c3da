from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

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
        self.assertNotIn("--dynamic_multitriplet", output)
        self.assertNotIn("--source_count1_weight", output)
        self.assertNotIn("dynamic_multitriplet", output)

    def test_dynamic_multitriplet_prepare_and_extractor_are_isolated(self) -> None:
        output = self.run_dry(
            "--dynamic_multitriplet",
            "--source_count1_weight",
            "1.0",
            "--source_count2_weight",
            "1.15",
            "--source_count3_weight",
            "1.25",
            "--source_count4plus_weight",
            "1.3",
        )

        self.assertIn("--dynamic_multitriplet", output)
        self.assertIn("--source_count1_weight 1.0", output)
        self.assertIn("--source_count2_weight 1.15", output)
        self.assertIn("--source_count3_weight 1.25", output)
        self.assertIn("--source_count4plus_weight 1.3", output)
        self.assertIn("extractor_ep25_plain_last_dynamic_multitriplet", output)

    def test_dynamic_prepare_requires_its_own_stage_and_analysis_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            run_dir.mkdir(parents=True)
            for path in (
                run_dir / "extract_train.jsonl",
                run_dir / "c3da_generator_train_label_to_text_gen.jsonl",
                run_dir / "c3da_generator_dev_label_to_text_gen.jsonl",
            ):
                path.write_text("{}\n", encoding="utf-8")
            (run_dir / "stage_status.json").write_text(
                '{"prepare_label_to_text_gen": true}',
                encoding="utf-8",
            )
            argv = [
                str(SCRIPT),
                "--output_root",
                temp_dir,
                "--pairs",
                "rest16:laptop14",
                "--dynamic_multitriplet",
                "--dry_run",
            ]
            with patch.object(sys, "argv", argv):
                args = stage1.parse_args()

            first_output = io.StringIO()
            with redirect_stdout(first_output):
                stage1.run_pair(args, "rest16", "laptop14")

            (run_dir / "stage_status.json").write_text(
                '{"prepare_dynamic_multitriplet_label_to_text_gen": true}',
                encoding="utf-8",
            )
            missing_analysis_output = io.StringIO()
            with redirect_stdout(missing_analysis_output):
                stage1.run_pair(args, "rest16", "laptop14")

            (run_dir / "extract_train_multitriplet_weight_analysis.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (run_dir / "stage_status.json").write_text(
                '{"prepare_label_to_text_gen": true}',
                encoding="utf-8",
            )
            old_stage_output = io.StringIO()
            with redirect_stdout(old_stage_output):
                stage1.run_pair(args, "rest16", "laptop14")

            (run_dir / "stage_status.json").write_text(
                '{"prepare_dynamic_multitriplet_label_to_text_gen": true}',
                encoding="utf-8",
            )
            complete_output = io.StringIO()
            with redirect_stdout(complete_output):
                stage1.run_pair(args, "rest16", "laptop14")

        self.assertIn("t5_aste_pipeline.py prepare", first_output.getvalue())
        self.assertIn("t5_aste_pipeline.py prepare", missing_analysis_output.getvalue())
        self.assertIn("t5_aste_pipeline.py prepare", old_stage_output.getvalue())
        self.assertNotIn("t5_aste_pipeline.py prepare", complete_output.getvalue())

    def test_legacy_prepare_still_uses_legacy_stage_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            run_dir.mkdir(parents=True)
            for path in (
                run_dir / "extract_train.jsonl",
                run_dir / "c3da_generator_train_label_to_text_gen.jsonl",
                run_dir / "c3da_generator_dev_label_to_text_gen.jsonl",
            ):
                path.write_text("{}\n", encoding="utf-8")
            (run_dir / "stage_status.json").write_text(
                '{"prepare_label_to_text_gen": true}',
                encoding="utf-8",
            )
            argv = [
                str(SCRIPT),
                "--output_root",
                temp_dir,
                "--pairs",
                "rest16:laptop14",
                "--dry_run",
            ]
            with patch.object(sys, "argv", argv):
                args = stage1.parse_args()
            output = io.StringIO()
            with redirect_stdout(output):
                stage1.run_pair(args, "rest16", "laptop14")

        self.assertNotIn("t5_aste_pipeline.py prepare", output.getvalue())

    def test_run_pair_accepts_legacy_namespace_without_dynamic_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                str(SCRIPT),
                "--output_root",
                temp_dir,
                "--pairs",
                "rest16:laptop14",
                "--dry_run",
            ]
            with patch.object(sys, "argv", argv):
                args = stage1.parse_args()
            for name in (
                "dynamic_multitriplet",
                "source_count1_weight",
                "source_count2_weight",
                "source_count3_weight",
                "source_count4plus_weight",
            ):
                delattr(args, name)

            output = io.StringIO()
            with redirect_stdout(output):
                stage1.run_pair(args, "rest16", "laptop14")

        self.assertNotIn("dynamic_multitriplet", output.getvalue())

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

    def test_complete_multi_contrastive_summary_tag_is_isolated(self) -> None:
        tag = stage1.append_sentiment_summary_tag(
            "complete_multi2_w025",
            0.01,
            source_only=True,
            class_balanced=True,
        )

        self.assertEqual(
            tag,
            "complete_multi2_w025_sentiment_contrastive_l001_source_balanced",
        )

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

    def test_mixed_generator_uses_isolated_files_model_and_resume(self) -> None:
        output = self.run_dry("--generator_prompt_style", "mixed")

        self.assertIn("--augment_prompt_style mixed", output)
        self.assertIn("c3da_generator_train_mixed_l2t_masked_aspect_masked_opinion.jsonl", output)
        self.assertIn("generator_mixed_l2t_masked_aspect_masked_opinion_ep8", output)
        self.assertIn("--resume_from_checkpoint auto", output)
        self.assertIn("--augment_prompt_style masked_mutual", output)
        self.assertIn("strict_aug150_w020_mixed_l2t_masked_aspect_masked_opinion", output)
        self.assertIn("--per_device_train_batch_size 1", output)
        self.assertIn("--per_device_eval_batch_size 2", output)
        self.assertIn("--gradient_accumulation_steps 16", output)
        self.assertIn("--fp16", output)
        self.assertIn("--gradient_checkpointing", output)

    def test_mixed_generator_can_reuse_upstream_extractor_and_pseudo_labels(self) -> None:
        upstream = r"runs\bgca_aste_stage1_domain_prompt_text_v1\rest16_to_laptop14"

        output = self.run_dry(
            "--generator_prompt_style",
            "mixed",
            "--reuse_upstream_run_dir",
            upstream,
        )

        upstream_extractor = upstream + r"\models\extractor_ep25_plain_last\best"
        self.assertNotIn("t5_aste_pipeline.py pseudo", output)
        self.assertNotIn(r"extract_train.jsonl --dev_file", output)
        self.assertIn(f"--augmentation_input_run_dir {upstream}", output)
        self.assertIn(f"--pseudo_train_file {upstream}\\target_pseudo_high_precision.jsonl", output)
        self.assertIn(f"--model_filter_path {upstream_extractor}", output)
        self.assertIn("generator_mixed_l2t_masked_aspect_masked_opinion_ep8", output)

    def test_encoder_pairing_ablation_reuses_best_final_train_and_isolates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            extractor_dir = run_dir / "models" / "extractor_ep25_plain_last" / "best"
            generator_dir = run_dir / "models" / "generator_label_to_text_gen_ep8" / "best"
            extractor_dir.mkdir(parents=True)
            generator_dir.mkdir(parents=True)
            for path in (
                run_dir / "extract_train.jsonl",
                run_dir / "c3da_generator_train_label_to_text_gen.jsonl",
                run_dir / "c3da_generator_dev_label_to_text_gen.jsonl",
                run_dir / "target_pseudo.jsonl",
                run_dir / "target_pseudo_high_precision.jsonl",
                run_dir / "target_pseudo_high_precision_analysis.json",
                run_dir / "final_train_strict_aug150_w020_label_to_text_gen.jsonl",
                run_dir / "final_dev_strict_aug150_w020_label_to_text_gen.jsonl",
                extractor_dir / "config.json",
                generator_dir / "config.json",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            (run_dir / "stage_status.json").write_text(
                '{"prepare_label_to_text_gen": true, "train_extractor_ep25_plain_last": true, '
                '"pseudo_extractor_ep25_plain_last": true, "train_generator_label_to_text_gen": true}',
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(SCRIPT),
                "--output_root",
                temp_dir,
                "--pairs",
                "rest16:laptop14",
                "--generator_prompt_style",
                "label_to_text",
                "--augment_prompt_style",
                "masked_mutual",
                "--domain_prefix_style",
                "text",
                "--lambda_sentiment_contrastive",
                "0.01",
                "--sentiment_contrastive_source_only",
                "--sentiment_contrastive_class_balanced",
                "--lambda_pairing_loss",
                "0.01",
                "--pairing_temperature",
                "0.1",
                "--pairing_source_only",
                "--dry_run",
            ]

            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        output = result.stdout
        command_lines = [line for line in output.splitlines() if line.startswith(sys.executable)]
        self.assertEqual(len(command_lines), 2)
        self.assertNotIn("t5_aste_pipeline.py prepare", output)
        self.assertNotIn("t5_aste_pipeline.py pseudo", output)
        self.assertNotIn("t5_aste_pipeline.py augment", output)
        self.assertIn("final_train_strict_aug150_w020_label_to_text_gen.jsonl", output)
        self.assertIn("--lambda_pairing_loss 0.01", output)
        self.assertIn("--pairing_temperature 0.1", output)
        self.assertIn("--pairing_source_only", output)
        self.assertIn("pairing_encoder_l001_source_only", output)

    def test_complete_multi_ablation_reuses_upstream_and_only_rebuilds_final_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            extractor_dir = run_dir / "models" / "extractor_ep25_plain_last" / "best"
            generator_dir = run_dir / "models" / "generator_label_to_text_gen_ep8" / "best"
            for path in (
                run_dir / "extract_train.jsonl",
                run_dir / "c3da_generator_train_label_to_text_gen.jsonl",
                run_dir / "c3da_generator_dev_label_to_text_gen.jsonl",
                run_dir / "target_pseudo.jsonl",
                run_dir / "target_pseudo_high_precision.jsonl",
                run_dir / "target_pseudo_high_precision_analysis.json",
                run_dir / "source_train.jsonl",
                run_dir / "source_dev.jsonl",
                run_dir / "c3da_two_channel_augmented_selected_strict_aug150_w020_label_to_text_gen.jsonl",
                run_dir / "aste_metrics_raw_label_to_text_gen.json",
                run_dir / "aste_metrics_fixed_label_to_text_gen.json",
                extractor_dir / "config.json",
                generator_dir / "config.json",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            (run_dir / "stage_status.json").write_text(
                '{"prepare_label_to_text_gen": true, "train_extractor_ep25_plain_last": true, '
                '"pseudo_extractor_ep25_plain_last": true, "train_generator_label_to_text_gen": true, '
                '"evaluate_label_to_text_gen": true}',
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(SCRIPT),
                "--output_root",
                temp_dir,
                "--pairs",
                "rest16:laptop14",
                "--generator_prompt_style",
                "label_to_text",
                "--augment_prompt_style",
                "masked_mutual",
                "--domain_prefix_style",
                "text",
                "--complete_multi_extra_weight",
                "0.25",
                "--dry_run",
            ]

            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        output = result.stdout
        command_lines = [line for line in output.splitlines() if line.startswith(sys.executable)]
        self.assertEqual(len(command_lines), 4)
        self.assertIn("select_complete_multi_pseudo", output)
        self.assertIn("hp1_complete2_dist5_w025", output)
        self.assertIn("build_final_train_from_files", output)
        self.assertIn("c3da_two_channel_augmented_selected_strict_aug150_w020_label_to_text_gen.jsonl", output)
        self.assertNotIn("t5_aste_pipeline.py augment", output)
        self.assertIn("complete_multi2_w025", output)
        self.assertIn("--resume_from_checkpoint auto", output)


if __name__ == "__main__":
    unittest.main()

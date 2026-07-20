from __future__ import annotations

import io
import json
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
    @staticmethod
    def _write_pseudo_metadata(
        run_dir: Path,
        model_path: Path,
        source_tag: str,
        status: str = "complete",
    ) -> None:
        provenance = {
            "model_path": str(model_path.resolve()),
            "pseudo_source_tag": source_tag,
        }
        (run_dir / "target_pseudo_analysis.json").write_text(
            json.dumps(provenance),
            encoding="utf-8",
        )
        (run_dir / "target_pseudo_generation_state.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "resolved_model_path": str(model_path.resolve()),
                    "pseudo_source_tag": source_tag,
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _dynamic_ready_run(run_dir: Path, config_tag: str) -> tuple[str, Path]:
        extractor_tag = f"extractor_ep25_aste_f1_{config_tag}"
        extractor_best = run_dir / "models" / extractor_tag / "best"
        required_outputs = (
            run_dir / f"extract_train_{config_tag}.jsonl",
            run_dir / f"extract_train_multitriplet_weight_analysis_{config_tag}.json",
            run_dir / "extract_dev.jsonl",
            run_dir / "source_train.jsonl",
            run_dir / "source_dev.jsonl",
            run_dir / "target_unlabeled.jsonl",
            run_dir / "target_train_gold_analysis.jsonl",
            run_dir / "target_test.jsonl",
            run_dir / "c3da_generator_train_label_to_text_gen.jsonl",
            run_dir / "c3da_generator_dev_label_to_text_gen.jsonl",
            extractor_best / "config.json",
            run_dir / "target_pseudo.jsonl",
            run_dir / "target_pseudo_high_precision.jsonl",
            run_dir / "target_pseudo_high_precision_analysis.json",
        )
        for path in required_outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        Stage1PairPseudoFilterTest._write_pseudo_metadata(
            run_dir,
            extractor_best,
            extractor_tag,
        )
        return extractor_tag, extractor_best

    @staticmethod
    def _dynamic_args(output_root: str, count3_weight: str = "1.25"):
        argv = [
            str(SCRIPT),
            "--output_root",
            output_root,
            "--pairs",
            "rest16:laptop14",
            "--dynamic_multitriplet",
            "--source_count3_weight",
            count3_weight,
            "--dry_run",
        ]
        with patch.object(sys, "argv", argv):
            return stage1.parse_args()

    @staticmethod
    def _reuse_args(output_root: str, upstream_run_dir: Path, dry_run: bool):
        argv = [
            str(SCRIPT),
            "--output_root",
            output_root,
            "--pairs",
            "rest16:laptop14",
            "--generator_prompt_style",
            "mixed",
            "--reuse_upstream_run_dir",
            str(upstream_run_dir),
            *(["--dry_run"] if dry_run else []),
        ]
        with patch.object(sys, "argv", argv):
            return stage1.parse_args()

    def _ready_legacy_upstream(self, run_dir: Path) -> tuple[str, Path]:
        extractor_tag = "extractor_ep25_plain_last"
        extractor_best = run_dir / "models" / extractor_tag / "best"
        for path in (
            extractor_best / "config.json",
            run_dir / "target_pseudo.jsonl",
            run_dir / "target_pseudo_high_precision.jsonl",
            run_dir / "target_pseudo_high_precision_analysis.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        self._write_pseudo_metadata(run_dir, extractor_best, extractor_tag)
        return extractor_tag, extractor_best

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
        extractor_command = next(
            line for line in output.splitlines()
            if "t5_absa_train.py" in line and "extractor_ep25_plain_last" in line
        )
        self.assertIn("--checkpoint_selection last", extractor_command)

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
        config_tag = "dynamic_multitriplet_c1w100_c2w115_c3w125_c4pw130"
        self.assertIn(f"extract_train_{config_tag}.jsonl", output)
        self.assertIn(f"extractor_ep25_aste_f1_{config_tag}", output)
        self.assertNotIn(f"extractor_ep25_plain_last_{config_tag}", output)
        training_commands = [line for line in output.splitlines() if "t5_absa_train.py" in line]
        extractor_command = next(line for line in training_commands if f"extractor_ep25_aste_f1_{config_tag}" in line)
        generator_command = next(line for line in training_commands if "generator_label_to_text_gen_ep8" in line)
        final_command = next(line for line in training_commands if "final_dann" in line)
        self.assertIn("--checkpoint_selection aste_f1", extractor_command)
        self.assertIn("--checkpoint_selection best", generator_command)
        self.assertIn("--checkpoint_selection best", final_command)
        self.assertIn("t5_aste_pipeline.py select_dynamic_pseudo", output)
        self.assertIn("pseudo_variants\\dynamic_dist5", output)
        self.assertIn("--pseudo_train_file", output)
        self.assertIn("strict_aug150_w020_label_to_text_gen_dynamic_dist5", output)

    def test_runner_cli_rejects_invalid_source_weights(self) -> None:
        invalid_cases = (
            ("--source_count1_weight", "nan"),
            ("--source_count2_weight", "inf"),
            ("--source_count3_weight", "0"),
            ("--source_count4plus_weight", "-1"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for option, value in invalid_cases:
                with self.subTest(option=option, value=value):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "--output_root",
                            temp_dir,
                            "--pairs",
                            "rest16:laptop14",
                            "--dynamic_multitriplet",
                            option,
                            value,
                            "--dry_run",
                        ],
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)

    def test_dynamic_prepare_requires_its_own_stage_and_analysis_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            run_dir.mkdir(parents=True)
            config_tag = "dynamic_multitriplet_c1w100_c2w115_c3w125_c4pw130"
            dynamic_stage = f"prepare_{config_tag}_label_to_text_gen"
            dynamic_extract = run_dir / f"extract_train_{config_tag}.jsonl"
            dynamic_analysis = run_dir / f"extract_train_multitriplet_weight_analysis_{config_tag}.json"
            for path in (
                run_dir / "extract_train.jsonl",
                run_dir / "extract_dev.jsonl",
                run_dir / "source_train.jsonl",
                run_dir / "source_dev.jsonl",
                run_dir / "target_unlabeled.jsonl",
                run_dir / "target_train_gold_analysis.jsonl",
                run_dir / "target_test.jsonl",
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

            dynamic_extract.write_text("{}\n", encoding="utf-8")
            (run_dir / "stage_status.json").write_text(
                json.dumps({dynamic_stage: True}),
                encoding="utf-8",
            )
            missing_analysis_output = io.StringIO()
            with redirect_stdout(missing_analysis_output):
                stage1.run_pair(args, "rest16", "laptop14")

            dynamic_analysis.write_text("{}\n", encoding="utf-8")
            (run_dir / "stage_status.json").write_text(
                '{"prepare_label_to_text_gen": true}',
                encoding="utf-8",
            )
            old_stage_output = io.StringIO()
            with redirect_stdout(old_stage_output):
                stage1.run_pair(args, "rest16", "laptop14")

            (run_dir / "stage_status.json").write_text(
                json.dumps({dynamic_stage: True}),
                encoding="utf-8",
            )
            complete_output = io.StringIO()
            with redirect_stdout(complete_output):
                stage1.run_pair(args, "rest16", "laptop14")

        self.assertIn("t5_aste_pipeline.py prepare", first_output.getvalue())
        self.assertIn("t5_aste_pipeline.py prepare", missing_analysis_output.getvalue())
        self.assertIn("t5_aste_pipeline.py prepare", old_stage_output.getvalue())
        self.assertNotIn("t5_aste_pipeline.py prepare", complete_output.getvalue())

    def test_different_dynamic_weight_config_does_not_reuse_stage_or_model(self) -> None:
        first_tag = "dynamic_multitriplet_c1w100_c2w115_c3w125_c4pw130"
        second_tag = "dynamic_multitriplet_c1w100_c2w115_c3wd1p251_c4pw130"
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            first_model = run_dir / "models" / f"extractor_ep25_aste_f1_{first_tag}" / "best" / "config.json"
            first_model.parent.mkdir(parents=True)
            first_model.write_text("{}\n", encoding="utf-8")
            (run_dir / "stage_status.json").write_text(
                json.dumps(
                    {
                        f"prepare_{first_tag}_label_to_text_gen": True,
                        f"train_extractor_ep25_aste_f1_{first_tag}": True,
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                str(SCRIPT),
                "--output_root",
                temp_dir,
                "--pairs",
                "rest16:laptop14",
                "--dynamic_multitriplet",
                "--source_count3_weight",
                "1.251",
                "--dry_run",
            ]
            with patch.object(sys, "argv", argv):
                args = stage1.parse_args()
            output = io.StringIO()
            with redirect_stdout(output):
                stage1.run_pair(args, "rest16", "laptop14")

        text_output = output.getvalue()
        self.assertIn("t5_aste_pipeline.py prepare", text_output)
        self.assertIn(f"extract_train_{second_tag}.jsonl", text_output)
        self.assertIn(f"extractor_ep25_aste_f1_{second_tag}", text_output)

    def test_dynamic_pseudo_does_not_accept_legacy_pseudo_stage(self) -> None:
        config_tag = "dynamic_multitriplet_c1w100_c2w115_c3w125_c4pw130"
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            extractor_tag = f"extractor_ep25_aste_f1_{config_tag}"
            extractor_config = run_dir / "models" / extractor_tag / "best" / "config.json"
            extractor_config.parent.mkdir(parents=True)
            required_prepare_outputs = (
                run_dir / f"extract_train_{config_tag}.jsonl",
                run_dir / f"extract_train_multitriplet_weight_analysis_{config_tag}.json",
                run_dir / "extract_dev.jsonl",
                run_dir / "source_train.jsonl",
                run_dir / "source_dev.jsonl",
                run_dir / "target_unlabeled.jsonl",
                run_dir / "target_train_gold_analysis.jsonl",
                run_dir / "target_test.jsonl",
                run_dir / "c3da_generator_train_label_to_text_gen.jsonl",
                run_dir / "c3da_generator_dev_label_to_text_gen.jsonl",
                extractor_config,
                run_dir / "target_pseudo.jsonl",
                run_dir / "target_pseudo_high_precision.jsonl",
                run_dir / "target_pseudo_high_precision_analysis.json",
            )
            for path in required_prepare_outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            (run_dir / "stage_status.json").write_text(
                json.dumps(
                    {
                        f"prepare_{config_tag}_label_to_text_gen": True,
                        f"train_{extractor_tag}": True,
                        "pseudo": True,
                    }
                ),
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
            output = io.StringIO()
            with redirect_stdout(output):
                stage1.run_pair(args, "rest16", "laptop14")

        self.assertIn("t5_aste_pipeline.py pseudo", output.getvalue())

    def test_dynamic_pseudo_reuse_requires_matching_provenance(self) -> None:
        config_tag = "dynamic_multitriplet_c1w100_c2w115_c3w125_c4pw130"
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            extractor_tag, extractor_best = self._dynamic_ready_run(run_dir, config_tag)
            status = {
                f"prepare_{config_tag}_label_to_text_gen": True,
                f"train_{extractor_tag}": True,
                f"pseudo_{extractor_tag}": True,
            }
            (run_dir / "stage_status.json").write_text(json.dumps(status), encoding="utf-8")
            (run_dir / "target_pseudo_analysis.json").write_text(
                json.dumps(
                    {
                        "model_path": str(extractor_best.resolve()),
                        "pseudo_source_tag": extractor_tag,
                    }
                ),
                encoding="utf-8",
            )
            args = self._dynamic_args(temp_dir)

            matching_output = io.StringIO()
            with redirect_stdout(matching_output):
                stage1.run_pair(args, "rest16", "laptop14")

            (run_dir / "target_pseudo_analysis.json").write_text(
                json.dumps(
                    {
                        "model_path": str((run_dir / "models" / "extractor_ep25_plain_last" / "best").resolve()),
                        "pseudo_source_tag": "extractor_ep25_plain_last",
                    }
                ),
                encoding="utf-8",
            )
            legacy_then_a_output = io.StringIO()
            with redirect_stdout(legacy_then_a_output):
                stage1.run_pair(args, "rest16", "laptop14")

        self.assertNotIn("t5_aste_pipeline.py pseudo", matching_output.getvalue())
        self.assertIn("t5_aste_pipeline.py pseudo", legacy_then_a_output.getvalue())
        self.assertIn(f"--pseudo_source_tag {extractor_tag}", legacy_then_a_output.getvalue())

    def test_dynamic_pseudo_rejects_other_dynamic_configuration_provenance(self) -> None:
        config_a = "dynamic_multitriplet_c1w100_c2w115_c3w125_c4pw130"
        config_b = "dynamic_multitriplet_c1w100_c2w115_c3wd1p251_c4pw130"
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            extractor_a, _extractor_a_best = self._dynamic_ready_run(run_dir, config_a)
            extractor_b = f"extractor_ep25_aste_f1_{config_b}"
            extractor_b_best = run_dir / "models" / extractor_b / "best"
            status = {
                f"prepare_{config_a}_label_to_text_gen": True,
                f"train_{extractor_a}": True,
                f"pseudo_{extractor_a}": True,
            }
            (run_dir / "stage_status.json").write_text(json.dumps(status), encoding="utf-8")
            (run_dir / "target_pseudo_analysis.json").write_text(
                json.dumps(
                    {
                        "model_path": str(extractor_b_best.resolve()),
                        "pseudo_source_tag": extractor_b,
                    }
                ),
                encoding="utf-8",
            )
            args = self._dynamic_args(temp_dir)
            output = io.StringIO()
            with redirect_stdout(output):
                stage1.run_pair(args, "rest16", "laptop14")

        self.assertIn("t5_aste_pipeline.py pseudo", output.getvalue())

    def test_dynamic_pseudo_rejects_interrupted_or_other_source_state(self) -> None:
        config_a = "dynamic_multitriplet_c1w100_c2w115_c3w125_c4pw130"
        config_b = "dynamic_multitriplet_c1w100_c2w115_c3wd1p251_c4pw130"
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            extractor_a, extractor_a_best = self._dynamic_ready_run(run_dir, config_a)
            extractor_b = f"extractor_ep25_aste_f1_{config_b}"
            extractor_b_best = run_dir / "models" / extractor_b / "best"
            (run_dir / "stage_status.json").write_text(
                json.dumps(
                    {
                        f"prepare_{config_a}_label_to_text_gen": True,
                        f"train_{extractor_a}": True,
                        f"pseudo_{extractor_a}": True,
                    }
                ),
                encoding="utf-8",
            )
            args = self._dynamic_args(temp_dir)

            self._write_pseudo_metadata(
                run_dir,
                extractor_b_best,
                extractor_b,
                status="in_progress",
            )
            (run_dir / "target_pseudo.jsonl").write_text('{"source": "b"}\n', encoding="utf-8")
            interrupted_output = io.StringIO()
            with redirect_stdout(interrupted_output):
                stage1.run_pair(args, "rest16", "laptop14")

            self._write_pseudo_metadata(run_dir, extractor_b_best, extractor_b)
            other_complete_output = io.StringIO()
            with redirect_stdout(other_complete_output):
                stage1.run_pair(args, "rest16", "laptop14")

            self._write_pseudo_metadata(run_dir, extractor_a_best, extractor_a)
            matching_output = io.StringIO()
            with redirect_stdout(matching_output):
                stage1.run_pair(args, "rest16", "laptop14")

        self.assertIn("t5_aste_pipeline.py pseudo", interrupted_output.getvalue())
        self.assertIn("t5_aste_pipeline.py pseudo", other_complete_output.getvalue())
        self.assertNotIn("t5_aste_pipeline.py pseudo", matching_output.getvalue())

    def test_dynamic_selection_validation_rejects_in_progress_or_wrong_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "dynamic_dist5"
            output_dir.mkdir(parents=True)
            for path in (
                output_dir / "target_pseudo_high_precision.jsonl",
                output_dir / "target_pseudo_high_precision_analysis.json",
            ):
                path.write_text("{}\n", encoding="utf-8")
            state_path = output_dir / "target_pseudo_generation_state.json"
            state = {
                "status": "in_progress",
                "selection_mode": "dynamic_high_precision",
                "base_pseudo_source_tag": "extractor_a",
                "min_pseudo_weight": 0.65,
                "max_token_distance": 5,
            }
            state_path.write_text(json.dumps(state), encoding="utf-8")

            valid, reason = stage1.validate_dynamic_pseudo_selection(
                output_dir,
                "extractor_a",
                0.65,
                5,
            )
            self.assertFalse(valid)
            self.assertIn("in_progress", reason)

            state["status"] = "complete"
            state["base_pseudo_source_tag"] = "extractor_b"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            valid, reason = stage1.validate_dynamic_pseudo_selection(
                output_dir,
                "extractor_a",
                0.65,
                5,
            )
            self.assertFalse(valid)
            self.assertIn("source tag", reason)

    def test_legacy_prepare_requires_every_shared_output(self) -> None:
        required_names = (
            "extract_train.jsonl",
            "extract_dev.jsonl",
            "source_train.jsonl",
            "source_dev.jsonl",
            "target_unlabeled.jsonl",
            "target_train_gold_analysis.jsonl",
            "target_test.jsonl",
            "c3da_generator_train_label_to_text_gen.jsonl",
            "c3da_generator_dev_label_to_text_gen.jsonl",
        )
        for missing_name in required_names:
            with self.subTest(missing=missing_name), tempfile.TemporaryDirectory() as temp_dir:
                run_dir = Path(temp_dir) / "rest16_to_laptop14"
                run_dir.mkdir(parents=True)
                for name in required_names:
                    if name != missing_name:
                        (run_dir / name).write_text("{}\n", encoding="utf-8")
                (run_dir / "stage_status.json").write_text(
                    '{"prepare_label_to_text_gen": true}',
                    encoding="utf-8",
                )
                args = self._dynamic_args(temp_dir)
                args.dynamic_multitriplet = False
                output = io.StringIO()
                with redirect_stdout(output):
                    stage1.run_pair(args, "rest16", "laptop14")

                self.assertIn("t5_aste_pipeline.py prepare", output.getvalue())

    def test_legacy_prepare_still_uses_legacy_stage_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            run_dir.mkdir(parents=True)
            for path in (
                run_dir / "extract_train.jsonl",
                run_dir / "extract_dev.jsonl",
                run_dir / "source_train.jsonl",
                run_dir / "source_dev.jsonl",
                run_dir / "target_unlabeled.jsonl",
                run_dir / "target_train_gold_analysis.jsonl",
                run_dir / "target_test.jsonl",
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
        self.assertIn("cannot validate upstream pseudo provenance", output)

    def test_reuse_upstream_accepts_complete_self_consistent_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir) / "upstream"
            self._ready_legacy_upstream(upstream)
            args = self._reuse_args(str(Path(temp_dir) / "output"), upstream, dry_run=True)
            output = io.StringIO()
            with redirect_stdout(output):
                stage1.run_pair(args, "rest16", "laptop14")

        self.assertNotIn("cannot validate upstream pseudo provenance", output.getvalue())
        self.assertNotIn("t5_aste_pipeline.py pseudo", output.getvalue())

    def test_reuse_upstream_rejects_in_progress_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir) / "upstream"
            extractor_tag, extractor_best = self._ready_legacy_upstream(upstream)
            self._write_pseudo_metadata(
                upstream,
                extractor_best,
                extractor_tag,
                status="in_progress",
            )
            args = self._reuse_args(str(Path(temp_dir) / "output"), upstream, dry_run=False)

            with patch.object(stage1, "run_command"):
                with self.assertRaisesRegex(RuntimeError, "in_progress"):
                    stage1.run_pair(args, "rest16", "laptop14")

    def test_reuse_upstream_rejects_path_or_tag_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for mismatch in ("path", "tag"):
                with self.subTest(mismatch=mismatch):
                    upstream = Path(temp_dir) / mismatch
                    extractor_tag, extractor_best = self._ready_legacy_upstream(upstream)
                    if mismatch == "path":
                        wrong_model = upstream / "models" / "other" / "best"
                        self._write_pseudo_metadata(upstream, wrong_model, extractor_tag)
                    else:
                        state = json.loads(
                            (upstream / "target_pseudo_generation_state.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        state["pseudo_source_tag"] = "other_extractor"
                        (upstream / "target_pseudo_generation_state.json").write_text(
                            json.dumps(state),
                            encoding="utf-8",
                        )
                    args = self._reuse_args(
                        str(Path(temp_dir) / f"output_{mismatch}"),
                        upstream,
                        dry_run=False,
                    )

                    with patch.object(stage1, "run_command"):
                        with self.assertRaisesRegex(RuntimeError, "pseudo provenance"):
                            stage1.run_pair(args, "rest16", "laptop14")

    def test_pseudo_validation_rejects_missing_or_damaged_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for case in ("missing", "damaged"):
                with self.subTest(case=case):
                    run_dir = Path(temp_dir) / case
                    model_path = run_dir / "models" / "extractor" / "best"
                    for path in (
                        run_dir / "target_pseudo.jsonl",
                        run_dir / "target_pseudo_high_precision.jsonl",
                        run_dir / "target_pseudo_high_precision_analysis.json",
                    ):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("{}\n", encoding="utf-8")
                    (run_dir / "target_pseudo_analysis.json").write_text(
                        json.dumps(
                            {
                                "model_path": str(model_path.resolve()),
                                "pseudo_source_tag": "extractor",
                            }
                        ),
                        encoding="utf-8",
                    )
                    if case == "damaged":
                        (run_dir / "target_pseudo_generation_state.json").write_text(
                            "[]",
                            encoding="utf-8",
                        )

                    valid, reason = stage1.validate_pseudo_provenance(
                        run_dir,
                        model_path,
                        "extractor",
                    )

                    self.assertFalse(valid)
                    self.assertTrue(reason)

    def test_encoder_pairing_ablation_reuses_best_final_train_and_isolates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rest16_to_laptop14"
            extractor_dir = run_dir / "models" / "extractor_ep25_plain_last" / "best"
            generator_dir = run_dir / "models" / "generator_label_to_text_gen_ep8" / "best"
            extractor_dir.mkdir(parents=True)
            generator_dir.mkdir(parents=True)
            for path in (
                run_dir / "extract_train.jsonl",
                run_dir / "extract_dev.jsonl",
                run_dir / "source_train.jsonl",
                run_dir / "source_dev.jsonl",
                run_dir / "target_unlabeled.jsonl",
                run_dir / "target_train_gold_analysis.jsonl",
                run_dir / "target_test.jsonl",
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
            self._write_pseudo_metadata(
                run_dir,
                extractor_dir,
                "extractor_ep25_plain_last",
            )
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
                run_dir / "extract_dev.jsonl",
                run_dir / "source_train.jsonl",
                run_dir / "source_dev.jsonl",
                run_dir / "target_unlabeled.jsonl",
                run_dir / "target_train_gold_analysis.jsonl",
                run_dir / "target_test.jsonl",
                run_dir / "c3da_generator_train_label_to_text_gen.jsonl",
                run_dir / "c3da_generator_dev_label_to_text_gen.jsonl",
                run_dir / "target_pseudo.jsonl",
                run_dir / "target_pseudo_high_precision.jsonl",
                run_dir / "target_pseudo_high_precision_analysis.json",
                run_dir / "c3da_two_channel_augmented_selected_strict_aug150_w020_label_to_text_gen.jsonl",
                run_dir / "aste_metrics_raw_label_to_text_gen.json",
                run_dir / "aste_metrics_fixed_label_to_text_gen.json",
                extractor_dir / "config.json",
                generator_dir / "config.json",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            self._write_pseudo_metadata(
                run_dir,
                extractor_dir,
                "extractor_ep25_plain_last",
            )
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

    def test_complete_multi_can_add_strict_dynamic_three_plus_extra(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
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
                "--complete_dynamic_extra_weight",
                "0.2",
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
        self.assertIn("select_dynamic_pseudo", output)
        self.assertIn("--dynamic_strict", output)
        self.assertIn("select_complete_multi_pseudo", output)
        self.assertIn("select_complete_dynamic_pseudo", output)
        self.assertIn("dynamic_strict_dist5", output)
        self.assertIn("complete_multi2_w025_dynamic_strict3plus_dist5_w020", output)
        self.assertIn(
            "final_train_strict_aug150_w020_label_to_text_gen_complete_multi2_w025_dynamic_strict3plus_dist5_w020.jsonl",
            output,
        )

    def test_complete_dynamic_requires_complete_multi_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            command = [
                sys.executable,
                str(SCRIPT),
                "--output_root",
                temp_dir,
                "--pairs",
                "rest16:laptop14",
                "--complete_dynamic_extra_weight",
                "0.2",
                "--dry_run",
            ]

            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --complete_multi_extra_weight", result.stderr)


if __name__ == "__main__":
    unittest.main()

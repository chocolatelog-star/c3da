import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import get_type_hints
from unittest.mock import patch

import t5_aste_pipeline as pipeline

from t5_aste_data import (
    micro_f1_by_triplet_count,
    triplet_count_bucket,
    triplet_count_diagnostics,
    triplets_to_text,
)


def _label(*triplets):
    return triplets_to_text(triplets)


class DynamicMultiTripletTest(unittest.TestCase):
    def test_pseudo_analysis_records_resolved_model_path_and_source_tag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            model_path = run_dir / "models" / "extractor_a" / "best"
            model_path.mkdir(parents=True)
            target_row = {"id": "t1", "text": "Bright screen."}
            pipeline.write_jsonl(run_dir / "target_unlabeled.jsonl", [target_row])
            pipeline.write_jsonl(
                run_dir / "target_train_gold_analysis.jsonl",
                [{**target_row, "label": _label(("screen", "bright", "pos"))}],
            )
            args = Namespace(
                run_dir=str(run_dir),
                model_path=str(model_path / ".." / "best"),
                pseudo_model_variant="last",
                max_target_unlabeled=0,
                no_task_prefix=True,
                batch_size=1,
                max_new_tokens=32,
                num_beams=1,
                cuda="0",
                no_constrained_decoding=True,
                length_penalty=1.0,
                pseudo_base_weight=0.5,
                high_precision_max_triplets=1,
                high_precision_max_token_distance=5,
                fixed_changed_min_score=0.65,
                fixed_changed_weight=0.35,
                pseudo_source_tag="extractor_a",
            )

            with patch.object(
                pipeline,
                "generate_texts",
                return_value=[_label(("screen", "bright", "pos"))],
            ):
                pipeline.pseudo(args)

            analysis = json.loads(
                (run_dir / "target_pseudo_analysis.json").read_text(encoding="utf-8")
            )
            state = json.loads(
                (run_dir / "target_pseudo_generation_state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(analysis["model_path"], str(model_path.resolve()))
        self.assertEqual(analysis["pseudo_source_tag"], "extractor_a")
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["resolved_model_path"], str(model_path.resolve()))
        self.assertEqual(state["pseudo_source_tag"], "extractor_a")

    def test_pseudo_leaves_in_progress_state_when_generation_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            model_path = run_dir / "models" / "extractor_a" / "best"
            model_path.mkdir(parents=True)
            pipeline.write_jsonl(
                run_dir / "target_unlabeled.jsonl",
                [{"id": "t1", "text": "Bright screen."}],
            )
            args = Namespace(
                run_dir=str(run_dir),
                model_path=str(model_path),
                pseudo_model_variant="last",
                max_target_unlabeled=0,
                no_task_prefix=True,
                batch_size=1,
                max_new_tokens=32,
                num_beams=1,
                cuda="0",
                no_constrained_decoding=True,
                length_penalty=1.0,
                pseudo_base_weight=0.5,
                high_precision_max_triplets=1,
                high_precision_max_token_distance=5,
                fixed_changed_min_score=0.65,
                fixed_changed_weight=0.35,
                pseudo_source_tag="extractor_a",
            )

            with patch.object(pipeline, "generate_texts", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    pipeline.pseudo(args)

            state = json.loads(
                (run_dir / "target_pseudo_generation_state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(state["status"], "in_progress")
        self.assertEqual(state["resolved_model_path"], str(model_path.resolve()))
        self.assertEqual(state["pseudo_source_tag"], "extractor_a")

    def test_source_triplet_count_weights_only_change_source_gold_rows(self):
        one_label = _label(("food", "great", "pos"))
        three_label = _label(
            ("food", "great", "pos"),
            ("service", "slow", "neg"),
            ("room", "clean", "pos"),
        )
        rows = [
            {"id": "s1", "label": one_label, "augmentation": "source_gold"},
            {"id": "s3", "label": three_label, "augmentation": "source_gold"},
            {
                "id": "p3",
                "label": three_label,
                "augmentation": "target_pseudo",
                "sample_weight": 0.65,
            },
            {
                "id": "a3",
                "label": three_label,
                "augmentation": "masked_aspect_channel",
                "sample_weight": 0.2,
            },
        ]

        weighted, stats = pipeline.assign_source_triplet_count_weights(rows)

        self.assertEqual(weighted[0]["sample_weight"], 1.0)
        self.assertEqual(weighted[1]["sample_weight"], 1.25)
        self.assertEqual(weighted[2]["sample_weight"], 0.65)
        self.assertEqual(weighted[3]["sample_weight"], 0.2)
        self.assertNotIn("source_triplet_count", weighted[2])
        self.assertNotIn("source_triplet_count", weighted[3])
        self.assertEqual([row["label"] for row in weighted], [row["label"] for row in rows])
        self.assertEqual(stats["count1"], {
            "rows": 1,
            "weight_mean": 1.0,
            "weight_min": 1.0,
            "weight_max": 1.0,
        })
        self.assertEqual(stats["count3"]["rows"], 1)
        self.assertEqual(stats["count3"]["weight_mean"], 1.25)
        for bucket in ("count2", "count4plus"):
            self.assertEqual(stats[bucket], {
                "rows": 0,
                "weight_mean": None,
                "weight_min": None,
                "weight_max": None,
            })
        self.assertEqual(stats["sample_weight_summary"]["count"], 2)
        self.assertEqual(
            sum(stats[bucket]["rows"] for bucket in ("count1", "count2", "count3", "count4plus")),
            stats["sample_weight_summary"]["count"],
        )

    def test_source_triplet_count_weights_reject_non_positive_or_non_finite_weights(self):
        rows = [{"id": "s1", "label": _label(("food", "great", "pos"))}]
        defaults = {
            "count1_weight": 1.0,
            "count2_weight": 1.15,
            "count3_weight": 1.25,
            "count4plus_weight": 1.30,
        }
        for name in defaults:
            for invalid in (float("nan"), float("inf"), 0.0, -0.1):
                with self.subTest(name=name, invalid=invalid):
                    weights = {**defaults, name: invalid}
                    with self.assertRaises(ValueError):
                        pipeline.assign_source_triplet_count_weights(rows, **weights)

    def test_dynamic_multitriplet_config_tag_is_deterministic_and_decimal_safe(self):
        self.assertEqual(
            pipeline.dynamic_multitriplet_config_tag(1.0, 1.15, 1.25, 1.30),
            "dynamic_multitriplet_c1w100_c2w115_c3w125_c4pw130",
        )
        self.assertNotEqual(
            pipeline.dynamic_multitriplet_config_tag(1.0, 1.15, 1.251, 1.30),
            pipeline.dynamic_multitriplet_config_tag(1.0, 1.15, 1.252, 1.30),
        )

    def test_pipeline_prepare_cli_rejects_invalid_source_weights(self):
        script = Path(__file__).resolve().parent / "t5_aste_pipeline.py"
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
                            str(script),
                            "prepare",
                            "--source_dataset",
                            "rest16",
                            "--target_dataset",
                            "laptop14",
                            "--run_dir",
                            temp_dir,
                            option,
                            value,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)

    @staticmethod
    def _prepare_args(run_dir: Path, **overrides) -> Namespace:
        values = {
            "source_dataset": "rest16",
            "target_dataset": "laptop14",
            "run_dir": str(run_dir),
            "dev_ratio": 0.1,
            "seed": 13,
            "augment_prompt_style": "label_to_text",
            "augment_channel_mode": "all",
            "domain_prefix_style": "none",
            "generator_output_tag": "",
            "no_task_prefix": True,
        }
        values.update(overrides)
        return Namespace(**values)

    @staticmethod
    def _split_rows():
        three_label = _label(
            ("food", "great", "pos"),
            ("service", "slow", "neg"),
            ("room", "clean", "pos"),
        )
        source_train = [
            {"id": "s1", "text": "Great food.", "label": _label(("food", "great", "pos"))},
            {"id": "s3", "text": "Mixed stay.", "label": three_label},
        ]
        source_dev = [
            {"id": "d1", "text": "Slow service.", "label": _label(("service", "slow", "neg"))},
        ]
        target_train = [
            {"id": "t1", "text": "Bright screen.", "label": _label(("screen", "bright", "pos"))},
        ]
        target_test = [
            {"id": "u1", "text": "Loud fan.", "label": _label(("fan", "loud", "neg"))},
        ]
        return {
            ("rest16", "train"): source_train,
            ("rest16", "dev"): source_dev,
            ("laptop14", "train"): target_train,
            ("laptop14", "test"): target_test,
        }

    def test_prepare_dynamic_multitriplet_writes_weighted_extract_train_and_analysis(self):
        splits = self._split_rows()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "dynamic"
            args = self._prepare_args(
                run_dir,
                dynamic_multitriplet=True,
                source_count1_weight=1.0,
                source_count2_weight=1.15,
                source_count3_weight=1.25,
                source_count4plus_weight=1.30,
            )
            with patch.object(pipeline, "load_split", side_effect=lambda dataset, split: splits[(dataset, split)]):
                pipeline.prepare(args)

            config_tag = pipeline.dynamic_multitriplet_config_tag(1.0, 1.15, 1.25, 1.30)
            extract_path = run_dir / f"extract_train_{config_tag}.jsonl"
            analysis_path = run_dir / f"extract_train_multitriplet_weight_analysis_{config_tag}.json"
            extract_rows = [json.loads(line) for line in extract_path.read_text(encoding="utf-8").splitlines()]
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            legacy_extract_exists = (run_dir / "extract_train.jsonl").exists()

        self.assertEqual([row["sample_weight"] for row in extract_rows], [1.0, 1.25])
        self.assertEqual(extract_rows[1]["source_triplet_count"], 3)
        self.assertEqual(extract_rows[1]["source_triplet_count_bucket"], "count3")
        expected_extract_rows = pipeline.to_extract_rows(
            splits[("rest16", "train")], use_task_prefix=False
        )
        self.assertEqual(extract_rows[1]["target"], expected_extract_rows[1]["target"])
        self.assertEqual(analysis["count3"]["rows"], 1)
        self.assertFalse(legacy_extract_exists)

    def test_prepare_keeps_legacy_and_dynamic_files_isolated_in_both_orders(self):
        splits = self._split_rows()
        config_tag = "dynamic_multitriplet_c1w100_c2w115_c3w125_c4pw130"
        dynamic_name = f"extract_train_{config_tag}.jsonl"

        def run_prepare(run_dir: Path, dynamic: bool) -> None:
            args = self._prepare_args(
                run_dir,
                dynamic_multitriplet=dynamic,
                source_count1_weight=1.0,
                source_count2_weight=1.15,
                source_count3_weight=1.25,
                source_count4plus_weight=1.30,
            )
            pipeline.prepare(args)

        with patch.object(pipeline, "load_split", side_effect=lambda dataset, split: splits[(dataset, split)]):
            with tempfile.TemporaryDirectory() as temp_dir:
                for index, sequence in enumerate(((True, False, True), (False, True, False))):
                    with self.subTest(sequence=sequence):
                        run_dir = Path(temp_dir) / str(index)
                        snapshots = {}
                        for dynamic in sequence:
                            run_prepare(run_dir, dynamic)
                            current_name = dynamic_name if dynamic else "extract_train.jsonl"
                            current_path = run_dir / current_name
                            self.assertTrue(current_path.exists())
                            for name, content in snapshots.items():
                                self.assertEqual((run_dir / name).read_bytes(), content)
                            snapshots[current_name] = current_path.read_bytes()

    def test_prepare_keeps_different_dynamic_weight_configs_isolated(self):
        splits = self._split_rows()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "configs"
            with patch.object(pipeline, "load_split", side_effect=lambda dataset, split: splits[(dataset, split)]):
                for count3_weight in (1.25, 1.251):
                    pipeline.prepare(
                        self._prepare_args(
                            run_dir,
                            dynamic_multitriplet=True,
                            source_count1_weight=1.0,
                            source_count2_weight=1.15,
                            source_count3_weight=count3_weight,
                            source_count4plus_weight=1.30,
                        )
                    )

            first_tag = pipeline.dynamic_multitriplet_config_tag(1.0, 1.15, 1.25, 1.30)
            second_tag = pipeline.dynamic_multitriplet_config_tag(1.0, 1.15, 1.251, 1.30)
            first_rows = [json.loads(line) for line in (run_dir / f"extract_train_{first_tag}.jsonl").read_text(encoding="utf-8").splitlines()]
            second_rows = [json.loads(line) for line in (run_dir / f"extract_train_{second_tag}.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(first_rows[1]["sample_weight"], 1.25)
        self.assertEqual(second_rows[1]["sample_weight"], 1.251)

    def test_prepare_legacy_namespace_keeps_extract_train_compatible(self):
        splits = self._split_rows()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "legacy"
            args = self._prepare_args(run_dir)
            with patch.object(pipeline, "load_split", side_effect=lambda dataset, split: splits[(dataset, split)]):
                pipeline.prepare(args)

            extract_rows = [json.loads(line) for line in (run_dir / "extract_train.jsonl").read_text(encoding="utf-8").splitlines()]
            analysis_exists = (run_dir / "extract_train_multitriplet_weight_analysis.json").exists()

        self.assertEqual(
            extract_rows,
            pipeline.to_extract_rows(splits[("rest16", "train")], use_task_prefix=False),
        )
        self.assertFalse(analysis_exists)

    def test_dynamic_high_precision_keeps_valid_four_triplets_and_filters_bad_triplet(self):
        four_label = _label(
            ("food", "Good", "pos"),
            ("service", "slow", "neg"),
            ("room", "clean", "pos"),
            ("staff", "friendly", "pos"),
        )
        partial_label = _label(
            ("keyboard", "cramped", "neg"),
            ("screen", "bright", "pos"),
            ("battery", "bright", "pos"),
        )
        rows = [
            {
                "id": "full",
                "text": "Good food, slow service, clean room, and friendly staff.",
                "label": four_label,
                "sample_weight": 0.65,
                "quality_flags": {"all_terms_in_text": True},
            },
            {
                "id": "partial",
                "text": "The battery runs all day, but the keyboard is cramped and the screen is bright.",
                "label": partial_label,
                "sample_weight": 0.65,
                "quality_flags": {"all_terms_in_text": True},
            },
        ]
        original_rows = [dict(row) for row in rows]

        selected, stats = pipeline.select_dynamic_high_precision_pseudo_rows(
            rows,
            min_weight=0.65,
            max_token_distance=5,
        )

        by_id = {row["id"]: row for row in selected}
        self.assertEqual(len(selected), 2)
        self.assertEqual(by_id["full"]["label"], pipeline.canonicalize_triplet_text(four_label))
        self.assertEqual(by_id["full"]["dynamic_triplet_count_after"], 4)
        self.assertEqual(by_id["partial"]["dynamic_triplet_count_before"], 3)
        self.assertEqual(by_id["partial"]["dynamic_triplet_count_after"], 2)
        self.assertIn("battery", by_id["partial"]["dynamic_original_label"])
        self.assertNotIn("battery", by_id["partial"]["label"])
        self.assertAlmostEqual(by_id["partial"]["sample_weight"], 0.368333, places=6)
        self.assertEqual(stats["fully_kept_rows"], 1)
        self.assertEqual(stats["partially_kept_rows"], 1)
        self.assertEqual(stats["removed_triplets_by_distance_too_far"], 1)
        self.assertEqual(rows, original_rows)

    def test_dynamic_high_precision_strict_rejects_partially_filtered_rows(self):
        partial_label = _label(
            ("keyboard", "cramped", "neg"),
            ("screen", "bright", "pos"),
            ("battery", "bright", "pos"),
        )
        rows = [
            {
                "id": "partial",
                "text": "The battery runs all day, but the keyboard is cramped and the screen is bright.",
                "label": partial_label,
                "sample_weight": 0.65,
                "quality_flags": {"all_terms_in_text": True},
            }
        ]

        selected, stats = pipeline.select_dynamic_high_precision_pseudo_rows(
            rows,
            min_weight=0.65,
            max_token_distance=5,
            strict=True,
        )

        self.assertEqual(selected, [])
        self.assertTrue(stats["strict"])
        self.assertEqual(stats["selected_rows"], 0)
        self.assertEqual(stats["rejected_partial_after_triplet_filter"], 1)
        self.assertEqual(stats["removed_triplets_by_distance_too_far"], 1)

    def test_dynamic_pseudo_filter_tag_records_strict_mode(self):
        self.assertEqual(pipeline.dynamic_pseudo_filter_tag(5), "dynamic_dist5")
        self.assertEqual(
            pipeline.dynamic_pseudo_filter_tag(5, strict=True),
            "dynamic_strict_dist5",
        )

    def test_complete_multi_dynamic_adds_only_complete_three_plus_rows(self):
        base_rows = [
            {
                "id": "hp1",
                "text": "Bright screen.",
                "label": _label(("screen", "bright", "pos")),
                "sample_weight": 0.65,
            }
        ]
        dynamic_rows = [
            {
                "id": "complete3",
                "text": "Bright screen, cramped keyboard, and long battery.",
                "label": _label(
                    ("screen", "bright", "pos"),
                    ("keyboard", "cramped", "neg"),
                    ("battery", "long", "pos"),
                ),
                "sample_weight": 0.65,
                "dynamic_high_precision_pseudo": True,
                "dynamic_strict_high_precision_pseudo": True,
                "dynamic_triplet_count_before": 3,
                "dynamic_triplet_count_after": 3,
            },
            {
                "id": "complete2",
                "text": "Bright screen and cramped keyboard.",
                "label": _label(("screen", "bright", "pos"), ("keyboard", "cramped", "neg")),
                "sample_weight": 0.65,
                "dynamic_high_precision_pseudo": True,
                "dynamic_strict_high_precision_pseudo": True,
                "dynamic_triplet_count_before": 2,
                "dynamic_triplet_count_after": 2,
            },
            {
                "id": "cropped3",
                "text": "Bright screen and cramped keyboard.",
                "label": _label(("screen", "bright", "pos"), ("keyboard", "cramped", "neg")),
                "sample_weight": 0.65,
                "dynamic_high_precision_pseudo": True,
                "dynamic_strict_high_precision_pseudo": False,
                "dynamic_triplet_count_before": 3,
                "dynamic_triplet_count_after": 2,
            },
            {
                "id": "hp1",
                "text": "Bright screen.",
                "label": _label(
                    ("screen", "bright", "pos"),
                    ("keyboard", "cramped", "neg"),
                    ("battery", "long", "pos"),
                ),
                "sample_weight": 0.65,
                "dynamic_high_precision_pseudo": True,
                "dynamic_strict_high_precision_pseudo": True,
                "dynamic_triplet_count_before": 3,
                "dynamic_triplet_count_after": 3,
            },
        ]

        merged, analysis = pipeline.build_complete_multitriplet_dynamic_pseudo_rows(
            base_rows,
            dynamic_rows,
            extra_weight=0.2,
            min_triplets=3,
        )

        self.assertEqual([row["id"] for row in merged], ["hp1", "complete3"])
        self.assertEqual(merged[1]["sample_weight"], 0.2)
        self.assertEqual(merged[1]["pseudo_mix_source"], "dynamic_strict_3plus_extra")
        self.assertEqual(analysis["dynamic_3plus_candidates"], 3)
        self.assertEqual(analysis["dynamic_extra_rows"], 1)
        self.assertEqual(analysis["dynamic_too_few_triplets_rejected"], 1)
        self.assertEqual(analysis["dynamic_not_strict_rejected"], 1)
        self.assertEqual(analysis["duplicate_rows_rejected"], 1)

    def test_select_dynamic_pseudo_command_writes_complete_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            output_dir = run_dir / "pseudo_variants" / "dynamic_dist5"
            row = {
                "id": "t1",
                "text": "Bright screen and cramped keyboard.",
                "label": _label(("screen", "Bright", "pos"), ("keyboard", "cramped", "neg")),
                "sample_weight": 0.65,
                "quality_flags": {"all_terms_in_text": True},
            }
            pipeline.write_jsonl(run_dir / "target_pseudo.jsonl", [row])
            pipeline.write_jsonl(
                run_dir / "target_train_gold_analysis.jsonl",
                [{**row, "label": _label(("screen", "Bright", "pos"), ("keyboard", "cramped", "neg"))}],
            )
            (run_dir / "target_pseudo_analysis.json").write_text(
                json.dumps(
                    {
                        "model_path": str((run_dir / "models" / "extractor" / "best").resolve()),
                        "pseudo_source_tag": "extractor_tag",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "target_pseudo_generation_state.json").write_text(
                json.dumps({"status": "complete"}),
                encoding="utf-8",
            )

            pipeline.select_dynamic_pseudo(
                Namespace(
                    run_dir=str(run_dir),
                    output_dir=str(output_dir),
                    min_pseudo_weight=0.65,
                    high_precision_max_token_distance=5,
                    dynamic_strict=False,
                )
            )

            rows = [json.loads(line) for line in (output_dir / "target_pseudo_high_precision.jsonl").read_text(encoding="utf-8").splitlines()]
            analysis = json.loads((output_dir / "target_pseudo_high_precision_analysis.json").read_text(encoding="utf-8"))
            state = json.loads((output_dir / "target_pseudo_generation_state.json").read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["dynamic_high_precision_pseudo"])
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["selection_mode"], "dynamic_high_precision")
        self.assertFalse(state["strict"])
        self.assertEqual(state["base_pseudo_source_tag"], "extractor_tag")
        self.assertIn("hidden_gold_eval", analysis)

    def test_select_complete_dynamic_pseudo_command_writes_combined_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            output_dir = run_dir / "pseudo_variants" / "hp1_complete2_dist5_w025_dynamic_strict3plus_dist5_w020"
            base_file = run_dir / "pseudo_variants" / "hp1_complete2_dist5_w025" / "target_pseudo_high_precision.jsonl"
            dynamic_file = run_dir / "pseudo_variants" / "dynamic_strict_dist5" / "target_pseudo_high_precision.jsonl"
            base_row = {
                "id": "hp1",
                "text": "Bright screen.",
                "label": _label(("screen", "bright", "pos")),
                "sample_weight": 0.65,
            }
            dynamic_row = {
                "id": "complete3",
                "text": "Bright screen, cramped keyboard, and long battery.",
                "label": _label(
                    ("screen", "bright", "pos"),
                    ("keyboard", "cramped", "neg"),
                    ("battery", "long", "pos"),
                ),
                "sample_weight": 0.65,
                "dynamic_high_precision_pseudo": True,
                "dynamic_strict_high_precision_pseudo": True,
                "dynamic_triplet_count_before": 3,
                "dynamic_triplet_count_after": 3,
            }
            pipeline.write_jsonl(base_file, [base_row])
            pipeline.write_jsonl(dynamic_file, [dynamic_row])

            pipeline.select_complete_dynamic_pseudo(
                Namespace(
                    run_dir=str(run_dir),
                    output_dir=str(output_dir),
                    base_pseudo_file=str(base_file),
                    dynamic_pseudo_file=str(dynamic_file),
                    dynamic_extra_weight=0.2,
                    dynamic_min_triplets=3,
                )
            )

            rows = [json.loads(line) for line in (output_dir / "target_pseudo_high_precision.jsonl").read_text(encoding="utf-8").splitlines()]
            analysis = json.loads((output_dir / "target_pseudo_high_precision_analysis.json").read_text(encoding="utf-8"))

        self.assertEqual([row["id"] for row in rows], ["hp1", "complete3"])
        self.assertEqual(rows[1]["pseudo_mix_source"], "dynamic_strict_3plus_extra")
        self.assertEqual(analysis["dynamic_extra_rows"], 1)
        self.assertEqual(analysis["dynamic_pseudo_file"], str(dynamic_file))

    def test_select_dynamic_pseudo_rejects_incomplete_base_generation_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            output_dir = run_dir / "pseudo_variants" / "dynamic_dist5"
            pipeline.write_jsonl(
                run_dir / "target_pseudo.jsonl",
                [
                    {
                        "id": "t1",
                        "text": "Bright screen.",
                        "label": _label(("screen", "Bright", "pos")),
                        "sample_weight": 0.65,
                        "quality_flags": {"all_terms_in_text": True},
                    }
                ],
            )
            (run_dir / "target_pseudo_generation_state.json").write_text(
                json.dumps({"status": "in_progress"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "in_progress"):
                pipeline.select_dynamic_pseudo(
                    Namespace(
                        run_dir=str(run_dir),
                        output_dir=str(output_dir),
                        min_pseudo_weight=0.65,
                        high_precision_max_token_distance=5,
                    )
                )

            state = json.loads((output_dir / "target_pseudo_generation_state.json").read_text(encoding="utf-8"))

        self.assertEqual(state["status"], "in_progress")

    def test_triplet_count_bucket_boundaries(self):
        expected = {
            0: "count1",
            1: "count1",
            2: "count2",
            3: "count3",
            4: "count4plus",
            7: "count4plus",
        }

        for count, bucket in expected.items():
            with self.subTest(count=count):
                self.assertEqual(triplet_count_bucket(count), bucket)

    def test_metric_return_types_are_precise(self):
        self.assertEqual(
            get_type_hints(micro_f1_by_triplet_count)["return"],
            dict[str, dict[str, int | float]],
        )
        self.assertEqual(
            get_type_hints(triplet_count_diagnostics)["return"],
            dict[str, int | float],
        )

    def test_empty_inputs_return_all_zero_metrics(self):
        metrics = micro_f1_by_triplet_count([], [])
        expected_bucket = {
            "rows": 0,
            "precision": 0.0,
            "recall": 0.0,
            "micro_f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

        self.assertEqual(
            set(metrics),
            {"count1", "count2", "count3", "count4plus"},
        )
        for bucket in metrics:
            with self.subTest(bucket=bucket):
                self.assertEqual(metrics[bucket], expected_bucket)

        self.assertEqual(
            triplet_count_diagnostics([], []),
            {
                "rows": 0,
                "exact_count_rows": 0,
                "under_generated_rows": 0,
                "over_generated_rows": 0,
                "exact_count_accuracy": 0.0,
            },
        )

    def test_micro_f1_is_grouped_by_gold_triplet_count(self):
        one_gold = _label(("food", "great", "pos"))
        three_gold = _label(
            ("food", "great", "pos"),
            ("service", "slow", "neg"),
            ("room", "clean", "pos"),
        )
        predictions = [one_gold, _label(("food", "great", "pos"))]

        metrics = micro_f1_by_triplet_count(predictions, [one_gold, three_gold])

        self.assertEqual(
            metrics["count1"],
            {
                "rows": 1,
                "precision": 1.0,
                "recall": 1.0,
                "micro_f1": 1.0,
                "tp": 1,
                "fp": 0,
                "fn": 0,
            },
        )
        self.assertEqual(
            metrics["count3"],
            {
                "rows": 1,
                "precision": 1.0,
                "recall": 1 / 3,
                "micro_f1": 0.5,
                "tp": 1,
                "fp": 0,
                "fn": 2,
            },
        )
        self.assertEqual(metrics["count2"]["rows"], 0)
        self.assertEqual(metrics["count2"]["micro_f1"], 0.0)
        self.assertEqual(metrics["count4plus"]["rows"], 0)
        self.assertEqual(metrics["count4plus"]["micro_f1"], 0.0)

    def test_triplet_count_diagnostics(self):
        one = _label(("food", "great", "pos"))
        two = _label(("food", "great", "pos"), ("service", "slow", "neg"))
        three = _label(
            ("food", "great", "pos"),
            ("service", "slow", "neg"),
            ("room", "clean", "pos"),
        )

        diagnostics = triplet_count_diagnostics(
            predictions=[one, one, three],
            golds=[one, two, two],
        )

        self.assertEqual(
            diagnostics,
            {
                "rows": 3,
                "exact_count_rows": 1,
                "under_generated_rows": 1,
                "over_generated_rows": 1,
                "exact_count_accuracy": 1 / 3,
            },
        )

    def test_length_mismatches_follow_micro_f1_zip_truncation(self):
        one = _label(("food", "great", "pos"))
        two = _label(("food", "great", "pos"), ("service", "slow", "neg"))
        expected_count1 = {
            "rows": 1,
            "precision": 1.0,
            "recall": 1.0,
            "micro_f1": 1.0,
            "tp": 1,
            "fp": 0,
            "fn": 0,
        }
        expected_diagnostics = {
            "rows": 1,
            "exact_count_rows": 1,
            "under_generated_rows": 0,
            "over_generated_rows": 0,
            "exact_count_accuracy": 1.0,
        }

        mismatched_inputs = (
            ([one, two], [one]),
            ([one], [one, two]),
        )
        for predictions, golds in mismatched_inputs:
            with self.subTest(predictions=len(predictions), golds=len(golds)):
                metrics = micro_f1_by_triplet_count(predictions, golds)
                self.assertEqual(metrics["count1"], expected_count1)
                self.assertEqual(sum(bucket["rows"] for bucket in metrics.values()), 1)
                self.assertEqual(
                    triplet_count_diagnostics(predictions, golds),
                    expected_diagnostics,
                )


if __name__ == "__main__":
    unittest.main()

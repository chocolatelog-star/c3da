import json
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
        ]

        weighted, stats = pipeline.assign_source_triplet_count_weights(rows)

        self.assertEqual(weighted[0]["sample_weight"], 1.0)
        self.assertEqual(weighted[1]["sample_weight"], 1.25)
        self.assertEqual(weighted[2]["sample_weight"], 0.65)
        self.assertNotIn("source_triplet_count", weighted[2])
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
        self.assertEqual(stats["sample_weight_summary"]["count"], 3)

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

            extract_rows = [json.loads(line) for line in (run_dir / "extract_train.jsonl").read_text(encoding="utf-8").splitlines()]
            analysis = json.loads((run_dir / "extract_train_multitriplet_weight_analysis.json").read_text(encoding="utf-8"))

        self.assertEqual([row["sample_weight"] for row in extract_rows], [1.0, 1.25])
        self.assertEqual(extract_rows[1]["source_triplet_count"], 3)
        self.assertEqual(extract_rows[1]["source_triplet_count_bucket"], "count3")
        expected_extract_rows = pipeline.to_extract_rows(
            splits[("rest16", "train")], use_task_prefix=False
        )
        self.assertEqual(extract_rows[1]["target"], expected_extract_rows[1]["target"])
        self.assertEqual(analysis["count3"]["rows"], 1)

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

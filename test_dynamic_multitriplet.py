import unittest
from typing import get_type_hints

from t5_aste_data import (
    micro_f1_by_triplet_count,
    triplet_count_bucket,
    triplet_count_diagnostics,
    triplets_to_text,
)


def _label(*triplets):
    return triplets_to_text(triplets)


class DynamicMultiTripletTest(unittest.TestCase):
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

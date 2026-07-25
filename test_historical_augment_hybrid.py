from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from t5_aste_data import read_jsonl, write_jsonl
from t5_aste_pipeline import extract_selected_augment_from_final_train, extract_selected_augment_rows


class HistoricalAugmentHybridTest(unittest.TestCase):
    def test_extracts_only_selected_augment_rows_and_strips_training_fields(self) -> None:
        selected = {
            "text": "The screen is bright.",
            "label": "<pos> screen <opinion> bright",
            "augmentation": "masked_aspect_channel",
            "selected_augmentation": True,
            "sample_weight": 0.2,
            "input": "old input",
            "target": "old target",
            "final_weight_flags": {"base_weight": 0.2},
        }
        rows, stats = extract_selected_augment_rows(
            [
                {"text": "Source.", "label": "<pos> source <opinion> good"},
                {
                    "text": "Pseudo.",
                    "label": "<pos> pseudo <opinion> good",
                    "augmentation": "target_pseudo",
                    "selected_augmentation": True,
                },
                selected,
                dict(selected),
            ],
            source_name="historical_final_train.jsonl",
            expected_rows=1,
            selected_weight=0.2,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_weight"], 0.2)
        self.assertEqual(rows[0]["hybrid_augment_source"], "historical_final_train.jsonl")
        self.assertNotIn("input", rows[0])
        self.assertNotIn("target", rows[0])
        self.assertNotIn("final_weight_flags", rows[0])
        self.assertEqual(stats["selected_rows"], 1)
        self.assertEqual(stats["duplicate_rows_rejected"], 1)
        self.assertEqual(stats["triplet_count_distribution"], {"1": 1})

    def test_rejects_unexpected_extracted_row_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2 selected augment rows, found 1"):
            extract_selected_augment_rows(
                [
                    {
                        "text": "The screen is bright.",
                        "label": "<pos> screen <opinion> bright",
                        "augmentation": "masked_aspect_channel",
                        "selected_augmentation": True,
                    }
                ],
                source_name="historical_final_train.jsonl",
                expected_rows=2,
                selected_weight=0.2,
            )

    def test_command_writes_rows_and_source_hash_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_file = root / "historical_final_train.jsonl"
            output_file = root / "historical_augment.jsonl"
            analysis_file = root / "historical_augment_analysis.json"
            write_jsonl(
                source_file,
                [
                    {
                        "text": "The screen is bright.",
                        "label": "<pos> screen <opinion> bright",
                        "augmentation": "masked_aspect_channel",
                        "selected_augmentation": True,
                    }
                ],
            )

            extract_selected_augment_from_final_train(
                argparse.Namespace(
                    source_final_train_file=str(source_file),
                    output_file=str(output_file),
                    analysis_file=str(analysis_file),
                    expected_rows=1,
                    selected_weight=0.2,
                )
            )

            self.assertEqual(len(read_jsonl(output_file)), 1)
            analysis = json.loads(analysis_file.read_text(encoding="utf-8"))
            self.assertEqual(analysis["selected_rows"], 1)
            self.assertEqual(len(analysis["source_sha256"]), 64)
            self.assertEqual(analysis["output_file"], str(output_file))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from t5_aste_pipeline import (
    build_complete_multitriplet_pseudo_rows,
    build_final_train_from_files,
    select_complete_multi_pseudo,
)


class CompleteMultitripletPseudoTest(unittest.TestCase):
    def test_keeps_hp1_and_adds_only_unchanged_complete_double_rows(self) -> None:
        hp1_rows = [
            {
                "id": "base-1",
                "text": "The battery is good.",
                "label": "<pos> battery <opinion> good",
                "sample_weight": 0.65,
            }
        ]
        hp2_rows = [
            {
                "id": "double-good",
                "text": "The screen is bright but the battery is weak.",
                "label": "<pos> screen <opinion> bright ; <neg> battery <opinion> weak",
                "sample_weight": 0.65,
                "high_precision_original_label": "<pos> screen <opinion> bright ; <neg> battery <opinion> weak",
                "high_precision_triplet_count_before": 2,
                "high_precision_triplet_count_after": 2,
            },
            {
                "id": "double-cropped",
                "text": "The keyboard is usable but the touchpad is too far away.",
                "label": "<pos> keyboard <opinion> usable",
                "sample_weight": 0.65,
                "high_precision_original_label": "<pos> keyboard <opinion> usable ; <neg> touchpad <opinion> too far away",
                "high_precision_triplet_count_before": 2,
                "high_precision_triplet_count_after": 1,
            },
            {
                "id": "base-1",
                "text": "The battery is good.",
                "label": "<pos> battery <opinion> good ; <neg> screen <opinion> dim",
                "sample_weight": 0.65,
                "high_precision_original_label": "<pos> battery <opinion> good ; <neg> screen <opinion> dim",
                "high_precision_triplet_count_before": 2,
                "high_precision_triplet_count_after": 2,
            },
        ]

        rows, analysis = build_complete_multitriplet_pseudo_rows(
            hp1_rows,
            hp2_rows,
            extra_weight=0.25,
        )

        self.assertEqual([row["id"] for row in rows], ["base-1", "double-good"])
        self.assertEqual(rows[0]["sample_weight"], 0.65)
        self.assertEqual(rows[1]["sample_weight"], 0.25)
        self.assertEqual(rows[1]["pseudo_mix_source"], "complete_multi2_extra")
        self.assertEqual(analysis["base_rows"], 1)
        self.assertEqual(analysis["candidate_rows"], 3)
        self.assertEqual(analysis["complete_multi2_candidates"], 2)
        self.assertEqual(analysis["cropped_multi2_rejected"], 1)
        self.assertEqual(analysis["duplicate_rows_rejected"], 1)
        self.assertEqual(analysis["extra_rows"], 1)
        self.assertEqual(analysis["final_rows"], 2)

    def test_rejects_invalid_extra_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "extra_weight"):
            build_complete_multitriplet_pseudo_rows([], [], extra_weight=0.0)

    def test_command_writes_isolated_rows_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            output_dir = run_dir / "pseudo_variants" / "hp1_complete2_dist5_w025"
            run_dir.mkdir()
            hp1_path = run_dir / "target_pseudo_high_precision.jsonl"
            hp1_path.write_text(
                json.dumps(
                    {
                        "id": "base-1",
                        "text": "The battery is good.",
                        "label": "<pos> battery <opinion> good",
                        "sample_weight": 0.65,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "target_pseudo.jsonl").write_text(
                json.dumps(
                    {
                        "id": "double-good",
                        "text": "The screen is bright and the battery is weak.",
                        "label": "<pos> screen <opinion> bright ; <neg> battery <opinion> weak",
                        "sample_weight": 0.65,
                        "quality_flags": {"all_terms_in_text": True},
                        "pred_fixed_changed": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "target_train_gold_analysis.jsonl").write_text(
                json.dumps(
                    {
                        "id": "double-good",
                        "text": "The screen is bright and the battery is weak.",
                        "label": "<pos> screen <opinion> bright ; <neg> battery <opinion> weak",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            select_complete_multi_pseudo(
                argparse.Namespace(
                    run_dir=str(run_dir),
                    output_dir=str(output_dir),
                    base_pseudo_file=str(hp1_path),
                    min_pseudo_weight=0.65,
                    high_precision_max_token_distance=5,
                    complete_multi_extra_weight=0.25,
                )
            )

            rows = [json.loads(line) for line in (output_dir / "target_pseudo_high_precision.jsonl").read_text(encoding="utf-8").splitlines()]
            analysis = json.loads((output_dir / "target_pseudo_high_precision_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual([row["id"] for row in rows], ["base-1", "double-good"])
            self.assertEqual(analysis["extra_rows"], 1)
            self.assertEqual(analysis["selected_rows"], 2)
            self.assertEqual(analysis["hidden_gold_eval"]["selected_rows"], 2)

    def test_build_final_train_reuses_selected_augment_without_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()

            def write_rows(name: str, rows: list[dict]) -> Path:
                path = run_dir / name
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                return path

            write_rows(
                "source_train.jsonl",
                [{"id": "source", "text": "Good battery.", "label": "<pos> battery <opinion> good"}],
            )
            write_rows(
                "source_dev.jsonl",
                [{"id": "dev", "text": "Dim screen.", "label": "<neg> screen <opinion> dim"}],
            )
            pseudo_path = write_rows(
                "pseudo.jsonl",
                [{"id": "pseudo", "text": "Weak battery.", "label": "<neg> battery <opinion> weak", "sample_weight": 0.25, "augmentation": "target_pseudo"}],
            )
            augment_path = write_rows(
                "augment.jsonl",
                [{"id": "augment", "text": "Bright display.", "label": "<pos> display <opinion> bright", "sample_weight": 0.2, "augmentation": "masked_aspect_channel"}],
            )

            build_final_train_from_files(
                argparse.Namespace(
                    run_dir=str(run_dir),
                    pseudo_train_file=str(pseudo_path),
                    selected_augment_file=str(augment_path),
                    final_train_output_tag="complete_multi2_w025",
                    no_final_train_source=False,
                    no_task_prefix=True,
                    final_multi_triplet_gain=0.0,
                    final_neutral_gain=0.0,
                    final_max_weight=1.0,
                )
            )

            train_path = run_dir / "final_train_complete_multi2_w025.jsonl"
            rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
            analysis = json.loads((run_dir / "final_train_composition_analysis_complete_multi2_w025.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 3)
            self.assertEqual(analysis["source_rows_used"], 1)
            self.assertEqual(analysis["pseudo_rows_used"], 1)
            self.assertEqual(analysis["selected_augmented_rows"], 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import t5_absa_train
import t5_aste_pipeline


class EvaluateOutputIsolationTest(unittest.TestCase):
    def test_output_tag_keeps_default_metrics_and_predictions_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            row = {
                "id": "t1",
                "text": "The battery is good.",
                "label": "<pos> battery <opinion> good",
            }
            (run_dir / "target_test.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )
            model_dir = run_dir / "model"
            model_dir.mkdir()
            args = Namespace(
                run_dir=str(run_dir),
                model_path=str(model_dir),
                batch_size=2,
                max_new_tokens=96,
                num_beams=4,
                length_penalty=1.0,
                cuda="0",
                no_constrained_decoding=True,
                no_task_prefix=True,
                output_tag="hp2_dist5",
            )
            original_generate = t5_aste_pipeline.generate_texts
            t5_aste_pipeline.generate_texts = lambda **kwargs: [row["label"]]
            try:
                t5_aste_pipeline.evaluate(args)
            finally:
                t5_aste_pipeline.generate_texts = original_generate

            self.assertTrue((run_dir / "aste_metrics_raw_hp2_dist5.json").exists())
            self.assertTrue((run_dir / "aste_metrics_fixed_hp2_dist5.json").exists())
            self.assertTrue((run_dir / "aste_predictions_hp2_dist5.jsonl").exists())
            self.assertTrue((run_dir / "aste_predictions_raw_fixed_hp2_dist5.jsonl").exists())
            self.assertTrue((run_dir / "aste_metrics_by_sentiment_hp2_dist5.json").exists())
            self.assertTrue((run_dir / "aste_error_analysis_hp2_dist5.json").exists())
            self.assertFalse((run_dir / "aste_metrics_raw.json").exists())
            self.assertFalse((run_dir / "aste_predictions.jsonl").exists())

            sentiment_metrics = json.loads(
                (run_dir / "aste_metrics_by_sentiment_hp2_dist5.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sentiment_metrics["raw"]["pos"]["micro_f1"], 1.0)
            self.assertEqual(sentiment_metrics["raw"]["neu"]["tp"], 0)

            error_analysis = json.loads(
                (run_dir / "aste_error_analysis_hp2_dist5.json").read_text(encoding="utf-8")
            )
            self.assertEqual(error_analysis["neutral_negation_false_positive_rows"], 0)

    def test_evaluate_writes_single_and_multi_triplet_structure_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            rows = [
                {
                    "id": "single",
                    "text": "The battery is good.",
                    "label": "<pos> battery <opinion> good",
                },
                {
                    "id": "multi",
                    "text": "The screen is bright but the keyboard is stiff.",
                    "label": "<pos> screen <opinion> bright ; <neg> keyboard <opinion> stiff",
                },
            ]
            (run_dir / "target_test.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            model_dir = run_dir / "model"
            model_dir.mkdir()
            args = Namespace(
                run_dir=str(run_dir),
                model_path=str(model_dir),
                batch_size=2,
                max_new_tokens=96,
                num_beams=4,
                length_penalty=1.0,
                cuda="0",
                no_constrained_decoding=True,
                no_task_prefix=True,
                output_tag="pairing",
            )
            original_generate = t5_aste_pipeline.generate_texts
            t5_aste_pipeline.generate_texts = lambda **kwargs: [row["label"] for row in rows]
            try:
                t5_aste_pipeline.evaluate(args)
            finally:
                t5_aste_pipeline.generate_texts = original_generate

            path = run_dir / "aste_metrics_by_structure_pairing.json"
            self.assertTrue(path.exists())
            metrics = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["single_triplet_rows"]["rows"], 1)
            self.assertEqual(metrics["multi_triplet_rows"]["rows"], 1)
            self.assertEqual(metrics["single_triplet_rows"]["raw"]["micro_f1"], 1.0)
            self.assertEqual(metrics["multi_triplet_rows"]["fixed"]["micro_f1"], 1.0)

    def test_weighted_trainer_strips_generation_only_inputs_before_prediction(self) -> None:
        inputs = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [4, 5, 6],
            "sample_weight": 0.7,
            "domain_weight": 0.8,
            "domain_label": 1,
            "structure_weight": 0.9,
            "pairing_mask": [1],
            "sentiment_contrastive_weights": [0.7],
        }

        cleaned = t5_absa_train.WeightedSeq2SeqTrainer._strip_generation_only_inputs(inputs)

        self.assertEqual(cleaned["input_ids"], [1, 2, 3])
        self.assertEqual(cleaned["attention_mask"], [1, 1, 1])
        self.assertEqual(cleaned["labels"], [4, 5, 6])
        self.assertNotIn("sample_weight", cleaned)
        self.assertNotIn("domain_weight", cleaned)
        self.assertNotIn("domain_label", cleaned)
        self.assertNotIn("structure_weight", cleaned)
        self.assertNotIn("pairing_mask", cleaned)
        self.assertNotIn("sentiment_contrastive_weights", cleaned)
        self.assertIn("sample_weight", inputs)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

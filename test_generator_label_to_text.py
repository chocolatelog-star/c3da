import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from t5_aste_pipeline import prepare


class GeneratorLabelToTextTest(unittest.TestCase):
    def test_prepare_generates_pure_label_to_text_rows_for_generator(self):
        source_train = [
            {"id": "s1", "text": "The battery life is long.", "label": "<pos> battery life <opinion> long"}
        ]
        source_dev = [
            {"id": "s2", "text": "The keyboard is stiff.", "label": "<neg> keyboard <opinion> stiff"}
        ]
        target_train = [
            {"id": "t1", "text": "The screen is bright.", "label": "<pos> screen <opinion> bright"}
        ]
        target_test = [
            {"id": "u1", "text": "The speaker is loud.", "label": "<pos> speaker <opinion> loud"}
        ]

        def fake_load_split(dataset, split):
            mapping = {
                ("rest16", "train"): source_train,
                ("rest16", "dev"): source_dev,
                ("laptop14", "train"): target_train,
                ("laptop14", "test"): target_test,
            }
            return mapping[(dataset, split)]

        import t5_aste_pipeline

        original = t5_aste_pipeline.load_split
        t5_aste_pipeline.load_split = fake_load_split
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                run_dir = Path(tmpdir) / "run"
                args = Namespace(
                    source_dataset="rest16",
                    target_dataset="laptop14",
                    run_dir=str(run_dir),
                    dev_ratio=0.1,
                    seed=13,
                    augment_prompt_style="label_to_text",
                    augment_channel_mode="all",
                    domain_prefix_style="none",
                    generator_output_tag="",
                    no_task_prefix=True,
                )
                prepare(args)
                rows = (run_dir / "generator_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
                self.assertTrue(rows)
                self.assertTrue(all('"channel": "label_to_text_generator"' in line for line in rows))
                self.assertTrue(all("generate aste sentence:" in line for line in rows))
        finally:
            t5_aste_pipeline.load_split = original

    def test_prepare_generates_tagged_mixed_rows_and_channel_counts(self):
        source_train = [
            {"id": "s1", "text": "The battery is great.", "label": "<pos> battery <opinion> great"},
            {"id": "s2", "text": "The screen is bright.", "label": "<pos> screen <opinion> bright"},
        ]
        source_dev = [
            {"id": "d1", "text": "The keyboard is stiff.", "label": "<neg> keyboard <opinion> stiff"},
            {"id": "d2", "text": "The trackpad is awful.", "label": "<neg> trackpad <opinion> awful"},
        ]
        target_train = [
            {"id": "t1", "text": "The processor is fast.", "label": "<pos> processor <opinion> fast"}
        ]
        target_test = [
            {"id": "u1", "text": "The speaker is loud.", "label": "<pos> speaker <opinion> loud"}
        ]

        def fake_load_split(dataset, split):
            mapping = {
                ("rest16", "train"): source_train,
                ("rest16", "dev"): source_dev,
                ("laptop14", "train"): target_train,
                ("laptop14", "test"): target_test,
            }
            return mapping[(dataset, split)]

        import t5_aste_pipeline

        original = t5_aste_pipeline.load_split
        t5_aste_pipeline.load_split = fake_load_split
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                run_dir = Path(tmpdir) / "run"
                tag = "mixed_l2t_masked_aspect_masked_opinion"
                args = Namespace(
                    source_dataset="rest16",
                    target_dataset="laptop14",
                    run_dir=str(run_dir),
                    dev_ratio=0.1,
                    seed=13,
                    augment_prompt_style="mixed",
                    augment_channel_mode="all",
                    domain_prefix_style="text",
                    generator_output_tag=tag,
                    no_task_prefix=True,
                )

                prepare(args)

                for stem in (
                    "generator_train",
                    "generator_dev",
                    "c3da_generator_train",
                    "c3da_generator_dev",
                ):
                    self.assertTrue((run_dir / f"{stem}_{tag}.jsonl").exists())
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["augment_prompt_style"], "mixed")
                self.assertEqual(
                    sum(manifest["generator_train_channel_counts"].values()),
                    manifest["generator_train"],
                )
                self.assertEqual(
                    sum(manifest["generator_dev_channel_counts"].values()),
                    manifest["generator_dev"],
                )
                self.assertAlmostEqual(sum(manifest["generator_train_channel_ratios"].values()), 1.0)
                self.assertAlmostEqual(sum(manifest["generator_dev_channel_ratios"].values()), 1.0)
                self.assertEqual(set(manifest["generator_train_channel_counts"]), {
                    "label_to_text_generator",
                    "masked_aspect_editor",
                    "masked_opinion_sentiment_editor",
                })
        finally:
            t5_aste_pipeline.load_split = original


if __name__ == "__main__":
    unittest.main()

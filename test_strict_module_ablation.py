from __future__ import annotations

import subprocess
import tempfile
import unittest
import json
from pathlib import Path

from summarize_strict_module_ablation import build_summary_rows, write_summary


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT = PROJECT_ROOT / "run_strict_module_ablation_queue.ps1"


class StrictModuleAblationTest(unittest.TestCase):
    def test_dry_run_contains_exact_four_training_variants(self) -> None:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-DryRun",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout

        self.assertEqual(output.count("DRYRUN TRAIN "), 4)
        self.assertEqual(output.count("DRYRUN EVALUATE "), 4)
        self.assertIn("source_pseudo_no_dann_no_contrast", output)
        self.assertIn("source_pseudo_aug_no_dann_no_contrast", output)
        self.assertIn("source_pseudo_dann_l003_no_contrast", output)
        self.assertIn("source_pseudo_aug_dann_l003_no_contrast", output)
        self.assertEqual(output.count("--lambda_domain_adv 0.0 --model_path"), 2)
        self.assertEqual(output.count("--lambda_domain_adv 0.03 --domain_adv_exclude_augment"), 2)
        self.assertEqual(output.count("--lambda_sentiment_contrastive 0.0"), 4)
        self.assertEqual(output.count("--resume_from_checkpoint auto"), 4)
        self.assertIn("CURRENT BEST E", output)

    def test_summary_contains_four_ablations_and_current_best(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            tags = [
                "strict_ablation_source_pseudo_no_dann_no_contrast",
                "strict_ablation_source_pseudo_aug_no_dann_no_contrast",
                "strict_ablation_source_pseudo_dann_l003_no_contrast",
                "strict_ablation_source_pseudo_aug_dann_l003_no_contrast",
                "strict_aug150_w020_label_to_text_gen_sentiment_contrastive_l001_source_balanced",
            ]
            for index, tag in enumerate(tags, start=1):
                (run_dir / f"aste_metrics_raw_{tag}.json").write_text(
                    json.dumps({"precision": 0.5, "recall": 0.4, "micro_f1": index / 10}),
                    encoding="utf-8",
                )
                (run_dir / f"aste_metrics_fixed_{tag}.json").write_text(
                    json.dumps({"precision": 0.55, "recall": 0.45, "micro_f1": index / 10 + 0.01}),
                    encoding="utf-8",
                )

            rows = build_summary_rows(run_dir)
            csv_path, md_path = write_summary(run_dir, rows)

            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[-1]["variant"], "E_current_best")
            self.assertTrue(csv_path.exists())
            self.assertIn("严格模块消融", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

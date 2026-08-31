import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RECIPE = ROOT / "configs" / "recipes" / "rest16_to_laptop14_best_v1.json"


class ReproducibleRecipeTest(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads(RECIPE.read_text(encoding="utf-8"))

    def test_observed_counts_never_become_selection_limits(self):
        golden = self.recipe["golden"]
        self.assertEqual(golden["base_pseudo"]["observed_golden_rows"], 421)
        self.assertNotIn("selection_limit", golden["base_pseudo"])
        self.assertEqual(golden["augment"]["selection_limit"], 150)
        self.assertEqual(golden["complete_pseudo"]["observed_golden_rows"], 494)
        self.assertNotIn("selection_limit", golden["complete_pseudo"])
        self.assertEqual(golden["final_train"]["observed_golden_rows"], 1499)

    def test_recipe_uses_only_raw_data_and_declared_models_as_external_inputs(self):
        text = RECIPE.read_text(encoding="utf-8").lower()
        self.assertNotIn(".worktrees", text)
        self.assertNotIn("reuse_upstream", text)
        self.assertNotIn("runs\\\\", text)
        self.assertEqual(self.recipe["source_dataset"], "rest16")
        self.assertEqual(self.recipe["target_dataset"], "laptop14")

    def test_historical_best_recipe_paths_are_portable(self):
        text = RECIPE.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", text))
        self.assertNotIn("BGCA-master", text)

    def test_golden_hashes_are_complete(self):
        required = {
            "extractor",
            "base_pseudo",
            "generator",
            "augment",
            "complete_pseudo",
            "final_train",
            "final_model",
            "predictions",
        }
        self.assertEqual(set(self.recipe["golden"]), required)
        for item in required - {"augment", "final_train"}:
            self.assertRegex(self.recipe["golden"][item]["sha256"], r"^[A-F0-9]{64}$")
        self.assertRegex(
            self.recipe["golden"]["augment"]["semantic_sha256"],
            r"^[A-F0-9]{64}$",
        )
        self.assertNotIn("sha256", self.recipe["golden"]["final_train"])
        self.assertRegex(
            self.recipe["golden"]["final_train"]["training_semantic_sha256"],
            r"^[A-F0-9]{64}$",
        )


if __name__ == "__main__":
    unittest.main()

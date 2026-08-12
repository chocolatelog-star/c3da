import hashlib
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RECIPES = ROOT / "configs" / "recipes" / "experiments"


class TargetAnchoredRecipeTest(unittest.TestCase):
    def _load(self, name):
        return json.loads((RECIPES / name).read_text(encoding="utf-8"))

    def test_three_recipes_are_cumulative_and_keep_best_training_parameters(self):
        step1 = self._load("laptop14_to_rest15_target_anchor_step1_v1.json")
        step2 = self._load("laptop14_to_rest15_target_anchor_step2_gap_v1.json")
        step3 = self._load("laptop14_to_rest15_target_anchor_step3_tiered_v1.json")
        self.assertEqual(step1["augment"]["target_anchor_mode"], "local_span_edit")
        self.assertFalse(step1["augment"]["gap_aware_selection"])
        self.assertFalse(step1["augment"]["tiered_weights"])
        self.assertTrue(step2["augment"]["gap_aware_selection"])
        self.assertFalse(step2["augment"]["tiered_weights"])
        self.assertTrue(step3["augment"]["gap_aware_selection"])
        self.assertTrue(step3["augment"]["tiered_weights"])
        for recipe in (step1, step2, step3):
            self.assertEqual(recipe["source_dataset"], "laptop14")
            self.assertEqual(recipe["target_dataset"], "rest15")
            self.assertEqual(recipe["seed"], 1000)
            self.assertEqual(recipe["final"]["pseudo_weight"], 0.65)
            self.assertAlmostEqual(recipe["final"]["pseudo_weight_scale"], 15 / 13)
            self.assertEqual(recipe["final"]["effective_pseudo_weight"], 0.75)
            self.assertEqual(recipe["final"]["lambda_domain_adv"], 0.03)
            self.assertEqual(recipe["final"]["lambda_sentiment_contrastive"], 0.01)
            self.assertEqual(recipe["training"]["train_batch_size"], 1)
            self.assertEqual(recipe["training"]["eval_batch_size"], 2)
            self.assertEqual(recipe["training"]["gradient_accumulation_steps"], 16)
            self.assertEqual(recipe["augment"]["model_filter_mode"], "exact")
            self.assertGreaterEqual(recipe["augment"]["min_selected_rows"], 1)

    def test_external_input_hashes_are_complete_and_identical_across_steps(self):
        recipes = [
            self._load("laptop14_to_rest15_target_anchor_step1_v1.json"),
            self._load("laptop14_to_rest15_target_anchor_step2_gap_v1.json"),
            self._load("laptop14_to_rest15_target_anchor_step3_tiered_v1.json"),
        ]
        expected = recipes[0]["external_inputs"]
        for recipe in recipes:
            self.assertEqual(recipe["external_inputs"], expected)
            for declaration in recipe["external_inputs"].values():
                digest = declaration["sha256"]
                self.assertEqual(len(digest), 64)
                int(digest, 16)
        required_model_inputs = {
            "t5_config",
            "t5_weights",
            "t5_generation_config",
            "t5_spiece",
            "t5_tokenizer_json",
            "nli_config",
            "nli_weights",
            "nli_added_tokens",
            "nli_special_tokens",
            "nli_spm",
            "nli_tokenizer_json",
            "nli_tokenizer_config",
        }
        self.assertTrue(required_model_inputs.issubset(expected))

    def test_step2_and_step3_are_siblings_of_step1_not_a_reuse_chain(self):
        step1 = self._load("laptop14_to_rest15_target_anchor_step1_v1.json")
        step2 = self._load("laptop14_to_rest15_target_anchor_step2_gap_v1.json")
        step3 = self._load("laptop14_to_rest15_target_anchor_step3_tiered_v1.json")
        step1_semantic_sha256 = hashlib.sha256(
            json.dumps(
                step1,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest().upper()
        self.assertEqual(step1["reuse"]["mode"], "full_from_scratch")
        for recipe in (step2, step3):
            self.assertEqual(recipe["reuse"]["mode"], "controlled_stage_reuse")
            self.assertEqual(recipe["reuse"]["parent_recipe_id"], step1["recipe_id"])
            self.assertEqual(
                recipe["reuse"]["parent_recipe_semantic_sha256"],
                step1_semantic_sha256,
            )
            self.assertEqual(
                recipe["reuse"]["parent_run_id"],
                "laptop14-rest15-target-anchor-step1-seed1000-v1",
            )
            self.assertEqual(recipe["reuse"]["through_stage"], "generator")

    def test_command_graph_changes_at_augment_and_uses_row_weights_in_step3(self):
        from run_reproducible_pipeline import build_best_v1_stages, load_recipe

        recipe = load_recipe(RECIPES / "laptop14_to_rest15_target_anchor_step3_tiered_v1.json")
        with tempfile.TemporaryDirectory() as temp:
            stages = build_best_v1_stages(ROOT, Path(temp), recipe, Path("python"), "0")
        commands = {stage.name: list(stage.argv) for stage in stages}
        augment = commands["augment"]
        self.assertIn("target_anchored_augment.py", augment[1])
        self.assertIn("--gap_aware_selection", augment)
        self.assertIn("--tiered_weights", augment)
        self.assertEqual(
            augment[augment.index("--model_filter_mode") + 1],
            "exact",
        )
        stage_map = {stage.name: stage for stage in stages}
        augment_output_names = {path.name for path in stage_map["augment"].outputs}
        self.assertIn("target_anchored_target_anchor_step3_v1_candidates.jsonl", augment_output_names)
        self.assertIn("target_anchored_target_anchor_step3_v1_exploratory_audit.jsonl", augment_output_names)
        self.assertEqual(len(stage_map["final_train"].outputs), 8)
        build = commands["build_final_train"]
        self.assertEqual(build[build.index("--selected_augment_weight") + 1], "0.0")
        self.assertEqual(commands["final_train"][commands["final_train"].index("--pseudo_weight") + 1], "0.65")
        self.assertEqual(
            commands["final_train"][commands["final_train"].index("--pseudo_weight_scale") + 1],
            str(15 / 13),
        )

    def test_single_entrypoint_selects_exactly_one_of_three_recipes(self):
        script = (ROOT / "run_target_anchored_pipeline.ps1").read_text(encoding="utf-8")
        self.assertIn("ValidateSet(1, 2, 3)", script)
        self.assertIn("target_anchor_step1_v1.json", script)
        self.assertIn("target_anchor_step2_gap_v1.json", script)
        self.assertIn("target_anchor_step3_tiered_v1.json", script)
        self.assertNotIn("Start-Process", script)

    def test_serial_monitor_runs_one_step_at_a_time_and_stops_on_failure(self):
        script = (ROOT / "run_target_anchored_all_serial.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("foreach ($Step in 1, 2, 3)", script)
        self.assertIn("while (-not $Child.HasExited)", script)
        self.assertIn("if ($Child.ExitCode -ne 0)", script)
        self.assertIn("health.json", script)


if __name__ == "__main__":
    unittest.main()

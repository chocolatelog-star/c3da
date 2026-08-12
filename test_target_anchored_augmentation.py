import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from target_anchored_augmentation import (
    build_target_anchored_candidates,
    classify_quality_tier,
    normalize_tiered_weights,
    select_gap_aware_rows,
    validate_anchor_edit_contract,
)
from t5_absa_train import CSA_AUGMENT_CHANNELS, JsonlSeq2SeqDataset
from target_anchored_augment import require_minimum_rows


def row(row_id, text, label, **extra):
    return {"id": row_id, "text": text, "label": label, **extra}


class TargetAnchoredCandidateTest(unittest.TestCase):
    def test_builds_local_edits_from_target_sentences_and_preserves_other_triplets(self):
        anchors = [
            row(
                7,
                "The food was tasty but the service was slow .",
                "<pos> food <opinion> tasty ; <neg> service <opinion> slow",
            )
        ]
        source = [row(1, "The screen was dim .", "<neg> screen <opinion> dim")]

        candidates, stats = build_target_anchored_candidates(
            anchors,
            source,
            per_anchor=4,
            seed=1000,
        )

        self.assertGreater(len(candidates), 0)
        self.assertTrue(all(item["base_text"] == anchors[0]["text"] for item in candidates))
        self.assertTrue(all(item["anchor_domain"] == "target" for item in candidates))
        self.assertTrue(all(item["edited_triplet_index"] in {0, 1} for item in candidates))
        for item in candidates:
            contract = validate_anchor_edit_contract(item)
            self.assertTrue(contract["passed"], contract)
            untouched = item["untouched_triplets"]
            self.assertTrue(untouched)
            self.assertTrue(all(t[0].lower() in item["text"].lower() for t in untouched))
            self.assertTrue(all(t[1].lower() in item["text"].lower() for t in untouched))
        self.assertEqual(stats["anchors"], 1)

    def test_target_opinions_are_primary_and_source_opinions_are_fallback_only(self):
        anchors = [
            row(1, "The food was slow .", "<neg> food <opinion> slow"),
            row(2, "The service was awful .", "<neg> service <opinion> awful"),
        ]
        source = [row(3, "The screen was dim .", "<neg> screen <opinion> dim")]

        candidates, _stats = build_target_anchored_candidates(
            anchors,
            source,
            per_anchor=20,
            seed=1000,
        )

        opinion_edits = [item for item in candidates if item["edit_type"] == "opinion"]
        self.assertTrue(opinion_edits)
        self.assertTrue(
            all(item["replacement_source"] == "target_pseudo" for item in opinion_edits)
        )
        self.assertTrue(all(item["new_triplet"][1] != "dim" for item in opinion_edits))

    def test_contract_rejects_global_rewrite_or_missing_untouched_triplet(self):
        candidate = {
            "text": "The food was excellent .",
            "base_text": "The food was tasty but the service was slow .",
            "old_triplet": ["food", "tasty", "pos"],
            "new_triplet": ["food", "excellent", "pos"],
            "untouched_triplets": [["service", "slow", "neg"]],
            "label": "<pos> food <opinion> excellent ; <neg> service <opinion> slow",
            "edit_type": "opinion",
        }

        contract = validate_anchor_edit_contract(candidate)

        self.assertFalse(contract["passed"])
        self.assertIn("untouched_span_missing", contract["reasons"])


class GapAwareSelectionTest(unittest.TestCase):
    def test_joint_solver_finds_feasible_mix_when_greedy_order_would_block_bases(self):
        specs = [
            (0, "aspect", False, 0.41272577790357334),
            (1, "neutral", True, 0.7015415121496803),
            (1, "neutral", False, 0.8949945760920935),
            (1, "aspect", False, 0.6875458457750431),
            (2, "aspect", True, 0.022981576913134294),
            (2, "opinion", False, 0.1969957367337014),
            (2, "aspect", True, 0.1757146662792488),
            (2, "neutral", True, 0.7868469480312121),
            (3, "neutral", False, 0.11029034195448484),
            (3, "neutral", True, 0.011010967696091556),
            (3, "neutral", True, 0.1922469316425307),
            (4, "opinion", True, 0.3464907520868177),
            (4, "aspect", False, 0.15915384659991993),
            (4, "aspect", False, 0.8635579922735233),
            (5, "aspect", True, 0.7719546760313628),
            (5, "aspect", True, 0.8358013940475433),
            (5, "aspect", True, 0.8577141416408877),
            (5, "opinion", False, 0.2624367865529239),
            (6, "opinion", False, 0.01014772084123805),
            (6, "neutral", True, 0.6936058945094618),
            (6, "aspect", False, 0.6154438398132859),
            (6, "aspect", False, 0.5748582016715789),
            (7, "opinion", False, 0.8060111646490538),
            (7, "neutral", True, 0.9523161183098076),
            (7, "neutral", False, 0.4738230597854559),
            (8, "opinion", True, 0.6315895612530479),
            (8, "opinion", False, 0.04129893123715389),
            (8, "opinion", False, 0.8983572536149018),
            (8, "neutral", True, 0.9238413568402002),
            (9, "aspect", True, 0.845436014964198),
        ]
        rows = [
            row(
                idx,
                f"sentence {idx}",
                "<pos> food <opinion> good",
                base_id=base_id,
                edit_type=edit_type,
                anchor_triplet_count=2 if is_multi else 1,
                quality_score=quality,
                contract_passed=True,
                model_filter_passed=True,
                model_filter_match="exact",
                nli_label="entailment",
            )
            for idx, (base_id, edit_type, is_multi, quality) in enumerate(specs)
        ]

        selected, stats = select_gap_aware_rows(
            rows,
            selection_limit=10,
            max_per_base=1,
        )

        counts = Counter(item["edit_type"] for item in selected)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({item["base_id"] for item in selected}), 10)
        self.assertEqual(counts, Counter({"aspect": 6, "opinion": 3, "neutral": 1}))
        self.assertGreaterEqual(stats["selected_multi_triplet_rows"], 5)
        self.assertTrue(stats["ratio_constraints_met"])
        self.assertTrue(stats["multi_triplet_target_met"])

    def test_selects_without_forcing_short_buckets_or_exceeding_total_cap(self):
        rows = []
        for idx in range(40):
            edit_type = "aspect" if idx < 24 else ("neutral" if idx < 28 else "opinion")
            sentiment = "neu" if edit_type == "neutral" else ("neg" if idx % 2 else "pos")
            rows.append(
                row(
                    idx,
                    f"sentence {idx}",
                    f"<{sentiment}> food <opinion> word{idx}",
                    edit_type=edit_type,
                    anchor_triplet_count=2 if idx % 2 == 0 else 1,
                    base_id=idx,
                    contract_passed=True,
                    model_filter_passed=True,
                    model_filter_match="exact",
                    nli_label="entailment",
                    quality_score=1.0 - idx / 1000,
                )
            )

        selected, stats = select_gap_aware_rows(
            rows,
            selection_limit=30,
            aspect_ratio=(0.55, 0.65),
            opinion_ratio=(0.25, 0.35),
            neutral_max_ratio=0.10,
            multi_triplet_target_ratio=0.50,
            max_per_base=1,
        )

        self.assertLessEqual(len(selected), 30)
        self.assertLessEqual(stats["selected_by_edit_type"].get("neutral", 0), 3)
        self.assertGreaterEqual(stats["selected_by_edit_type"].get("aspect", 0), 16)
        self.assertGreaterEqual(stats["selected_multi_triplet_rows"], 15)
        self.assertFalse(stats["forced_low_quality_fill"])

    def test_ratios_are_based_on_available_rows_when_limit_is_not_filled(self):
        rows = []
        for idx in range(20):
            edit_type = "neutral" if idx < 4 else ("aspect" if idx < 15 else "opinion")
            rows.append(
                row(
                    idx,
                    f"sentence {idx}",
                    "<neu> food <opinion> okay"
                    if edit_type == "neutral"
                    else "<pos> food <opinion> good",
                    edit_type=edit_type,
                    anchor_triplet_count=2 if idx % 2 == 0 else 1,
                    base_id=idx,
                    contract_passed=True,
                    model_filter_passed=True,
                    model_filter_match="exact",
                    nli_label="entailment",
                    quality_score=1.0,
                )
            )

        selected, stats = select_gap_aware_rows(
            rows,
            selection_limit=100,
            aspect_ratio=(0.55, 0.65),
            opinion_ratio=(0.25, 0.35),
            neutral_max_ratio=0.10,
            multi_triplet_target_ratio=0.50,
            max_per_base=1,
        )

        self.assertEqual(stats["effective_selection_target"], 17)
        self.assertEqual(len(selected), 17)
        edit_counts = Counter(item["edit_type"] for item in selected)
        self.assertGreaterEqual(edit_counts["aspect"] / len(selected), 0.55)
        self.assertLessEqual(edit_counts["aspect"] / len(selected), 0.65)
        self.assertGreaterEqual(edit_counts["opinion"] / len(selected), 0.25)
        self.assertLessEqual(edit_counts["opinion"] / len(selected), 0.35)
        self.assertLessEqual(edit_counts["neutral"] / len(selected), 0.10)
        self.assertTrue(stats["multi_triplet_target_met"])

    def test_empty_or_too_small_augmentation_fails_before_final_training(self):
        with self.assertRaisesRegex(RuntimeError, "target-anchor candidates"):
            require_minimum_rows("target-anchor candidates", [], 1)
        with self.assertRaisesRegex(RuntimeError, "selected target-anchor augmentation"):
            require_minimum_rows("selected target-anchor augmentation", [{"id": 1}], 2)


class TieredWeightTest(unittest.TestCase):
    def test_assigns_quality_tiers_and_caps_total_effective_mass(self):
        high = {
            "edit_type": "opinion",
            "contract_passed": True,
            "model_filter_passed": True,
            "model_filter_match": "exact",
            "nli_label": "entailment",
            "quality_flags": {"all_terms_in_text": True},
        }
        medium = {
            **high,
            "model_filter_match": "opinion_span_compatible",
            "nli_label": "neutral",
        }
        exploratory = {**high, "contract_passed": False}
        self.assertEqual(classify_quality_tier(high), "high")
        self.assertEqual(classify_quality_tier(medium), "medium")
        self.assertEqual(classify_quality_tier(exploratory), "exploratory")
        counterfactual_neutral_high = {
            **high,
            "edit_type": "neutral",
            "nli_label": "entailment",
            "nli_counterfactual_consistent": True,
        }
        counterfactual_neutral_medium = {
            **counterfactual_neutral_high,
            "nli_label": "neutral",
            "nli_counterfactual_consistent": False,
        }
        counterfactual_neutral_invalid = {
            **counterfactual_neutral_high,
            "nli_label": "contradiction",
        }
        self.assertEqual(classify_quality_tier(counterfactual_neutral_high), "high")
        self.assertEqual(classify_quality_tier(counterfactual_neutral_medium), "medium")
        self.assertEqual(
            classify_quality_tier(counterfactual_neutral_invalid), "exploratory"
        )

        rows = [dict(high, id=i, label="<pos> food <opinion> good") for i in range(200)]
        weighted, stats = normalize_tiered_weights(
            rows,
            high_weight=0.30,
            medium_weight=0.15,
            neutral_high_weight=0.18,
            neutral_medium_weight=0.10,
            total_weight_cap=30.0,
        )

        self.assertEqual(len(weighted), 200)
        self.assertLessEqual(sum(item["sample_weight"] for item in weighted), 30.0 + 1e-6)
        self.assertTrue(math.isclose(stats["normalized_weight_mass"], 30.0, abs_tol=1e-6))
        self.assertLess(stats["normalization_scale"], 1.0)


class FinalTrainingIntegrationTest(unittest.TestCase):
    class TinyTokenizer:
        pad_token_id = 0

        def __call__(self, text, **_kwargs):
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    def test_target_anchor_channels_are_augmentation_and_pw065_rows_scale_to_pw075(self):
        expected_channels = {
            "target_anchored_aspect_channel",
            "target_anchored_opinion_channel",
            "target_anchored_neutral_channel",
        }
        self.assertTrue(expected_channels.issubset(CSA_AUGMENT_CHANNELS))
        rows = [
            {
                "input": "target sentence",
                "target": "<pos> food <opinion> good",
                "augmentation": "target_pseudo",
                "sample_weight": 0.65,
            }
        ]
        dataset = JsonlSeq2SeqDataset(
            rows,
            self.TinyTokenizer(),
            16,
            16,
            1.0,
            0.65,
            0.2,
            pseudo_weight_scale=15 / 13,
        )
        self.assertAlmostEqual(dataset.sample_weight(rows[0]), 0.75)


if __name__ == "__main__":
    unittest.main()

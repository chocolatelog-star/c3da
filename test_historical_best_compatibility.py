import json
import unittest
from pathlib import Path

from t5_aste_augment import build_augmentation_requests
from t5_aste_pipeline import filter_augmented_rows_for_compatibility


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "test_fixtures" / "historical_best_augment_cases.jsonl"


def load_fixture() -> dict:
    line = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)


def project_requests(rows: list[dict]) -> list[dict]:
    projected = []
    for row in rows:
        item = {
            "input": row["input"],
            "label": row["label"],
            "channel": row["channel"],
        }
        if "opinion_replacement_mode" in row:
            item["opinion_replacement_mode"] = row["opinion_replacement_mode"]
        projected.append(item)
    return projected


class HistoricalBestCompatibilityTest(unittest.TestCase):
    def test_historical_profile_skips_new_opinion_boundary_filter(self):
        row = {
            "text": "The screen is fine.",
            "label": "<pos> screen <opinion> with fine",
            "augmentation": "masked_opinion_sentiment_channel",
        }
        self.assertEqual(filter_augmented_rows_for_compatibility([row], ""), [])
        self.assertEqual(
            filter_augmented_rows_for_compatibility(
                [row], "historical_best_v1"
            ),
            [row],
        )

    def test_historical_profile_matches_audit_fixture_without_changing_default(self):
        fixture = load_fixture()
        self.assertTrue(fixture["audit_fixture_only"])
        default_rows = build_augmentation_requests(**fixture["inputs"])
        compat_rows = build_augmentation_requests(
            **fixture["inputs"], compatibility_profile="historical_best_v1"
        )
        self.assertEqual(
            project_requests(compat_rows), fixture["expected_historical_requests"]
        )
        self.assertEqual(
            project_requests(default_rows), fixture["expected_current_default_requests"]
        )
        self.assertNotEqual(project_requests(default_rows), project_requests(compat_rows))


if __name__ == "__main__":
    unittest.main()

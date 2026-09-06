from t5_aste_augment import (
    build_label_invariant_prompt,
    filter_augmented_text_quality,
    is_prompt_leak,
)
from t5_aste_pipeline import validate_structure_preserving_augmentation


def test_concept_prompt_avoids_template_keywords():
    prompt = build_label_invariant_prompt("<pos> battery life <opinion> long")

    assert "battery life" in prompt
    assert "long" in prompt
    assert "label:" not in prompt.lower()
    assert "aspect terms" not in prompt.lower()
    assert "opinion terms" not in prompt.lower()


def test_prompt_leak_filter_rejects_generated_template_fragments():
    text = "label: pos> battery life opinion> long ; paraphrase with label terms"

    passed, reason = filter_augmented_text_quality(text)

    assert is_prompt_leak(text) is True
    assert passed is False
    assert reason == "prompt_leak"


def test_prompt_leak_filter_keeps_natural_sentence():
    text = "The battery life is long and reliable."

    passed, reason = filter_augmented_text_quality(text)

    assert is_prompt_leak(text) is False
    assert passed is True
    assert reason == ""


def test_structure_preserving_augmentation_keeps_untouched_triplets():
    parent = [
        ("battery", "long", "pos"),
        ("screen", "bright", "pos"),
    ]
    result = validate_structure_preserving_augmentation(
        parent, ("battery", "long", "pos"), ("battery", "excellent", "pos"),
        "<pos> battery <opinion> excellent ; <pos> screen <opinion> bright",
    )
    assert result["structure_passed"] is True
    assert result["untouched_retention"] == 1.0
    assert result["count_preserved"] is True
    assert result["unplanned_triplets"] == []


def test_structure_preserving_augmentation_rejects_missing_untouched_triplet():
    result = validate_structure_preserving_augmentation(
        [("battery", "long", "pos"), ("screen", "bright", "pos")],
        ("battery", "long", "pos"), ("battery", "excellent", "pos"),
        "<pos> battery <opinion> excellent",
    )
    assert result["structure_passed"] is False
    assert result["untouched_retention"] < 1.0


def test_structure_preserving_augmentation_rejects_unplanned_and_count_change():
    result = validate_structure_preserving_augmentation(
        [("battery", "long", "pos")],
        ("battery", "long", "pos"), ("battery", "excellent", "pos"),
        "<pos> battery <opinion> excellent ; <neg> camera <opinion> awful",
    )
    assert result["structure_passed"] is False
    assert result["count_preserved"] is False
    assert result["unplanned_triplets"]

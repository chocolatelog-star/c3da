from t5_aste_augment import (
    build_label_invariant_prompt,
    filter_augmented_text_quality,
    is_prompt_leak,
)


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

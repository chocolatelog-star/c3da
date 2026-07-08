from t5_aste_pipeline import (
    filter_augmented_rows_by_model_predictions,
    filter_augmented_rows_by_model_predictions_channel_aware,
    filter_augmented_rows_with_optional_channel_awareness,
)


def test_model_filter_fixed_mode_keeps_rows_after_span_fix():
    rows = [
        {
            "text": "battery life is long",
            "label": "<pos> battery life <opinion> long",
            "augmentation": "aspect_channel",
        },
        {
            "text": "screen is dark",
            "label": "<pos> screen <opinion> bright",
            "augmentation": "opinion_sentiment_channel",
        },
    ]
    predictions = [
        "<pos> battery <opinion> long",
        "<neg> screen <opinion> dark",
    ]

    kept, removed, stats = filter_augmented_rows_by_model_predictions(rows, predictions, mode="fixed")

    assert [row["text"] for row in kept] == ["battery life is long"]
    assert kept[0]["model_filter_pred_fixed"] == "<pos> battery life <opinion> long"
    assert removed[0]["model_filter_reason"] == "label_mismatch"
    assert stats["kept_rows"] == 1
    assert stats["removed_rows"] == 1
    assert stats["mode"] == "fixed"


def test_model_filter_exact_mode_requires_raw_label_match():
    rows = [
        {
            "text": "battery life is long",
            "label": "<pos> battery life <opinion> long",
            "augmentation": "aspect_channel",
        },
    ]
    predictions = ["<pos> battery <opinion> long"]

    kept, removed, stats = filter_augmented_rows_by_model_predictions(rows, predictions, mode="exact")

    assert kept == []
    assert removed[0]["model_filter_pred_raw"] == "<pos> battery <opinion> long"
    assert stats["kept_rows"] == 0


def test_model_filter_allows_opinion_span_extension_for_label_composition():
    rows = [
        {
            "text": "The notebook is lacking in quality but the processing is fast.",
            "label": "<neg> notebook <opinion> lacking ; <pos> processing <opinion> fast",
            "augmentation": "label_composition_channel",
        },
    ]
    predictions = ["<neg> notebook <opinion> lacking in quality ; <pos> processing <opinion> fast"]

    kept, removed, stats = filter_augmented_rows_by_model_predictions(rows, predictions, mode="fixed")

    assert removed == []
    assert kept[0]["model_filter_passed"] is True
    assert kept[0]["model_filter_match"] == "span_compatible"
    assert stats["kept_rows"] == 1


def test_model_filter_keeps_strict_matching_for_non_composition_channels():
    rows = [
        {
            "text": "The notebook is lacking in quality.",
            "label": "<neg> notebook <opinion> lacking",
            "augmentation": "masked_aspect_channel",
        },
    ]
    predictions = ["<neg> notebook <opinion> lacking in quality"]

    kept, removed, stats = filter_augmented_rows_by_model_predictions(rows, predictions, mode="fixed")

    assert kept == []
    assert removed[0]["model_filter_reason"] == "label_mismatch"
    assert stats["kept_rows"] == 0


def test_channel_aware_model_filter_relabels_opinion_channel_from_extractor():
    rows = [
        {
            "text": "The keyboard is terrible.",
            "label": "<neg> keyboard <opinion> slow",
            "augmentation": "masked_opinion_sentiment_channel",
            "new_triplet": ["keyboard", "slow", "neg"],
        },
    ]
    predictions = ["<neg> keyboard <opinion> terrible"]

    kept, removed, stats = filter_augmented_rows_by_model_predictions_channel_aware(rows, predictions, mode="fixed")

    assert removed == []
    assert kept[0]["label"] == "<neg> keyboard <opinion> terrible"
    assert kept[0]["model_filter_label_source"] == "opinion_extractor"
    assert stats["opinion_label_replaced"] == 1


def test_channel_aware_model_filter_rejects_opinion_channel_without_target_sentiment():
    rows = [
        {
            "text": "The keyboard is terrible.",
            "label": "<neg> keyboard <opinion> terrible",
            "augmentation": "masked_opinion_sentiment_channel",
            "new_triplet": ["keyboard", "terrible", "neg"],
        },
    ]
    predictions = ["<pos> keyboard <opinion> terrible"]

    kept, removed, stats = filter_augmented_rows_by_model_predictions_channel_aware(rows, predictions, mode="fixed")

    assert kept == []
    assert removed[0]["model_filter_reason"] == "missing_target_aspect_sentiment"
    assert stats["removed_rows"] == 1


def test_optional_channel_awareness_uses_strict_filter_by_default():
    rows = [
        {
            "text": "The notebook is lacking in quality.",
            "label": "<neg> notebook <opinion> lacking",
            "augmentation": "masked_aspect_channel",
            "new_triplet": ["notebook", "lacking", "neg"],
        },
    ]
    predictions = ["<neg> notebook <opinion> lacking in quality"]

    kept, removed, stats = filter_augmented_rows_with_optional_channel_awareness(
        rows,
        predictions,
        mode="fixed",
        channel_aware=False,
    )

    assert kept == []
    assert removed[0]["model_filter_reason"] == "label_mismatch"
    assert stats.get("channel_aware") is False


def test_optional_channel_awareness_enables_aspect_sentiment_compatible_filter():
    rows = [
        {
            "text": "The notebook is lacking in quality.",
            "label": "<neg> notebook <opinion> lacking",
            "augmentation": "masked_aspect_channel",
            "new_triplet": ["notebook", "lacking", "neg"],
        },
    ]
    predictions = ["<neg> notebook <opinion> lacking in quality"]

    kept, removed, stats = filter_augmented_rows_with_optional_channel_awareness(
        rows,
        predictions,
        mode="fixed",
        channel_aware=True,
    )

    assert removed == []
    assert kept[0]["model_filter_match"] == "aspect_sentiment_compatible"
    assert stats["channel_aware"] is True


def test_channel_aware_sentence_fusion_keeps_full_match_with_punctuation():
    rows = [
        {
            "text": "The battery life is excellent and the 2 gb of ram is plenty.",
            "label": "<pos> 2 gb of ram <opinion> plenty ; <pos> battery life <opinion> excellent",
            "augmentation": "sentence_fusion_composition_channel",
            "sample_weight": 0.15,
        },
    ]
    predictions = ["<pos> 2 gb of ram <opinion> plenty ; <pos> battery life <opinion> excellent."]

    kept, removed, stats = filter_augmented_rows_by_model_predictions_channel_aware(rows, predictions, mode="fixed")

    assert removed == []
    assert kept[0]["label"] == "<pos> 2 gb of ram <opinion> plenty ; <pos> battery life <opinion> excellent"
    assert kept[0]["model_filter_match"] == "exact"
    assert stats["fusion_full_kept"] == 1


def test_channel_aware_sentence_fusion_drops_partial_exact_match():
    rows = [
        {
            "text": "The temp is not ideal, but I found the ibookg4 very attractive.",
            "label": "<pos> ibookg4 <opinion> attractive ; <neg> temp <opinion> not ideal",
            "augmentation": "sentence_fusion_composition_channel",
            "sample_weight": 0.15,
        },
    ]
    predictions = ["<pos> ibookg4 <opinion> attractive"]

    kept, removed, stats = filter_augmented_rows_by_model_predictions_channel_aware(rows, predictions, mode="fixed")

    assert kept == []
    assert removed[0]["model_filter_reason"] == "fusion_partial_match_dropped"
    assert removed[0]["model_filter_partial_triplets"] == 1
    assert stats["fusion_partial_dropped"] == 1

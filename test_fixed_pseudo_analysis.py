from pathlib import Path

from t5_aste_pipeline import (
    build_final_training_rows,
    build_pseudo_analysis,
    build_training_pseudo_row,
    evaluate_selected_pseudo_against_hidden_gold,
    filter_augmented_rows_by_model_predictions_channel_aware,
    read_extra_augmented_rows,
    read_pseudo_rows_for_training,
    resolve_extractor_model_path,
    tagged_output_path,
    select_high_precision_pseudo_rows,
    select_high_confidence_pseudo_rows,
    select_train_pseudo_rows,
)
from t5_aste_data import write_jsonl


def test_build_pseudo_analysis_reports_raw_and_fixed_scores():
    target_rows = [{"id": 0, "text": "the battery life is long"}]
    pseudo_rows = [{"id": 0, "text": "the battery life is long", "label": "<pos> battery life <opinion> lng"}]
    gold_rows = {0: {"label": "<pos> battery life <opinion> long"}}

    analysis, rows = build_pseudo_analysis(target_rows, pseudo_rows, gold_rows)

    assert analysis["pseudo_micro_f1_against_hidden_gold"]["micro_f1"] == 0.0
    assert analysis["fixed_pseudo_micro_f1_against_hidden_gold"]["micro_f1"] == 1.0
    assert rows[0]["pseudo_fixed"] == "<pos> battery life <opinion> long"


def test_resolve_extractor_model_path_prefers_last_variant(tmp_path):
    run_dir = tmp_path / "run"
    last = run_dir / "models" / "extractor_ep25_last" / "best"
    last.mkdir(parents=True)

    resolved = resolve_extractor_model_path(run_dir, variant="last")

    assert resolved == last


def test_resolve_extractor_model_path_prefers_direct_best(tmp_path):
    run_dir = tmp_path / "run"
    best = run_dir / "models" / "extractor" / "best"
    best.mkdir(parents=True)

    resolved = resolve_extractor_model_path(run_dir, variant="best")

    assert resolved == best


def test_read_extra_augmented_rows_supports_semicolon_paths_and_weight(tmp_path):
    first = tmp_path / "masked.jsonl"
    second = tmp_path / "composition.jsonl"
    write_jsonl(
        first,
        [{"text": "The screen is bright.", "label": "<pos> screen <opinion> bright", "augmentation": "masked_aspect_channel"}],
    )
    write_jsonl(
        second,
        [
            {
                "text": "The screen is bright but the battery is bad.",
                "label": "<pos> screen <opinion> bright ; <neg> battery <opinion> bad",
                "augmentation": "label_composition_channel",
            }
        ],
    )

    rows, stats = read_extra_augmented_rows(f"{first};{second}", sample_weight=0.35)

    assert len(rows) == 2
    assert all(row["sample_weight"] == 0.35 for row in rows)
    assert rows[0]["extra_augmented"] is True
    assert stats["rows"] == 2
    assert stats["augmentation_distribution"] == {"masked_aspect_channel": 1, "label_composition_channel": 1}


def test_build_final_training_rows_can_exclude_source_rows():
    source_rows = [{"text": "The food is good.", "label": "<pos> food <opinion> good"}]
    pseudo_rows = [{"text": "The battery is long.", "label": "<pos> battery <opinion> long"}]
    augmented_rows = [{"text": "The screen is bright.", "label": "<pos> screen <opinion> bright"}]

    final_rows = build_final_training_rows(source_rows, pseudo_rows, augmented_rows, include_source=False)

    assert [row["text"] for row in final_rows] == ["The battery is long.", "The screen is bright."]


def test_tagged_output_path_appends_suffix_without_overwriting_default():
    run_dir = Path(r"J:\tmp")

    tagged = tagged_output_path(run_dir, "c3da_two_channel_augmented_selected.jsonl", "single")
    plain = tagged_output_path(run_dir, "c3da_two_channel_augmented_selected.jsonl", "")

    assert str(tagged).endswith("c3da_two_channel_augmented_selected_single.jsonl")
    assert str(plain).endswith("c3da_two_channel_augmented_selected.jsonl")


def test_build_training_pseudo_row_uses_fixed_label_and_preserves_raw_label():
    row = {"id": 0, "text": "The keyboard is too slick ."}
    raw_label = "<neg> keyboard <opinion> too slick"

    pseudo_row = build_training_pseudo_row(row, raw_label)

    assert pseudo_row["label"] == "<neg> keyboard <opinion> slick"
    assert pseudo_row["label_raw"] == "<neg> keyboard <opinion> too slick"
    assert pseudo_row["label_fixed"] == "<neg> keyboard <opinion> slick"
    assert pseudo_row["fixed_changed"] is True


def test_select_high_confidence_pseudo_rows_keeps_clean_raw_fixed_agreement():
    rows = [
        {
            "id": 1,
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
            "fixed_changed": False,
            "sample_weight": 0.65,
            "quality_flags": {"all_terms_in_text": True, "triplet_count": 1},
        },
        {
            "id": 2,
            "text": "Easy to start up and does not overheat as much as other laptops.",
            "label": "<pos> laptops <opinion> easy",
            "fixed_changed": True,
            "sample_weight": 0.65,
            "quality_flags": {"all_terms_in_text": True, "triplet_count": 1},
        },
        {
            "id": 3,
            "text": "The use is simple.",
            "label": "<pos> use <opinion> simple",
            "fixed_changed": False,
            "sample_weight": 0.65,
            "quality_flags": {"all_terms_in_text": True, "triplet_count": 1},
        },
        {
            "id": 4,
            "text": "The screen is bright.",
            "label": "<pos> screen <opinion> bright",
            "fixed_changed": False,
            "sample_weight": 0.55,
            "quality_flags": {"all_terms_in_text": True, "triplet_count": 1},
        },
    ]

    selected, stats = select_high_confidence_pseudo_rows(rows, min_weight=0.65)

    assert [row["id"] for row in selected] == [1]
    assert selected[0]["selected_pseudo"] is True
    assert stats["input_rows"] == 4
    assert stats["selected_rows"] == 1
    assert stats["rejected_fixed_changed"] == 1
    assert stats["rejected_noisy_term"] == 1
    assert stats["rejected_low_weight"] == 1


def test_select_train_pseudo_rows_adds_high_score_fixed_changed_low_weight():
    rows = [
        {
            "id": 1,
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
            "fixed_changed": False,
            "sample_weight": 0.65,
            "quality_flags": {"all_terms_in_text": True, "triplet_count": 1},
        },
        {
            "id": 2,
            "text": "The keyboard is too slick.",
            "label": "<neg> keyboard <opinion> slick",
            "label_raw": "<neg> keyboard <opinion> too slick",
            "fixed_changed": True,
            "sample_weight": 0.65,
            "quality_flags": {"all_terms_in_text": True, "triplet_count": 1},
        },
        {
            "id": 3,
            "text": "The use is simple.",
            "label": "<pos> use <opinion> simple",
            "fixed_changed": False,
            "sample_weight": 0.65,
            "quality_flags": {"all_terms_in_text": True, "triplet_count": 1},
        },
    ]

    strict_selected, strict_stats = select_high_confidence_pseudo_rows(rows, min_weight=0.65, max_triplets=3)
    train_selected, train_stats = select_train_pseudo_rows(rows, min_weight=0.65, fixed_changed_weight=0.35)

    assert [row["id"] for row in strict_selected] == [1]
    assert [row["id"] for row in train_selected] == [1, 2]
    assert strict_stats["selected_rows"] == 1
    assert train_stats["selected_rows"] == 2
    assert train_selected[1]["sample_weight"] == 0.35
    assert train_selected[1]["train_selected_reason"] == "fixed_changed_high_confidence"
    assert train_stats["added_fixed_changed_high_confidence"] == 1


def test_evaluate_selected_pseudo_against_hidden_gold_reports_retained_f1():
    selected_rows = [
        {"id": 1, "text": "The battery is long.", "label": "<pos> battery <opinion> long"},
        {"id": 2, "text": "The keyboard is bad.", "label": "<pos> keyboard <opinion> bad"},
    ]
    gold_rows = {
        1: {"label": "<pos> battery <opinion> long"},
        2: {"label": "<neg> keyboard <opinion> bad"},
        3: {"label": "<pos> screen <opinion> bright"},
    }

    stats = evaluate_selected_pseudo_against_hidden_gold(selected_rows, gold_rows, name="strict")

    assert stats["name"] == "strict"
    assert stats["selected_rows"] == 2
    assert stats["hidden_gold_rows"] == 3
    assert stats["exact_match_rows"] == 1
    assert stats["raw_scores"]["tp"] == 1
    assert stats["raw_scores"]["fp"] == 1
    assert stats["raw_scores"]["fn"] == 1
    assert round(stats["raw_scores"]["micro_f1"], 6) == 0.5


def test_rsda_t5_label_composition_rejects_partial_triplet_extraction():
    rows = [
        {
            "text": "The osx is very easy, but the notebook is missing.",
            "label": "<neg> notebook <opinion> missing ; <pos> osx <opinion> easy",
            "augmentation": "rsda_t5_label_composition_channel",
        }
    ]
    predictions = ["<neg> notebook <opinion> missing"]

    kept, removed, stats = filter_augmented_rows_by_model_predictions_channel_aware(
        rows,
        predictions,
        mode="fixed",
    )

    assert kept == []
    assert len(removed) == 1
    assert stats["removed_rows"] == 1


def test_rsda_t5_label_composition_keeps_all_aspect_sentiment_matches_with_opinion_span():
    rows = [
        {
            "text": "The screen is very bright and the keyboard is quite responsive.",
            "label": "<pos> keyboard <opinion> responsive ; <pos> screen <opinion> bright",
            "augmentation": "rsda_t5_label_composition_channel",
        }
    ]
    predictions = ["<pos> keyboard <opinion> quite responsive ; <pos> screen <opinion> very bright"]

    kept, removed, stats = filter_augmented_rows_by_model_predictions_channel_aware(
        rows,
        predictions,
        mode="fixed",
    )

    assert len(kept) == 1
    assert removed == []
    assert kept[0]["model_filter_match"] == "aspect_sentiment_opinion_span"
    assert kept[0]["label"] == "<pos> keyboard <opinion> responsive ; <pos> screen <opinion> bright"
    assert stats["kept_rows"] == 1


def test_read_pseudo_rows_for_training_defaults_to_strict(tmp_path):
    write_jsonl(tmp_path / "target_pseudo_selected.jsonl", [{"id": 1, "label": "<pos> a <opinion> good"}])
    write_jsonl(
        tmp_path / "target_pseudo_train_selected.jsonl",
        [
            {"id": 1, "label": "<pos> a <opinion> good"},
            {"id": 2, "label": "<neg> b <opinion> bad"},
        ],
    )

    strict_rows = read_pseudo_rows_for_training(tmp_path, "strict")
    train_selected_rows = read_pseudo_rows_for_training(tmp_path, "train_selected")

    assert [row["id"] for row in strict_rows] == [1]
    assert [row["id"] for row in train_selected_rows] == [1, 2]


def test_read_pseudo_rows_for_training_uses_explicit_file(tmp_path):
    explicit_path = tmp_path / "custom_high_precision.jsonl"
    write_jsonl(explicit_path, [{"id": 9, "label": "<pos> custom <opinion> good"}])
    write_jsonl(tmp_path / "target_pseudo_selected.jsonl", [{"id": 1, "label": "<pos> a <opinion> good"}])

    rows = read_pseudo_rows_for_training(tmp_path, "strict", explicit_path)

    assert [row["id"] for row in rows] == [9]


def test_select_high_precision_pseudo_rows_filters_triplets_by_distance():
    rows = [
        {
            "id": 1,
            "text": "The battery is long.",
            "label": "<pos> battery <opinion> long",
            "fixed_changed": False,
            "sample_weight": 0.65,
            "quality_flags": {"all_terms_in_text": True, "triplet_count": 1},
        },
        {
            "id": 2,
            "text": "The keyboard is useful but the screen is bright.",
            "label": "<pos> keyboard <opinion> bright ; <pos> screen <opinion> bright",
            "fixed_changed": False,
            "sample_weight": 0.65,
            "quality_flags": {"all_terms_in_text": True, "triplet_count": 2},
        },
        {
            "id": 3,
            "text": "The use is cheap.",
            "label": "<pos> use <opinion> cheap",
            "fixed_changed": False,
            "sample_weight": 0.65,
            "quality_flags": {"all_terms_in_text": True, "triplet_count": 1},
        },
    ]

    selected, stats = select_high_precision_pseudo_rows(rows, min_weight=0.65, max_token_distance=2)

    assert [row["id"] for row in selected] == [1, 2]
    assert selected[1]["label"] == "<pos> screen <opinion> bright"
    assert selected[1]["high_precision_original_label"] == "<pos> keyboard <opinion> bright ; <pos> screen <opinion> bright"
    assert stats["input_rows"] == 3
    assert stats["selected_rows"] == 2
    assert stats["removed_triplets_by_distance"] == 1
    assert stats["rejected_noisy_term"] == 1

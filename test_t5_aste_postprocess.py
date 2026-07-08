from t5_aste_data import parse_triplet_text_list
from t5_aste_postprocess import evaluate_raw_and_fixed, fix_pred_triplets


def test_fix_pred_triplets_recovers_nearest_sentence_span():
    sentence = "Setup is not yet discovered by the keyboard utility"
    pred = "<neg> setup <opinion> discovred"

    fixed = fix_pred_triplets(pred, sentence)

    assert parse_triplet_text_list(fixed) == [("setup", "discovered", "neg")]


def test_fix_pred_triplets_aligns_singular_to_sentence_plural():
    sentence = "It is bundled with lots of very good applications ."
    pred = "<pos> application <opinion> good"

    fixed = fix_pred_triplets(pred, sentence)

    assert parse_triplet_text_list(fixed) == [("applications", "good", "pos")]


def test_evaluate_raw_and_fixed_reports_bgca_style_scores():
    rows = [
        {
            "text": "Setup is not yet discovered by the keyboard utility",
            "gold": "<neg> setup <opinion> discovered",
            "pred": "<neg> setup <opinion> discovred",
        }
    ]

    result = evaluate_raw_and_fixed(rows)

    assert result["raw_scores"]["micro_f1"] == 0.0
    assert result["fixed_scores"]["micro_f1"] == 1.0
    assert result["predictions"][0]["pred_fixed"] == "<neg> setup <opinion> discovered"


def test_fix_pred_triplets_contracts_safe_aspect_modifier_when_opinion_is_same_word():
    sentence = "The large screen gives you the option to comfortably watch movies ."
    pred = "<pos> large screen <opinion> large"

    fixed = fix_pred_triplets(pred, sentence)

    assert parse_triplet_text_list(fixed) == [("screen", "large", "pos")]


def test_fix_pred_triplets_expands_safe_right_context_aspect():
    sentence = "Product support very poor as each phone call costs me long distance ."
    pred = "<neg> product <opinion> poor"

    fixed = fix_pred_triplets(pred, sentence)

    assert parse_triplet_text_list(fixed) == [("product support", "poor", "neg")]


def test_fix_pred_triplets_normalizes_safe_opinion_boundary():
    sentence = "The keyboard is too slick ."
    pred = "<neg> keyboard <opinion> too slick"

    fixed = fix_pred_triplets(pred, sentence)

    assert parse_triplet_text_list(fixed) == [("keyboard", "slick", "neg")]


def test_fix_pred_triplets_preserves_meaningful_long_negative_opinion():
    sentence = "I can barely use any usb devices because they will not stay connected properly ."
    pred = "<neg> usb devices <opinion> not stay connected properly"

    fixed = fix_pred_triplets(pred, sentence)

    assert parse_triplet_text_list(fixed) == [("usb devices", "not stay connected properly", "neg")]

import numpy as np
import torch
from transformers import AutoTokenizer, EvalPrediction

import t5_absa_train as train

from t5_absa_train import (
    DataCollatorForSeq2SeqWithPairing,
    JsonlSeq2SeqDataset,
    encoder_pairing_contrastive_loss,
    find_token_subsequence_span,
    grouped_representation_consistency_loss,
    joint_weighted_loss,
    pairing_contrastive_loss,
    summarize_sample_weights,
    summarize_generation_weights,
    weighted_loss_mean,
)
from t5_aste_pipeline import (
    assign_augment_quality,
    assign_final_training_weights,
    assign_pseudo_quality,
    build_mixed_recall_pseudo_rows,
    select_high_value_augmented_rows,
)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text=None, text_target=None, max_length=None, truncation=True):
        value = text_target if text_target is not None else text
        return {"input_ids": [ord(ch) % 97 + 1 for ch in str(value)]}

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 97 + 1 for ch in str(text)]


class FakeMetricTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"
    unk_token = "<unk>"
    task_tokens = ("<pos>", "<neg>", "<neu>", "<opinion>")

    def __init__(self, decoded: dict[int, str]):
        self.decoded = decoded
        self.decode_calls = []

    def decode(self, token_ids, skip_special_tokens=True):
        row = np.asarray(token_ids).tolist()
        self.decode_calls.append((row, skip_special_tokens))
        text = self.decoded.get(next((token for token in row if token), 0), "")
        if skip_special_tokens:
            for token in self.task_tokens:
                text = text.replace(token, " ")
        return f"{self.pad_token} {text} {self.eos_token}"

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in token_ids]


def test_aste_compute_metrics_reports_structure_groups():
    single = "<pos> battery <opinion> long"
    triple = (
        "<pos> screen <opinion> bright ; "
        "<neg> keyboard <opinion> stiff ; "
        "<neu> price <opinion> average"
    )
    missing_one = "<pos> screen <opinion> bright ; <neg> keyboard <opinion> stiff"
    tokenizer = FakeMetricTokenizer({1: single, 2: missing_one, 3: triple})
    compute_metrics = train.build_aste_compute_metrics(tokenizer)

    metrics = compute_metrics(
        EvalPrediction(
            predictions=np.array([[1, 0], [2, 0]]),
            label_ids=np.array([[1, -100], [3, -100]]),
        )
    )

    assert abs(metrics["micro_f1"] - (6 / 7)) < 1e-12
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.75
    assert metrics["multi_micro_f1"] == 0.8
    assert metrics["count1_micro_f1"] == 1.0
    assert metrics["count3_micro_f1"] == 0.8
    assert metrics["exact_count_accuracy"] == 0.5
    assert metrics["under_generated_rows"] == 1
    assert metrics["over_generated_rows"] == 0
    assert abs(metrics["selection_score"] - ((6 / 7) + 0.001 * 0.8)) < 1e-12


def test_aste_compute_metrics_handles_empty_tuple_and_ignore_tokens():
    label = "<pos> battery <opinion> long"
    tokenizer = FakeMetricTokenizer({1: label})
    compute_metrics = train.build_aste_compute_metrics(tokenizer)

    metrics = compute_metrics(
        EvalPrediction(
            predictions=(np.array([[1, 0]]), np.array([[99]])),
            label_ids=np.array([[1, -100]]),
        )
    )

    assert metrics["micro_f1"] == 1.0
    assert tokenizer.decode_calls[1][0] == [1, tokenizer.pad_token_id]
    assert all(call[1] is False for call in tokenizer.decode_calls)

    empty_metrics = compute_metrics(
        EvalPrediction(
            predictions=(np.empty((0, 2), dtype=int),),
            label_ids=np.empty((0, 2), dtype=int),
        )
    )
    assert empty_metrics["micro_f1"] == 0.0
    assert empty_metrics["multi_micro_f1"] == 0.0
    assert empty_metrics["exact_count_accuracy"] == 0.0


def test_aste_compute_metrics_preserves_tokens_with_local_t5_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        r"J:\nlp\models\t5-base-py",
        local_files_only=True,
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<pos>", "<neg>", "<neu>", "<opinion>"]}
    )
    label = "<pos> battery life <opinion> long"
    token_ids = np.asarray([tokenizer.encode(label, add_special_tokens=True)])

    metrics = train.build_aste_compute_metrics(tokenizer)(
        EvalPrediction(predictions=token_ids, label_ids=token_ids.copy())
    )

    assert metrics["micro_f1"] == 1.0
    assert metrics["selection_score"] > 0.0


def test_aste_compute_metrics_handles_torch_logits_and_validates_inputs():
    label = "<pos> battery <opinion> long"
    tokenizer = FakeMetricTokenizer({1: label})
    logits = torch.tensor(
        [[[0.0, 5.0, 1.0], [5.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[1, -100]])
    metrics = train.build_aste_compute_metrics(tokenizer)(
        EvalPrediction(predictions=(logits,), label_ids=(labels,))
    )
    assert metrics["micro_f1"] == 1.0

    no_pad = FakeMetricTokenizer({1: label})
    no_pad.pad_token_id = None
    try:
        train.build_aste_compute_metrics(no_pad)
    except ValueError as exc:
        assert "pad_token_id" in str(exc)
    else:
        raise AssertionError("missing pad_token_id must be rejected")

    compute_metrics = train.build_aste_compute_metrics(tokenizer)
    invalid_cases = (
        EvalPrediction(predictions=np.array([1, 0]), label_ids=np.array([[1, 0]])),
        EvalPrediction(predictions=np.array([[1, 0], [1, 0]]), label_ids=np.array([[1, 0]])),
    )
    for prediction in invalid_cases:
        try:
            compute_metrics(prediction)
        except ValueError as exc:
            assert "dimension" in str(exc) or "batch" in str(exc)
        else:
            raise AssertionError("invalid metric inputs must be rejected")


def test_checkpoint_selection_config_supports_aste_f1_without_loading_t5():
    last = train.build_checkpoint_selection_config("last")
    best = train.build_checkpoint_selection_config("best")
    aste_f1 = train.build_checkpoint_selection_config("aste_f1")

    assert last["load_best_model_at_end"] is False
    assert last["metric_for_best_model"] == "eval_loss"
    assert best["load_best_model_at_end"] is True
    assert best["metric_for_best_model"] == "eval_loss"
    assert best["greater_is_better"] is False
    assert aste_f1 == {
        "predict_with_generate": True,
        "generation_num_beams": 1,
        "generation_max_length": 128,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_selection_score",
        "greater_is_better": True,
    }


def test_pseudo_quality_sets_bounded_dynamic_weight():
    rows = [
        {"text": "battery life is long", "label": "<pos> battery life <opinion> long", "augmentation": "target_pseudo"},
        {"text": "battery life is long", "label": "<pos> screen <opinion> dark", "augmentation": "target_pseudo"},
    ]

    weighted = assign_pseudo_quality(rows, base_weight=0.5)

    assert weighted[0]["sample_weight"] > weighted[1]["sample_weight"]
    assert 0.5 <= weighted[0]["sample_weight"] <= 0.8
    assert weighted[1]["sample_weight"] >= 0.5
    assert weighted[0]["quality_flags"]["all_terms_in_text"] is True


def test_augment_quality_uses_nli_label_and_consistency():
    rows = [
        {"text": "the battery life is long", "label": "<pos> battery life <opinion> long", "augmentation": "aspect_channel", "nli_label": "entailment"},
        {"text": "the battery life is long", "label": "<pos> screen <opinion> dark", "augmentation": "aspect_channel", "nli_label": "neutral"},
    ]

    weighted = assign_augment_quality(rows, base_weight=0.2)

    assert weighted[0]["sample_weight"] > weighted[1]["sample_weight"]
    assert 0.2 <= weighted[0]["sample_weight"] <= 0.35
    assert weighted[1]["sample_weight"] >= 0.2
    assert weighted[0]["quality_flags"]["nli_entailment"] is True


def test_augment_quality_rewards_model_filter_raw_agreement():
    rows = [
        {
            "text": "the battery life is long",
            "label": "<pos> battery life <opinion> long",
            "augmentation": "masked_aspect_channel",
            "nli_label": "entailment",
            "model_filter_passed": True,
            "model_filter_pred_raw": "<pos> battery life <opinion> long",
            "model_filter_pred_fixed": "<pos> battery life <opinion> long",
        },
        {
            "text": "the battery life is long",
            "label": "<pos> battery life <opinion> long",
            "augmentation": "masked_aspect_channel",
            "nli_label": "entailment",
            "model_filter_passed": True,
            "model_filter_pred_raw": "<pos> battery <opinion> long",
            "model_filter_pred_fixed": "<pos> battery life <opinion> long",
        },
    ]

    weighted = assign_augment_quality(rows, base_weight=0.2)

    assert weighted[0]["sample_weight"] > weighted[1]["sample_weight"]
    assert weighted[0]["quality_flags"]["model_filter_raw_exact"] is True
    assert weighted[1]["quality_flags"]["model_filter_fixed_exact"] is True
    assert weighted[1]["quality_flags"]["model_filter_raw_exact"] is False


def test_augment_quality_records_replacement_rank_score_without_rewarding_it():
    rows = [
        {
            "text": "the keyboard is responsive",
            "label": "<pos> keyboard <opinion> responsive",
            "augmentation": "masked_aspect_channel",
            "nli_label": "entailment",
            "replacement_rank": {"score": 9.0},
        },
        {
            "text": "the warranty is responsive",
            "label": "<pos> warranty <opinion> responsive",
            "augmentation": "masked_aspect_channel",
            "nli_label": "entailment",
            "replacement_rank": {"score": 1.0},
        },
    ]

    weighted = assign_augment_quality(rows, base_weight=0.2)

    assert weighted[0]["sample_weight"] == weighted[1]["sample_weight"]
    assert weighted[0]["quality_flags"]["replacement_rank_score"] == 9.0


def test_dataset_prefers_row_sample_weight_over_source_weight():
    rows = [{"input": "extract aste: x", "target": "<pos> x <opinion> good", "sample_weight": 0.37}]
    dataset = JsonlSeq2SeqDataset(rows, TinyTokenizer(), 16, 16, 1.0, 0.5, 0.2)

    item = dataset[0]
    summary = summarize_sample_weights(rows, 1.0, 0.5, 0.2)

    assert item["sample_weight"] == 0.37
    assert summary["sample_weight_mean"] == 0.37


def test_dataset_can_force_domain_weights_over_row_sample_weight():
    rows = [{"input": "extract aste: x", "target": "<pos> x <opinion> good", "sample_weight": 0.65, "augmentation": "target_pseudo"}]
    dataset = JsonlSeq2SeqDataset(rows, TinyTokenizer(), 16, 16, 1.0, 0.4, 0.2, force_domain_weights=True)
    summary = summarize_sample_weights(rows, 1.0, 0.4, 0.2, force_domain_weights=True)

    assert dataset[0]["sample_weight"] == 0.4
    assert summary["sample_weight_mean"] == 0.4
    assert summary["force_domain_weights"] is True


def test_masked_channels_use_augment_fallback_weight_in_training_summary():
    rows = [
        {"input": "x", "target": "y", "augmentation": "masked_aspect_channel"},
        {"input": "x", "target": "y", "augmentation": "masked_opinion_sentiment_channel"},
    ]

    dataset = JsonlSeq2SeqDataset(rows, TinyTokenizer(), 16, 16, 1.0, 0.5, 0.2)
    summary = summarize_sample_weights(rows, 1.0, 0.5, 0.2)

    assert dataset[0]["sample_weight"] == 0.2
    assert dataset[1]["sample_weight"] == 0.2
    assert summary["c3da_augment"] == 2
    assert summary["c3da_augment_weight_mean"] == 0.2


def test_neutral_generation_weight_does_not_enable_multi_triplet_gain_for_other_rows():
    rows = [
        {
            "input": "x",
            "target": "<pos> a <opinion> good ; <neg> b <opinion> bad",
            "sample_weight": 1.0,
        },
        {
            "input": "y",
            "target": "<neu> c <opinion> average",
            "sample_weight": 1.0,
        },
    ]
    dataset = JsonlSeq2SeqDataset(
        rows,
        TinyTokenizer(),
        16,
        16,
        1.0,
        0.5,
        0.2,
        multi_triplet_loss_gain=0.1,
        max_effective_weight=1.0,
        neutral_generation_loss_gain=1.0,
        neutral_generation_max_effective_weight=2.0,
    )

    assert dataset[0]["domain_weight"] == 1.0
    assert dataset[1]["domain_weight"] == 2.0
    assert dataset[0]["structure_weight"] == 1.0
    assert dataset[1]["structure_weight"] == 1.0
    summary = summarize_generation_weights(dataset)
    assert summary["neutral_rows"] == 1
    assert summary["neutral_weight_mean"] == 2.0
    assert summary["non_neutral_weight_max"] == 1.0


def test_select_high_value_augmented_rows_prefers_raw_exact_and_diverse_bases():
    rows = [
        {
            "text": "a",
            "label": "<pos> a <opinion> good",
            "base_text": "base-1",
            "sample_weight": 0.31,
            "quality_score": 1.4,
            "quality_flags": {"model_filter_raw_exact": True, "model_filter_fixed_exact": True},
        },
        {
            "text": "b",
            "label": "<pos> b <opinion> good",
            "base_text": "base-1",
            "sample_weight": 0.34,
            "quality_score": 1.5,
            "quality_flags": {"model_filter_raw_exact": False, "model_filter_fixed_exact": True},
        },
        {
            "text": "c",
            "label": "<pos> c <opinion> good",
            "base_text": "base-2",
            "sample_weight": 0.29,
            "quality_score": 1.2,
            "quality_flags": {"model_filter_raw_exact": True, "model_filter_fixed_exact": True},
        },
    ]

    selected, stats = select_high_value_augmented_rows(
        rows,
        max_rows=2,
        max_per_base=1,
        selected_weight=0.35,
    )

    assert [row["text"] for row in selected] == ["a", "c"]
    assert all(row["sample_weight"] == 0.35 for row in selected)
    assert all(row["selected_augmentation"] is True for row in selected)
    assert stats["input_rows"] == 3
    assert stats["selected_rows"] == 2
    assert stats["skipped_by_base_limit"] == 1


def test_select_high_value_augmented_rows_does_not_rank_by_replacement_score():
    rows = [
        {
            "text": "low",
            "label": "<pos> warranty <opinion> responsive",
            "base_text": "base-1",
            "sample_weight": 0.34,
            "quality_score": 1.4,
            "quality_flags": {"model_filter_raw_exact": True, "replacement_rank_score": 1.0},
        },
        {
            "text": "high",
            "label": "<pos> keyboard <opinion> responsive",
            "base_text": "base-2",
            "sample_weight": 0.31,
            "quality_score": 1.2,
            "quality_flags": {"model_filter_raw_exact": True, "replacement_rank_score": 9.0},
        },
    ]

    selected, _stats = select_high_value_augmented_rows(rows, max_rows=1, max_per_base=1)

    assert selected[0]["text"] == "low"


def test_select_high_value_augmented_rows_can_require_raw_exact():
    rows = [
        {
            "text": "raw",
            "label": "<pos> a <opinion> good",
            "base_text": "base-1",
            "sample_weight": 0.31,
            "quality_score": 1.0,
            "quality_flags": {"model_filter_raw_exact": True},
        },
        {
            "text": "fixed",
            "label": "<pos> b <opinion> good",
            "base_text": "base-2",
            "sample_weight": 0.35,
            "quality_score": 2.0,
            "quality_flags": {"model_filter_fixed_exact": True},
        },
    ]

    selected, stats = select_high_value_augmented_rows(rows, max_rows=2, require_raw_exact=True)

    assert [row["text"] for row in selected] == ["raw"]
    assert stats["input_rows"] == 2
    assert stats["candidate_rows"] == 1
    assert stats["require_raw_exact"] is True


def test_select_high_value_augmented_rows_can_require_model_filter_passed():
    rows = [
        {
            "text": "passed",
            "label": "<pos> a <opinion> good ; <neg> b <opinion> bad",
            "base_text": "base-1",
            "sample_weight": 0.2,
            "quality_score": 1.0,
            "model_filter_passed": True,
            "model_filter_match": "span_compatible",
            "quality_flags": {"model_filter_raw_exact": False},
        },
        {
            "text": "failed",
            "label": "<pos> c <opinion> good ; <neg> d <opinion> bad",
            "base_text": "base-2",
            "sample_weight": 0.2,
            "quality_score": 2.0,
            "model_filter_passed": False,
        },
    ]

    selected, stats = select_high_value_augmented_rows(rows, max_rows=2, require_model_filter_passed=True)

    assert [row["text"] for row in selected] == ["passed"]
    assert stats["candidate_rows"] == 1
    assert stats["require_model_filter_passed"] is True


def test_assign_final_training_weights_boosts_multi_triplet_and_neutral_rows():
    rows = [
        {"text": "a", "label": "<pos> a <opinion> good", "sample_weight": 1.0},
        {
            "text": "b",
            "label": "<pos> a <opinion> good ; <neg> b <opinion> bad",
            "sample_weight": 0.65,
            "augmentation": "target_pseudo",
        },
        {"text": "c", "label": "<neu> c <opinion> average", "sample_weight": 0.65, "augmentation": "target_pseudo"},
    ]

    weighted, stats = assign_final_training_weights(rows, multi_triplet_gain=0.25, neutral_gain=0.5, max_weight=1.2)

    assert weighted[0]["sample_weight"] == 1.0
    assert weighted[1]["sample_weight"] == 0.8125
    assert weighted[1]["final_weight_flags"]["multi_triplet"] is True
    assert weighted[2]["sample_weight"] == 0.975
    assert weighted[2]["final_weight_flags"]["contains_neutral"] is True
    assert stats["multi_triplet_rows"] == 1
    assert stats["neutral_rows"] == 1


def test_build_mixed_recall_pseudo_rows_adds_low_weight_extra_rows():
    high_precision_rows = [
        {
            "id": "a",
            "text": "The keyboard is good.",
            "label": "<pos> keyboard <opinion> good",
            "sample_weight": 0.65,
        }
    ]
    recall_rows = [
        {
            "id": "a",
            "text": "The keyboard is good.",
            "label": "<pos> keyboard <opinion> good",
            "sample_weight": 0.65,
        },
        {
            "id": "b",
            "text": "The screen is bright and the battery is poor.",
            "label": "<pos> screen <opinion> bright ; <neg> battery <opinion> poor",
            "sample_weight": 0.65,
        },
    ]

    mixed, stats = build_mixed_recall_pseudo_rows(
        high_precision_rows,
        recall_rows,
        recall_extra_weight=0.25,
        recall_extra_max_rows=10,
    )

    assert len(mixed) == 2
    assert mixed[0]["sample_weight"] == 0.65
    assert mixed[0]["pseudo_mix_source"] == "high_precision"
    assert mixed[1]["sample_weight"] == 0.25
    assert mixed[1]["pseudo_mix_source"] == "recall_extra"
    assert stats["high_precision_rows"] == 1
    assert stats["recall_extra_added"] == 1


def test_weighted_loss_affects_batch_size_one():
    loss = weighted_loss_mean(torch.tensor([10.0]), torch.tensor([0.2]))

    assert float(loss) == 2.0


def test_grouped_representation_consistency_loss_only_uses_paired_groups():
    representations = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1], [1, 1], [1, 1]])
    paired_groups = torch.tensor([7, 7, 9])
    unpaired_groups = torch.tensor([1, 2, 3])

    paired_loss = grouped_representation_consistency_loss(representations, attention_mask, paired_groups)
    unpaired_loss = grouped_representation_consistency_loss(representations, attention_mask, unpaired_groups)

    assert paired_loss > 0
    assert unpaired_loss == 0


def test_dataset_returns_domain_and_structure_weights():
    rows = [
        {"input": "x", "target": "<pos> a <opinion> good ; <neu> b <opinion> average", "sample_weight": 0.65, "augmentation": "target_pseudo"}
    ]
    dataset = JsonlSeq2SeqDataset(
        rows,
        TinyTokenizer(),
        16,
        16,
        1.0,
        0.5,
        0.2,
        multi_triplet_loss_gain=0.1,
        neutral_loss_gain=0.15,
        max_effective_weight=1.0,
    )

    item = dataset[0]

    assert item["sample_weight"] == 0.65
    assert item["domain_weight"] == 0.65
    assert item["structure_weight"] == 0.8125


def test_joint_weighted_loss_adds_structure_loss():
    per_sample_loss = torch.tensor([2.0, 4.0])
    domain_weight = torch.tensor([0.5, 1.0])
    structure_weight = torch.tensor([1.0, 0.5])

    loss = joint_weighted_loss(per_sample_loss, domain_weight, structure_weight, lambda_structure=0.25)

    assert float(loss) == 3.0


def test_dataset_returns_encoder_pairing_spans_for_source_multi_triplets():
    rows = [
        {
            "input": "The battery life is very long but the screen is dark.",
            "target": "<pos> battery life <opinion> very long ; <neg> screen <opinion> dark",
            "sample_weight": 1.0,
        }
    ]
    tokenizer = TinyTokenizer()
    dataset = JsonlSeq2SeqDataset(
        rows,
        tokenizer,
        96,
        96,
        1.0,
        0.5,
        0.2,
        max_pairing_triplets=4,
        pairing_source_only=True,
    )

    item = dataset[0]

    input_ids = tokenizer.encode(rows[0]["input"], add_special_tokens=False)
    assert item["pairing_aspect_spans"] == [
        list(find_token_subsequence_span(input_ids, tokenizer.encode("battery life", add_special_tokens=False))),
        list(find_token_subsequence_span(input_ids, tokenizer.encode("screen", add_special_tokens=False))),
    ]
    assert item["pairing_opinion_spans"] == [
        list(find_token_subsequence_span(input_ids, tokenizer.encode("very long", add_special_tokens=False))),
        list(find_token_subsequence_span(input_ids, tokenizer.encode("dark", add_special_tokens=False))),
    ]
    assert item["pairing_mask"][:2] == [1, 1]


def test_pairing_source_only_excludes_pseudo_augment_and_single_triplets():
    rows = [
        {
            "input": "The battery is long and the screen is dark.",
            "target": "<pos> battery <opinion> long ; <neg> screen <opinion> dark",
            "sample_weight": 0.65,
            "augmentation": "target_pseudo",
        },
        {
            "input": "The battery is long and the screen is dark.",
            "target": "<pos> battery <opinion> long ; <neg> screen <opinion> dark",
            "sample_weight": 0.2,
            "augmentation": "masked_aspect_channel",
        },
        {
            "input": "The battery is long.",
            "target": "<pos> battery <opinion> long",
            "sample_weight": 1.0,
        },
    ]
    dataset = JsonlSeq2SeqDataset(
        rows,
        TinyTokenizer(),
        96,
        96,
        1.0,
        0.5,
        0.2,
        pairing_source_only=True,
    )

    for index in range(len(rows)):
        item = dataset[index]
        assert item["pairing_aspect_spans"] == []
        assert item["pairing_opinion_spans"] == []
        assert item["pairing_mask"] == []


def test_dataset_skips_pairing_features_for_low_confidence_or_single_triplet_rows():
    rows = [
        {
            "input": "x",
            "target": "<pos> battery life <opinion> long",
            "sample_weight": 0.5,
            "augmentation": "label_to_text_channel",
        }
    ]
    dataset = JsonlSeq2SeqDataset(rows, TinyTokenizer(), 16, 96, 1.0, 0.5, 0.2, max_pairing_triplets=4)

    item = dataset[0]

    assert item["pairing_aspect_spans"] == []
    assert item["pairing_opinion_spans"] == []
    assert item["pairing_mask"] == []


def test_pairing_collator_pads_pairing_fields():
    class BaseCollator:
        def __call__(self, features):
            return {
                "input_ids": torch.tensor([feature["input_ids"] for feature in features]),
                "labels": torch.tensor([feature["labels"] for feature in features]),
            }

    collator = DataCollatorForSeq2SeqWithPairing(BaseCollator())
    batch = collator(
        [
            {
                "input_ids": [1],
                "labels": [2],
                "pairing_aspect_spans": [[1, 2]],
                "pairing_opinion_spans": [[3, 4]],
                "pairing_mask": [1],
            },
            {
                "input_ids": [1],
                "labels": [2],
                "pairing_aspect_spans": [[5, 6], [7, 8]],
                "pairing_opinion_spans": [[9, 10], [11, 12]],
                "pairing_mask": [1, 1],
            },
        ]
    )

    assert batch["pairing_aspect_spans"].tolist() == [[[1, 2], [0, 0]], [[5, 6], [7, 8]]]
    assert batch["pairing_mask"].tolist() == [[1, 0], [1, 1]]


def test_pairing_contrastive_loss_prefers_correct_opinion_pair():
    hidden = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ]
    )
    aspect_spans = torch.tensor([[[0, 1], [2, 3]]])
    opinion_spans = torch.tensor([[[0, 1], [2, 3]]])
    mask = torch.tensor([[1, 1]])

    good_loss = pairing_contrastive_loss(hidden, aspect_spans, opinion_spans, mask)
    swapped_loss = pairing_contrastive_loss(hidden, aspect_spans, torch.tensor([[[2, 3], [0, 1]]]), mask)

    assert good_loss < swapped_loss


def test_encoder_pairing_loss_prefers_correct_pairs():
    hidden = torch.tensor(
        [[
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]]
    )
    aspect_spans = torch.tensor([[[0, 1], [1, 2]]])
    opinion_spans = torch.tensor([[[2, 3], [3, 4]]])
    mask = torch.tensor([[1, 1]])

    good_loss, stats = encoder_pairing_contrastive_loss(
        hidden, aspect_spans, opinion_spans, mask, return_stats=True
    )
    swapped_loss = encoder_pairing_contrastive_loss(
        hidden,
        aspect_spans,
        torch.tensor([[[3, 4], [2, 3]]]),
        mask,
    )

    assert good_loss < swapped_loss
    assert stats["pairing_aspect_accuracy"] == 1.0
    assert stats["pairing_opinion_accuracy"] == 1.0
    assert stats["pairing_active_rows"] == 1.0
    assert stats["pairing_active_pairs"] == 2.0


def test_encoder_pairing_loss_supports_multiple_positives():
    hidden = torch.tensor(
        [[
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]]
    )
    aspect_spans = torch.tensor([[[0, 1], [0, 1], [3, 4]]])
    opinion_spans = torch.tensor([[[1, 2], [2, 3], [4, 5]]])
    mask = torch.tensor([[1, 1, 1]])

    loss, stats = encoder_pairing_contrastive_loss(
        hidden, aspect_spans, opinion_spans, mask, return_stats=True
    )

    assert torch.isfinite(loss)
    assert stats["pairing_aspect_accuracy"] == 1.0
    assert stats["pairing_opinion_accuracy"] == 1.0


def test_encoder_pairing_loss_is_zero_without_real_negatives():
    hidden = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    aspect_spans = torch.tensor([[[0, 1], [0, 1]]])
    opinion_spans = torch.tensor([[[1, 2], [2, 3]]])
    mask = torch.tensor([[1, 1]])

    loss, stats = encoder_pairing_contrastive_loss(
        hidden, aspect_spans, opinion_spans, mask, return_stats=True
    )

    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    assert stats["pairing_active_rows"] == 0.0

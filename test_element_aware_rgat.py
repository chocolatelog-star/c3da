import copy

import torch
from transformers import T5Config

from element_aware_rgat import (
    align_gold_elements_to_graph_words,
    balanced_element_focus_loss,
    multi_element_coverage_loss,
)
from syntactic_graph_adapter import (
    SyntacticGraphAdapter,
    SyntacticGraphT5ForConditionalGeneration,
    graph_model_config,
)


class _DiscardTrace:
    def record(self, *_args, **_kwargs):
        return None


def _parser_tokens():
    return [
        {"index": 0, "text": "The", "start": 0, "end": 3},
        {"index": 1, "text": "food", "start": 4, "end": 8},
        {"index": 2, "text": "is", "start": 9, "end": 11},
        {"index": 3, "text": "very", "start": 12, "end": 16},
        {"index": 4, "text": "good", "start": 17, "end": 21},
        {"index": 5, "text": "but", "start": 22, "end": 25},
        {"index": 6, "text": "service", "start": 26, "end": 33},
        {"index": 7, "text": "is", "start": 34, "end": 36},
        {"index": 8, "text": "slow", "start": 37, "end": 41},
    ]


def _graph_inputs():
    return {
        "word_to_subword": torch.tensor([[[0], [1], [2]]]),
        "word_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
        "edge_src": torch.tensor([[0, 1, 2]]),
        "edge_dst": torch.tensor([[2, 2, 2]]),
        "relation_id": torch.tensor([[0, 0, 0]]),
        "dependency_relation_id": torch.tensor([[0, 0, 0]]),
        "pos_pair_id": torch.tensor([[0, 0, 0]]),
        "edge_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
    }


def test_gold_element_alignment_marks_unique_nodes_and_reports_ambiguous_unmatched():
    result = align_gold_elements_to_graph_words(
        text="The food is very good but service is slow",
        parser_tokens=_parser_tokens(),
        triplets=[
            ("food", "very good", "pos"),
            ("service", "slow", "neg"),
            ("missing", "is", "neu"),
        ],
    )

    assert result["node_labels"] == [0, 1, 0, 1, 1, 0, 1, 0, 1]
    assert result["node_loss_mask"] == [1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert result["element_spans"] == [[1, 2], [3, 5], [6, 7], [8, 9]]
    assert result["stats"] == {
        "gold_aspects": 3,
        "aligned_aspects": 2,
        "unmatched_aspects": 1,
        "ambiguous_aspects": 0,
        "gold_opinions": 3,
        "aligned_opinions": 2,
        "unmatched_opinions": 0,
        "ambiguous_opinions": 1,
    }


def test_balanced_focus_loss_gives_equal_positive_and_negative_mass():
    salience = torch.tensor([[0.8, 0.2, 0.6, 0.4]])
    labels = torch.tensor([[1, 0, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)

    loss, stats = balanced_element_focus_loss(salience, labels, mask)

    expected = 0.5 * (-torch.log(torch.tensor([0.8, 0.6]))).mean()
    expected += 0.5 * (-torch.log(torch.tensor([0.8, 0.6]))).mean()
    assert torch.allclose(loss, expected)
    assert stats["positive_count"] == 2
    assert stats["negative_count"] == 2


def test_coverage_loss_uses_every_aligned_element_only_for_source_multi_rows():
    salience = torch.tensor([[0.8, 0.4, 0.6, 0.2], [0.9, 0.1, 0.5, 0.5]])
    spans = torch.tensor([[[0, 2], [2, 3]], [[0, 1], [1, 2]]])
    span_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool)
    source_mask = torch.tensor([1, 1], dtype=torch.bool)
    triplet_count = torch.tensor([2, 1])

    loss, stats = multi_element_coverage_loss(
        salience,
        spans,
        span_mask,
        source_mask,
        triplet_count,
    )

    assert torch.allclose(loss, torch.tensor(((1 - 0.6) + (1 - 0.6)) / 2))
    assert stats["active_element_count"] == 2
    assert stats["active_row_count"] == 1


def test_target_rows_have_zero_auxiliary_losses():
    salience = torch.tensor([[0.8, 0.2]])
    labels = torch.tensor([[1, 0]])
    node_mask = torch.tensor([[1, 1]], dtype=torch.bool)
    source_mask = torch.tensor([0], dtype=torch.bool)
    spans = torch.tensor([[[0, 1], [1, 2]]])
    span_mask = torch.tensor([[1, 1]], dtype=torch.bool)

    focus, _ = balanced_element_focus_loss(salience, labels, node_mask & source_mask[:, None])
    coverage, _ = multi_element_coverage_loss(
        salience,
        spans,
        span_mask,
        source_mask,
        torch.tensor([2]),
    )

    assert focus.item() == 0.0
    assert coverage.item() == 0.0


def test_zero_initialized_salience_head_is_attention_and_graph_equivalent():
    torch.manual_seed(1000)
    control = SyntacticGraphAdapter(8, 8, 2, 4, 2, 0.0, element_aware_enabled=False).eval()
    treatment = SyntacticGraphAdapter(8, 8, 2, 4, 2, 0.0, element_aware_enabled=True).eval()
    treatment.load_state_dict(control.state_dict(), strict=False)
    hidden = torch.randn(1, 3, 8)
    fields = _graph_inputs()

    control_output = control(
        hidden,
        torch.ones(1, 3, dtype=torch.long),
        trace=_DiscardTrace(),
        **fields,
    )
    treatment_output = treatment(hidden, torch.ones(1, 3, dtype=torch.long), **fields)

    assert torch.all(treatment_output.salience_logits == 0)
    assert torch.all(treatment_output.salience_scores == 0.5)
    assert torch.allclose(control_output.attention_probabilities, treatment_output.attention_probabilities, atol=1e-7, rtol=0)
    assert torch.allclose(control_output.graph_hidden, treatment_output.graph_hidden, atol=1e-7, rtol=0)
    assert torch.equal(control_output.fused_hidden, treatment_output.fused_hidden)


def test_disabled_element_attention_does_not_retain_attention_tensor_by_default():
    adapter = SyntacticGraphAdapter(8, 8, 2, 4, 2, 0.0, element_aware_enabled=False).eval()
    output = adapter(
        torch.randn(1, 3, 8),
        torch.ones(1, 3, dtype=torch.long),
        **_graph_inputs(),
    )

    assert output.attention_probabilities is None


def test_attention_bias_uses_message_source_node_salience():
    adapter = SyntacticGraphAdapter(8, 8, 2, 4, 2, 0.0, element_aware_enabled=True).eval()
    with torch.no_grad():
        adapter.node_projection.weight.copy_(torch.eye(8))
        adapter.node_projection.bias.zero_()
        adapter.query_projection.weight.zero_()
        adapter.query_projection.bias.zero_()
        adapter.key_projection.weight.zero_()
        adapter.key_projection.bias.zero_()
        adapter.dependency_bias.weight.zero_()
        adapter.pos_pair_bias.weight.zero_()
        adapter.salience_head.weight.zero_()
        adapter.salience_head.bias.zero_()
        adapter.salience_head.weight[0, 0] = 4.0
        adapter.output_projection.weight.normal_(0.0, 0.02)
    hidden = torch.zeros(1, 3, 8)
    hidden[0, 0, 0] = 2.0
    hidden[0, 1, 0] = -2.0
    fields = _graph_inputs()

    output = adapter(hidden, torch.ones(1, 3, dtype=torch.long), **fields)
    attention = output.attention_probabilities[0, :, 0]

    assert attention[0] > attention[1]
    assert torch.isfinite(attention).all()


def test_salience_state_round_trip_preserves_nonzero_parameters():
    adapter = SyntacticGraphAdapter(8, 8, 2, 4, 2, 0.0, element_aware_enabled=True)
    with torch.no_grad():
        adapter.salience_head.weight.fill_(0.125)
        adapter.salience_head.bias.fill_(-0.25)
    restored = copy.deepcopy(adapter)

    assert torch.equal(restored.salience_head.weight, adapter.salience_head.weight)
    assert torch.equal(restored.salience_head.bias, adapter.salience_head.bias)


def test_element_aware_t5_forward_exposes_salience_without_gold_masks():
    config = T5Config(
        vocab_size=32,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    graph_model_config(config, 2, element_aware_enabled=True, focus_enabled=True, coverage_enabled=True)
    model = SyntacticGraphT5ForConditionalGeneration(config).eval()
    input_ids = torch.tensor([[2, 3, 4]])
    labels = torch.tensor([[5, 1]])
    fields = {"graph_" + name: value for name, value in _graph_inputs().items()}

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        return_dict=True,
        **fields,
    )

    assert outputs.element_salience_scores.shape == (1, 3)
    assert torch.all(outputs.element_salience_scores == 0.5)
    assert torch.isfinite(outputs.logits).all()

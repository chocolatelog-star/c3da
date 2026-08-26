import copy

import torch
from transformers import T5Config, T5ForConditionalGeneration

from syntactic_graph_adapter import (
    SyntacticGraphAdapter,
    SyntacticGraphT5ForConditionalGeneration,
    graph_model_config,
)


def graph_inputs(batch_size=2, token_count=6, word_count=3):
    word_to_subword = torch.tensor(
        [
            [[0, 1], [2, 3], [4, -1]],
            [[0, 1], [2, -1], [4, 5]],
        ][:batch_size],
        dtype=torch.long,
    )
    word_mask = torch.tensor([[1, 1, 1], [1, 1, 1]][:batch_size], dtype=torch.bool)
    edge_src = torch.tensor(
        [[0, 1, 2, 1, 0, 2], [0, 1, 2, 1, 0, 2]][:batch_size], dtype=torch.long
    )
    edge_dst = torch.tensor(
        [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]][:batch_size], dtype=torch.long
    )
    edge_mask = torch.ones_like(edge_src, dtype=torch.bool)
    relation_id = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]][:batch_size], dtype=torch.long
    )
    dependency_relation_id = relation_id.clone()
    pos_pair_id = relation_id.clone()
    return {
        "word_to_subword": word_to_subword,
        "word_mask": word_mask,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "relation_id": relation_id,
        "dependency_relation_id": dependency_relation_id,
        "pos_pair_id": pos_pair_id,
        "edge_mask": edge_mask,
    }


def test_zero_initialized_output_preserves_encoder_hidden_and_prefix_padding():
    torch.manual_seed(1000)
    adapter = SyntacticGraphAdapter(
        hidden_size=8,
        graph_hidden_size=8,
        attention_heads=2,
        head_size=4,
        num_relations=8,
        dropout=0.0,
    ).eval()
    hidden = torch.randn(2, 6, 8)
    hidden[:, 5] = 17.0
    fields = graph_inputs()

    output = adapter(hidden, attention_mask=torch.ones(2, 6, dtype=torch.long), **fields)

    assert torch.equal(output.fused_hidden, hidden)
    assert torch.equal(output.fused_hidden[:, 5], hidden[:, 5])
    assert torch.all(output.graph_hidden[:, 3:] == 0)


def test_backward_reaches_zero_initialized_output_projection():
    torch.manual_seed(1000)
    adapter = SyntacticGraphAdapter(
        hidden_size=8,
        graph_hidden_size=8,
        attention_heads=2,
        head_size=4,
        num_relations=8,
        dropout=0.0,
    )
    hidden = torch.randn(2, 6, 8, requires_grad=True)
    fields = graph_inputs()
    loss = adapter(hidden, attention_mask=torch.ones(2, 6, dtype=torch.long), **fields).fused_hidden.square().mean()
    loss.backward()

    assert adapter.output_projection.weight.grad is not None
    assert torch.isfinite(adapter.output_projection.weight.grad).all()
    assert adapter.output_projection.weight.grad.abs().sum() > 0


def test_batch_graphs_do_not_cross_sample_when_one_graph_changes():
    torch.manual_seed(1000)
    adapter = SyntacticGraphAdapter(
        hidden_size=8,
        graph_hidden_size=8,
        attention_heads=2,
        head_size=4,
        num_relations=8,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        adapter.output_projection.weight.normal_(0.0, 0.02)
    hidden = torch.randn(2, 6, 8)
    fields = graph_inputs()
    batched = adapter(hidden, attention_mask=torch.ones(2, 6, dtype=torch.long), **fields).fused_hidden

    single_fields = {key: value[0:1].clone() for key, value in fields.items()}
    single = adapter(hidden[0:1], attention_mask=torch.ones(1, 6, dtype=torch.long), **single_fields).fused_hidden
    assert torch.allclose(batched[0], single[0], atol=1e-6, rtol=1e-6)


def test_adapter_state_dict_round_trip_is_exact():
    torch.manual_seed(1000)
    adapter = SyntacticGraphAdapter(8, 8, 2, 4, 8, 0.0).eval()
    restored = copy.deepcopy(adapter).eval()
    hidden = torch.randn(2, 6, 8)
    fields = graph_inputs()
    first = adapter(hidden, attention_mask=torch.ones(2, 6, dtype=torch.long), **fields).fused_hidden
    second = restored(hidden, attention_mask=torch.ones(2, 6, dtype=torch.long), **fields).fused_hidden
    assert torch.equal(first, second)


def test_zero_initialized_t5_wrapper_matches_control_logits_and_loss():
    torch.manual_seed(1000)
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
    control = T5ForConditionalGeneration(config).eval()
    graph_config = copy.deepcopy(config)
    graph_model_config(graph_config, 8)
    treatment = SyntacticGraphT5ForConditionalGeneration(graph_config).eval()
    treatment.load_state_dict(control.state_dict(), strict=False)
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]])
    labels = torch.tensor([[1, 2, 3], [1, 2, 3]])
    fields = {"graph_" + key: value for key, value in graph_inputs().items()}
    control_output = control(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels)
    treatment_output = treatment(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        **fields,
    )
    assert torch.equal(control_output.logits, treatment_output.logits)
    assert torch.equal(control_output.loss, treatment_output.loss)


def test_graph_t5_generation_uses_graph_fields_without_changing_output_shape():
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
    graph_model_config(config, 8)
    model = SyntacticGraphT5ForConditionalGeneration(config).eval()
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]])
    fields = {"graph_" + key: value for key, value in graph_inputs().items()}
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=3,
        **fields,
    )
    assert generated.shape[0] == 2
    assert generated.shape[1] <= 4

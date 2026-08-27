"""CPU regression tests for the read-only M1 numerical trace."""

from unittest.mock import patch

import torch
import torch.nn as nn
from transformers import T5Config

from m1_graph_fp16_numerical_trace import (
    NumericalTrace,
    build_trace_report,
    record_target_pseudo_result,
    summarize_tensor_stats,
)
from syntactic_graph_adapter import (
    SyntacticGraphAdapter,
    SyntacticGraphT5ForConditionalGeneration,
    graph_model_config,
)


def _small_graph_inputs(batch_size=1, token_count=4, word_count=2):
    hidden = torch.arange(batch_size * token_count * 8, dtype=torch.float32).reshape(
        batch_size, token_count, 8
    ) / 100.0
    word_to_subword = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long).expand(batch_size, -1, -1).clone()
    word_mask = torch.ones(batch_size, word_count, dtype=torch.bool)
    edge_src = torch.tensor([[0, 1, 0]], dtype=torch.long).expand(batch_size, -1).clone()
    edge_dst = torch.tensor([[1, 0, 0]], dtype=torch.long).expand(batch_size, -1).clone()
    relation_id = torch.zeros(batch_size, 3, dtype=torch.long)
    dependency_relation_id = torch.zeros(batch_size, 3, dtype=torch.long)
    pos_pair_id = torch.zeros(batch_size, 3, dtype=torch.long)
    edge_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    attention_mask = torch.ones(batch_size, token_count, dtype=torch.long)
    return {
        "hidden": hidden,
        "attention_mask": attention_mask,
        "word_to_subword": word_to_subword,
        "word_mask": word_mask,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "relation_id": relation_id,
        "dependency_relation_id": dependency_relation_id,
        "pos_pair_id": pos_pair_id,
        "edge_mask": edge_mask,
    }


def test_finite_trace_records_all_finite_statistics():
    trace = NumericalTrace("fp32")
    stats = trace.record("query_projection", torch.ones(1, 2, 4), axes=("batch", "node", "feature"))

    assert stats["dtype"] == "torch.float32"
    assert stats["finite_count"] == 8
    assert stats["total_count"] == 8
    assert stats["nan_count"] == 0
    assert stats["first_nonfinite"] is None
    assert trace.finalize()["pass"] is True


def test_artificial_overflow_marks_first_nonfinite_stage_and_location():
    trace = NumericalTrace("fp16")
    trace.record("query_projection", torch.ones(1, 1, 4), axes=("batch", "node", "feature"))
    trace.record(
        "query_key_product",
        torch.tensor([[[float("inf")]]], dtype=torch.float16),
        axes=("batch", "edge", "head"),
    )
    trace.record("final_loss", torch.tensor(float("nan")), axes=())

    result = trace.finalize()
    assert result["first_nonfinite_stage"] == "query_key_product"
    assert result["stages"]["query_key_product"]["first_nonfinite"]["batch"] == 0
    assert result["stages"]["query_key_product"]["first_nonfinite"]["edge"] == 0
    assert result["stages"]["final_loss"]["nan_count"] == 1


def test_syntactic_adapter_can_emit_required_intermediate_trace_stages():
    adapter = SyntacticGraphAdapter(
        hidden_size=8,
        graph_hidden_size=8,
        attention_heads=2,
        head_size=4,
        num_relations=1,
        dropout=0.0,
    )
    trace = NumericalTrace("fp32")
    adapter(**_small_graph_inputs(), trace=trace)
    stages = trace.finalize()["stages"]
    for stage in (
        "pooled_word_hidden",
        "node_projection",
        "query_projection",
        "key_projection",
        "value_projection",
        "edge_query",
        "edge_key",
        "edge_value",
        "query_key_product",
        "attention_logits_before_scaling",
        "attention_logits_scaled",
        "dependency_bias",
        "pos_pair_bias",
        "final_attention_logits",
        "softmax_input_float32_logits",
        "attention_probabilities",
        "relation_embeddings",
        "edge_messages",
        "aggregated_messages",
        "graph_hidden",
        "dropout_graph_hidden",
        "output_projection_input",
        "output_projection_output",
        "residual",
        "gate",
        "word_delta",
        "fused_hidden",
    ):
        assert stage in stages
        assert stages[stage]["finite_count"] == stages[stage]["total_count"]


def test_trace_attention_probabilities_accepts_fp16_attention_with_fp32_logits():
    class FloatProjection(nn.Module):
        def forward(self, input_tensor):
            return input_tensor.float()

    adapter = SyntacticGraphAdapter(
        hidden_size=8,
        graph_hidden_size=8,
        attention_heads=2,
        head_size=4,
        num_relations=1,
        dropout=0.0,
    )
    trace = NumericalTrace("fp16")
    fields = _small_graph_inputs()
    adapter.query_projection = FloatProjection()
    adapter.key_projection = FloatProjection()
    adapter.value_projection = FloatProjection()
    projected_nodes = torch.arange(16, dtype=torch.float16).reshape(1, 2, 8) / 100
    original_softmax = torch.softmax

    def fp16_softmax(input_tensor, dim):
        return original_softmax(input_tensor, dim=dim).to(torch.float16)

    with patch("syntactic_graph_adapter.torch.softmax", side_effect=fp16_softmax):
        adapter._graph_attention(
            projected_nodes,
            fields["edge_src"],
            fields["edge_dst"],
            fields["relation_id"],
            fields["dependency_relation_id"],
            fields["pos_pair_id"],
            fields["edge_mask"],
            fields["word_mask"],
            trace=trace,
        )

    stage = trace.finalize()["stages"]["attention_probabilities"]
    assert stage["dtype"] == "torch.float32"
    assert stage["finite_count"] == stage["total_count"]


def test_t5_trace_records_encoder_decoder_and_loss_stages():
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
    graph_model_config(config, 1)
    model = SyntacticGraphT5ForConditionalGeneration(config).eval()
    fields = _small_graph_inputs()
    hidden = fields.pop("hidden")
    del hidden
    graph_fields = {"graph_" + key: value for key, value in fields.items()}
    trace = NumericalTrace("fp32")
    output = model(
        input_ids=torch.tensor([[2, 3, 4, 5]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        labels=torch.tensor([[1, 2]]),
        graph_trace=trace,
        **graph_fields,
    )

    stages = trace.finalize()["stages"]
    assert output.loss is not None
    assert "t5_encoder_last_hidden_state" in stages
    assert "decoder_logits" in stages
    assert "final_loss" in stages


def test_trace_does_not_change_model_parameters():
    model = nn.Linear(4, 3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    trace = NumericalTrace("fp32")
    trace.record("node_projection", model(torch.ones(2, 4)), axes=("batch", "feature"))
    after = [parameter.detach().clone() for parameter in model.parameters()]

    assert all(torch.equal(left, right) for left, right in zip(before, after))


def test_target_pseudo_exception_is_structured_and_not_swallowed():
    def failing_call():
        raise ValueError("pseudo inference fixture failure")

    result = record_target_pseudo_result(failing_call)

    assert result["status"] == "ERROR"
    assert result["exception_type"] == "ValueError"
    assert result["message"] == "pseudo inference fixture failure"


def test_trace_report_is_machine_readable_and_distinguishes_modes():
    fp32 = NumericalTrace("fp32")
    fp32.record("final_loss", torch.tensor(1.0), axes=())
    fp16 = NumericalTrace("fp16")
    fp16.record("final_loss", torch.tensor(2.0), axes=())

    report = build_trace_report(
        fp32=fp32.finalize(),
        fp16=fp16.finalize(),
        model_hash_before="before",
        model_hash_after="after",
        target_test_access=False,
    )

    assert report["fp32_pass"] is True
    assert report["fp16_pass"] is True
    assert report["target_test_access"] is False
    assert report["model_parameter_hashes_match"] is False


def test_tensor_summary_handles_nonfinite_values_without_replacement():
    stats = summarize_tensor_stats(
        "overflow",
        torch.tensor([float("nan"), float("inf"), float("-inf"), 2.0]),
        axes=("token",),
    )

    assert stats["finite_count"] == 1
    assert stats["nan_count"] == 1
    assert stats["posinf_count"] == 1
    assert stats["neginf_count"] == 1
    assert stats["min"] == 2.0
    assert stats["max"] == 2.0
    assert stats["first_nonfinite"]["token"] == 0

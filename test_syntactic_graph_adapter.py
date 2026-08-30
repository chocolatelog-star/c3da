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


def _tiny_t5_config():
    return T5Config(
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


def _write_tiny_base_checkpoint(tmp_path):
    base_dir = tmp_path / "base-t5"
    T5ForConditionalGeneration(_tiny_t5_config()).save_pretrained(base_dir)
    return base_dir


def _load_graph_model_from_base(base_dir, seed=1000):
    config = T5Config.from_pretrained(base_dir, local_files_only=True)
    graph_model_config(config, 8)
    torch.manual_seed(seed)
    return SyntacticGraphT5ForConditionalGeneration.from_pretrained(
        base_dir,
        config=config,
        local_files_only=True,
    )


def _graph_parameter_state(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.syntactic_graph_adapter.named_parameters()
    }


def test_graph_adapter_explicit_reset_parameters_is_deterministic_and_zeroes_output():
    first = SyntacticGraphAdapter(8, 8, 2, 4, 8, 0.0)
    second = SyntacticGraphAdapter(8, 8, 2, 4, 8, 0.0)
    torch.manual_seed(1000)
    first.reset_parameters()
    torch.manual_seed(1000)
    second.reset_parameters()

    first_state = _graph_parameter_state(type("Model", (), {"syntactic_graph_adapter": first})())
    second_state = _graph_parameter_state(type("Model", (), {"syntactic_graph_adapter": second})())
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    assert all(torch.isfinite(parameter).all() for parameter in first.parameters())
    assert all(parameter.detach().abs().max() < 1.0e6 for parameter in first.parameters())
    assert torch.count_nonzero(first.output_projection.weight) == 0


def test_base_checkpoint_loading_initializes_graph_parameters_deterministically(tmp_path):
    base_dir = _write_tiny_base_checkpoint(tmp_path)
    first = _load_graph_model_from_base(base_dir, seed=1000)
    second = _load_graph_model_from_base(base_dir, seed=1000)
    first_state = _graph_parameter_state(first)
    second_state = _graph_parameter_state(second)

    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    assert all(torch.isfinite(parameter).all() for parameter in first_state.values())
    assert all(parameter.abs().max() < 1.0e6 for parameter in first_state.values())
    assert torch.count_nonzero(first.syntactic_graph_adapter.output_projection.weight) == 0
    assert first.graph_parameter_initialization["initialized_from_base_checkpoint"] is True
    assert first.graph_parameter_initialization["graph_checkpoint_detected"] is False


def test_base_checkpoint_low_cpu_memory_loading_is_not_misclassified_as_partial(tmp_path):
    base_dir = _write_tiny_base_checkpoint(tmp_path)
    config = T5Config.from_pretrained(base_dir, local_files_only=True)
    graph_model_config(
        config,
        8,
        element_aware_enabled=True,
        focus_enabled=True,
        coverage_enabled=True,
    )

    loaded = SyntacticGraphT5ForConditionalGeneration.from_pretrained(
        base_dir,
        config=config,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    assert loaded.graph_parameter_initialization["initialized_from_base_checkpoint"] is True
    assert loaded.graph_parameter_initialization["graph_checkpoint_detected"] is False
    assert loaded.graph_parameter_initialization["salience_initialization"] == "zero"
    assert all(not parameter.is_meta for parameter in loaded.parameters())
    loaded.to(torch.device("cpu"))


def test_base_checkpoint_loading_uses_seed_for_nonzero_graph_parameters(tmp_path):
    base_dir = _write_tiny_base_checkpoint(tmp_path)
    first = _load_graph_model_from_base(base_dir, seed=1000)
    second = _load_graph_model_from_base(base_dir, seed=1001)
    first_state = _graph_parameter_state(first)
    second_state = _graph_parameter_state(second)

    nonzero_names = [name for name, parameter in first_state.items() if torch.count_nonzero(parameter)]
    assert nonzero_names
    assert any(not torch.equal(first_state[name], second_state[name]) for name in nonzero_names)


def test_graph_checkpoint_round_trip_preserves_every_graph_parameter(tmp_path):
    config = _tiny_t5_config()
    graph_model_config(config, 8)
    source = SyntacticGraphT5ForConditionalGeneration(config).eval()
    with torch.no_grad():
        for index, parameter in enumerate(source.syntactic_graph_adapter.parameters(), start=1):
            parameter.fill_(index / 100.0)
    checkpoint_dir = tmp_path / "graph-checkpoint"
    source.save_pretrained(checkpoint_dir)

    loaded = SyntacticGraphT5ForConditionalGeneration.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )
    source_state = _graph_parameter_state(source)
    loaded_state = _graph_parameter_state(loaded)
    assert source_state.keys() == loaded_state.keys()
    assert all(torch.equal(source_state[name], loaded_state[name]) for name in source_state)
    assert loaded.graph_parameter_initialization["initialized_from_base_checkpoint"] is False
    assert loaded.graph_parameter_initialization["graph_checkpoint_detected"] is True


def test_old_graph_checkpoint_upgrades_to_zero_salience_deterministically(tmp_path):
    old_config = _tiny_t5_config()
    graph_model_config(old_config, 8)
    old_model = SyntacticGraphT5ForConditionalGeneration(old_config).eval()
    checkpoint_dir = tmp_path / "old-graph-checkpoint"
    old_model.save_pretrained(checkpoint_dir)

    new_config = T5Config.from_pretrained(checkpoint_dir, local_files_only=True)
    graph_model_config(new_config, 8, element_aware_enabled=True, focus_enabled=True, coverage_enabled=True)
    loaded = SyntacticGraphT5ForConditionalGeneration.from_pretrained(
        checkpoint_dir,
        config=new_config,
        local_files_only=True,
    )

    assert torch.count_nonzero(loaded.syntactic_graph_adapter.salience_head.weight) == 0
    assert torch.count_nonzero(loaded.syntactic_graph_adapter.salience_head.bias) == 0
    assert loaded.graph_parameter_initialization["salience_initialization"] == (
        "old_graph_checkpoint_missing_salience_zero_initialized"
    )


def test_new_element_aware_checkpoint_round_trip_preserves_salience(tmp_path):
    config = _tiny_t5_config()
    graph_model_config(config, 8, element_aware_enabled=True, focus_enabled=True, coverage_enabled=True)
    source = SyntacticGraphT5ForConditionalGeneration(config).eval()
    with torch.no_grad():
        source.syntactic_graph_adapter.salience_head.weight.fill_(0.125)
        source.syntactic_graph_adapter.salience_head.bias.fill_(-0.25)
    checkpoint_dir = tmp_path / "new-element-aware-checkpoint"
    source.save_pretrained(checkpoint_dir)

    loaded = SyntacticGraphT5ForConditionalGeneration.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )

    assert torch.equal(
        loaded.syntactic_graph_adapter.salience_head.weight,
        source.syntactic_graph_adapter.salience_head.weight,
    )
    assert torch.equal(
        loaded.syntactic_graph_adapter.salience_head.bias,
        source.syntactic_graph_adapter.salience_head.bias,
    )
    assert loaded.graph_parameter_initialization["salience_initialization"] == "checkpoint_loaded"


def test_partial_salience_checkpoint_is_rejected(tmp_path):
    config = _tiny_t5_config()
    graph_model_config(config, 8, element_aware_enabled=True, focus_enabled=True, coverage_enabled=True)
    source = SyntacticGraphT5ForConditionalGeneration(config).eval()
    checkpoint_dir = tmp_path / "partial-salience-checkpoint"
    source.save_pretrained(checkpoint_dir, safe_serialization=False)
    weights_path = checkpoint_dir / "pytorch_model.bin"
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    removed_name = "syntactic_graph_adapter.salience_head.bias"
    del state[removed_name]
    torch.save(state, weights_path)

    try:
        SyntacticGraphT5ForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
        )
    except RuntimeError as exc:
        assert "partial element salience head" in str(exc)
        assert removed_name in str(exc)
    else:
        raise AssertionError("partial salience checkpoints must be rejected")


def test_partial_graph_checkpoint_is_rejected_without_resetting_graph_parameters(tmp_path):
    config = _tiny_t5_config()
    graph_model_config(config, 8)
    source = SyntacticGraphT5ForConditionalGeneration(config).eval()
    checkpoint_dir = tmp_path / "partial-graph-checkpoint"
    source.save_pretrained(checkpoint_dir, safe_serialization=False)

    weights_path = checkpoint_dir / "pytorch_model.bin"
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    removed_name = "syntactic_graph_adapter.node_projection.weight"
    assert removed_name in state
    del state[removed_name]
    torch.save(state, weights_path)

    try:
        SyntacticGraphT5ForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
        )
    except RuntimeError as exc:
        assert "partial syntactic graph adapter" in str(exc)
        assert removed_name in str(exc)
    else:
        raise AssertionError("partial graph checkpoints must be rejected")


def test_base_checkpoint_zero_update_graph_path_is_finite_and_control_equivalent(tmp_path):
    base_dir = _write_tiny_base_checkpoint(tmp_path)
    control = T5ForConditionalGeneration.from_pretrained(base_dir, local_files_only=True).eval()
    treatment = _load_graph_model_from_base(base_dir, seed=1000).eval()
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
    assert torch.isfinite(treatment_output.logits).all()
    assert torch.isfinite(treatment_output.loss)
    assert torch.equal(control_output.logits, treatment_output.logits)
    assert torch.equal(control_output.loss, treatment_output.loss)


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

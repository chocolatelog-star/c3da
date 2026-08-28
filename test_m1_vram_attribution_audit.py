"""CPU regression tests for the M1 VRAM attribution diagnostic."""

import gc
import inspect

import torch

import m1_vram_attribution_audit as audit
from m1_vram_attribution_audit import (
    MemoryRecorder,
    MemoryTrace,
    SyntheticMemoryBackend,
    TRACE_TO_CALLPOINT,
    analyze_memory_attribution,
    audit_python_container_tensors,
    build_v2_report_gates,
    count_autograd_nodes,
    materialize_adamw_state_for_audit,
    record_lifecycle_event,
    run_zero_update_optimizer_step,
)


def test_memory_recorder_records_tensor_bytes_and_batch_counts():
    backend = SyntheticMemoryBackend(
        allocated=[1024],
        reserved=[2048],
        peak_allocated=[1536],
        peak_reserved=[3072],
        stats=[{"active_bytes.all.current": 1024, "inactive_split_bytes.all.current": 1024}],
    )
    recorder = MemoryRecorder(
        backend,
        device=torch.device("cpu"),
        batch_counts={"tokens": 4, "nodes": 2, "edges": 3},
    )
    tensor = torch.zeros((2, 3), dtype=torch.float16)
    event = recorder.mark("node_edge_projection", tensor=tensor)
    assert event["memory"]["allocated_bytes"] == 1024
    assert event["memory"]["inactive_split_bytes"] == 1024
    assert event["tensor"]["shape"] == [2, 3]
    assert event["tensor"]["dtype"] == "torch.float16"
    assert event["tensor"]["theoretical_bytes"] == 12
    assert event["batch_counts"] == {"tokens": 4, "nodes": 2, "edges": 3}


def test_memory_attribution_finds_first_significant_treatment_growth():
    control = [
        {"step": 0, "callpoint": "batch_to_gpu", "memory": {"allocated_bytes": 100}},
        {"step": 0, "callpoint": "node_edge_projection", "memory": {"allocated_bytes": 120}},
        {"step": 0, "callpoint": "attention_logits", "memory": {"allocated_bytes": 150}},
    ]
    treatment = [
        {"step": 0, "callpoint": "batch_to_gpu", "memory": {"allocated_bytes": 100}},
        {"step": 0, "callpoint": "node_edge_projection", "memory": {"allocated_bytes": 140}},
        {"step": 0, "callpoint": "attention_logits", "memory": {"allocated_bytes": 220}},
    ]
    result = analyze_memory_attribution(
        control,
        treatment,
        significant_growth_bytes=50,
        total_memory_bytes=1000,
    )
    assert result["first_significant_growth_callpoint"] == "attention_logits"
    assert result["graph_module_increment_peak_allocated_bytes"] == 70
    assert result["leak_suspected"] is False


def test_memory_attribution_marks_repeated_growth_as_leak_suspected():
    control = [
        {"step": step, "callpoint": "backward", "memory": {"allocated_bytes": 100}}
        for step in range(3)
    ]
    treatment = [
        {"step": 0, "callpoint": "backward", "memory": {"allocated_bytes": 100}},
        {"step": 1, "callpoint": "backward", "memory": {"allocated_bytes": 200}},
        {"step": 2, "callpoint": "backward", "memory": {"allocated_bytes": 300}},
    ]
    result = analyze_memory_attribution(
        control,
        treatment,
        significant_growth_bytes=50,
        total_memory_bytes=1000,
    )
    assert result["leak_suspected"] is True
    assert result["classification"] == "retained_growth_suspected"


def test_python_container_audit_and_autograd_count_are_explicit():
    tensor = torch.ones(2, requires_grad=True)
    loss = (tensor * tensor).sum()
    container_report = audit_python_container_tensors({"cpu_tensor": tensor})
    assert container_report["gpu_tensor_count"] == 0
    assert container_report["tensor_count"] == 1
    assert count_autograd_nodes(loss) >= 1
    del loss, tensor
    gc.collect()


def test_lightweight_trace_maps_graph_stages_without_retaining_tensors():
    backend = SyntheticMemoryBackend(
        allocated=[10],
        reserved=[20],
        peak_allocated=[10],
        peak_reserved=[20],
    )
    recorder = MemoryRecorder(backend, device=torch.device("cpu"), batch_counts={"tokens": 1, "nodes": 1, "edges": 1})
    trace = MemoryTrace(recorder, step=0)
    tensor = torch.zeros((1, 2), dtype=torch.float16)
    trace.record("query_key_product", tensor, axes=("batch", "edge"))
    assert recorder.events[0]["callpoint"] == "attention_logits"
    assert recorder.events[0]["trace_stage"] == "query_key_product"
    assert "query_key_product" in TRACE_TO_CALLPOINT
    del tensor, trace
    gc.collect()


def test_memory_attribution_marks_allocator_fragmentation_separately():
    events = [
        {
            "step": 0,
            "callpoint": "graph_fusion",
            "memory": {
                "allocated_bytes": 700,
                "reserved_bytes": 1000,
                "peak_allocated_bytes": 700,
                "peak_reserved_bytes": 1000,
                "inactive_split_bytes": 300,
            },
        }
    ]
    result = analyze_memory_attribution(events, events, total_memory_bytes=10000)
    assert result["fragmentation_suspected"] is True
    assert result["classification"] == "allocator_fragmentation_suspected"


def test_memory_attribution_reports_fp32_promotion_candidates_and_expected_stages():
    events = [
        {
            "step": 0,
            "callpoint": "node_edge_projection",
            "trace_stage": "node_projection",
            "tensor": {"dtype": "torch.float16"},
            "memory": {"allocated_bytes": 1},
        },
        {
            "step": 0,
            "callpoint": "attention_logits",
            "trace_stage": "dependency_bias",
            "tensor": {"dtype": "torch.float32"},
            "memory": {"allocated_bytes": 2},
        },
        {
            "step": 0,
            "callpoint": "attention_logits",
            "trace_stage": "softmax_input_float32_logits",
            "tensor": {"dtype": "torch.float32"},
            "memory": {"allocated_bytes": 3},
        },
    ]
    result = analyze_memory_attribution(events, events, significant_growth_bytes=100)
    promotion = result["treatment_fp32_promotion"]
    assert promotion["implicit_promotion_suspected"] is True
    assert promotion["implicit_promotion_candidates"][0]["trace_stage"] == "dependency_bias"
    assert promotion["expected_fp32_trace_observations"][0]["trace_stage"] == "softmax_input_float32_logits"


def test_memory_attribution_reports_gradient_graph_and_optimizer_overlap():
    events = [
        {
            "step": 0,
            "callpoint": "backward",
            "memory": {"allocated_bytes": 10},
            "runtime": {
                "parameter_bytes": 100,
                "gradient_bytes": 20,
                "optimizer_state_bytes": 30,
                "autograd_graph_nodes": 4,
            },
        }
    ]
    result = analyze_memory_attribution(events, events, significant_growth_bytes=100)
    overlap = result["treatment_runtime_overlap"]
    assert overlap["optimizer_state_observable"] is True
    assert overlap["gradient_and_autograd_overlap_event_count"] == 1
    assert overlap["parameter_gradient_optimizer_autograd_overlap_event_count"] == 1


def test_zero_update_optimizer_step_does_not_change_parameters():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = run_zero_update_optimizer_step(model, optimizer)
    assert result["optimizer_updates"] == 0
    assert result["parameter_updates"] == 0
    for parameter, expected in zip(model.parameters(), before):
        assert torch.equal(parameter, expected)


def test_control_model_call_does_not_receive_graph_trace_keyword():
    source = inspect.getsource(audit._run_variant_steps)
    assert 'model_kwargs["graph_trace"] = graph_trace' in source
    assert "if graph_enabled:" in source
    assert "graph_trace=graph_trace" not in source


def test_batch_counts_use_real_project_collator_graph_fields():
    from t5_absa_train import DataCollatorForSeq2SeqWithPairing

    class BaseCollator:
        def __call__(self, features):
            return {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.ones((2, 2), dtype=torch.long),
                "labels": torch.tensor([[1, 2], [-100, -100]]),
                "sample_weight": torch.tensor([1.0, 0.0]),
                "domain_weight": torch.tensor([1.0, 0.0]),
                "domain_label": torch.tensor([0, 1]),
            }

    def feature(offset):
        return {
            "input_ids": [offset + 1, offset + 2],
            "labels": [1, 2] if offset == 0 else [-100, -100],
            "sample_weight": 1.0 if offset == 0 else 0.0,
            "domain_weight": 1.0 if offset == 0 else 0.0,
            "domain_label": offset,
            "word_to_subword": [[0]] * 27,
            "word_mask": [True] * 27,
            "edge_src": [0] * 131,
            "edge_dst": [0] * 131,
            "relation_id": [0] * 131,
            "dependency_relation_id": [0] * 131,
            "pos_pair_id": [0] * 131,
            "edge_mask": [True] * 131,
        }

    batch = DataCollatorForSeq2SeqWithPairing(BaseCollator())([feature(0), feature(1)])
    counts = audit._batch_counts(batch)
    assert tuple(batch["graph_word_mask"].shape) == (2, 27)
    assert tuple(batch["graph_edge_mask"].shape) == (2, 131)
    assert counts["nodes"] == 27
    assert counts["edges"] == 131


def test_v2_report_gate_blocks_zero_graph_counts_and_requires_trace_match():
    blocked = build_v2_report_gates(
        control_result={"graph_stage_count": 0, "parameter_hashes_match": True, "optimizer_updates": 0, "scheduler_steps": 0, "parameter_updates": 0},
        treatment_result={"graph_stage_count": 0, "parameter_hashes_match": True, "optimizer_updates": 0, "scheduler_steps": 0, "parameter_updates": 0, "graph_counts_match_trace": True},
        batch_counts={"nodes": 0, "edges": 0},
        target_test_access=False,
    )
    assert blocked["status"] == "BLOCKED"
    assert blocked["gates"]["treatment_graph_counts_positive"] is False

    passed = build_v2_report_gates(
        control_result={"graph_stage_count": 0, "parameter_hashes_match": True, "optimizer_updates": 0, "scheduler_steps": 0, "parameter_updates": 0},
        treatment_result={"graph_stage_count": 1, "parameter_hashes_match": True, "optimizer_updates": 0, "scheduler_steps": 0, "parameter_updates": 0, "graph_counts_match_trace": True},
        batch_counts={"nodes": 27, "edges": 131},
        target_test_access=False,
    )
    assert passed["status"] == "PASS"
    assert all(passed["gates"].values())


def test_materialize_adamw_state_for_audit_does_not_change_formal_model():
    model = torch.nn.Linear(3, 2)
    before = audit._parameter_state_sha256(model)
    result = materialize_adamw_state_for_audit(
        model,
        device=torch.device("cpu"),
        learning_rate=0.001,
    )
    assert result["optimizer_state_bytes"] > 0
    assert result["formal_model_hash_before"] == before
    assert result["formal_model_hash_after"] == before
    assert result["formal_model_unchanged"] is True


def test_lifecycle_event_records_live_cuda_and_control_reachability_fields():
    backend = SyntheticMemoryBackend(
        allocated=[10], reserved=[20], peak_allocated=[10], peak_reserved=[20]
    )
    recorder = MemoryRecorder(backend, device=torch.device("cpu"))
    model = torch.nn.Linear(2, 2)
    event = record_lifecycle_event(
        recorder,
        "control_model_loaded",
        model=model,
        trainer=None,
        optimizer=None,
    )
    assert isinstance(event["lifecycle"]["live_cuda_tensor_count"], int)
    assert event["lifecycle"]["live_cuda_tensor_count"] >= 0
    assert event["lifecycle"]["control_model_reachable"] is True
    assert event["lifecycle"]["control_trainer_reachable"] is False
    assert "live_cuda_tensor_bytes" in event["lifecycle"]


def test_trace_shape_counts_match_fixed_real_graph_batch():
    events = [
        {
            "trace_stage": "pooled_word_hidden",
            "tensor": {"shape": [2, 27, 768]},
        },
        {
            "trace_stage": "edge_query",
            "tensor": {"shape": [2, 131, 4, 64]},
        },
    ]
    counts = audit._trace_shape_counts(events)
    assert counts["observed"] is True
    assert counts["consistent"] is True
    assert counts["nodes"] == 27
    assert counts["edges"] == 131


def test_v2_report_gate_blocks_trace_shape_mismatch_even_with_positive_batch_counts():
    result = build_v2_report_gates(
        control_result={"graph_stage_count": 0, "parameter_hashes_all_match": True, "optimizer_updates": 0, "scheduler_steps": 0, "parameter_updates": 0},
        treatment_result={"graph_stage_count": 1, "parameter_hashes_all_match": True, "optimizer_updates": 0, "scheduler_steps": 0, "parameter_updates": 0, "graph_counts_match_trace": False},
        batch_counts={"nodes": 27, "edges": 131},
        target_test_access=False,
    )
    assert result["status"] == "BLOCKED"
    assert result["gates"]["batch_counts_match_trace"] is False


def test_lifecycle_pre_registered_rule_classifies_control_cleanup_drop():
    events = [
        {
            "callpoint": "control_diagnostic_end_before_locals_exit",
            "memory": {"allocated_bytes": 2 * 1024**3, "reserved_bytes": 2 * 1024**3},
        },
        {
            "callpoint": "cuda_empty_cache_after",
            "memory": {"allocated_bytes": 0, "reserved_bytes": 0},
            "lifecycle": {
                "control_model_reachable": False,
                "control_optimizer_reachable": False,
                "control_trainer_reachable": False,
            },
        },
    ]
    result = audit.classify_lifecycle_attribution(
        events,
        isolated_treatment_steady_allocated_bytes=100,
        total_memory_bytes=8 * 1024**3,
    )
    assert result["classification"] == "CONTROL_LIFECYCLE_RETENTION_IDENTIFIED"
    assert result["control_cleanup_allocated_drop_bytes"] > 1024**3


def test_lifecycle_report_does_not_call_dead_weakref_a_cuda_reference():
    events = [
        {
            "callpoint": "control_diagnostic_end_before_locals_exit",
            "memory": {"allocated_bytes": 2 * 1024**3, "reserved_bytes": 2 * 1024**3},
        },
        {
            "callpoint": "cuda_empty_cache_after",
            "memory": {
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "live_cuda_tensor_count": 0,
                "live_cuda_tensor_bytes": 0,
            },
            "lifecycle": {
                # This is the old weak-reference reporting shape.  A weakref
                # that is dead and no live CUDA tensor is not a retained object.
                "control_model_reachable": True,
                "control_optimizer_reachable": True,
                "control_trainer_reachable": True,
                "live_cuda_tensor_count": 0,
                "live_cuda_tensor_bytes": 0,
            },
        },
    ]
    result = audit.classify_lifecycle_attribution(
        events,
        isolated_treatment_steady_allocated_bytes=100,
        total_memory_bytes=8 * 1024**3,
    )
    assert result["control_cuda_references_remain"] is False
    assert result["classification"] == "CONTROL_LIFECYCLE_RETENTION_IDENTIFIED"

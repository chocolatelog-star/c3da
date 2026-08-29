from __future__ import annotations

import hashlib
import json
import tempfile
import copy
import io
import gc
import os
import sys
import weakref
from types import SimpleNamespace
from pathlib import Path

import torch
import pytest
from torch.utils.data import Dataset
from transformers import TrainerCallback, T5Config, T5ForConditionalGeneration, Seq2SeqTrainingArguments

from m1_phase_a_control_terminal_lookahead_salvage import audit_v6_control
from m1_syntactic_rgat_pseudo_quick_ablation import (
    PHASE_A_STOP_CODE,
    PHASE_B_REQUEST_CODE,
    audit_control_identity,
    build_phase_a_scope,
    build_variant_config,
    build_phase_a_pseudo_output_paths,
    decide_phase_a,
    evaluate_phase_a_gates,
    load_or_initialize_stage_status,
    build_stage_identity,
    validate_stage_identity,
    validate_stage_status_shape,
    validate_external_control_dann_audit,
    _validate_control_treatment_dann_reports,
    _read_dann_batch_audit,
    _model_hashes,
    _training_argv,
    validate_initialization_pair,
    _validate_recipe,
    validate_input_split,
    validate_phase_a_graph_cache,
    prepare_legacy_diagnostic_resume,
    evaluate_control_lifecycle_gate,
    append_control_return_lifecycle_event,
    classify_terminal_lookahead,
    run_isolated_phase_a_worker,
    _serialize_rows,
    _write_inputs,
    _write_variant_inputs,
    _pipeline_argv,
    stage_producer_commit_for_validation,
)
from syntactic_graph_adapter import load_seq2seq_model
from t5_absa_train import (
    PairedDomainBatchSampler,
    WeightedSeq2SeqTrainer,
    build_initialization_audit,
    compute_dann_expected_max_steps,
    find_latest_complete_dann_checkpoint,
    initialize_domain_adversarial_head,
    recover_dann_audit_journal,
    cleanup_phase_a_training_runtime,
    finalize_phase_a_training_runtime,
    phase_a_rng_state_hashes,
)


def test_external_control_terminal_lookahead_normalization_is_bounded():
    control = {
        "epochs": [{"completion": "partial", "planned_batches": 3, "issued_batches": 3, "processed_batches": 2, "logical_batches": 3, "source_rows": 3, "target_rows": 3, "source_unique_rows": 3, "target_unique_rows": 3, "batches": [{"logical_batch_id": i} for i in range(3)]}],
        "terminal_lookahead_audit": {"safe": True, "lookahead_not_consumed": True, "dangling_logical_batch_ids": [2]},
    }
    treatment = copy.deepcopy(control)
    treatment["epochs"][0].update({"issued_batches": 2, "logical_batches": 2, "source_rows": 2, "target_rows": 2, "source_unique_rows": 2, "target_unique_rows": 2, "batches": treatment["epochs"][0]["batches"][:2]})
    normalized, audit = _normalize_external_control_terminal_lookahead(control, treatment)
    assert normalized["epochs"][0]["issued_batches"] == 2
    assert len(normalized["epochs"][0]["batches"]) == 2
    assert audit["trimmed_logical_batch_ids"] == [2]

def test_control_lifecycle_red_reproduces_reachable_runtime_before_treatment():
    """RED guard: a reachable Control runtime must never authorize Treatment."""
    runtime = {"model": object(), "optimizer": object(), "trainer": object()}
    audit = {
        "events": [
            {
                "callpoint": "control_return_after_return",
                "memory": {
                    "allocated_bytes": 300 * 1024 * 1024,
                    "reserved_bytes": 300 * 1024 * 1024,
                    "live_cuda_tensor_count": 1,
                    "live_cuda_tensor_bytes": 1024,
                },
                "references": {
                    "control_model_reachable": True,
                    "control_optimizer_reachable": True,
                    "control_trainer_reachable": True,
                },
            },
            {
                "callpoint": "control_cuda_empty_cache_after",
                "memory": {
                    "allocated_bytes": 300 * 1024 * 1024,
                    "reserved_bytes": 300 * 1024 * 1024,
                    "live_cuda_tensor_count": 1,
                    "live_cuda_tensor_bytes": 1024,
                },
                "references": {
                    "control_model_reachable": True,
                    "control_optimizer_reachable": True,
                    "control_trainer_reachable": True,
                },
            },
        ],
        "runtime_references": runtime,
    }
    gate = evaluate_control_lifecycle_gate(
        audit,
        baseline={"allocated_bytes": 0, "reserved_bytes": 0},
    )
    assert gate["passed"] is False
    assert gate["treatment_allowed"] is False
    assert gate["next_action"] == "CONTROL_TREATMENT_SUBPROCESS_ISOLATION_REQUIRED"


def test_phase_a_cleanup_releases_runtime_and_preserves_rng():
    class Runtime:
        pass

    model = Runtime()
    optimizer = Runtime()
    trainer = Runtime()
    model_ref = weakref.ref(model)
    optimizer_ref = weakref.ref(optimizer)
    trainer_ref = weakref.ref(trainer)
    runtime = {"model": model, "optimizer": optimizer, "trainer": trainer}
    before_rng = phase_a_rng_state_hashes(include_cuda=False)
    audit = cleanup_phase_a_training_runtime(runtime, cuda=False)
    del model, optimizer, trainer
    gc.collect()
    assert runtime == {}
    assert model_ref() is None
    assert optimizer_ref() is None
    assert trainer_ref() is None
    assert audit["cleanup_performed"] is True
    assert audit["rng_state_before_cleanup"] == before_rng
    assert audit["rng_state_after_cleanup"] == before_rng


def test_control_return_event_is_added_without_retaining_runtime_objects():
    audit = {"events": []}
    event = append_control_return_lifecycle_event(
        audit,
        baseline={"allocated_bytes": 10, "reserved_bytes": 20},
    )
    assert event["callpoint"] == "control_return_after_return"
    assert audit["events"][-1] == event
    assert "runtime_object" not in json.dumps(audit)


def test_control_lifecycle_gate_allows_treatment_only_after_clean_release():
    audit = {
        "cleanup_performed": True,
        "rng_state_unchanged": True,
        "references_after_cleanup": {
            "model": False,
            "optimizer": False,
            "trainer": False,
        },
        "events": [
            {
                "callpoint": "control_cuda_empty_cache_after",
                "memory": {
                    "allocated_bytes": 64 * 1024 * 1024,
                    "reserved_bytes": 96 * 1024 * 1024,
                    "live_cuda_tensor_count": 0,
                    "live_cuda_tensor_bytes": 0,
                },
                "references": {
                    "model": False,
                    "optimizer": False,
                    "trainer": False,
                },
            },
            {
                "callpoint": "phase_a_return_after_local_release",
                "memory": {
                    "allocated_bytes": 64 * 1024 * 1024,
                    "reserved_bytes": 96 * 1024 * 1024,
                    "live_cuda_tensor_count": 0,
                    "live_cuda_tensor_bytes": 0,
                },
                "references": {
                    "weakref_alive": {
                        "model": False,
                        "optimizer": False,
                        "trainer": False,
                    },
                },
            },
        ],
    }
    gate = evaluate_control_lifecycle_gate(
        audit,
        baseline={"allocated_bytes": 0, "reserved_bytes": 0},
    )
    assert gate["passed"] is True
    assert gate["treatment_allowed"] is True


def test_phase_a_cleanup_preserves_model_and_audit_artifact_bytes(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model_before = [parameter.detach().clone() for parameter in model.parameters()]
    audit_path = tmp_path / "dann_batch_audit.json"
    audit_path.write_text(json.dumps({"seed": 1000, "batches": [[0, 1]]}), encoding="utf-8")
    audit_bytes = audit_path.read_bytes()
    runtime = {"model": model, "optimizer": optimizer}
    lifecycle = cleanup_phase_a_training_runtime(runtime, cuda=False)
    for parameter, expected in zip(model.parameters(), model_before):
        assert torch.equal(parameter, expected)
    assert audit_path.read_bytes() == audit_bytes
    assert lifecycle["rng_state_unchanged"] is True


def test_phase_a_lifecycle_final_event_uses_outer_weakrefs_after_local_release():
    class Runtime:
        pass

    trainer = Runtime()
    model = Runtime()
    optimizer = Runtime()
    dataloader = Runtime()
    callback = Runtime()
    runtime = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "dataloader": dataloader,
        "callbacks": [callback],
    }
    lifecycle = cleanup_phase_a_training_runtime(
        runtime,
        cuda=False,
        variant="control",
        defer_finalization=True,
    )
    early = next(event for event in lifecycle["events"] if event["callpoint"] == "control_gc_after")
    assert early["references"]["model"] is True
    assert early["references"]["trainer"] is True
    trainer = model = optimizer = dataloader = callback = None
    gc.collect()
    lifecycle = finalize_phase_a_training_runtime(lifecycle, include_cuda=False)
    final = next(event for event in lifecycle["events"] if event["callpoint"] == "phase_a_return_after_local_release")
    assert final["references"]["weakref_alive"]["model"] is False
    assert final["references"]["weakref_alive"]["trainer"] is False
    assert final["memory"]["live_cuda_tensor_count"] == 0
    gate = evaluate_control_lifecycle_gate(
        {**lifecycle, "runner_return_recorded": True},
        baseline={"allocated_bytes": 0, "reserved_bytes": 0},
    )
    assert gate["passed"] is True


def test_phase_a_lifecycle_gate_rejects_final_event_with_retained_object():
    class Runtime:
        pass

    held_model = Runtime()
    runtime = {"model": held_model}
    lifecycle = cleanup_phase_a_training_runtime(
        runtime,
        cuda=False,
        variant="control",
        defer_finalization=True,
    )
    lifecycle = finalize_phase_a_training_runtime(lifecycle, include_cuda=False)
    final = next(event for event in lifecycle["events"] if event["callpoint"] == "phase_a_return_after_local_release")
    assert final["references"]["weakref_alive"]["model"] is True
    gate = evaluate_control_lifecycle_gate(
        {**lifecycle, "runner_return_recorded": True},
        baseline={"allocated_bytes": 0, "reserved_bytes": 0},
    )
    assert gate["passed"] is False
    assert "control_runtime_reference_remains_after_cleanup" in gate["reasons"]


def test_phase_a_lifecycle_gate_rejects_hardcoded_empty_final_references():
    lifecycle = {
        "cleanup_performed": True,
        "rng_state_unchanged": True,
        "events": [
            {
                "callpoint": "control_cuda_empty_cache_after",
                "memory": {
                    "allocated_bytes": 0,
                    "reserved_bytes": 0,
                    "live_cuda_tensor_count": 0,
                    "live_cuda_tensor_bytes": 0,
                },
                "references": {"model": True},
            },
            {
                "callpoint": "phase_a_return_after_local_release",
                "memory": {
                    "allocated_bytes": 0,
                    "reserved_bytes": 0,
                    "live_cuda_tensor_count": 0,
                    "live_cuda_tensor_bytes": 0,
                },
                "references": {},
            },
        ],
    }
    gate = evaluate_control_lifecycle_gate(
        lifecycle,
        baseline={"allocated_bytes": 0, "reserved_bytes": 0},
    )
    assert gate["passed"] is False
    assert "missing_final_weakref_evidence" in gate["reasons"]


def test_phase_a_cleanup_clears_trainer_and_accelerator_runtime_holders():
    class Runtime:
        pass

    trainer = Runtime()
    accelerator = Runtime()
    model = Runtime()
    optimizer = Runtime()
    scheduler = Runtime()
    scaler = Runtime()
    dataloader = Runtime()
    batch = Runtime()
    callback = Runtime()
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.scaler = scaler
    trainer._train_dataloader = dataloader
    trainer._past = [batch]
    trainer._models = [model]
    trainer._optimizers = [optimizer]
    trainer._dataloaders = [dataloader]
    trainer.callback_handler = SimpleNamespace(callbacks=[callback])
    trainer.accelerator = accelerator
    accelerator._models = [model]
    accelerator._optimizers = [optimizer]
    accelerator._dataloaders = [dataloader]
    accelerator._schedulers = [scheduler]
    accelerator._scalers = [scaler]
    accelerator.optimizer = optimizer
    accelerator.lr_scheduler = scheduler
    accelerator.scaler = scaler

    runtime = {"trainer": trainer, "model": model, "accelerator": accelerator}
    cleanup_phase_a_training_runtime(runtime, cuda=False)

    assert trainer.model is None
    assert trainer.optimizer is None
    assert trainer.lr_scheduler is None
    assert trainer.scaler is None
    assert trainer._train_dataloader is None
    assert trainer._past is None
    assert trainer._models is None
    assert trainer._optimizers is None
    assert trainer._dataloaders is None
    assert trainer.callback_handler is None
    assert trainer.accelerator is None
    assert accelerator._models is None
    assert accelerator._optimizers is None
    assert accelerator._dataloaders is None
    assert accelerator._schedulers is None
    assert accelerator._scalers is None
    assert accelerator.optimizer is None
    assert accelerator.lr_scheduler is None
    assert accelerator.scaler is None


def _metrics():
    return {
        "source_dev": {
            "control": {
                "strict_triplet_f1": 0.50,
                "multi_triplet_sentence_recall": 0.40,
                "absence_rates": {"overall": 0.60, "aspect": 0.40, "opinion": 0.30},
            },
            "treatment": {
                "strict_triplet_f1": 0.495,
                "multi_triplet_sentence_recall": 0.425,
                "absence_rates": {"overall": 0.55, "aspect": 0.35, "opinion": 0.30},
            },
        },
        "target_unlabeled_pseudo": {
            "control": {"qualified_total_rows": 100, "qualified_multi_rows": 20},
            "treatment": {"qualified_total_rows": 96, "qualified_multi_rows": 21},
        },
    }


def test_phase_a_four_gates_pass_and_request_phase_b_only():
    result = evaluate_phase_a_gates(_metrics())
    assert all(item["status"] == "PASS" for item in result["gates"].values())
    decision = decide_phase_a(result)
    assert decision["status"] == "PASS"
    assert decision["next_action"] == PHASE_B_REQUEST_CODE
    assert decision["hard_stop"] is False


def test_a4_zero_control_multi_denominator_is_undefined_and_blocked():
    metrics = _metrics()
    metrics["target_unlabeled_pseudo"]["control"]["qualified_multi_rows"] = 0
    result = evaluate_phase_a_gates(metrics)
    gate = result["gates"]["A4"]
    assert gate["status"] == "FAIL"
    assert gate["actual"]["multi_ratio"] is None
    assert gate["matches"]["multi_ratio"] is False
    assert decide_phase_a(result)["status"] == "BLOCKED"


def test_a4_integer_boundary_keeps_nonzero_ratio_thresholds():
    metrics = _metrics()
    metrics["target_unlabeled_pseudo"]["control"] = {
        "qualified_total_rows": 20,
        "qualified_multi_rows": 20,
    }
    metrics["target_unlabeled_pseudo"]["treatment"] = {
        "qualified_total_rows": 18,
        "qualified_multi_rows": 21,
    }
    assert evaluate_phase_a_gates(metrics)["gates"]["A4"]["status"] == "FAIL"
    metrics["target_unlabeled_pseudo"]["treatment"]["qualified_total_rows"] = 20
    assert evaluate_phase_a_gates(metrics)["gates"]["A4"]["status"] == "PASS"


def test_each_phase_a_gate_failure_hard_stops_upstream():
    for field, value in (
        (("source_dev", "treatment", "strict_triplet_f1"), 0.48),
        (("source_dev", "treatment", "multi_triplet_sentence_recall"), 0.40),
        (("source_dev", "treatment", "absence_rates"), {"overall": 0.61, "aspect": 0.41, "opinion": 0.31}),
        (("target_unlabeled_pseudo", "treatment", "qualified_multi_rows"), 20),
    ):
        metrics = _metrics()
        if field[-1] == "absence_rates":
            metrics[field[0]][field[1]][field[2]] = value
        else:
            metrics[field[0]][field[1]][field[2]] = value
        if field[-1] == "qualified_multi_rows":
            metrics["target_unlabeled_pseudo"]["treatment"]["qualified_total_rows"] = 100
        result = evaluate_phase_a_gates(metrics)
        assert any(item["status"] == "FAIL" for item in result["gates"].values())
        decision = decide_phase_a(result)
        assert decision["status"] == "BLOCKED"
        assert decision["next_action"] == PHASE_A_STOP_CODE
        assert decision["hard_stop"] is True


def test_graph_scope_and_control_treatment_only_differ_by_graph_enabled():
    scope = build_phase_a_scope()
    assert scope["control"]["graph_enabled"] is False
    assert all(scope["treatment"][name]["graph_enabled"] for name in (
        "source_extractor_training",
        "source_dev_evaluation",
        "target_unlabeled_dann",
        "target_pseudo_inference",
    ))
    assert all(not value for value in scope["forbidden"].values())
    assert scope["target_test_access"] is False
    assert set(scope["formal_callpoint_paths"]) == {
        "source_extractor_training",
        "source_dev_evaluation",
        "target_unlabeled_dann",
        "target_pseudo_inference",
    }
    control = build_variant_config(False)
    treatment = build_variant_config(True)
    assert {key for key in control if control[key] != treatment[key]} == {"graph_enabled"}


def test_control_identity_requires_every_field_and_unknown_forces_rerun():
    expected = {
        "direction": "laptop14 -> rest15",
        "seed": 1000,
        "data_split": "source_train+source_dev+target_unlabeled",
        "recipe_sha256": "recipe",
        "checkpoint_selection_rule": "last",
        "tokenizer_sha256": "tokenizer",
        "model_sha256": "model",
        "code_semantics": "158654021fc5f26bf1cfb8e803d7d1b592bd8534",
        "artifact_sha256": "artifact",
    }
    matching = audit_control_identity(expected, dict(expected))
    assert matching["reuse_allowed"] is True
    changed = dict(expected)
    changed["recipe_sha256"] = "changed"
    assert audit_control_identity(expected, changed)["reuse_allowed"] is False
    unknown = dict(expected)
    unknown["model_sha256"] = None
    assert audit_control_identity(expected, unknown)["reuse_allowed"] is False


def test_resume_identity_and_phase_b_boundary():
    identity = {"code": "abc", "recipe": "def"}
    with tempfile.TemporaryDirectory() as directory:
        status_path = Path(directory) / "stage_status.json"
        state = load_or_initialize_stage_status(status_path, identity, resume=False)
        assert state["completed_stages"] == []
        status_path.write_text(
            json.dumps({"schema_version": 2, "identity": identity, "completed_stages": [], "stages": {}}),
            encoding="utf-8",
        )
        resumed = load_or_initialize_stage_status(status_path, identity, resume=True)
        assert resumed["completed_stages"] == []
        assert load_or_initialize_stage_status(status_path, {"code": "changed", "recipe": "def"}, resume=True) is None


def test_resume_repair_migrates_only_code_identity_and_preserves_completed_producers():
    old_identity = {"task_id": "phase-a", "code_commit": "old", "recipe_sha256": "recipe"}
    new_identity = {"task_id": "phase-a", "code_commit": "new", "recipe_sha256": "recipe"}
    with tempfile.TemporaryDirectory() as directory:
        status_path = Path(directory) / "stage_status.json"
        status_path.write_text(
            json.dumps({
                "schema_version": 2,
                "identity": old_identity,
                "completed_stages": ["treatment_training"],
                "stages": {"treatment_training": {"producer_commit": "old"}},
            }),
            encoding="utf-8",
        )
        resumed = load_or_initialize_stage_status(
            status_path,
            new_identity,
            resume=True,
            repair_from_commit="old",
        )
        assert resumed["identity"] == new_identity
        assert resumed["repair_history"][-1]["from_commit"] == "old"
        assert resumed["repair_history"][-1]["to_commit"] == "new"
        assert resumed["repair_history"][-1]["completed_stages"] == ["treatment_training"]
        assert resumed["stages"]["treatment_training"]["producer_commit"] == "old"
        assert stage_producer_commit_for_validation(resumed, "treatment_training", "new") == "old"


def test_resume_repair_rejects_stage_producer_outside_audited_chain():
    state = {
        "stages": {"treatment_training": {"producer_commit": "unknown"}},
        "repair_history": [{
            "from_commit": "old",
            "to_commit": "new",
            "completed_stages": ["treatment_training"],
        }],
    }
    try:
        stage_producer_commit_for_validation(state, "treatment_training", "new")
    except RuntimeError as exc:
        assert "outside the audited repair chain" in str(exc)
    else:
        raise AssertionError("unaudited stage producer was accepted")


def test_graph_pipeline_uses_frozen_base_tokenizer_for_cache_identity():
    args = SimpleNamespace(
        graph_cache_dir=Path("graph-cache"),
        parser_dir=Path("parser"),
        model_path=Path("base-t5"),
        cuda="0",
        recipe_data={
            "training": {"extractor_eval_batch_size": 2, "target_pseudo_batch_size": 1},
            "pseudo": {
                "num_beams": 4,
                "max_new_tokens": 96,
                "length_penalty": 1.0,
                "max_target_unlabeled": 0,
                "pseudo_model_variant": "best",
                "pseudo_source_tag": "phase-a",
                "base_weight": 0.65,
                "high_precision_max_triplets": 1,
                "high_precision_max_token_distance": 5,
                "fixed_changed_min_score": 0.65,
                "fixed_changed_weight": 0.35,
            },
        },
    )
    argv = _pipeline_argv(args, Path("treatment"), "evaluate", True, Path("checkpoint"), "source_dev")
    index = argv.index("--syntactic_graph_cache_tokenizer_path")
    assert argv[index + 1] == "base-t5"


def test_target_test_is_rejected_before_execution():
    try:
        validate_input_split("target_test")
    except ValueError as exc:
        assert "target_test" in str(exc)
    else:
        raise AssertionError("target_test was not rejected")


def test_dann_batch_sampler_emits_exact_source_and_target_pairs_with_shared_order():
    control = PairedDomainBatchSampler(3, 3, source_batch_size=1, target_batch_size=1, seed=1000)
    treatment = PairedDomainBatchSampler(3, 3, source_batch_size=1, target_batch_size=1, seed=1000)

    control_batches = list(control)
    treatment_batches = list(treatment)

    assert control_batches == treatment_batches
    assert all(len(batch) == 2 for batch in control_batches)
    assert all(batch[0] < 3 and batch[1] >= 3 for batch in control_batches)
    assert control.epoch_reports[0]["source_rows"] == 3
    assert control.epoch_reports[0]["target_rows"] == 3
    assert control.epoch_reports[0]["logical_batches"] == 3
    assert control.epoch_reports[0]["incomplete_batches"] == 0


def test_dann_batch_sampler_rejects_single_domain_batches():
    try:
        PairedDomainBatchSampler(1, 0, source_batch_size=1, target_batch_size=1, seed=1000)
    except ValueError as exc:
        assert "both source and target" in str(exc)
    else:
        raise AssertionError("a DANN sampler without a target domain must fail")


def test_dann_sampler_explicit_epoch_is_repeatable_and_checkpointable():
    sampler = PairedDomainBatchSampler(
        4,
        4,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        source_row_ids=["s1", "s2", "s3", "s4"],
        target_row_ids=["t1", "t2", "t3", "t4"],
    )
    sampler.set_epoch(0)
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    sampler.set_epoch(2)
    epoch_two = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == epoch_zero
    assert epoch_two != epoch_zero

    state = sampler.state_dict()
    restored = PairedDomainBatchSampler(
        4,
        4,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        source_row_ids=["s1", "s2", "s3", "s4"],
        target_row_ids=["t1", "t2", "t3", "t4"],
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert list(restored) == epoch_zero


def test_dann_sampler_rejects_checkpoint_identity_or_batch_mismatch():
    sampler = PairedDomainBatchSampler(2, 2, source_batch_size=1, target_batch_size=1, seed=1000)
    state = sampler.state_dict()
    for field, value in (("seed", 2000), ("source_count", 3), ("target_batch_size", 2), ("source_row_ids", ["changed", 1])):
        changed = dict(state)
        changed[field] = value
        try:
            sampler.load_state_dict(changed)
        except ValueError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"sampler state mismatch must hard-fail: {field}")


def test_dataloader_extra_iteration_does_not_advance_explicit_sampler_epoch():
    sampler = PairedDomainBatchSampler(3, 3, source_batch_size=1, target_batch_size=1, seed=1000)
    sampler.set_epoch(2)
    expected = list(sampler)
    sampler.set_epoch(2)
    _ = list(sampler)
    sampler.set_epoch(2)
    assert list(sampler) == expected


def test_dann_sampler_records_each_physical_traversal_without_epoch_overwrite(tmp_path):
    audit_path = tmp_path / "dann_batch_audit.json"
    sampler = PairedDomainBatchSampler(
        2,
        2,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        audit_path=audit_path,
    )
    list(sampler)
    list(sampler)

    report = recover_dann_audit_journal(audit_path.with_suffix(".journal.jsonl"))
    assert [item["physical_traversal_index"] for item in report["epochs"]] == [0, 1]
    assert [item["sampling_epoch"] for item in report["epochs"]] == [0, 1]
    assert all(item["completion"] == "complete" for item in report["epochs"])


def test_dann_sampler_persists_last_partial_traversal_before_generator_cleanup(tmp_path):
    audit_path = tmp_path / "dann_batch_audit.json"
    sampler = PairedDomainBatchSampler(
        3,
        3,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        audit_path=audit_path,
    )
    iterator = iter(sampler)
    next(iterator)

    report = recover_dann_audit_journal(audit_path.with_suffix(".journal.jsonl"))
    partial = report["epochs"][-1]
    assert partial["completion"] == "partial"
    assert partial["planned_batches"] == 3
    assert partial["issued_batches"] == 1
    assert partial["processed_batches"] == 0
    assert partial["optimizer_global_step_start"] is None
    assert partial["optimizer_global_step_end"] is None


def test_dann_sampler_restore_keeps_physical_traversal_identity_monotonic(tmp_path):
    audit_path = tmp_path / "dann_batch_audit.json"
    sampler = PairedDomainBatchSampler(
        2,
        2,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        audit_path=audit_path,
    )
    list(sampler)
    state = sampler.state_dict()
    audit = sampler.audit_report()

    restored = PairedDomainBatchSampler(
        2,
        2,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        audit_path=audit_path,
    )
    restored.load_state_dict(state)
    restored.load_audit_report(audit)
    list(restored)

    report = restored.audit_report()
    assert [item["physical_traversal_index"] for item in report["epochs"]] == [0, 1]
    assert [item["sampling_epoch"] for item in report["epochs"]] == [0, 1]


def test_dann_sampling_order_keeps_legacy_trainer_epoch_semantics_at_real_boundary():
    labels = list(range(7)) + [6] + list(range(8, 15)) + [14] + list(range(16, 23)) + [22, 22]
    assert sorted(set(labels)) == list(range(7)) + list(range(8, 15)) + list(range(16, 23))
    assert compute_dann_expected_max_steps(
        source_count=906,
        target_count=906,
        source_batch_size=1,
        target_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=25,
    ) == 1400
    terminal_state = {"global_step": 1400, "max_steps": 1400}
    terminal_sampler = PairedDomainBatchSampler(906, 906, source_batch_size=1, target_batch_size=1, seed=1000)
    terminal_sampler.bind_training_state_provider(lambda: terminal_state)
    assert list(terminal_sampler) == []
    assert terminal_sampler.audit_report()["epochs"] == []

    sampler = PairedDomainBatchSampler(
        906,
        906,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
    )
    remaining = iter(labels)
    sampler.bind_sampling_epoch_provider(lambda: next(remaining))
    observed = []
    expected = []
    for label in labels:
        batches = list(sampler)
        observed.append(hashlib.sha256(json.dumps(batches, separators=(",", ":")).encode()).hexdigest())
        source_order = list(range(906))
        target_order = list(range(906))
        import random
        random.Random(1000 + label).shuffle(source_order)
        random.Random(1000 + label).shuffle(target_order)
        expected.append(hashlib.sha256(json.dumps(
            [[source_order[i], 906 + target_order[i]] for i in range(906)],
            separators=(",", ":"),
        ).encode()).hexdigest())
    assert observed == expected
    report = sampler.audit_report()
    assert [item["physical_traversal_index"] for item in report["epochs"]] == list(range(25))
    assert [item["sampling_epoch"] for item in report["epochs"]] == labels


def test_dann_audit_journal_is_append_only_and_recovers_crash_between_issue_and_ack(tmp_path):
    audit_path = tmp_path / "dann_batch_audit.json"
    sampler = PairedDomainBatchSampler(
        3,
        3,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        audit_path=audit_path,
    )
    iterator = iter(sampler)
    next(iterator)
    sampler.flush_audit_snapshot()
    journal_path = audit_path.with_suffix(".journal.jsonl")
    assert journal_path.is_file()
    stats = sampler.audit_io_stats()
    assert stats["journal_write_count"] <= 4
    assert stats["snapshot_write_count"] <= 1
    recovered = recover_dann_audit_journal(journal_path)
    partial = recovered["epochs"][-1]
    assert partial["issued_batches"] == 1
    assert partial["processed_batches"] == 0
    assert partial["completion"] == "partial"
    assert stats["journal_bytes"] < stats["snapshot_bytes"] * 4


def test_dann_audit_ack_is_distinct_from_issued_batches():
    state = {"global_step": 0, "max_steps": 1}
    sampler = PairedDomainBatchSampler(2, 2, source_batch_size=1, target_batch_size=1, seed=1000)
    sampler.bind_training_state_provider(lambda: state)
    iterator = iter(sampler)
    next(iterator)
    assert sampler.audit_report()["epochs"][-1]["issued_batches"] == 1
    assert sampler.audit_report()["epochs"][-1]["processed_batches"] == 0
    sampler.acknowledge_next_batch()
    assert sampler.audit_report()["epochs"][-1]["processed_batches"] == 1


def test_dann_sampler_stops_before_issuing_after_terminal_max_steps():
    state = {"global_step": 2, "max_steps": 2}
    sampler = PairedDomainBatchSampler(
        5, 5, source_batch_size=1, target_batch_size=1, seed=1000
    )
    sampler.bind_training_state_provider(lambda: state)
    iterator = iter(sampler)
    with pytest.raises(StopIteration):
        next(iterator)
    assert sampler.audit_report()["epochs"] == []


def test_dann_complete_requires_issued_processed_and_planned_equality():
    sampler = PairedDomainBatchSampler(
        2, 2, source_batch_size=1, target_batch_size=1, seed=1000
    )
    list(sampler)
    report = copy.deepcopy(sampler.audit_report())
    report["epochs"][0]["processed_batches"] = 1
    restored = PairedDomainBatchSampler(
        2, 2, source_batch_size=1, target_batch_size=1, seed=1000
    )
    with pytest.raises(ValueError, match="complete|accounting"):
        restored.load_audit_report(report)


def test_formal_dann_validator_rejects_complete_count_mismatch(tmp_path):
    sampler = PairedDomainBatchSampler(1, 1, source_batch_size=1, target_batch_size=1, seed=1000)
    list(sampler)
    report = sampler.audit_report()
    report["epochs"][0]["processed_batches"] = 0
    (tmp_path / "dann_batch_audit.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="complete.*mismatch"):
        _read_dann_batch_audit(tmp_path)


def test_dann_resume_reissues_issued_but_unprocessed_batch(tmp_path):
    audit_path = tmp_path / "dann_batch_audit.json"
    state = {"global_step": 0, "max_steps": 3}
    sampler = PairedDomainBatchSampler(
        3, 3, source_batch_size=1, target_batch_size=1, seed=1000, audit_path=audit_path
    )
    sampler.bind_training_state_provider(lambda: state)
    original_batch = next(iter(sampler))
    snapshot = sampler.state_dict()
    report = sampler.audit_report()

    restored = PairedDomainBatchSampler(
        3, 3, source_batch_size=1, target_batch_size=1, seed=1000, audit_path=audit_path
    )
    restored.bind_training_state_provider(lambda: state)
    restored.load_state_dict(snapshot)
    restored.load_audit_report(report)
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()
    restored.set_resume_checkpoint_identity(checkpoint_dir, "checkpoint-hash")
    replayed = next(iter(restored))
    assert replayed == original_batch
    assert restored.audit_report()["epochs"][-1]["processed_batches"] == 0
    restored.acknowledge_next_batch()
    assert restored.audit_report()["epochs"][-1]["processed_batches"] == 1


def test_dann_validator_rejects_non_integer_or_non_contiguous_optimizer_step_ranges(tmp_path):
    state = {"global_step": 0, "max_steps": 1}
    sampler = PairedDomainBatchSampler(2, 2, source_batch_size=1, target_batch_size=1, seed=1000)
    sampler.bind_training_state_provider(lambda: state)
    iterator = iter(sampler)
    next(iterator)
    sampler.acknowledge_next_batch()
    state["global_step"] = 1
    report = sampler.audit_report()
    report["epochs"][0]["optimizer_global_step_start"] = "0"
    report_text = json.dumps(report)
    control_dir = tmp_path / "control"
    treatment_dir = tmp_path / "treatment"
    control_dir.mkdir()
    treatment_dir.mkdir()
    for directory in (control_dir, treatment_dir):
        (directory / "dann_batch_audit.json").write_text(report_text, encoding="utf-8")
    try:
        _validate_control_treatment_dann_reports(
            {"control": control_dir, "treatment": treatment_dir},
            {"dann_batch_audit_path": str(control_dir / "dann_batch_audit.json"), "dann_batch_audit_sha256": hashlib.sha256(report_text.encode()).hexdigest()},
            expected_source_count=2,
            expected_target_count=2,
            expected_source_row_ids=[0, 1],
            expected_target_row_ids=[0, 1],
            require_training_state=True,
            expected_max_steps=1,
        )
    except RuntimeError as exc:
        assert "step" in str(exc).lower() or "optimizer" in str(exc).lower()
    else:
        raise AssertionError("invalid optimizer step range must be rejected")


def test_control_treatment_alignment_rejects_partial_physical_traversal(tmp_path):
    control_dir = tmp_path / "control"
    treatment_dir = tmp_path / "treatment"
    control_dir.mkdir()
    treatment_dir.mkdir()
    for directory in (control_dir, treatment_dir):
        (directory / "source_train.jsonl").write_text('{"id":"s1"}\n{"id":"s2"}\n', encoding="utf-8")
        (directory / "target_unlabeled.jsonl").write_text('{"id":"t1"}\n{"id":"t2"}\n', encoding="utf-8")
    sampler = PairedDomainBatchSampler(
        2,
        2,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        source_row_ids=["s1", "s2"],
        target_row_ids=["t1", "t2"],
    )
    iterator = iter(sampler)
    next(iterator)
    report_text = json.dumps(sampler.audit_report())
    for directory in (control_dir, treatment_dir):
        (directory / "dann_batch_audit.json").write_text(report_text, encoding="utf-8")
    try:
        _validate_control_treatment_dann_reports(
            {"control": control_dir, "treatment": treatment_dir},
            {
                "dann_batch_audit_path": str(control_dir / "dann_batch_audit.json"),
                "dann_batch_audit_sha256": hashlib.sha256(report_text.encode()).hexdigest(),
            },
            expected_source_count=2,
            expected_target_count=2,
            expected_source_row_ids=["s1", "s2"],
            expected_target_row_ids=["t1", "t2"],
        )
    except RuntimeError as exc:
        assert "partial" in str(exc).lower() or "complete" in str(exc).lower()
    else:
        raise AssertionError("partial physical traversals must not be formal evidence")


def test_terminal_partial_traversal_is_formal_only_when_trainer_reached_max_steps(tmp_path):
    control_dir = tmp_path / "control"
    treatment_dir = tmp_path / "treatment"
    control_dir.mkdir()
    treatment_dir.mkdir()
    for directory in (control_dir, treatment_dir):
        (directory / "source_train.jsonl").write_text('{"id":"s1"}\n{"id":"s2"}\n{"id":"s3"}\n', encoding="utf-8")
        (directory / "target_unlabeled.jsonl").write_text('{"id":"t1"}\n{"id":"t2"}\n{"id":"t3"}\n', encoding="utf-8")
    training_state = {"global_step": 0, "max_steps": 1}
    sampler = PairedDomainBatchSampler(
        3,
        3,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        source_row_ids=["s1", "s2", "s3"],
        target_row_ids=["t1", "t2", "t3"],
    )
    sampler.bind_training_state_provider(lambda: training_state)
    iterator = iter(sampler)
    next(iterator)
    sampler.acknowledge_next_batch()
    training_state["global_step"] = 1
    report_text = json.dumps(sampler.audit_report())
    for directory in (control_dir, treatment_dir):
        (directory / "dann_batch_audit.json").write_text(report_text, encoding="utf-8")

    result = _validate_control_treatment_dann_reports(
        {"control": control_dir, "treatment": treatment_dir},
        {
            "dann_batch_audit_path": str(control_dir / "dann_batch_audit.json"),
            "dann_batch_audit_sha256": hashlib.sha256(report_text.encode()).hexdigest(),
        },
        expected_source_count=3,
        expected_target_count=3,
        expected_source_row_ids=["s1", "s2", "s3"],
        expected_target_row_ids=["t1", "t2", "t3"],
        require_training_state=True,
        expected_max_steps=1,
    )

    assert result["status"] == "matched"
    assert result["treatment"]["epochs"][-1]["completion"] == "partial"


def test_terminal_partial_checkpoint_is_resume_valid_only_with_processed_equals_issued(tmp_path):
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    state = {"global_step": 0, "max_steps": 1}
    sampler = PairedDomainBatchSampler(3, 3, source_batch_size=1, target_batch_size=1, seed=1000)
    sampler.bind_training_state_provider(lambda: state)
    iterator = iter(sampler)
    next(iterator)
    sampler.acknowledge_next_batch()
    state["global_step"] = 1
    audit = sampler.audit_report()
    sampler_state = sampler.state_dict()
    trainer_state = {"global_step": 1}
    model_path = checkpoint / "pytorch_model.bin"
    model_path.write_bytes(b"cpu-test-model")
    sampler_path = checkpoint / "dann_batch_sampler_state.json"
    audit_path = checkpoint / "dann_batch_audit.json"
    trainer_state_path = checkpoint / "trainer_state.json"
    gradient_state_path = checkpoint / "dann_gradient_state.pt"
    sampler_path.write_text(json.dumps(sampler_state), encoding="utf-8")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    trainer_state_path.write_text(json.dumps(trainer_state), encoding="utf-8")
    torch.save({"schema_version": 1, "accumulation_remainder": 0, "gradients": {}}, gradient_state_path)
    manifest = {
        "schema_version": 3,
        "complete": True,
        "resume_complete": True,
        "training_terminal_partial": True,
        "audit_protocol": "physical_dataloader_traversal_v2",
        "seed": 1000,
        "source_count": 3,
        "target_count": 3,
        "source_batch_size": 1,
        "target_batch_size": 1,
        "source_row_ids": [0, 1, 2],
        "target_row_ids": [0, 1, 2],
        "completed_epochs": [],
        "completed_epoch_count": 0,
        "completed_physical_traversals": [],
        "audit_traversal_count": 1,
        "trainer_global_step": 1,
        "trainer_max_steps": 1,
        "last_traversal_completion": "partial",
        "sampler_state_sha256": hashlib.sha256(sampler_path.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "trainer_state_sha256": hashlib.sha256(trainer_state_path.read_bytes()).hexdigest(),
        "gradient_state_artifact": gradient_state_path.name,
        "gradient_state_sha256": hashlib.sha256(gradient_state_path.read_bytes()).hexdigest(),
        "model_artifact": "pytorch_model.bin",
        "model_artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (checkpoint / "dann_checkpoint_state.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert find_latest_complete_dann_checkpoint(tmp_path, sampler) == checkpoint


def test_legacy_dann_audit_cannot_be_formal_phase_a_pass(tmp_path):
    control_dir = tmp_path / "control"
    treatment_dir = tmp_path / "treatment"
    control_dir.mkdir()
    treatment_dir.mkdir()
    for directory in (control_dir, treatment_dir):
        (directory / "source_train.jsonl").write_text('{"id":"s1"}\n', encoding="utf-8")
        (directory / "target_unlabeled.jsonl").write_text('{"id":"t1"}\n', encoding="utf-8")
    report = {
        "schema_version": 1,
        "seed": 1000,
        "source_batch_size": 1,
        "target_batch_size": 1,
        "source_count": 1,
        "target_count": 1,
        "source_row_ids": ["s1"],
        "target_row_ids": ["t1"],
        "epochs": [{
            "epoch": 0,
            "source_batch_size": 1,
            "target_batch_size": 1,
            "source_rows": 1,
            "target_rows": 1,
            "source_unique_rows": 1,
            "target_unique_rows": 1,
            "logical_batches": 1,
            "incomplete_batches": 0,
            "batches": [{
                "logical_batch_id": 0,
                "source_indices": [0],
                "target_indices": [1],
                "source_row_ids": ["s1"],
                "target_row_ids": ["t1"],
                "source_count": 1,
                "target_count": 1,
            }],
        }],
    }
    report_text = json.dumps(report)
    for directory in (control_dir, treatment_dir):
        (directory / "dann_batch_audit.json").write_text(report_text, encoding="utf-8")
    try:
        _validate_control_treatment_dann_reports(
            {"control": control_dir, "treatment": treatment_dir},
            {
                "dann_batch_audit_path": str(control_dir / "dann_batch_audit.json"),
                "dann_batch_audit_sha256": hashlib.sha256(report_text.encode()).hexdigest(),
            },
            expected_epochs=1,
            expected_source_count=1,
            expected_target_count=1,
            expected_source_row_ids=["s1"],
            expected_target_row_ids=["t1"],
        )
    except RuntimeError as exc:
        assert "legacy" in str(exc).lower()
    else:
        raise AssertionError("legacy DANN evidence must not be formally accepted")


def test_legacy_diagnostic_resume_writes_migration_report_without_touching_stage_status(tmp_path):
    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir()
    stage_status = run_dir / "stage_status.json"
    original_status = '{"status":"in_progress","completed_stages":["control_training"]}'
    stage_status.write_text(original_status, encoding="utf-8")
    (run_dir / "git_identity.json").write_text(json.dumps({"commit": "3d4153c"}), encoding="utf-8")
    (run_dir / "control").mkdir()
    (run_dir / "control" / "dann_batch_audit.json").write_text('{"schema_version":1}', encoding="utf-8")

    report_path = prepare_legacy_diagnostic_resume(run_dir, current_commit="new-commit")

    assert report_path == run_dir / "legacy_diagnostic_migration.json"
    assert stage_status.read_text(encoding="utf-8") == original_status
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "BLOCKED"
    assert report["formal_evidence"] is False
    assert report["resume_allowed"] is False
    assert report["source_commit"] == "3d4153c"
    assert report["current_commit"] == "new-commit"


def test_trainer_dataloader_preserves_the_complete_paired_domain_batch():
    class PairDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                "input_ids": [index + 1, 2],
                "labels": [1, 2],
                "sample_weight": 1.0 if index == 0 else 0.0,
                "domain_weight": 1.0 if index == 0 else 0.0,
                "domain_label": 0 if index == 0 else 1,
            }

    class PairCollator:
        def __call__(self, features):
            return {
                key: torch.tensor([feature[key] for feature in features])
                for key in ("input_ids", "labels", "sample_weight", "domain_weight", "domain_label")
            }

    with tempfile.TemporaryDirectory() as directory:
        model = T5ForConditionalGeneration(
            T5Config(
                vocab_size=8,
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
        )
        args = Seq2SeqTrainingArguments(
            output_dir=directory,
            per_device_train_batch_size=1,
            report_to=[],
            remove_unused_columns=True,
        )
        trainer = WeightedSeq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=PairDataset(),
            data_collator=PairCollator(),
            dann_batch_sampler=PairedDomainBatchSampler(
                1,
                1,
                source_batch_size=1,
                target_batch_size=1,
                seed=1000,
            ),
        )
        batch = next(iter(trainer.get_train_dataloader()))

    assert tuple(batch["input_ids"].shape) == (2, 2)
    assert batch["domain_label"].tolist() == [0, 1]
    assert batch["domain_weight"].tolist() == [1.0, 0.0]


def test_paired_dann_generation_loss_matches_source_only_loss():
    config = T5Config(
        vocab_size=16,
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
    source_model = T5ForConditionalGeneration(config).eval()
    paired_model = T5ForConditionalGeneration(config).eval()
    paired_model.load_state_dict(source_model.state_dict())
    source_args = Seq2SeqTrainingArguments(
        output_dir=tempfile.mkdtemp(),
        per_device_train_batch_size=1,
        no_cuda=True,
        report_to=[],
    )
    paired_args = Seq2SeqTrainingArguments(
        output_dir=tempfile.mkdtemp(),
        per_device_train_batch_size=1,
        no_cuda=True,
        report_to=[],
    )
    source_trainer = WeightedSeq2SeqTrainer(model=source_model, args=source_args)
    paired_trainer = WeightedSeq2SeqTrainer(
        model=paired_model,
        args=paired_args,
        dann_batch_sampler=PairedDomainBatchSampler(
            1,
            1,
            source_batch_size=1,
            target_batch_size=1,
            seed=1000,
        ),
    )
    source_batch = {
        "input_ids": torch.tensor([[2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3]]),
        "sample_weight": torch.tensor([1.0]),
        "domain_weight": torch.tensor([1.0]),
    }
    paired_batch = {
        "input_ids": torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3], [-100, -100, -100]]),
        "sample_weight": torch.tensor([1.0, 0.0]),
        "domain_weight": torch.tensor([1.0, 0.0]),
    }
    source_loss = source_trainer.compute_loss(source_model, source_batch)
    paired_loss = paired_trainer.compute_loss(paired_model, paired_batch)
    assert torch.allclose(source_loss, paired_loss, atol=1e-7, rtol=1e-7)
    assert not torch.allclose(paired_loss, source_loss * 0.5, atol=1e-7, rtol=1e-7)


def test_real_tiny_t5_control_treatment_initialization_audit_isolated(tmp_path):
    base_dir = tmp_path / "base-t5"
    config = T5Config(
        vocab_size=16,
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
    T5ForConditionalGeneration(config).save_pretrained(base_dir)
    torch.manual_seed(1000)
    control = load_seq2seq_model(str(base_dir), use_syntactic_graph_adapter=False)
    treatment = load_seq2seq_model(str(base_dir), use_syntactic_graph_adapter=True, relation_vocab_size=8)
    initialize_domain_adversarial_head(control, hidden_size=8, classifier_hidden_size=8, seed=1001)
    initialize_domain_adversarial_head(treatment, hidden_size=8, classifier_hidden_size=8, seed=1001)
    control_audit = build_initialization_audit(control, variant="control", seed=1000)
    treatment_audit = build_initialization_audit(treatment, variant="treatment", seed=1000)
    assert control_audit["shared_t5_parameter_sha256"] == treatment_audit["shared_t5_parameter_sha256"]
    assert control_audit["dann_head_parameter_sha256"] == treatment_audit["dann_head_parameter_sha256"]
    assert control_audit["parameter_groups"]["syntactic_graph_adapter"]["parameter_names"] == []
    assert treatment_audit["parameter_groups"]["syntactic_graph_adapter"]["parameter_names"]
    assert all(item["finite"] for item in treatment_audit["graph_parameter_stats"])
    assert treatment_audit["graph_parameter_initialization"]["initialized_from_base_checkpoint"] is True
    control_audit_path = base_dir.parent / "control-init.json"
    treatment_audit_path = base_dir.parent / "treatment-init.json"
    control_audit_path.write_text(json.dumps(control_audit), encoding="utf-8")
    treatment_audit_path.write_text(json.dumps(treatment_audit), encoding="utf-8")
    assert validate_initialization_pair(control_audit_path, treatment_audit_path)["status"] == "matched"


def test_domain_head_initialization_restores_training_rng_state():
    model = T5ForConditionalGeneration(T5Config(vocab_size=8, d_model=8, d_kv=4, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=2, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    initialize_domain_adversarial_head(model, hidden_size=8, classifier_hidden_size=8, seed=1001)
    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_phase_a_recipe_rejects_all_frozen_boundary_and_parameter_mutations():
    recipe_path = Path(__file__).parent / "configs" / "recipes" / "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_v1.json"
    base = json.loads(recipe_path.read_text(encoding="utf-8"))
    mutations = (
        lambda value: value["external_inputs"].__setitem__("target_test_access", True),
        lambda value: value["data_boundary"].__setitem__("generator", True),
        lambda value: value["variants"]["control"].__setitem__("graph_enabled", True),
        lambda value: value["external_inputs"]["source_train"].__setitem__("path", "changed.txt"),
        lambda value: value["pseudo"].__setitem__("constrained_decoding", True),
        lambda value: value["variants"]["treatment"].__setitem__("graph_layers", 2),
        lambda value: value["models"].__setitem__("t5_base", "changed-model"),
        lambda value: value["training"].__setitem__("lambda_domain_adv", 0.04),
        lambda value: value["training"].__setitem__("extractor_train_batch_size", 2),
        lambda value: value["training"].__setitem__("max_source_length", 127),
        lambda value: value["training"].__setitem__("pairing_temperature", 0.2),
        lambda value: value["training"].__setitem__("multi_triplet_loss_gain", 0.1),
        lambda value: value["pseudo"].__setitem__("length_penalty", 1.2),
        lambda value: value["models"].__setitem__("generator", "changed-model"),
        lambda value: value["external_inputs"]["source_train"].__setitem__("sha256", "changed"),
    )
    for mutate in mutations:
        changed = copy.deepcopy(base)
        mutate(changed)
        try:
            _validate_recipe(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("frozen recipe mutation must hard-fail")


def test_phase_a_recipe_is_accepted_and_training_argv_has_explicit_lengths():
    recipe_path = Path(__file__).parent / "configs" / "recipes" / "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_v1.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    _validate_recipe(recipe)
    args = SimpleNamespace(
        recipe_data=recipe,
        model_path=recipe["models"]["t5_base"],
        graph_cache_dir="graph-cache",
        parser_dir="parser",
        cuda="0",
    )
    argv = _training_argv(args, Path("variant"), True)
    assert "--max_source_length" in argv and argv[argv.index("--max_source_length") + 1] == "128"
    assert "--max_target_length" in argv and argv[argv.index("--max_target_length") + 1] == "96"
    assert argv[argv.index("--lambda_domain_adv") + 1] == "0.03"


def test_phase_a_graph_cache_preflight_rejects_missing_identity_bundle(tmp_path):
    with pytest.raises(RuntimeError, match="missing required artifacts"):
        validate_phase_a_graph_cache(
            tmp_path / "missing-cache",
            {"source_train": [], "source_dev": [], "target_unlabeled": []},
            {},
        )


def test_t5_model_identity_includes_all_loaded_files():
    hashes = _model_hashes(Path(r"J:\nlp\models\t5-base-py"))
    assert {"config.json", "pytorch_model.bin", "generation_config.json", "spiece.model", "tokenizer.json"}.issubset(hashes)


def test_dann_audit_validates_real_counts_ids_coverage_and_epochs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        control_dir = root / "control"
        treatment_dir = root / "treatment"
        control_dir.mkdir()
        treatment_dir.mkdir()
        rows = {
            "source_train": [{"id": "s1"}, {"id": "s2"}],
            "target_unlabeled": [{"id": "t1"}, {"id": "t2"}],
        }
        for variant_dir in (control_dir, treatment_dir):
            for name, values in rows.items():
                (variant_dir / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
        sampler = PairedDomainBatchSampler(2, 2, source_batch_size=1, target_batch_size=1, seed=1000, source_row_ids=["s1", "s2"], target_row_ids=["t1", "t2"])
        list(sampler)
        sampler.set_epoch(1)
        list(sampler)
        report_text = json.dumps(sampler.audit_report())
        (control_dir / "dann_batch_audit.json").write_text(report_text, encoding="utf-8")
        (treatment_dir / "dann_batch_audit.json").write_text(report_text, encoding="utf-8")
        result = _validate_control_treatment_dann_reports(
            {"control": control_dir, "treatment": treatment_dir},
            {"dann_batch_audit_path": str(control_dir / "dann_batch_audit.json"), "dann_batch_audit_sha256": hashlib.sha256(report_text.encode()).hexdigest()},
            expected_epochs=2,
            expected_source_count=2,
            expected_target_count=2,
            expected_source_row_ids=["s1", "s2"],
            expected_target_row_ids=["t1", "t2"],
        )
        assert result["status"] == "matched"


def test_stage_identity_records_and_validates_artifact_hashes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "model.bin"
        artifact.write_bytes(b"stable")
        input_file = root / "source_train.jsonl"
        input_file.write_text("source", encoding="utf-8")
        recipe_file = root / "recipe.json"
        recipe_file.write_text("recipe", encoding="utf-8")
        record = build_stage_identity(
            "treatment_training",
            ["python", "train", "--seed", "1000"],
            {"source_train": input_file},
            hashlib.sha256(b"recipe").hexdigest(),
            artifact,
            artifact,
            "commit-a",
            recipe_path=recipe_file,
        )

        assert validate_stage_identity(record) is True
        artifact.write_bytes(b"changed")
        try:
            validate_stage_identity(record)
        except RuntimeError as exc:
            assert "treatment_training" in str(exc)
        else:
            raise AssertionError("modified treatment artifact must hard-fail resume")


def test_pseudo_stage_identity_covers_all_a4_outputs_and_rejects_each_change():
    required_names = (
        "target_pseudo.jsonl",
        "target_pseudo_selected.jsonl",
        "target_pseudo_high_precision.jsonl",
        "target_pseudo_train_selected.jsonl",
        "target_pseudo_selected_analysis.json",
        "target_pseudo_generation_state.json",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output_paths = {name: root / name for name in required_names}
        for name, path in output_paths.items():
            path.write_text(name, encoding="utf-8")
        input_file = root / "target_unlabeled.jsonl"
        input_file.write_text("target", encoding="utf-8")
        recipe_file = root / "recipe.json"
        recipe_file.write_text("recipe", encoding="utf-8")
        record = build_stage_identity(
            "treatment_target_pseudo_inference",
            ["python", "pseudo"],
            {"target_unlabeled": input_file},
            hashlib.sha256(b"recipe").hexdigest(),
            output_paths["target_pseudo_selected.jsonl"],
            root,
            "commit-a",
            recipe_path=recipe_file,
            output_artifacts=output_paths,
        )
        assert set(record["output_artifacts"]) == set(required_names)
        assert validate_stage_identity(record) is True
        for name, path in output_paths.items():
            path.write_text(name + "-changed", encoding="utf-8")
            try:
                validate_stage_identity(record)
            except RuntimeError as exc:
                assert name in str(exc)
            else:
                raise AssertionError(f"modified pseudo output must hard-fail: {name}")
            path.write_text(name, encoding="utf-8")


def test_training_stage_identity_covers_dann_audit_and_rejects_change():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model_dir = root / "extractor" / "best"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        audit = root / "dann_batch_audit.json"
        audit.write_text(json.dumps({"seed": 1000, "epochs": [{"epoch": 0}]}), encoding="utf-8")
        initialization = root / "phase_a_initialization_audit.json"
        initialization.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        input_file = root / "source_train.jsonl"
        input_file.write_text("source", encoding="utf-8")
        recipe_file = root / "recipe.json"
        recipe_file.write_text("recipe", encoding="utf-8")
        record = build_stage_identity(
            "treatment_training",
            ["python", "train", "--paired_domain_batches"],
            {"source_train": input_file},
            hashlib.sha256(b"recipe").hexdigest(),
            model_dir,
            model_dir,
            "commit-a",
            recipe_path=recipe_file,
            output_artifacts={"extractor_best": model_dir, "dann_batch_audit": audit, "phase_a_initialization_audit": initialization},
        )
        assert set(record["output_artifacts"]) == {"extractor_best", "dann_batch_audit", "phase_a_initialization_audit"}
        audit.write_text(json.dumps({"seed": 1000, "epochs": [{"epoch": 1}]}), encoding="utf-8")
        try:
            validate_stage_identity(record)
        except RuntimeError as exc:
            assert "dann_batch_audit" in str(exc)
        else:
            raise AssertionError("modified DANN audit must hard-fail resume")


def test_pseudo_output_path_contract_names_all_required_a4_artifacts():
    paths = build_phase_a_pseudo_output_paths(Path("variant"))
    assert set(paths) == {
        "target_pseudo.jsonl",
        "target_pseudo_selected.jsonl",
        "target_pseudo_high_precision.jsonl",
        "target_pseudo_train_selected.jsonl",
        "target_pseudo_selected_analysis.json",
        "target_pseudo_generation_state.json",
    }


def test_external_control_without_dann_audit_hard_fails():
    try:
        validate_external_control_dann_audit({"resolved_model_path": "missing-model"})
    except RuntimeError as exc:
        assert "DANN" in str(exc)
    else:
        raise AssertionError("external Control without DANN audit must hard-fail")


def test_control_and_treatment_dann_row_ids_or_order_must_match():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        control_dir = root / "control"
        treatment_dir = root / "treatment"
        control_dir.mkdir()
        treatment_dir.mkdir()
        control_report = {
            "schema_version": 1,
            "seed": 1000,
            "source_count": 1,
            "target_count": 1,
            "source_row_ids": ["s1"],
            "target_row_ids": ["t1"],
            "epochs": [{
                "epoch": 0,
                "source_batch_size": 1,
                "target_batch_size": 1,
                "source_rows": 1,
                "target_rows": 1,
                "source_unique_rows": 1,
                "target_unique_rows": 1,
                "logical_batches": 1,
                "incomplete_batches": 0,
                "batches": [{"logical_batch_id": 0, "source_indices": [0], "target_indices": [1], "source_count": 1, "target_count": 1, "source_row_ids": ["s1"], "target_row_ids": ["t1"]}],
            }],
        }
        treatment_report = json.loads(json.dumps(control_report))
        treatment_report["epochs"][0]["batches"][0]["source_row_ids"] = ["s2"]
        (control_dir / "dann_batch_audit.json").write_text(json.dumps(control_report), encoding="utf-8")
        (treatment_dir / "dann_batch_audit.json").write_text(json.dumps(treatment_report), encoding="utf-8")
        try:
            _validate_control_treatment_dann_reports(
                {"control": control_dir, "treatment": treatment_dir},
                {"dann_batch_audit_path": str(control_dir / "dann_batch_audit.json")},
                expected_source_count=1,
                expected_target_count=1,
                expected_source_row_ids=["s1"],
                expected_target_row_ids=["t1"],
            )
        except RuntimeError as exc:
            assert "batch" in str(exc).lower()
        else:
            raise AssertionError("different DANN row IDs must hard-fail")


def test_two_identical_but_wrong_dann_reports_fail_against_real_input_identity():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        control_dir = root / "control"
        treatment_dir = root / "treatment"
        control_dir.mkdir()
        treatment_dir.mkdir()
        report = {
            "schema_version": 1,
            "seed": 1000,
            "source_batch_size": 1,
            "target_batch_size": 1,
            "source_count": 1,
            "target_count": 1,
            "source_row_ids": ["wrong-source"],
            "target_row_ids": ["wrong-target"],
            "epochs": [{
                "epoch": 0,
                "source_batch_size": 1,
                "target_batch_size": 1,
                "source_rows": 1,
                "target_rows": 1,
                "source_unique_rows": 1,
                "target_unique_rows": 1,
                "logical_batches": 1,
                "incomplete_batches": 0,
                "batches": [{
                    "logical_batch_id": 0,
                    "source_indices": [0],
                    "target_indices": [1],
                    "source_row_ids": ["wrong-source"],
                    "target_row_ids": ["wrong-target"],
                    "source_count": 1,
                    "target_count": 1,
                }],
            }],
        }
        for path in (control_dir / "dann_batch_audit.json", treatment_dir / "dann_batch_audit.json"):
            path.write_text(json.dumps(report), encoding="utf-8")
        with (control_dir / "dann_batch_audit.json").open("rb") as handle:
            control_hash = hashlib.sha256(handle.read()).hexdigest()
        try:
            _validate_control_treatment_dann_reports(
                {"control": control_dir, "treatment": treatment_dir},
                {
                    "dann_batch_audit_path": str(control_dir / "dann_batch_audit.json"),
                    "dann_batch_audit_sha256": control_hash,
                },
                expected_epochs=1,
                expected_source_count=2,
                expected_target_count=2,
                expected_source_row_ids=["source-1", "source-2"],
                expected_target_row_ids=["target-1", "target-2"],
            )
        except RuntimeError as exc:
            assert "identity" in str(exc).lower() or "row" in str(exc).lower()
        else:
            raise AssertionError("identical but incorrect DANN reports must fail")


class _TinyPairedDataset(Dataset):
    def __init__(self):
        self.items = [
            {"input_ids": [2, 3, 4, 5], "labels": [1, 2, 3], "sample_weight": 1.0, "domain_weight": 1.0},
            {"input_ids": [3, 4, 5, 6], "labels": [1, 2, 3], "sample_weight": 1.0, "domain_weight": 1.0},
            {"input_ids": [7, 8, 9, 10], "labels": [-100, -100, -100], "sample_weight": 0.0, "domain_weight": 0.0},
            {"input_ids": [8, 9, 10, 11], "labels": [-100, -100, -100], "sample_weight": 0.0, "domain_weight": 0.0},
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _tiny_pair_collator(features):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in features], dtype=torch.long),
        "attention_mask": torch.ones(len(features), 4, dtype=torch.long),
        "labels": torch.tensor([item["labels"] for item in features], dtype=torch.long),
        "sample_weight": torch.tensor([item["sample_weight"] for item in features]),
        "domain_weight": torch.tensor([item["domain_weight"] for item in features]),
    }


def _run_tiny_paired_trainer(output_dir, epochs, checkpoint=None, stop_after_epoch=None):
    config = T5Config(
        vocab_size=16,
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
    if checkpoint is None:
        torch.manual_seed(1000)
        model = T5ForConditionalGeneration(config)
    else:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    sampler = PairedDomainBatchSampler(
        2,
        2,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        source_row_ids=["s1", "s2"],
        target_row_ids=["t1", "t2"],
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        no_cuda=True,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=5,
        logging_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
        save_safetensors=False,
        seed=1000,
        data_seed=1000,
    )
    callbacks = []
    if stop_after_epoch is not None:
        class StopAfterEpoch(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                if state.epoch >= stop_after_epoch:
                    control.should_training_stop = True
                return control
        callbacks.append(StopAfterEpoch())
    trainer = WeightedSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=_TinyPairedDataset(),
        data_collator=_tiny_pair_collator,
        dann_batch_sampler=sampler,
        callbacks=callbacks,
    )
    if checkpoint is not None:
        trainer.load_dann_batch_sampler_state(checkpoint)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None)
    return trainer


def test_real_trainer_three_epoch_resume_matches_continuous_run(tmp_path):
    continuous = _run_tiny_paired_trainer(tmp_path / "continuous", 3)
    interrupted = _run_tiny_paired_trainer(tmp_path / "interrupted", 3, stop_after_epoch=1)
    checkpoint = find_latest_complete_dann_checkpoint(tmp_path / "interrupted", interrupted.dann_batch_sampler)
    resumed = _run_tiny_paired_trainer(tmp_path / "interrupted", 3, checkpoint=checkpoint)
    assert resumed.get_dann_batch_audit() == continuous.get_dann_batch_audit()
    continuous_state = continuous.model.state_dict()
    resumed_state = resumed.model.state_dict()
    assert continuous_state.keys() == resumed_state.keys()
    assert all(torch.equal(continuous_state[name], resumed_state[name]) for name in continuous_state)


class _NonDivisiblePairedDataset(Dataset):
    def __init__(self, count=5):
        self.items = []
        for index in range(count):
            self.items.append({"input_ids": [2 + index, 3, 4, 5], "labels": [1, 2, 3], "sample_weight": 1.0, "domain_weight": 1.0})
        for index in range(count):
            self.items.append({"input_ids": [7 + index, 8, 9, 10], "labels": [-100, -100, -100], "sample_weight": 0.0, "domain_weight": 0.0})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _run_non_divisible_paired_trainer(
    output_dir,
    checkpoint=None,
    stop_after_epoch=None,
    max_steps=2,
    paired_count=5,
    paired_batch_size=1,
    audit_path=None,
    count_training_steps=False,
):
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
    if checkpoint is None:
        torch.manual_seed(1000)
        model = T5ForConditionalGeneration(config)
    else:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    sampler = PairedDomainBatchSampler(
        paired_count,
        paired_count,
        source_batch_size=paired_batch_size,
        target_batch_size=paired_batch_size,
        seed=1000,
        audit_path=audit_path,
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=3,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        no_cuda=True,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_steps=2,
        save_total_limit=5,
        logging_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
        save_safetensors=False,
        seed=1000,
        data_seed=1000,
    )
    callbacks = []
    class StopAtTerminalStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= max_steps:
                control.should_training_stop = True
            return control
    callbacks.append(StopAtTerminalStep())
    if stop_after_epoch is not None:
        class StopAfterEpoch(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                if state.epoch >= stop_after_epoch:
                    control.should_training_stop = True
                return control
        callbacks.append(StopAfterEpoch())
    trainer = WeightedSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=_NonDivisiblePairedDataset(paired_count),
        data_collator=_tiny_pair_collator,
        dann_batch_sampler=sampler,
        callbacks=callbacks,
    )
    step_counter = {"count": 0}
    if count_training_steps:
        original_training_step = trainer.training_step

        def counting_training_step(*args, **kwargs):
            step_counter["count"] += 1
            return original_training_step(*args, **kwargs)

        trainer.training_step = counting_training_step
    if checkpoint is not None:
        trainer.load_dann_batch_sampler_state(checkpoint)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None)
    if count_training_steps:
        trainer.training_step_call_count = step_counter["count"]
    return trainer


def test_fresh_nondivisible_three_epoch_training_has_zero_replay_and_expected_training_steps(tmp_path):
    audit_path = tmp_path / "fresh" / "dann_batch_audit.json"
    trainer = _run_non_divisible_paired_trainer(
        tmp_path / "fresh",
        max_steps=9,
        paired_count=5,
        audit_path=audit_path,
        count_training_steps=True,
    )
    journal_path = audit_path.with_suffix(".journal.jsonl")
    events = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    replay_events = [event for event in events if event["event"] == "batch_replayed"]
    assert replay_events == []
    assert trainer.training_step_call_count == 15
    assert trainer.state.global_step == 9


def test_phase_a_dann_gate_reads_journal_and_rejects_fresh_replay(tmp_path):
    audit_path = tmp_path / "run" / "dann_batch_audit.json"
    sampler = PairedDomainBatchSampler(1, 1, source_batch_size=1, target_batch_size=1, seed=1000, audit_path=audit_path)
    list(sampler)
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()
    sampler.set_resume_checkpoint_identity(checkpoint_dir, "checkpoint-hash")
    sampler._resume_replay_batch_ids = [(0, 0)]
    iterator = iter(sampler)
    next(iterator)
    sampler.flush_audit_snapshot()
    try:
        _read_dann_batch_audit(
            audit_path,
            require_journal=True,
            require_fresh_replay_free=True,
        )
    except RuntimeError as exc:
        assert "fresh" in str(exc).lower()
    else:
        raise AssertionError("fresh Phase A DANN audit must reject journal replay events")


def test_checkpoint_state_with_zero_remainder_is_empty_and_does_not_mutate_live_sampler():
    sampler = PairedDomainBatchSampler(5, 5, source_batch_size=1, target_batch_size=1, seed=1000)
    list(sampler)
    live_before = sampler.state_dict()
    audit_before = copy.deepcopy(sampler.audit_report())
    checkpoint_state = sampler.build_checkpoint_state(accumulation_remainder=0)
    assert checkpoint_state["resume_replay_batch_ids"] == []
    assert checkpoint_state["resume_reissue_batch_ids"] == []
    assert sampler.state_dict() == live_before
    assert sampler.audit_report() == audit_before
    assert sampler.state_dict()["resume_replay_batch_ids"] == []


def test_checkpoint_replay_state_isolated_until_explicit_restore():
    sampler = PairedDomainBatchSampler(5, 5, source_batch_size=1, target_batch_size=1, seed=1000)
    list(sampler)
    live_before = sampler.state_dict()
    audit_before = copy.deepcopy(sampler.audit_report())
    checkpoint_state = sampler.build_checkpoint_state(accumulation_remainder=2)
    assert checkpoint_state["resume_replay_batch_ids"] == [[0, 3], [0, 4]]
    assert sampler.state_dict() == live_before
    assert sampler.audit_report() == audit_before

    restored = PairedDomainBatchSampler(5, 5, source_batch_size=1, target_batch_size=1, seed=1000)
    restored.load_state_dict(checkpoint_state)
    assert restored.state_dict() == checkpoint_state


def test_replay_and_reissue_require_explicit_checkpoint_identity():
    sampler = PairedDomainBatchSampler(1, 1, source_batch_size=1, target_batch_size=1, seed=1000)
    for field in ("resume_replay_batch_ids", "resume_reissue_batch_ids"):
        state = sampler.state_dict()
        state[field] = [[0, 0]]
        restored = PairedDomainBatchSampler(1, 1, source_batch_size=1, target_batch_size=1, seed=1000)
        restored.load_state_dict(state)
        try:
            next(iter(restored))
        except RuntimeError as exc:
            assert "explicit checkpoint" in str(exc)
        else:
            raise AssertionError(f"{field} must require explicit checkpoint identity")


def test_checkpoint_remainder_uses_explicit_microbatch_counter_not_global_step():
    model = T5ForConditionalGeneration(
        T5Config(
            vocab_size=16,
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
    )
    trainer = WeightedSeq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=tempfile.mkdtemp(),
            per_device_train_batch_size=1,
            no_cuda=True,
            gradient_accumulation_steps=16,
            report_to=[],
        ),
        dann_batch_sampler=PairedDomainBatchSampler(1, 1, source_batch_size=1, target_batch_size=1, seed=1000),
    )
    trainer.state.global_step = 999
    trainer._dann_microbatches_since_optimizer_step = 0
    assert trainer.checkpoint_accumulation_remainder() == 0
    trainer._dann_microbatches_since_optimizer_step = 3
    assert trainer.checkpoint_accumulation_remainder() == 3


def test_906_16_25_boundary_has_no_fresh_replay_and_expected_terminal_traversal():
    state = {"global_step": 0, "max_steps": 1400, "gradient_accumulation_steps": 16}
    sampler = PairedDomainBatchSampler(906, 906, source_batch_size=1, target_batch_size=1, seed=1000)
    sampler.bind_training_state_provider(lambda: state)
    for epoch in range(25):
        sampler.set_epoch(epoch)
        microbatches = 0
        for _batch in sampler:
            sampler.acknowledge_next_batch()
            microbatches += 1
            is_full_update = microbatches % 16 == 0
            is_tail_update = microbatches == 906
            if is_full_update or is_tail_update:
                state["global_step"] += 1
            if state["global_step"] >= state["max_steps"]:
                break
    report = sampler.audit_report()
    assert len(report["epochs"]) == 25
    assert [item["sampling_epoch"] for item in report["epochs"]] == list(range(25))
    assert [item["issued_batches"] for item in report["epochs"][:-1]] == [906] * 24
    assert report["epochs"][-1]["issued_batches"] == 512
    assert state["global_step"] == 1400
    live_before = sampler.state_dict()
    checkpoint_state = sampler.build_checkpoint_state(accumulation_remainder=0)
    assert checkpoint_state["resume_replay_batch_ids"] == []
    assert sampler.state_dict() == live_before


def test_real_trainer_nondivisible_terminal_partial_is_formally_resumable(tmp_path):
    continuous = _run_non_divisible_paired_trainer(tmp_path / "continuous")
    interrupted = _run_non_divisible_paired_trainer(tmp_path / "interrupted", stop_after_epoch=1)
    continuous_audit = continuous.get_dann_batch_audit()
    assert continuous_audit["trainer_global_step"] == 2
    assert continuous_audit["epochs"][-1]["issued_batches"] == 4
    assert continuous_audit["epochs"][-1]["processed_batches"] == 4
    checkpoint = find_latest_complete_dann_checkpoint(
        tmp_path / "interrupted", interrupted.dann_batch_sampler
    )
    assert checkpoint.name == "checkpoint-2"
    assert continuous.state.epoch != int(continuous.state.epoch)


def test_real_trainer_nondivisible_continuous_resume_matches_all_states_and_batches(tmp_path):
    # Formal boundary: five physical batches with accumulation two, so the
    # planned traversal is deliberately non-divisible by the accumulation
    # window and the terminal traversal is partial.
    continuous = _run_non_divisible_paired_trainer(tmp_path / "continuous", max_steps=6, paired_count=5, paired_batch_size=1)
    interrupted = _run_non_divisible_paired_trainer(tmp_path / "interrupted", max_steps=6, stop_after_epoch=1.5, paired_count=5, paired_batch_size=1)
    checkpoint = find_latest_complete_dann_checkpoint(
        tmp_path / "interrupted", interrupted.dann_batch_sampler
    )
    resumed = _run_non_divisible_paired_trainer(
        tmp_path / "interrupted", checkpoint=checkpoint, max_steps=6, paired_count=5, paired_batch_size=1
    )
    continuous_audit = continuous.get_dann_batch_audit()
    resumed_audit = resumed.get_dann_batch_audit()
    assert continuous.dann_batch_sampler.epoch_reports[0]["planned_batches"] == 5
    assert resumed.dann_batch_sampler.epoch_reports[0]["planned_batches"] == 5
    assert continuous.args.gradient_accumulation_steps == resumed.args.gradient_accumulation_steps == 2
    assert continuous.state.global_step == resumed.state.global_step == 6
    continuous_processed = [
        batch["logical_batch_id"]
        for epoch in continuous_audit["epochs"]
        for batch in epoch["batches"][:epoch["processed_batches"]]
    ]
    resumed_processed = [
        batch["logical_batch_id"]
        for epoch in resumed_audit["epochs"]
        for batch in epoch["batches"][:epoch["processed_batches"]]
    ]
    assert continuous_processed == resumed_processed
    assert continuous_audit == resumed_audit
    assert continuous.model.state_dict().keys() == resumed.model.state_dict().keys()
    assert all(torch.equal(continuous.model.state_dict()[name], resumed.model.state_dict()[name]) for name in continuous.model.state_dict())
    continuous_optimizer_buffer = io.BytesIO()
    resumed_optimizer_buffer = io.BytesIO()
    torch.save(continuous.optimizer.state_dict(), continuous_optimizer_buffer)
    torch.save(resumed.optimizer.state_dict(), resumed_optimizer_buffer)
    assert continuous_optimizer_buffer.getvalue() == resumed_optimizer_buffer.getvalue()
    assert continuous.lr_scheduler.state_dict() == resumed.lr_scheduler.state_dict()


def test_corrupt_latest_dann_checkpoint_falls_back_to_previous_complete_checkpoint(tmp_path):
    trainer = _run_tiny_paired_trainer(tmp_path / "run", 2)
    latest = find_latest_complete_dann_checkpoint(tmp_path / "run", trainer.dann_batch_sampler)
    assert latest.name == "checkpoint-4"
    (latest / "dann_checkpoint_state.json").write_text("{}", encoding="utf-8")
    fallback = find_latest_complete_dann_checkpoint(tmp_path / "run", trainer.dann_batch_sampler)
    assert fallback.name == "checkpoint-2"


def test_missing_custom_checkpoint_audit_cannot_be_resumed(tmp_path):
    trainer = _run_tiny_paired_trainer(tmp_path / "run", 1)
    checkpoint = find_latest_complete_dann_checkpoint(tmp_path / "run", trainer.dann_batch_sampler)
    (checkpoint / "dann_batch_audit.json").write_text("{}", encoding="utf-8")
    try:
        find_latest_complete_dann_checkpoint(tmp_path / "run", trainer.dann_batch_sampler)
    except RuntimeError as exc:
        assert "no complete" in str(exc).lower()
    else:
        raise AssertionError("checkpoint without a valid custom audit must not resume")


def test_external_control_stage_identity_uses_resolved_model_path():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        external_model = root / "external-control" / "best"
        external_model.mkdir(parents=True)
        (external_model / "config.json").write_text("{}", encoding="utf-8")
        input_file = root / "source_train.jsonl"
        input_file.write_text("source", encoding="utf-8")
        recipe_file = root / "recipe.json"
        recipe_file.write_text("recipe", encoding="utf-8")
        record = build_stage_identity(
            "control_training",
            ["reuse_external_control", str(external_model)],
            {"source_train": input_file},
            hashlib.sha256(b"recipe").hexdigest(),
            external_model,
            external_model,
            "commit-a",
            recipe_path=recipe_file,
        )
        assert record["resolved_model_path"] == str(external_model.resolve())
        assert validate_stage_identity(record) is True
        (external_model / "config.json").write_text("{\"changed\":true}", encoding="utf-8")
        try:
            validate_stage_identity(record)
        except RuntimeError as exc:
            assert "model_artifact_sha256" in str(exc)
        else:
            raise AssertionError("changed external Control model must hard-fail resume")


def test_resume_hard_fails_when_prediction_or_pseudo_artifact_changes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        recipe_file = root / "recipe.json"
        recipe_file.write_text("recipe", encoding="utf-8")
        input_file = root / "source_dev.jsonl"
        input_file.write_text("source-dev", encoding="utf-8")
        model_dir = root / "model" / "best"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        for stage in ("control_source_dev_evaluation", "treatment_source_dev_evaluation", "control_target_pseudo_inference", "treatment_target_pseudo_inference"):
            artifact = root / f"{stage}.jsonl"
            artifact.write_text(stage, encoding="utf-8")
            record = build_stage_identity(
                stage,
                ["python", stage],
                {"source": input_file},
                hashlib.sha256(b"recipe").hexdigest(),
                artifact,
                model_dir,
                "commit-a",
                recipe_path=recipe_file,
            )
            artifact.write_text(stage + "-changed", encoding="utf-8")
            try:
                validate_stage_identity(record)
            except RuntimeError as exc:
                assert stage in str(exc)
            else:
                raise AssertionError(f"modified {stage} artifact must hard-fail resume")


def test_legacy_stage_status_without_per_stage_identity_cannot_resume():
    identity = {"code": "abc", "recipe": "def"}
    with tempfile.TemporaryDirectory() as directory:
        status_path = Path(directory) / "stage_status.json"
        status_path.write_text(
            json.dumps({"schema_version": 1, "identity": identity, "completed_stages": ["treatment_training"]}),
            encoding="utf-8",
        )
        try:
            load_or_initialize_stage_status(status_path, identity, resume=True)
        except RuntimeError as exc:
            assert "per-stage identity" in str(exc)
        else:
            raise AssertionError("legacy stage status must not be resumed")


def test_stage_status_unknown_or_malformed_completed_stage_hard_fails():
    stages = ("control_training", "treatment_training")
    unknown_state = {
        "completed_stages": ["unknown_stage"],
        "stages": {},
    }
    try:
        validate_stage_status_shape(unknown_state, stages)
    except RuntimeError as exc:
        assert "unknown completed stages" in str(exc)
    else:
        raise AssertionError("unknown completed stage must hard-fail")

    malformed_state = {
        "completed_stages": ["control_training"],
        "stages": {"control_training": {"stage": "treatment_training"}},
    }
    try:
        validate_stage_status_shape(malformed_state, stages)
    except RuntimeError as exc:
        assert "wrong stage name" in str(exc)
    else:
        raise AssertionError("malformed completed stage must hard-fail")


def test_v6_single_terminal_lookahead_is_not_counted_as_training():
    report = {
        "trainer_global_step": 1400,
        "trainer_max_steps": 1400,
        "epochs": [{
            "physical_traversal_index": 24,
            "completion": "partial",
            "planned_batches": 906,
            "issued_batches": 321,
            "processed_batches": 320,
            "optimizer_global_step_start": 1380,
            "optimizer_global_step_end": 1400,
            "batches": [{"logical_batch_id": index} for index in range(321)],
        }],
    }
    decision = classify_terminal_lookahead(report, gradient_accumulation_steps=16)
    assert decision["safe"] is True
    assert decision["classification"] == "terminal_lookahead_not_consumed"
    assert decision["dangling_logical_batch_ids"] == [320]
    assert decision["processed_batches"] == 320


def test_formal_dann_gate_accepts_one_journal_proven_terminal_lookahead(tmp_path):
    audit_path = tmp_path / "dann_batch_audit.json"
    # Recreate the legacy V6 prefetch shape: the old sampler did not know the
    # accumulation factor early enough to stop the terminal DataLoader fetch.
    training_state = {"global_step": 0, "max_steps": 1, "gradient_accumulation_steps": None}
    sampler = PairedDomainBatchSampler(
        3,
        3,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        source_row_ids=["s0", "s1", "s2"],
        target_row_ids=["t0", "t1", "t2"],
        audit_path=audit_path,
    )
    sampler.bind_training_state_provider(lambda: training_state)
    iterator = iter(sampler)
    next(iterator)
    sampler.acknowledge_next_batch()
    next(iterator)  # DataLoader lookahead: issued but never passed to training_step.
    training_state["global_step"] = 1
    training_state["gradient_accumulation_steps"] = 1
    sampler.flush_audit_snapshot()

    report = _read_dann_batch_audit(
        audit_path,
        expected_seed=1000,
        expected_source_count=3,
        expected_target_count=3,
        expected_source_row_ids=["s0", "s1", "s2"],
        expected_target_row_ids=["t0", "t1", "t2"],
        require_training_state=True,
        expected_max_steps=1,
        expected_planned_batches=3,
        gradient_accumulation_steps=1,
        allow_legacy=False,
        require_journal=True,
        require_fresh_replay_free=True,
    )
    decision = report["terminal_lookahead_audit"]
    assert decision["safe"] is True
    assert decision["lookahead_not_consumed"] is True
    assert decision["dangling_logical_batch_ids"] == [1]


def test_sampler_prevents_v6_shaped_terminal_lookahead_before_issue():
    training_state = {"global_step": 1380, "max_steps": 1400, "gradient_accumulation_steps": 16}
    sampler = PairedDomainBatchSampler(
        906,
        605,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
    )
    sampler.bind_training_state_provider(lambda: training_state)
    iterator = iter(sampler)
    first = next(iterator)
    assert first
    sampler.acknowledge_next_batch()
    training_state["global_step"] = 1399
    for _ in range(319):
        next(iterator)
        sampler.acknowledge_next_batch()
    with pytest.raises(StopIteration):
        next(iterator)
    report = sampler.audit_report()["epochs"][-1]
    assert report["issued_batches"] == 320
    assert report["processed_batches"] == 320


def test_v6_salvage_audit_requires_and_records_control_only_evidence(tmp_path):
    run_dir = tmp_path / "v6"
    control = run_dir / "control"
    audit_path = control / "dann_batch_audit.json"
    training_state = {"global_step": 0, "max_steps": 1, "gradient_accumulation_steps": None}
    sampler = PairedDomainBatchSampler(
        3,
        3,
        source_batch_size=1,
        target_batch_size=1,
        seed=1000,
        source_row_ids=["s0", "s1", "s2"],
        target_row_ids=["t0", "t1", "t2"],
        audit_path=audit_path,
    )
    sampler.bind_training_state_provider(lambda: training_state)
    iterator = iter(sampler)
    next(iterator)
    sampler.acknowledge_next_batch()
    next(iterator)
    training_state.update(global_step=1, gradient_accumulation_steps=1)
    sampler.flush_audit_snapshot()

    (run_dir / "stage_status.json").write_text(json.dumps({
        "identity": {
            "code_commit": "9caba1c508d096a4d360d7940d8c9d9eb4be8333",
            "scope": {"target_test_access": False, "forbidden": {"target_test": False}},
        },
        "completed_stages": ["control_training"],
    }), encoding="utf-8")
    checkpoint = control / "models" / "extractor" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    torch.save({"accumulation_remainder": 0, "gradients": {}}, checkpoint / "dann_gradient_state.pt")
    best = control / "models" / "extractor" / "best"
    best.mkdir(parents=True)
    (best / "model.safetensors").write_bytes(b"model")
    output = tmp_path / "salvage.json"
    result = audit_v6_control(run_dir, output)
    assert result["status"] == "PASS"
    assert result["classification"]["lookahead_not_consumed"] is True
    assert result["source_run_remains_blocked"] is True
    assert output.is_file()


def test_multiple_terminal_unacknowledged_batches_are_rejected():
    report = {
        "trainer_global_step": 1400,
        "trainer_max_steps": 1400,
        "epochs": [{
            "physical_traversal_index": 24,
            "completion": "partial",
            "planned_batches": 906,
            "issued_batches": 322,
            "processed_batches": 320,
            "optimizer_global_step_start": 1380,
            "optimizer_global_step_end": 1400,
            "batches": [{"logical_batch_id": index} for index in range(322)],
        }],
    }
    assert classify_terminal_lookahead(report, gradient_accumulation_steps=16)["safe"] is False


def test_phase_a_training_worker_runs_in_a_distinct_process(tmp_path):
    result_path = tmp_path / "worker_result.json"
    child_code = (
        "import json,os,sys; "
        "json.dump({'status':'PASS','variant':'treatment','pid':os.getpid()},"
        "open(sys.argv[1],'w',encoding='utf-8'))"
    )
    result = run_isolated_phase_a_worker(
        [sys.executable, "-c", child_code, str(result_path)],
        result_path=result_path,
        expected_variant="treatment",
    )
    assert result["status"] == "PASS"
    assert result["pid"] != os.getpid()


def test_phase_a_input_artifact_hash_matches_serialized_identity_on_windows(tmp_path):
    rows = {
        "source_train": [{"id": 0, "text": "a", "target": "b"}],
        "source_dev": [{"id": 0, "text": "c", "target": "d"}],
        "target_unlabeled": [{"id": 0, "text": "e"}],
    }
    _write_inputs(rows, tmp_path)
    for split, split_rows in rows.items():
        artifact = (tmp_path / "inputs" / f"{split}.jsonl").read_bytes()
        assert artifact == _serialize_rows(split_rows)


def test_phase_a_resume_accepts_crlf_only_and_rejects_semantic_change(tmp_path):
    rows = {
        "source_train": [{"id": 0, "text": "a", "target": "b"}],
        "source_dev": [{"id": 0, "text": "c", "target": "d"}],
        "target_unlabeled": [{"id": 0, "text": "e"}],
    }
    _write_inputs(rows, tmp_path)
    source = tmp_path / "inputs" / "source_train.jsonl"
    source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    _write_inputs(rows, tmp_path, resume=True)
    source.write_text('{"id":0,"text":"changed","target":"b"}\r\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="resume input artifact mismatch"):
        _write_inputs(rows, tmp_path, resume=True)


def test_variant_inputs_are_byte_identical_and_resume_safe(tmp_path):
    rows = {
        "source_train": [{"id": 0, "text": "a", "target": "b"}],
        "source_dev": [{"id": 0, "text": "c", "target": "d"}],
        "target_unlabeled": [{"id": 0, "text": "e"}],
    }
    _write_inputs(rows, tmp_path)
    variant = tmp_path / "treatment"
    _write_variant_inputs(variant, tmp_path)
    _write_variant_inputs(variant, tmp_path, resume=True)
    for split in rows:
        assert (variant / f"{split}.jsonl").read_bytes() == (tmp_path / "inputs" / f"{split}.jsonl").read_bytes()


@pytest.mark.parametrize("global_step,processed", [(1399, 320), (1400, 319)])
def test_terminal_lookahead_rejects_nonterminal_or_incomplete_accumulation(global_step, processed):
    issued = processed + 1
    report = {
        "trainer_global_step": global_step,
        "trainer_max_steps": 1400,
        "epochs": [{
            "physical_traversal_index": 24,
            "completion": "partial",
            "planned_batches": 906,
            "issued_batches": issued,
            "processed_batches": processed,
            "optimizer_global_step_start": 1380,
            "optimizer_global_step_end": global_step,
            "batches": [{"logical_batch_id": index} for index in range(issued)],
        }],
    }
    assert classify_terminal_lookahead(report, gradient_accumulation_steps=16)["safe"] is False


def test_phase_a_worker_rejects_parent_pid_or_wrong_variant(tmp_path):
    for name, payload, expected in (
        ("same_pid", {"status": "PASS", "variant": "treatment", "pid": os.getpid()}, "not process-isolated"),
        ("wrong_variant", {"status": "PASS", "variant": "control", "pid": os.getpid() + 1}, "identity mismatch"),
    ):
        result_path = tmp_path / f"{name}.json"
        child_code = (
            "import json,sys; "
            f"json.dump({payload!r},open(sys.argv[1],'w',encoding='utf-8'))"
        )
        with pytest.raises(RuntimeError, match=expected):
            run_isolated_phase_a_worker(
                [sys.executable, "-c", child_code, str(result_path)],
                result_path=result_path,
                expected_variant="treatment",
            )

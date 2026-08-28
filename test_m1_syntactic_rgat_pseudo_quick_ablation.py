from __future__ import annotations

import hashlib
import json
import tempfile
import copy
from types import SimpleNamespace
from pathlib import Path

import torch
import pytest
from torch.utils.data import Dataset
from transformers import TrainerCallback, T5Config, T5ForConditionalGeneration, Seq2SeqTrainingArguments

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
    prepare_legacy_diagnostic_resume,
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
)


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
    sampler_path.write_text(json.dumps(sampler_state), encoding="utf-8")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    trainer_state_path.write_text(json.dumps(trainer_state), encoding="utf-8")
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
    def __init__(self):
        self.items = []
        for index in range(5):
            self.items.append({"input_ids": [2 + index, 3, 4, 5], "labels": [1, 2, 3], "sample_weight": 1.0, "domain_weight": 1.0})
        for index in range(5):
            self.items.append({"input_ids": [7 + index, 8, 9, 10], "labels": [-100, -100, -100], "sample_weight": 0.0, "domain_weight": 0.0})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _run_non_divisible_paired_trainer(output_dir, checkpoint=None, stop_after_epoch=None):
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
    sampler = PairedDomainBatchSampler(5, 5, source_batch_size=1, target_batch_size=1, seed=1000)
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=3,
        max_steps=2,
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
            if state.global_step >= 2:
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
        train_dataset=_NonDivisiblePairedDataset(),
        data_collator=_tiny_pair_collator,
        dann_batch_sampler=sampler,
        callbacks=callbacks,
    )
    if checkpoint is not None:
        trainer.load_dann_batch_sampler_state(checkpoint)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None)
    return trainer


def test_real_trainer_nondivisible_terminal_processing_gap_is_rejected(tmp_path):
    continuous = _run_non_divisible_paired_trainer(tmp_path / "continuous")
    interrupted = _run_non_divisible_paired_trainer(tmp_path / "interrupted", stop_after_epoch=1)
    continuous_audit = continuous.get_dann_batch_audit()
    assert continuous_audit["trainer_global_step"] == 2
    assert continuous_audit["epochs"][-1]["issued_batches"] == 5
    assert continuous_audit["epochs"][-1]["processed_batches"] == 4
    with pytest.raises(RuntimeError, match="complete|terminal|acknowledged"):
        find_latest_complete_dann_checkpoint(tmp_path / "interrupted", interrupted.dann_batch_sampler)
    assert continuous.state.epoch != int(continuous.state.epoch)


def test_corrupt_latest_dann_checkpoint_falls_back_to_previous_complete_checkpoint(tmp_path):
    trainer = _run_tiny_paired_trainer(tmp_path / "run", 2)
    latest = find_latest_complete_dann_checkpoint(tmp_path / "run", trainer.dann_batch_sampler)
    assert latest.name == "checkpoint-2"
    (latest / "dann_checkpoint_state.json").write_text("{}", encoding="utf-8")
    try:
        find_latest_complete_dann_checkpoint(tmp_path / "run", trainer.dann_batch_sampler)
    except RuntimeError as exc:
        assert "no complete" in str(exc).lower()
    else:
        raise AssertionError("corrupting the only identity-valid checkpoint must hard-fail")


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

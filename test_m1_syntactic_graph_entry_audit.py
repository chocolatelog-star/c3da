import hashlib
import tempfile
from types import SimpleNamespace
from pathlib import Path

import torch

from m1_syntactic_graph_entry_audit import (
    ENTRY_GATE_NAMES,
    EXPECTED_PARSER_SHA256,
    FORMAL_CALLPOINT_PATHS,
    _file_identity,
    assemble_audit_report,
    build_entry_report,
    parameter_state_sha256,
)


def test_entry_report_has_exactly_fifteen_machine_readable_gates():
    gate_values = {name: True for name in ENTRY_GATE_NAMES}
    report = build_entry_report(
        gate_values=gate_values,
        measurements={"optimizer_updates": 0, "scheduler_steps": 0},
        callpoints={
            "source_extractor_training": "PASS",
            "source_dev_evaluation": "PASS",
            "target_unlabeled_dann": "PASS",
            "target_pseudo_inference": "PASS",
        },
        metadata={"target_test_access": False},
    )

    assert len(ENTRY_GATE_NAMES) == 15
    assert list(report["gates"]) == ENTRY_GATE_NAMES
    assert all(value["status"] == "PASS" for value in report["gates"].values())
    assert report["status"] == "PASS"
    assert report["measurements"]["optimizer_updates"] == 0
    assert report["metadata"]["target_test_access"] is False


def test_entry_report_marks_failed_gate_and_never_masks_it():
    gate_values = {name: True for name in ENTRY_GATE_NAMES}
    gate_values["zero_update"] = False
    report = build_entry_report(
        gate_values=gate_values,
        measurements={"optimizer_updates": 1, "scheduler_steps": 0},
        callpoints={},
        metadata={"target_test_access": False},
    )

    assert report["status"] == "BLOCKED"
    assert report["gates"]["zero_update"]["status"] == "FAIL"
    assert report["gates"]["zero_update"]["value"] is False


def test_parameter_state_hash_is_ordered_and_changes_when_a_parameter_changes():
    first = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    second = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    second.load_state_dict(first.state_dict())
    before = parameter_state_sha256(first)
    assert before == parameter_state_sha256(second)

    with torch.no_grad():
        second[0].weight[0, 0] += 1.0
    after = parameter_state_sha256(second)
    assert before != after
    assert len(before) == hashlib.sha256().digest_size * 2


def test_run_audit_report_assembly_reads_gradient_checkpointing_from_model_measurements():
    args = SimpleNamespace(
        fp16=True,
        gradient_checkpointing=True,
        lambda_domain_adv=0.03,
    )
    model_measurements = {
        "measurements": {
            "control_loss": 1.0,
            "treatment_loss": 1.0,
            "repeat_loss": 1.0,
            "control_treatment_max_abs_logit_diff": 0.0,
            "repeat_max_abs_logit_diff": 0.0,
            "aste_gradient_norm": 1.0,
            "dann_gradient_norm": 1.0,
            "target_labels_are_all_ignore_index": True,
            "gradient_checkpointing_enabled": True,
            "gpu_total_memory_bytes": 0,
            "gpu_peak_reserved_bytes": 0,
        },
        "parameter_hash_before": "same",
        "parameter_hash_after": "same",
    }
    cache_measurements = {
        "interruption_observed": True,
        "byte_identical_repeat": True,
        "inspect": {
            "coverage_ok": True,
            "edge_legality_ok": True,
            "reverse_selfloop_ok": True,
            "forbidden_graph_fields": [],
            "node_count": {},
            "edge_count": {},
        },
    }
    recipe = {
        "data_boundary": {
            "target_test_access": False,
            "generator": False,
            "augmentation": False,
            "nli": False,
            "final_aste": False,
        }
    }
    report = assemble_audit_report(
        args=args,
        recipe=recipe,
        manifest={"target_test_access": False},
        cache_measurements=cache_measurements,
        parser_identity={"stanza_version": "1.14.0", "sha256": EXPECTED_PARSER_SHA256},
        callpoints={name: True for name in (
            "source_extractor_training",
            "source_dev_evaluation",
            "target_unlabeled_dann",
            "target_pseudo_inference",
        )},
        model_measurements=model_measurements,
        device=torch.device("cpu"),
        identity_measurements={"all_matches": True},
    )

    assert report["measurements"]["gradient_checkpointing_enabled"] is True
    assert report["status"] == "BLOCKED"
    assert report["gates"]["fp16_entry"]["status"] == "FAIL"


def test_audit_records_the_four_formal_callpoint_paths():
    assert FORMAL_CALLPOINT_PATHS == {
        "source_extractor_training": "t5_absa_train.WeightedSeq2SeqTrainer.compute_loss",
        "source_dev_evaluation": "t5_absa_train.WeightedSeq2SeqTrainer.prediction_step",
        "target_unlabeled_dann": "t5_absa_train.WeightedSeq2SeqTrainer.compute_loss",
        "target_pseudo_inference": "t5_aste_pipeline.generate_texts",
    }


def test_identity_recomputes_file_sha256_and_rejects_a_mismatch():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "artifact.bin"
        path.write_bytes(b"actual")
        identity = _file_identity(path, "0" * 64)

    assert identity["actual_sha256"] == hashlib.sha256(b"actual").hexdigest()
    assert identity["expected_sha256"] == "0" * 64
    assert identity["matches"] is False

import hashlib

import torch

from m1_syntactic_graph_entry_audit import (
    ENTRY_GATE_NAMES,
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

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from m1_syntactic_rgat_pseudo_quick_ablation import (
    PHASE_A_STOP_CODE,
    PHASE_B_REQUEST_CODE,
    audit_control_identity,
    build_phase_a_scope,
    build_variant_config,
    decide_phase_a,
    evaluate_phase_a_gates,
    load_or_initialize_stage_status,
    validate_input_split,
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
        status_path.write_text(json.dumps({"identity": identity, "completed_stages": ["control"]}), encoding="utf-8")
        resumed = load_or_initialize_stage_status(status_path, identity, resume=True)
        assert resumed["completed_stages"] == ["control"]
        assert load_or_initialize_stage_status(status_path, {"code": "changed", "recipe": "def"}, resume=True) is None


def test_target_test_is_rejected_before_execution():
    try:
        validate_input_split("target_test")
    except ValueError as exc:
        assert "target_test" in str(exc)
    else:
        raise AssertionError("target_test was not rejected")

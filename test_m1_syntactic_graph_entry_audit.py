import hashlib
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import torch

from m1_syntactic_graph_entry_audit import (
    AuditConfigurationError,
    ENTRY_GATE_NAMES,
    EXPECTED_PARSER_SHA256,
    FORMAL_CALLPOINT_PATHS,
    _file_identity,
    assemble_audit_report,
    build_entry_report,
    ensure_audit_recipe,
    parameter_state_sha256,
    validate_audit_recipe,
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


def _valid_audit_parameters():
    args = SimpleNamespace(
        source_dataset="laptop14",
        target_dataset="rest15",
        seed=1000,
        lambda_domain_adv=0.03,
        fp16=True,
        gradient_checkpointing=True,
        extractor_train_batch_size=1,
        extractor_eval_batch_size=2,
        dann_source_batch_size=1,
        dann_target_batch_size=1,
        target_pseudo_batch_size=1,
        max_source_length=128,
        max_target_length=96,
    )
    recipe = {
        "source_dataset": "laptop14",
        "target_dataset": "rest15",
        "seed": 1000,
        "training": {
            "extractor_train_batch_size": 1,
            "extractor_eval_batch_size": 2,
            "target_unlabeled_dann": {
                "source_batch_size": 1,
                "target_batch_size": 1,
            },
            "target_pseudo_batch_size": 1,
            "max_source_length": 128,
            "max_target_length": 96,
            "fp16": True,
            "gradient_checkpointing": True,
            "lambda_domain_adv": 0.03,
        },
    }
    return args, recipe


def test_audit_recipe_rejects_wrong_seed_on_cpu():
    args, recipe = _valid_audit_parameters()
    args.seed = 999

    try:
        ensure_audit_recipe(args, recipe)
    except AuditConfigurationError as exc:
        assert exc.validation["matches"]["seed"] is False
        assert exc.validation["all_matches"] is False
    else:
        raise AssertionError("wrong seed must be rejected before GPU/data work")


def test_audit_recipe_rejects_wrong_batch_parameters_on_cpu():
    args, recipe = _valid_audit_parameters()
    args.extractor_train_batch_size = 2
    args.extractor_eval_batch_size = 1
    args.dann_source_batch_size = 2
    args.dann_target_batch_size = 2
    args.target_pseudo_batch_size = 2

    try:
        ensure_audit_recipe(args, recipe)
    except AuditConfigurationError as exc:
        for field in (
            "extractor_train_batch_size",
            "extractor_eval_batch_size",
            "dann_source_batch_size",
            "dann_target_batch_size",
            "target_pseudo_batch_size",
        ):
            assert exc.validation["matches"][field] is False
    else:
        raise AssertionError("wrong batch parameters must be rejected")


def test_audit_recipe_rejects_wrong_lengths_on_cpu():
    args, recipe = _valid_audit_parameters()
    args.max_source_length = 127
    args.max_target_length = 95

    try:
        ensure_audit_recipe(args, recipe)
    except AuditConfigurationError as exc:
        assert exc.validation["matches"]["max_source_length"] is False
        assert exc.validation["matches"]["max_target_length"] is False
    else:
        raise AssertionError("wrong sequence lengths must be rejected")


def test_audit_recipe_report_records_actual_expected_and_matches():
    args, recipe = _valid_audit_parameters()

    validation = validate_audit_recipe(args, recipe)

    assert validation["all_matches"] is True
    assert validation["actual"]["seed"] == 1000
    assert validation["expected"]["seed"] == 1000
    assert validation["matches"]["dann_source_batch_size"] is True


def test_run_audit_wrong_seed_blocks_before_data_or_cuda_on_cpu():
    from m1_syntactic_graph_entry_audit import run_audit

    args, recipe = _valid_audit_parameters()
    args.seed = 999
    args.recipe_path = None
    with tempfile.TemporaryDirectory() as temporary:
        recipe_path = Path(temporary) / "recipe.json"
        recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
        args.recipe_path = str(recipe_path)
        with patch(
            "m1_syntactic_graph_entry_audit._prepare_rows",
            side_effect=AssertionError("data preparation must not run"),
        ), patch(
            "m1_syntactic_graph_entry_audit.torch.cuda.is_available",
            side_effect=AssertionError("CUDA probing must not run"),
        ):
            try:
                run_audit(args)
            except AuditConfigurationError as exc:
                assert exc.validation["matches"]["seed"] is False
            else:
                raise AssertionError("wrong seed must block audit startup")

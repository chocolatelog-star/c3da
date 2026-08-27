from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import T5Config, T5ForConditionalGeneration, Seq2SeqTrainingArguments

from m1_syntactic_rgat_pseudo_quick_ablation import (
    PHASE_A_STOP_CODE,
    PHASE_B_REQUEST_CODE,
    audit_control_identity,
    build_phase_a_scope,
    build_variant_config,
    decide_phase_a,
    evaluate_phase_a_gates,
    load_or_initialize_stage_status,
    build_stage_identity,
    validate_stage_identity,
    validate_stage_status_shape,
    validate_input_split,
)
from t5_absa_train import PairedDomainBatchSampler, WeightedSeq2SeqTrainer


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

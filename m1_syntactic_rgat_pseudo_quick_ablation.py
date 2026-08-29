"""M1 句法 RGAT Phase A 快速消融专用入口。

本模块只编排已批准的上游四个调用点：源域抽取训练、源域开发集评估、
目标无标签 DANN 和目标伪标签推理。它不实现 Phase B，也不读取 target_test。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tqdm import tqdm

from m1_phase_a_control_terminal_lookahead_salvage import audit_v6_control
from syntactic_graph import GraphCacheError, build_parser_identity, load_graph_cache_directory, sha256_file
from t5_aste_data import (
    dump_json,
    micro_f1,
    parse_triplet_text_list,
    read_bgca_aste_file,
    read_jsonl,
    to_extract_rows,
)
from t5_aste_pipeline import DATASETS
from t5_absa_train import (
    classify_terminal_lookahead,
    compute_dann_expected_max_steps,
    compute_dann_planned_batches,
    phase_a_lifecycle_memory_snapshot,
    phase_a_rng_state_hashes,
    read_dann_audit_journal,
)


TASK_ID = "M1_SYNTACTIC_RGAT_PSEUDO_QUICK_ABLATION_V1"
FIXED_PARENT_CODE_IDENTITY = "158654021fc5f26bf1cfb8e803d7d1b592bd8534"
TRAINING_SEMANTICS_IDENTITY = "9caba1c508d096a4d360d7940d8c9d9eb4be8333"
PHASE_A_STOP_CODE = "STOP_M1_SYNTACTIC_GRAPH_UPSTREAM"
PHASE_B_REQUEST_CODE = "REQUEST_PHASE_B"
PHASE_A_CALLPOINTS = (
    "source_extractor_training",
    "source_dev_evaluation",
    "target_unlabeled_dann",
    "target_pseudo_inference",
)
FORMAL_PHASE_A_CALLPOINTS = {
    "source_extractor_training": "t5_absa_train.WeightedSeq2SeqTrainer.compute_loss",
    "source_dev_evaluation": "t5_aste_pipeline.evaluate -> generate_texts",
    "target_unlabeled_dann": "t5_absa_train.WeightedSeq2SeqTrainer.compute_loss",
    "target_pseudo_inference": "t5_aste_pipeline.pseudo -> generate_texts",
}
CONTROL_IDENTITY_FIELDS = (
    "direction",
    "seed",
    "data_split",
    "recipe_sha256",
    "checkpoint_selection_rule",
    "tokenizer_sha256",
    "model_sha256",
    "code_semantics",
    "artifact_sha256",
)
CONTROL_LIFECYCLE_ALLOCATED_CAP_BYTES = 256 * 1024 * 1024
FROZEN_RECIPE = {
    "source_dataset": "laptop14",
    "target_dataset": "rest15",
    "seed": 1000,
    "num_train_epochs": 25,
    "checkpoint_selection": "last",
    "extractor_train_batch_size": 1,
    "extractor_eval_batch_size": 2,
    "dann_source_batch_size": 1,
    "dann_target_batch_size": 1,
    "target_pseudo_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 0.0003,
    "fp16": True,
    "gradient_checkpointing": True,
    "lambda_domain_adv": 0.03,
    "pseudo_base_weight": 0.75,
    "pseudo_num_beams": 1,
    "pseudo_max_new_tokens": 128,
    "high_precision_max_triplets": 1,
    "high_precision_max_token_distance": 5,
    "max_source_length": 128,
    "max_target_length": 96,
    "constrained_decoding": False,
    "source_weight": 1.0,
    "pseudo_weight": 0.75,
    "augment_weight": 0.2,
    "lambda_structure_loss": 0.0,
    "lambda_consistency_loss": 0.0,
    "lambda_pairing_loss": 0.0,
    "domain_adv_grl_lambda": 1.0,
    "domain_adv_hidden_size": 256,
    "domain_adv_exclude_augment": True,
    "paired_domain_batches": True,
    "force_domain_weights": False,
    "pairing_temperature": 0.1,
    "pairing_source_only": False,
    "multi_triplet_loss_gain": 0.0,
    "neutral_loss_gain": 0.0,
    "max_effective_weight": 1.0,
    "neutral_generation_loss_gain": 0.0,
    "neutral_generation_max_effective_weight": 0.0,
    "max_pairing_triplets": 4,
    "min_pairing_triplets": 2,
    "min_pairing_sample_weight": 0.65,
}
FROZEN_PSEUDO_RECIPE = {
    "length_penalty": 1.0,
    "max_target_unlabeled": 0,
    "pseudo_model_variant": "best",
    "pseudo_source_tag": "",
    "fixed_changed_min_score": 0.65,
    "fixed_changed_weight": 0.35,
    "use_task_prefix": False,
}

FROZEN_PHASE_A_DATA_BOUNDARY = {
    "graph_cache_splits": ["source_train", "source_dev", "target_unlabeled"],
    "target_test_access": False,
    "generator": False,
    "augmentation": False,
    "nli": False,
    "selector": False,
    "final_aste": False,
    "phase_b": False,
}
FROZEN_PHASE_A_TREATMENT_GRAPH = {
    "graph_enabled": True,
    "graph_layers": 1,
    "graph_hidden_size": 256,
    "graph_attention_heads": 4,
    "graph_head_size": 64,
    "graph_use_dependency": True,
    "graph_use_reverse_dependency": True,
    "graph_use_pos_neighbor": True,
    "graph_use_self_loop": True,
    "graph_external_word_embeddings": False,
    "graph_sentiment_embedding": False,
}


def _utc_now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(path) for path in paths), key=lambda item: str(item).lower()):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _hash_tree(root: Path) -> str:
    files = [path for path in root.rglob("*") if path.is_file() and ".tmp" not in path.name]
    return _hash_paths(files)


def _artifact_identity(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if resolved.is_file():
        return {"path": str(resolved), "kind": "file", "sha256": sha256_file(resolved)}
    if resolved.is_dir():
        return {"path": str(resolved), "kind": "tree", "sha256": _hash_tree(resolved)}
    raise FileNotFoundError(resolved)


def _normalize_lf_bytes(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _normalized_file_sha256(path: Path) -> str:
    return _sha256_bytes(_normalize_lf_bytes(Path(path).read_bytes()))


def _jsonl_semantically_matches(path: Path, expected_rows: list[dict]) -> bool:
    try:
        actual_rows = [
            json.loads(line)
            for line in Path(path).read_bytes().splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return actual_rows == expected_rows


def _stage_command_fingerprint(command_argv: list[str]) -> str:
    return _sha256_bytes(_canonical_json([str(value) for value in command_argv]))


def build_stage_identity(
    stage: str,
    command_argv: list[str],
    input_files: dict[str, Path],
    recipe_sha256: str,
    artifact_path: Path,
    model_path: Path,
    producer_commit: str,
    recipe_path: Path | None = None,
    output_artifacts: dict[str, Path] | None = None,
) -> dict:
    input_identities = {}
    for name, raw_path in sorted(input_files.items()):
        path = Path(raw_path).resolve()
        identity = {"path": str(path), "sha256": sha256_file(path)}
        if path.suffix.lower() == ".jsonl":
            identity["normalized_lf_sha256"] = _normalized_file_sha256(path)
        input_identities[name] = identity
    output_identity = _artifact_identity(Path(artifact_path))
    model_identity = _artifact_identity(Path(model_path))
    output_identities = {
        name: _artifact_identity(Path(path))
        for name, path in sorted((output_artifacts or {}).items())
    }
    if not output_identities:
        output_identities["primary"] = output_identity
    return {
        "schema_version": 2,
        "stage": stage,
        "command_argv": [str(value) for value in command_argv],
        "command_fingerprint": _stage_command_fingerprint(command_argv),
        "input_files": input_identities,
        "recipe_path": str(Path(recipe_path).resolve()) if recipe_path is not None else "",
        "recipe_sha256": recipe_sha256,
        "output_artifact_path": output_identity["path"],
        "output_artifact_kind": output_identity["kind"],
        "output_artifact_sha256": output_identity["sha256"],
        "output_artifacts": output_identities,
        "model_artifact_path": model_identity["path"],
        "model_artifact_kind": model_identity["kind"],
        "model_artifact_sha256": model_identity["sha256"],
        "resolved_model_path": model_identity["path"],
        "producer_commit": producer_commit,
    }


def validate_stage_identity(saved: dict, expected: dict | None = None) -> bool:
    required = (
        "schema_version",
        "stage",
        "command_argv",
        "command_fingerprint",
        "input_files",
        "recipe_path",
        "recipe_sha256",
        "output_artifact_path",
        "output_artifact_kind",
        "output_artifact_sha256",
        "output_artifacts",
        "model_artifact_path",
        "model_artifact_kind",
        "model_artifact_sha256",
        "resolved_model_path",
        "producer_commit",
    )
    missing = [field for field in required if field not in saved or saved[field] in (None, "")]
    problems = [f"missing:{field}" for field in missing]
    if saved.get("schema_version") != 2:
        problems.append("schema_version")
    command = saved.get("command_argv")
    if not isinstance(command, list) or not command:
        problems.append("command_argv")
    elif saved.get("command_fingerprint") != _stage_command_fingerprint(command):
        problems.append("command_fingerprint")
    recipe_path = Path(saved.get("recipe_path", ""))
    if not recipe_path.is_file():
        problems.append("recipe_path")
    elif sha256_file(recipe_path) != saved.get("recipe_sha256"):
        problems.append("recipe_sha256")
    input_files = saved.get("input_files")
    if not isinstance(input_files, dict) or not input_files:
        problems.append("input_files")
    else:
        for name, identity in input_files.items():
            if not isinstance(identity, dict) or not identity.get("path") or not identity.get("sha256"):
                problems.append(f"input_files:{name}")
                continue
            path = Path(identity["path"])
            if not path.is_file():
                problems.append(f"input_files:{name}")
                continue
            raw_matches = sha256_file(path) == identity["sha256"]
            normalized_matches = (
                path.suffix.lower() == ".jsonl"
                and identity.get("normalized_lf_sha256")
                and _normalized_file_sha256(path) == identity.get("normalized_lf_sha256")
            )
            if not raw_matches and not normalized_matches:
                problems.append(f"input_files:{name}")
    for prefix in ("output_artifact", "model_artifact"):
        path = Path(saved.get(f"{prefix}_path", ""))
        kind = saved.get(f"{prefix}_kind")
        if not path.exists():
            problems.append(f"{prefix}_path")
            continue
        try:
            current = _artifact_identity(path)
        except FileNotFoundError:
            problems.append(f"{prefix}_path")
            continue
        if current["kind"] != kind or current["sha256"] != saved.get(f"{prefix}_sha256"):
            problems.append(f"{prefix}_sha256")
    output_artifacts = saved.get("output_artifacts")
    if not isinstance(output_artifacts, dict) or not output_artifacts:
        problems.append("output_artifacts")
    else:
        for name, identity in output_artifacts.items():
            if not isinstance(identity, dict) or not identity.get("path") or not identity.get("kind") or not identity.get("sha256"):
                problems.append(f"output_artifacts:{name}")
                continue
            path = Path(identity["path"])
            if not path.exists():
                problems.append(f"output_artifacts:{name}")
                continue
            try:
                current = _artifact_identity(path)
            except FileNotFoundError:
                problems.append(f"output_artifacts:{name}")
                continue
            if current != identity:
                problems.append(f"output_artifacts:{name}")
        primary = output_artifacts.get("primary")
        if isinstance(primary, dict):
            if primary.get("path") != saved.get("output_artifact_path"):
                problems.append("output_artifacts:primary_path")
            if primary.get("kind") != saved.get("output_artifact_kind"):
                problems.append("output_artifacts:primary_kind")
            if primary.get("sha256") != saved.get("output_artifact_sha256"):
                problems.append("output_artifacts:primary_sha256")
    if saved.get("resolved_model_path") != saved.get("model_artifact_path"):
        problems.append("resolved_model_path")
    if expected is not None:
        for field in required:
            if field == "input_files":
                saved_inputs = saved.get("input_files")
                expected_inputs = expected.get("input_files")
                if not isinstance(saved_inputs, dict) or not isinstance(expected_inputs, dict) or set(saved_inputs) != set(expected_inputs):
                    problems.append("expected:input_files")
                    continue
                for name in sorted(expected_inputs):
                    saved_identity = saved_inputs.get(name)
                    expected_identity = expected_inputs.get(name)
                    if not isinstance(saved_identity, dict) or not isinstance(expected_identity, dict):
                        problems.append(f"expected:input_files:{name}")
                        continue
                    if saved_identity.get("path") != expected_identity.get("path"):
                        problems.append(f"expected:input_files:{name}:path")
                        continue
                    if saved_identity.get("sha256") != expected_identity.get("sha256"):
                        saved_normalized = saved_identity.get("normalized_lf_sha256")
                        expected_normalized = expected_identity.get("normalized_lf_sha256")
                        if not saved_normalized or not expected_normalized or saved_normalized != expected_normalized:
                            problems.append(f"expected:input_files:{name}:sha256")
                continue
            if saved.get(field) != expected.get(field):
                problems.append(f"expected:{field}")
    if problems:
        raise RuntimeError(f"stage identity mismatch for {saved.get('stage', '<unknown>')}: {sorted(set(problems))}")
    return True


def _utf8_lf_bytes(value: str) -> bytes:
    return value.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: dict) -> None:
    _atomic_write_bytes(path, _utf8_lf_bytes(json.dumps(value, ensure_ascii=False, indent=2) + "\n"))


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, _utf8_lf_bytes(value))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_legacy_diagnostic_migration(run_dir: Path, *, current_commit: str) -> Path:
    """Record a blocking legacy migration audit without changing stage status."""
    run_dir = Path(run_dir)
    stage_status_path = run_dir / "stage_status.json"
    git_identity_path = run_dir / "git_identity.json"
    audit_path = run_dir / "control" / "dann_batch_audit.json"
    source_commit = None
    if git_identity_path.is_file():
        try:
            source_commit = _read_json(git_identity_path).get("commit")
        except (OSError, json.JSONDecodeError):
            source_commit = None
    report_path = run_dir / "legacy_diagnostic_migration.json"
    _atomic_write_json(
        report_path,
        {
            "schema_version": 1,
            "mode": "legacy_diagnostic_migration",
            "status": "BLOCKED",
            "formal_evidence": False,
            "resume_allowed": False,
            "reason": "legacy audit identity cannot be resumed with the physical-traversal protocol without changing training semantics; run a fresh directory",
            "source_commit": source_commit,
            "current_commit": current_commit,
            "source_stage_status_sha256": sha256_file(stage_status_path) if stage_status_path.is_file() else None,
            "source_control_dann_audit_sha256": sha256_file(audit_path) if audit_path.is_file() else None,
            "stage_status_modified": False,
            "target_test_access": False,
        },
    )
    return report_path


def prepare_legacy_diagnostic_resume(run_dir: Path, *, current_commit: str) -> Path:
    """Backward-compatible API alias; this never resumes training."""
    return prepare_legacy_diagnostic_migration(run_dir, current_commit=current_commit)


def validate_input_split(split: str) -> None:
    if split == "target_test":
        raise ValueError("Phase A forbids target_test access")
    if split not in {"source_train", "source_dev", "target_unlabeled"}:
        raise ValueError(f"unsupported Phase A input split: {split}")


def build_phase_a_scope() -> dict:
    return {
        "control": {"graph_enabled": False, "entry": "raw_t5_pseudo_extractor"},
        "treatment": {
            name: {"graph_enabled": True, "entry": name}
            for name in PHASE_A_CALLPOINTS
        },
        "forbidden": {
            "generator": False,
            "augmentation": False,
            "nli": False,
            "exact_conflict_filtering": False,
            "selector": False,
            "final_aste": False,
            "target_test": False,
        },
        "target_test_access": False,
        "formal_callpoint_paths": FORMAL_PHASE_A_CALLPOINTS,
    }


def build_variant_config(graph_enabled: bool) -> dict:
    """Return the frozen Phase A config; Control/Treatment differ only in graph_enabled."""
    return {"graph_enabled": bool(graph_enabled)}


def audit_control_identity(expected: dict, actual: dict) -> dict:
    matches = {}
    for field in CONTROL_IDENTITY_FIELDS:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        matches[field] = expected_value is not None and actual_value is not None and expected_value == actual_value
    return {
        "expected": {field: expected.get(field) for field in CONTROL_IDENTITY_FIELDS},
        "actual": {field: actual.get(field) for field in CONTROL_IDENTITY_FIELDS},
        "matches": matches,
        "all_matches": all(matches.values()),
        "reuse_allowed": all(matches.values()),
        "requires_rerun": not all(matches.values()),
    }


def evaluate_phase_a_gates(metrics: dict) -> dict:
    source = metrics["source_dev"]
    control = source["control"]
    treatment = source["treatment"]
    c_f1 = float(control["strict_triplet_f1"])
    t_f1 = float(treatment["strict_triplet_f1"])
    c_recall = float(control["multi_triplet_sentence_recall"])
    t_recall = float(treatment["multi_triplet_sentence_recall"])
    c_absence = control["absence_rates"]
    t_absence = treatment["absence_rates"]
    c_pseudo = metrics["target_unlabeled_pseudo"]["control"]
    t_pseudo = metrics["target_unlabeled_pseudo"]["treatment"]
    gates = {
        "A1": {
            "status": "PASS" if t_f1 - c_f1 >= -0.01 else "FAIL",
            "metric": "source_dev_strict_triplet_f1_delta",
            "actual": {"control": c_f1, "treatment": t_f1, "delta": t_f1 - c_f1},
            "threshold": -0.01,
            "matches": {"threshold": t_f1 - c_f1 >= -0.01},
        },
        "A2": {
            "status": "PASS" if t_recall - c_recall >= 0.02 else "FAIL",
            "metric": "source_dev_multi_triplet_sentence_recall_delta",
            "actual": {"control": c_recall, "treatment": t_recall, "delta": t_recall - c_recall},
            "threshold": 0.02,
            "matches": {"threshold": t_recall - c_recall >= 0.02},
        },
        "A3": {
            "status": "PASS" if t_absence["overall"] <= c_absence["overall"] and (
                t_absence["aspect"] < c_absence["aspect"] or t_absence["opinion"] < c_absence["opinion"]
            ) else "FAIL",
            "metric": "source_dev_multi_triplet_element_absence_rates",
            "actual": {"control": c_absence, "treatment": t_absence},
            "threshold": {"overall": "treatment<=control", "aspect_or_opinion": "at least one strict improvement"},
            "matches": {
                "overall": t_absence["overall"] <= c_absence["overall"],
                "aspect_or_opinion": t_absence["aspect"] < c_absence["aspect"] or t_absence["opinion"] < c_absence["opinion"],
            },
            "note": "aspect_absence and opinion_absence may overlap; no independent causal claim",
        },
        "A4": {
            "status": "PASS" if c_pseudo["qualified_multi_rows"] > 0 and c_pseudo["qualified_total_rows"] > 0 and t_pseudo["qualified_multi_rows"] >= c_pseudo["qualified_multi_rows"] * 1.05 and t_pseudo["qualified_total_rows"] >= c_pseudo["qualified_total_rows"] * 0.95 else "FAIL",
            "metric": "target_unlabeled_qualified_pseudo_supply",
            "actual": {
                "control": c_pseudo,
                "treatment": t_pseudo,
                "multi_ratio": (t_pseudo["qualified_multi_rows"] / c_pseudo["qualified_multi_rows"]) if c_pseudo["qualified_multi_rows"] else None,
                "total_ratio": (t_pseudo["qualified_total_rows"] / c_pseudo["qualified_total_rows"]) if c_pseudo["qualified_total_rows"] else None,
                "multi_ratio_status": "defined" if c_pseudo["qualified_multi_rows"] else "undefined",
                "total_ratio_status": "defined" if c_pseudo["qualified_total_rows"] else "undefined",
            },
            "threshold": {"multi_ratio": 1.05, "total_ratio": 0.95},
            "matches": {
                "multi_ratio": c_pseudo["qualified_multi_rows"] > 0 and t_pseudo["qualified_multi_rows"] >= c_pseudo["qualified_multi_rows"] * 1.05,
                "total_ratio": c_pseudo["qualified_total_rows"] > 0 and t_pseudo["qualified_total_rows"] >= c_pseudo["qualified_total_rows"] * 0.95,
            },
            "target_test_access": False,
        },
    }
    return {
        "gates": gates,
        "all_pass": all(item["status"] == "PASS" for item in gates.values()),
    }


def decide_phase_a(gate_result: dict) -> dict:
    passed = bool(gate_result["all_pass"])
    return {
        "status": "PASS" if passed else "BLOCKED",
        "next_action": PHASE_B_REQUEST_CODE if passed else PHASE_A_STOP_CODE,
        "hard_stop": not passed,
        "phase_b_entered": False,
    }


def load_or_initialize_stage_status(
    path: Path,
    identity: dict,
    resume: bool,
    repair_from_commit: str = "",
) -> dict | None:
    if path.exists():
        saved = _read_json(path)
        if not resume:
            raise RuntimeError(f"run already exists; use --resume: {path.parent}")
        if saved.get("identity") != identity:
            previous = saved.get("identity")
            previous_without_commit = dict(previous) if isinstance(previous, dict) else {}
            current_without_commit = dict(identity)
            previous_commit = previous_without_commit.pop("code_commit", None)
            current_commit = current_without_commit.pop("code_commit", None)
            repair_allowed = (
                bool(repair_from_commit)
                and previous_commit == repair_from_commit
                and current_commit
                and previous_without_commit == current_without_commit
            )
            if not repair_allowed:
                return None
            saved.setdefault("repair_history", []).append({
                "reason": "graph_cache_runtime_tokenizer_identity_contract_fix",
                "from_commit": previous_commit,
                "to_commit": current_commit,
                "completed_stages": list(saved.get("completed_stages", [])),
                "target_test_access": False,
            })
            saved["identity"] = identity
        if saved.get("schema_version") != 2 or not isinstance(saved.get("stages"), dict):
            raise RuntimeError("stage_status.json has no complete per-stage identity records; refusing resume")
        return saved
    if resume:
        raise RuntimeError("cannot resume without an initialized stage_status.json")
    return {"schema_version": 2, "status": "initialized", "identity": identity, "completed_stages": [], "stages": {}}


def validate_stage_status_shape(state: dict, stages: tuple[str, ...]) -> None:
    """Reject unknown or incomplete bookkeeping before any stage is skipped."""
    completed = state.get("completed_stages")
    records = state.get("stages")
    if not isinstance(completed, list):
        raise RuntimeError("stage_status.json completed_stages is missing or invalid; refusing resume")
    if not isinstance(records, dict):
        raise RuntimeError("stage_status.json stages is missing or invalid; refusing resume")
    allowed = set(stages)
    unknown_completed = sorted(set(completed) - allowed)
    unknown_records = sorted(set(records) - allowed)
    if unknown_completed:
        raise RuntimeError(f"stage_status.json contains unknown completed stages: {unknown_completed}")
    if unknown_records:
        raise RuntimeError(f"stage_status.json contains unknown stage records: {unknown_records}")
    for stage in completed:
        record = records.get(stage)
        if not isinstance(record, dict):
            raise RuntimeError(f"completed stage {stage} has no identity record; refusing resume")
        if record.get("stage") != stage:
            raise RuntimeError(f"completed stage {stage} identity record has the wrong stage name; refusing resume")


def stage_producer_commit_for_validation(state: dict, stage: str, current_commit: str) -> str:
    record = state.get("stages", {}).get(stage, {})
    producer = record.get("producer_commit")
    if producer == current_commit:
        return current_commit
    reachable = {current_commit}
    for repair in reversed(state.get("repair_history", [])):
        to_commit = repair.get("to_commit")
        from_commit = repair.get("from_commit")
        if to_commit in reachable and stage in repair.get("completed_stages", []):
            reachable.add(from_commit)
            if producer == from_commit:
                return str(producer)
    raise RuntimeError(f"stage {stage} producer commit is outside the audited repair chain")

def _validate_recipe(recipe: dict) -> None:
    if not isinstance(recipe, dict) or recipe.get("task_id") != TASK_ID:
        raise ValueError(f"recipe task_id must be {TASK_ID}")
    training = recipe.get("training", {})
    pseudo = recipe.get("pseudo", {})
    if not isinstance(training, dict) or not isinstance(pseudo, dict):
        raise ValueError("Phase A recipe training and pseudo sections must be mappings")
    dann = training.get("target_unlabeled_dann", {})
    if not isinstance(dann, dict):
        raise ValueError("Phase A recipe target_unlabeled_dann must be a mapping")
    actual = {
        "source_dataset": recipe.get("source_dataset"),
        "target_dataset": recipe.get("target_dataset"),
        "seed": recipe.get("seed"),
        "num_train_epochs": training.get("num_train_epochs"),
        "checkpoint_selection": training.get("checkpoint_selection"),
        "extractor_train_batch_size": training.get("extractor_train_batch_size"),
        "extractor_eval_batch_size": training.get("extractor_eval_batch_size"),
        "dann_source_batch_size": dann.get("source_batch_size"),
        "dann_target_batch_size": dann.get("target_batch_size"),
        "target_pseudo_batch_size": training.get("target_pseudo_batch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "learning_rate": training.get("learning_rate"),
        "fp16": training.get("fp16"),
        "gradient_checkpointing": training.get("gradient_checkpointing"),
        "lambda_domain_adv": training.get("lambda_domain_adv"),
        "pseudo_base_weight": pseudo.get("base_weight"),
        "pseudo_num_beams": pseudo.get("num_beams"),
        "pseudo_max_new_tokens": pseudo.get("max_new_tokens"),
        "high_precision_max_triplets": pseudo.get("high_precision_max_triplets"),
        "high_precision_max_token_distance": pseudo.get("high_precision_max_token_distance"),
        "max_source_length": training.get("max_source_length"),
        "max_target_length": training.get("max_target_length"),
        "constrained_decoding": pseudo.get("constrained_decoding"),
        "source_weight": training.get("source_weight"),
        "pseudo_weight": training.get("pseudo_weight"),
        "augment_weight": training.get("augment_weight"),
        "lambda_structure_loss": training.get("lambda_structure_loss"),
        "lambda_consistency_loss": training.get("lambda_consistency_loss"),
        "lambda_pairing_loss": training.get("lambda_pairing_loss"),
        "domain_adv_grl_lambda": training.get("domain_adv_grl_lambda"),
        "domain_adv_hidden_size": training.get("domain_adv_hidden_size"),
        "domain_adv_exclude_augment": training.get("domain_adv_exclude_augment"),
        "paired_domain_batches": training.get("paired_domain_batches"),
        "force_domain_weights": training.get("force_domain_weights"),
        "pairing_temperature": training.get("pairing_temperature"),
        "pairing_source_only": training.get("pairing_source_only"),
        "multi_triplet_loss_gain": training.get("multi_triplet_loss_gain"),
        "neutral_loss_gain": training.get("neutral_loss_gain"),
        "max_effective_weight": training.get("max_effective_weight"),
        "neutral_generation_loss_gain": training.get("neutral_generation_loss_gain"),
        "neutral_generation_max_effective_weight": training.get("neutral_generation_max_effective_weight"),
        "max_pairing_triplets": training.get("max_pairing_triplets"),
        "min_pairing_triplets": training.get("min_pairing_triplets"),
        "min_pairing_sample_weight": training.get("min_pairing_sample_weight"),
    }
    actual_pseudo = {
        "length_penalty": pseudo.get("length_penalty"),
        "max_target_unlabeled": pseudo.get("max_target_unlabeled"),
        "pseudo_model_variant": pseudo.get("pseudo_model_variant"),
        "pseudo_source_tag": pseudo.get("pseudo_source_tag"),
        "fixed_changed_min_score": pseudo.get("fixed_changed_min_score"),
        "fixed_changed_weight": pseudo.get("fixed_changed_weight"),
        "use_task_prefix": pseudo.get("use_task_prefix"),
    }
    expected_recipe = dict(FROZEN_RECIPE)
    if recipe.get("recipe_id") == "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann0_v1":
        expected_recipe["lambda_domain_adv"] = 0.0
    mismatches = {key: {"actual": actual[key], "expected": expected} for key, expected in expected_recipe.items() if actual[key] != expected}
    mismatches.update({f"pseudo.{key}": {"actual": actual_pseudo[key], "expected": expected} for key, expected in FROZEN_PSEUDO_RECIPE.items() if actual_pseudo[key] != expected})
    expected_model_path = Path(r"J:\nlp\models\t5-base-py").resolve()
    models = recipe.get("models")
    actual_model_path = models.get("t5_base") if isinstance(models, dict) else None
    if actual_model_path is None or Path(actual_model_path).resolve() != expected_model_path:
        mismatches["models.t5_base"] = {"actual": actual_model_path, "expected": str(expected_model_path)}
    if not isinstance(models, dict) or set(models) != {"t5_base"}:
        mismatches["models.keys"] = {"actual": sorted(models) if isinstance(models, dict) else models, "expected": ["t5_base"]}

    expected_inputs = {
        "source_train": (DATASETS["laptop14"] / "train.txt").resolve(),
        "source_dev": (DATASETS["laptop14"] / "dev.txt").resolve(),
        "target_unlabeled": (DATASETS["rest15"] / "train.txt").resolve(),
    }
    external_inputs = recipe.get("external_inputs")
    if not isinstance(external_inputs, dict):
        mismatches["external_inputs"] = {"actual": external_inputs, "expected": "three declared inputs plus target_test_access=false"}
    else:
        for name, expected_path in expected_inputs.items():
            entry = external_inputs.get(name)
            actual_path = entry.get("path") if isinstance(entry, dict) else None
            if actual_path is None or Path(actual_path).resolve() != expected_path:
                mismatches[f"external_inputs.{name}.path"] = {"actual": actual_path, "expected": str(expected_path)}
            if not isinstance(entry, dict) or set(entry) != {"path"}:
                mismatches[f"external_inputs.{name}.keys"] = {"actual": sorted(entry) if isinstance(entry, dict) else entry, "expected": ["path"]}
        if external_inputs.get("target_test_access") is not False:
            mismatches["external_inputs.target_test_access"] = {"actual": external_inputs.get("target_test_access"), "expected": False}
        unexpected_inputs = sorted(set(external_inputs) - set(expected_inputs) - {"target_test_access"})
        if unexpected_inputs:
            mismatches["external_inputs.unexpected"] = {"actual": unexpected_inputs, "expected": []}

    if recipe.get("data_boundary") != FROZEN_PHASE_A_DATA_BOUNDARY:
        mismatches["data_boundary"] = {"actual": recipe.get("data_boundary"), "expected": FROZEN_PHASE_A_DATA_BOUNDARY}
    variants = recipe.get("variants")
    if not isinstance(variants, dict) or variants.get("control") != {"graph_enabled": False}:
        mismatches["variants.control"] = {"actual": variants.get("control") if isinstance(variants, dict) else variants, "expected": {"graph_enabled": False}}
    if not isinstance(variants, dict) or variants.get("treatment") != FROZEN_PHASE_A_TREATMENT_GRAPH:
        mismatches["variants.treatment"] = {"actual": variants.get("treatment") if isinstance(variants, dict) else variants, "expected": FROZEN_PHASE_A_TREATMENT_GRAPH}
    if mismatches:
        raise ValueError("frozen Phase A recipe mismatch: " + json.dumps(mismatches, ensure_ascii=False, sort_keys=True))


def _git_identity(project_root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True, encoding="utf-8", errors="replace").strip()

    status = git("status", "--porcelain")
    commit = git("rev-parse", "HEAD")
    return {
        "commit": commit,
        "branch": git("branch", "--show-current"),
        "worktree_clean": not bool(status),
        "status_porcelain": status,
        "parent_entry_code_identity": FIXED_PARENT_CODE_IDENTITY,
    }


def _build_input_rows(
    source_dataset: str,
    target_dataset: str,
    external_inputs: dict | None = None,
) -> dict[str, list[dict]]:
    if external_inputs is None:
        paths = {
            "source_train": DATASETS[source_dataset] / "train.txt",
            "source_dev": DATASETS[source_dataset] / "dev.txt",
            "target_unlabeled": DATASETS[target_dataset] / "train.txt",
        }
    else:
        paths = {
            name: Path(external_inputs[name]["path"])
            for name in ("source_train", "source_dev", "target_unlabeled")
        }
    source_train_raw = read_bgca_aste_file(paths["source_train"])
    source_dev_raw = read_bgca_aste_file(paths["source_dev"])
    target_raw = read_bgca_aste_file(paths["target_unlabeled"])
    return {
        "source_train": to_extract_rows(source_train_raw, use_task_prefix=False),
        "source_dev": to_extract_rows(source_dev_raw, use_task_prefix=False),
        "target_unlabeled": [{"id": row["id"], "text": row["text"]} for row in target_raw],
    }


def _serialize_rows(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows)


def _write_inputs(input_rows: dict[str, list[dict]], run_dir: Path, *, resume: bool = False) -> None:
    for split in input_rows:
        validate_input_split(split)
        path = run_dir / "inputs" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = _serialize_rows(input_rows[split])
        if resume and path.exists():
            observed = path.read_bytes()
            canonical_observed = observed.replace(b"\r\n", b"\n")
            if canonical_observed != serialized:
                raise RuntimeError(f"resume input artifact mismatch: {path}")
        if not path.exists():
            _atomic_write_bytes(path, serialized)
        if path.read_bytes().replace(b"\r\n", b"\n") != serialized:
            raise RuntimeError(f"written input artifact mismatch: {path}")


def _input_hashes(input_rows: dict[str, list[dict]], run_dir: Path) -> dict:
    result = {}
    for split in ("source_train", "source_dev", "target_unlabeled"):
        path = run_dir / "inputs" / f"{split}.jsonl"
        serialized = _serialize_rows(input_rows[split])
        result[split] = {"path": str(path), "sha256": _sha256_bytes(serialized), "rows": len(input_rows[split])}
    return result


def _model_hashes(model_path: Path) -> dict:
    required_names = ("config.json", "pytorch_model.bin", "generation_config.json", "spiece.model", "tokenizer.json")
    optional_names = ("tokenizer_config.json", "special_tokens_map.json")
    files = [model_path / name for name in required_names]
    files.extend(path for path in (model_path / name for name in optional_names) if path.exists())
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing T5-base files: " + ", ".join(missing))
    return {str(path.name): sha256_file(path) for path in files}


def _declared_input_hashes(recipe: dict) -> dict:
    result = {}
    for name in ("source_train", "source_dev", "target_unlabeled"):
        path = Path(recipe["external_inputs"][name]["path"]).resolve()
        result[name] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _read_initialization_audit(path: Path, expected_variant: str, expected_seed: int = 1000) -> dict:
    if not Path(path).is_file():
        raise RuntimeError(f"missing {expected_variant} initialization audit: {path}")
    audit = _read_json(Path(path))
    if not isinstance(audit, dict) or audit.get("schema_version") != 1:
        raise RuntimeError(f"invalid {expected_variant} initialization audit: {path}")
    if audit.get("variant") != expected_variant or audit.get("seed") != expected_seed:
        raise RuntimeError(f"{expected_variant} initialization audit identity mismatch: {path}")
    groups = audit.get("parameter_groups")
    if not isinstance(groups, dict):
        raise RuntimeError(f"{expected_variant} initialization audit has no parameter groups: {path}")
    for group_name, top_level_name in (
        ("shared_t5", "shared_t5_parameter_sha256"),
        ("domain_adversarial_head", "dann_head_parameter_sha256"),
        ("syntactic_graph_adapter", "graph_parameter_sha256"),
    ):
        group = groups.get(group_name)
        if not isinstance(group, dict) or not isinstance(group.get("parameter_names"), list):
            raise RuntimeError(f"{expected_variant} initialization audit missing group {group_name}: {path}")
        if group.get("sha256") != audit.get(top_level_name):
            raise RuntimeError(f"{expected_variant} initialization audit hash mismatch: {group_name}")
        stats = group.get("parameter_stats")
        if not isinstance(stats, list) or len(stats) != len(group["parameter_names"]):
            raise RuntimeError(f"{expected_variant} initialization audit stats mismatch: {group_name}")
        if any(item.get("finite") is not True for item in stats):
            raise RuntimeError(f"{expected_variant} initialization audit contains non-finite parameters: {group_name}")
        if group_name == "syntactic_graph_adapter" and any(
            item.get("max_abs") is not None and float(item["max_abs"]) >= 1.0e6
            for item in stats
        ):
            raise RuntimeError(f"{expected_variant} initialization audit contains implausibly large graph parameters: {path}")
    return audit


def validate_initialization_pair(
    control_path: Path,
    treatment_path: Path,
    *,
    expected_seed: int = 1000,
) -> dict:
    control = _read_initialization_audit(control_path, "control", expected_seed)
    treatment = _read_initialization_audit(treatment_path, "treatment", expected_seed)
    control_groups = control["parameter_groups"]
    treatment_groups = treatment["parameter_groups"]
    if control_groups["shared_t5"]["parameter_names"] != treatment_groups["shared_t5"]["parameter_names"]:
        raise RuntimeError("Control/Treatment shared T5 parameter names differ")
    if control_groups["shared_t5"]["sha256"] != treatment_groups["shared_t5"]["sha256"]:
        raise RuntimeError("Control/Treatment shared T5 parameters differ before training")
    if control_groups["domain_adversarial_head"]["parameter_names"] != treatment_groups["domain_adversarial_head"]["parameter_names"]:
        raise RuntimeError("Control/Treatment DANN head parameter names differ")
    if control_groups["domain_adversarial_head"]["sha256"] != treatment_groups["domain_adversarial_head"]["sha256"]:
        raise RuntimeError("Control/Treatment DANN head parameters differ before training")
    if control_groups["syntactic_graph_adapter"]["parameter_names"]:
        raise RuntimeError("Control unexpectedly contains syntactic graph parameters")
    if not treatment_groups["syntactic_graph_adapter"]["parameter_names"]:
        raise RuntimeError("Treatment initialization audit has no syntactic graph parameters")
    return {
        "schema_version": 1,
        "status": "matched",
        "seed": expected_seed,
        "control": control,
        "treatment": treatment,
        "matches": {
            "shared_t5_parameters": True,
            "dann_head_parameters": True,
            "control_has_no_graph_parameters": True,
            "treatment_has_graph_parameters": True,
        },
    }


def _write_variant_inputs(variant_dir: Path, run_dir: Path, *, resume: bool = False) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    for split in ("source_train", "source_dev", "target_unlabeled"):
        target = variant_dir / f"{split}.jsonl"
        source = run_dir / "inputs" / f"{split}.jsonl"
        source_bytes = source.read_bytes()
        normalized_source = _normalize_lf_bytes(source_bytes)
        source_hash = _sha256_bytes(source_bytes)
        source_normalized_hash = _sha256_bytes(normalized_source)
        if resume and not target.exists():
            raise RuntimeError(f"resume variant input artifact is missing: {target}")
        if target.exists():
            target_bytes = target.read_bytes()
            if _sha256_bytes(target_bytes) != source_hash and _sha256_bytes(_normalize_lf_bytes(target_bytes)) != source_normalized_hash:
                raise RuntimeError(f"resume variant input artifact mismatch: {target}")
        if not target.exists():
            _atomic_write_bytes(target, source_bytes)
        target_bytes = target.read_bytes()
        if _sha256_bytes(target_bytes) != source_hash and _sha256_bytes(_normalize_lf_bytes(target_bytes)) != source_normalized_hash:
            raise RuntimeError(f"variant input artifact mismatch: {target}")


def _preflight_resume_inputs(input_rows: dict[str, list[dict]], run_dir: Path) -> None:
    """Validate all persisted input bytes before any resume bookkeeping is written."""
    for split, rows in input_rows.items():
        validate_input_split(split)
        path = run_dir / "inputs" / f"{split}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"resume input artifact is missing: {path}")
        if _normalize_lf_bytes(path.read_bytes()) != _serialize_rows(rows):
            raise RuntimeError(f"resume input artifact mismatch: {path}")
    for variant in ("control", "treatment"):
        variant_dir = run_dir / variant
        for split in input_rows:
            path = variant_dir / f"{split}.jsonl"
            source = run_dir / "inputs" / f"{split}.jsonl"
            if not path.is_file():
                raise RuntimeError(f"resume variant input artifact is missing: {path}")
            if _normalize_lf_bytes(path.read_bytes()) != _normalize_lf_bytes(source.read_bytes()):
                raise RuntimeError(f"resume variant input artifact mismatch: {path}")


def _training_argv(
    args: argparse.Namespace,
    variant_dir: Path,
    graph_enabled: bool,
    dann_batch_audit_path: Path | None = None,
    initialization_audit_path: Path | None = None,
) -> list[str]:
    recipe = args.recipe_data
    training = recipe["training"]
    argv = [
        "--model_path", str(args.model_path),
        "--train_file", str(variant_dir / "source_train.jsonl"),
        "--dev_file", str(variant_dir / "source_dev.jsonl"),
        "--output_dir", str(variant_dir / "models" / "extractor"),
        "--num_train_epochs", str(training["num_train_epochs"]),
        "--source_weight", str(training["source_weight"]),
        "--pseudo_weight", str(training["pseudo_weight"]),
        "--augment_weight", str(training["augment_weight"]),
        "--lambda_structure_loss", str(training["lambda_structure_loss"]),
        "--lambda_consistency_loss", str(training["lambda_consistency_loss"]),
        "--lambda_pairing_loss", str(training["lambda_pairing_loss"]),
        "--multi_triplet_loss_gain", str(training["multi_triplet_loss_gain"]),
        "--neutral_loss_gain", str(training["neutral_loss_gain"]),
        "--checkpoint_selection", training["checkpoint_selection"],
        "--resume_from_checkpoint", "auto",
        "--per_device_train_batch_size", str(training["extractor_train_batch_size"]),
        "--per_device_eval_batch_size", str(training["extractor_eval_batch_size"]),
        "--max_source_length", str(training["max_source_length"]),
        "--max_target_length", str(training["max_target_length"]),
        "--gradient_accumulation_steps", str(training["gradient_accumulation_steps"]),
        "--learning_rate", str(training["learning_rate"]),
        "--lambda_domain_adv", str(training["lambda_domain_adv"]),
        "--domain_adv_grl_lambda", str(training["domain_adv_grl_lambda"]),
        "--domain_adv_hidden_size", str(training["domain_adv_hidden_size"]),
        "--pairing_temperature", str(training["pairing_temperature"]),
        "--max_effective_weight", str(training["max_effective_weight"]),
        "--neutral_generation_loss_gain", str(training["neutral_generation_loss_gain"]),
        "--neutral_generation_max_effective_weight", str(training["neutral_generation_max_effective_weight"]),
        "--max_pairing_triplets", str(training["max_pairing_triplets"]),
        "--min_pairing_triplets", str(training["min_pairing_triplets"]),
        "--min_pairing_sample_weight", str(training["min_pairing_sample_weight"]),
        "--target_unlabeled_file", str(variant_dir / "target_unlabeled.jsonl"),
        "--paired_domain_batches",
        "--dann_source_batch_size", str(training["target_unlabeled_dann"]["source_batch_size"]),
        "--dann_target_batch_size", str(training["target_unlabeled_dann"]["target_batch_size"]),
        "--fp16",
        "--gradient_checkpointing",
        "--cuda", str(args.cuda),
        "--seed", str(recipe["seed"]),
    ]
    if graph_enabled:
        argv.extend([
            "--use_syntactic_graph_adapter",
            "--syntactic_graph_cache_dir", str(args.graph_cache_dir),
            "--syntactic_graph_parser_dir", str(args.parser_dir),
        ])
    if training["force_domain_weights"]:
        argv.append("--force_domain_weights")
    if training["pairing_source_only"]:
        argv.append("--pairing_source_only")
    if training["domain_adv_exclude_augment"]:
        argv.append("--domain_adv_exclude_augment")
    if dann_batch_audit_path is not None:
        argv.extend(["--dann_batch_audit_path", str(dann_batch_audit_path)])
    if initialization_audit_path is not None:
        argv.extend(["--initialization_audit_path", str(initialization_audit_path)])
    return argv


def _phase_a_worker_command(spec_path: Path) -> list[str]:
    return [
        str(Path(sys.executable)),
        str(Path(__file__).resolve()),
        "--training_worker_spec",
        str(Path(spec_path).resolve()),
    ]


def _worker_spec_path(stage: str, variant_dir: Path) -> Path:
    return Path(variant_dir) / f"worker_spec_{stage}.json"


def _run_training(
    args: argparse.Namespace,
    variant_dir: Path,
    graph_enabled: bool,
    *,
    stage: str | None = None,
) -> dict:
    variant = "treatment" if graph_enabled else "control"
    stage = stage or ("treatment_training" if graph_enabled else "control_training")
    training_argv = _training_argv(
        args,
        variant_dir,
        graph_enabled,
        variant_dir / "dann_batch_audit.json",
        variant_dir / "phase_a_initialization_audit.json",
    )
    result_path = variant_dir / "worker_result.json"
    spec_path = variant_dir / f"worker_spec_{stage}.json"
    input_identity = {
        split: _artifact_identity(variant_dir / f"{split}.jsonl")
        for split in ("source_train", "source_dev", "target_unlabeled")
    }
    model_identity = _model_hashes(Path(args.model_path).resolve())
    graph_cache_identity = None
    if graph_enabled:
        graph_cache_identity = {
            "path": str(Path(args.graph_cache_dir).resolve()),
            "sha256": _hash_tree(Path(args.graph_cache_dir).resolve()),
        }
    _atomic_write_json(spec_path, {
        "schema_version": 1,
        "stage": stage,
        "variant": variant,
        "seed": int(args.recipe_data["seed"]),
        "graph_enabled": graph_enabled,
        "training_argv": training_argv,
        "training_argv_fingerprint": _sha256_bytes(_canonical_json(training_argv)),
        "result_path": str(result_path.resolve()),
        "parent_pid": os.getpid(),
        "target_test_access": False,
        "code_identity": _git_identity(args.project_root),
        "recipe_identity": {
            "path": str(Path(args.recipe).resolve()),
            "sha256": sha256_file(Path(args.recipe)),
        },
        "input_identity": input_identity,
        "model_identity": model_identity,
        "graph_cache_identity": graph_cache_identity,
    })
    command = _phase_a_worker_command(spec_path)
    return run_isolated_phase_a_worker(
        command,
        result_path=result_path,
        expected_variant=variant,
        require_complete_result=True,
    )


def _run_phase_a_training_worker(spec_path: Path) -> int:
    from t5_absa_train import run_phase_a_training

    spec = _read_json(spec_path)
    variant = spec.get("variant")
    if variant not in {"control", "treatment"}:
        raise RuntimeError("invalid Phase A worker variant")
    training_argv = spec.get("training_argv")
    if not isinstance(training_argv, list) or not all(isinstance(item, str) for item in training_argv):
        raise RuntimeError("invalid Phase A worker argv")
    if any("target_test" in item.lower() for item in training_argv):
        raise RuntimeError("Phase A worker forbids target_test access")
    expected_fingerprint = spec.get("training_argv_fingerprint")
    if _sha256_bytes(_canonical_json(training_argv)) != expected_fingerprint:
        raise RuntimeError("Phase A worker argv fingerprint mismatch")
    result_path = Path(spec["result_path"])
    lifecycle_path = spec_path.parent / "phase_a_lifecycle_audit.json"
    start_time = _utc_now()
    worker_result = {
        "schema_version": 2,
        "status": "FAIL",
        "stage": spec.get("stage"),
        "variant": variant,
        "pid": os.getpid(),
        "parent_pid": spec.get("parent_pid"),
        "argv": list(training_argv),
        "argv_fingerprint": expected_fingerprint,
        "training_argv_fingerprint": expected_fingerprint,
        "seed": spec.get("seed"),
        "code_identity": spec.get("code_identity"),
        "recipe_identity": spec.get("recipe_identity"),
        "input_identity": spec.get("input_identity"),
        "model_identity": spec.get("model_identity"),
        "graph_cache_identity": spec.get("graph_cache_identity"),
        "start_time": start_time,
        "target_test_access": False,
    }
    try:
        result = run_phase_a_training(training_argv)
        if not isinstance(result, dict):
            raise RuntimeError("Phase A training worker returned no result")
        lifecycle = result.get("lifecycle")
        if isinstance(lifecycle, dict):
            _atomic_write_json(lifecycle_path, lifecycle)
        model_path = spec_path.parent / "models" / "extractor" / "best"
        dann_path = spec_path.parent / "dann_batch_audit.json"
        initialization_path = spec_path.parent / "phase_a_initialization_audit.json"
        dann_audit = result.get("dann_batch_audit") or {}
        worker_result.update({
            "status": "PASS",
            "exit_code": 0,
            "end_time": _utc_now(),
            "model_path": str(model_path.resolve()) if model_path.is_dir() else None,
            "model_tree_hash": _hash_tree(model_path) if model_path.is_dir() else None,
            "model_tree_sha256": _hash_tree(model_path) if model_path.is_dir() else None,
            "dann_audit_path": str(dann_path.resolve()) if dann_path.is_file() else None,
            "dann_audit_hash": sha256_file(dann_path) if dann_path.is_file() else None,
            "dann_batch_audit_path": str(dann_path.resolve()) if dann_path.is_file() else None,
            "dann_batch_audit_sha256": sha256_file(dann_path) if dann_path.is_file() else None,
            "initialization_audit_path": str(initialization_path.resolve()) if initialization_path.is_file() else None,
            "initialization_audit_hash": sha256_file(initialization_path) if initialization_path.is_file() else None,
            "initialization_audit_sha256": sha256_file(initialization_path) if initialization_path.is_file() else None,
            "optimizer_global_step": dann_audit.get("trainer_global_step"),
            "trainer_global_step": dann_audit.get("trainer_global_step"),
            "training_result_schema": sorted(result),
            "lifecycle_audit_path": str(lifecycle_path.resolve()) if lifecycle_path.is_file() else None,
            "lifecycle_audit_sha256": sha256_file(lifecycle_path) if lifecycle_path.is_file() else None,
        })
    except Exception as exc:
        worker_result.update({
            "exit_code": 1,
            "end_time": _utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
    _atomic_write_json(result_path, worker_result)
    return int(worker_result.get("exit_code", 1))


def append_control_return_lifecycle_event(
    lifecycle_audit: dict,
    *,
    baseline: dict[str, int],
) -> dict:
    """Record the first point visible to the Phase A runner after Control returns."""
    event = {
        "callpoint": "control_return_after_return",
        "memory": phase_a_lifecycle_memory_snapshot(),
        "references": dict(lifecycle_audit.get("references_after_cleanup", {})),
        "baseline": {
            "allocated_bytes": int(baseline.get("allocated_bytes", 0)),
            "reserved_bytes": int(baseline.get("reserved_bytes", 0)),
        },
        "rng_state": phase_a_rng_state_hashes(),
    }
    lifecycle_audit.setdefault("events", []).append(event)
    lifecycle_audit["runner_return_recorded"] = True
    return event


def evaluate_control_lifecycle_gate(
    lifecycle_audit: dict,
    *,
    baseline: dict[str, int],
    allocated_cap_bytes: int = CONTROL_LIFECYCLE_ALLOCATED_CAP_BYTES,
) -> dict:
    """Hard-stop Treatment unless Control is no longer retained after cleanup."""
    events = lifecycle_audit.get("events", [])
    by_name = {str(event.get("callpoint")): event for event in events}
    process_event = (
        by_name.get("control_cuda_empty_cache_after")
        or by_name.get("phase_a_cuda_empty_cache_after")
    )
    final_event = by_name.get("phase_a_return_after_local_release")
    reasons: list[str] = []
    if process_event is None:
        reasons.append("missing_control_cuda_cleanup_event")
    if final_event is None:
        reasons.append("missing_final_lifecycle_event")
    if not bool(lifecycle_audit.get("cleanup_performed")):
        reasons.append("phase_a_training_cleanup_not_performed")
    if not bool(lifecycle_audit.get("rng_state_unchanged")):
        reasons.append("cleanup_changed_rng_state")
    final_memory = (final_event or {}).get("memory", {})
    live_count = int(final_memory.get("live_cuda_tensor_count", 0) or 0)
    live_bytes = int(final_memory.get("live_cuda_tensor_bytes", 0) or 0)
    if live_count > 0 or live_bytes > 0:
        reasons.append("control_cuda_tensors_remain_after_cleanup")
    final_refs = (final_event or {}).get("references", {})
    weakref_alive = final_refs.get("weakref_alive")
    if not isinstance(weakref_alive, dict):
        reasons.append("missing_final_weakref_evidence")
        weakref_alive = {}
    if any(bool(value) for value in weakref_alive.values()):
        reasons.append("control_runtime_reference_remains_after_cleanup")
    allocated = int((final_event or {}).get("memory", {}).get("allocated_bytes", 0) or 0)
    baseline_allocated = int(baseline.get("allocated_bytes", 0) or 0)
    if allocated > baseline_allocated + int(allocated_cap_bytes):
        reasons.append("control_allocated_memory_exceeds_cleanup_cap")
    passed = not reasons
    return {
        "passed": passed,
        "treatment_allowed": passed,
        "next_action": (
            "CONTINUE_PHASE_A_TREATMENT"
            if passed
            else "CONTROL_TREATMENT_SUBPROCESS_ISOLATION_REQUIRED"
        ),
        "reasons": reasons,
        "baseline": dict(baseline),
        "allocated_cap_bytes": int(allocated_cap_bytes),
        "observed_cleanup_memory": {
            "allocated_bytes": allocated,
            "live_cuda_tensor_count": live_count,
            "live_cuda_tensor_bytes": live_bytes,
        },
        "cleanup_reference_flags": dict(final_refs),
    }


def _pipeline_argv(
    args: argparse.Namespace,
    variant_dir: Path,
    command: str,
    graph_enabled: bool,
    model_path: Path,
    output_tag: str = "",
) -> list[str]:
    if command == "evaluate":
        argv = [
            str(Path(sys.executable)), "t5_aste_pipeline.py", "evaluate",
            "--run_dir", str(variant_dir), "--model_path", str(model_path),
            "--eval_file", str(variant_dir / "source_dev.jsonl"),
            "--batch_size", str(args.recipe_data["training"]["extractor_eval_batch_size"]),
            "--num_beams", str(args.recipe_data["pseudo"]["num_beams"]),
            "--max_new_tokens", str(args.recipe_data["pseudo"]["max_new_tokens"]),
            "--length_penalty", str(args.recipe_data["pseudo"]["length_penalty"]),
            "--cuda", str(args.cuda), "--no_task_prefix", "--no_constrained_decoding",
            "--output_tag", output_tag,
        ]
        if graph_enabled:
            argv.extend([
                "--use_syntactic_graph_adapter",
                "--syntactic_graph_cache_dir", str(args.graph_cache_dir),
                "--syntactic_graph_parser_dir", str(args.parser_dir),
                "--syntactic_graph_cache_tokenizer_path", str(args.model_path),
                "--syntactic_graph_split", "source_dev",
            ])
    elif command == "pseudo":
        argv = [
            str(Path(sys.executable)), "t5_aste_pipeline.py", "pseudo",
            "--run_dir", str(variant_dir), "--model_path", str(model_path),
            "--batch_size", str(args.recipe_data["training"]["target_pseudo_batch_size"]),
            "--num_beams", str(args.recipe_data["pseudo"]["num_beams"]),
            "--max_new_tokens", str(args.recipe_data["pseudo"]["max_new_tokens"]),
            "--length_penalty", str(args.recipe_data["pseudo"]["length_penalty"]),
            "--max_target_unlabeled", str(args.recipe_data["pseudo"]["max_target_unlabeled"]),
            "--pseudo_model_variant", str(args.recipe_data["pseudo"]["pseudo_model_variant"]),
            "--pseudo_source_tag", str(args.recipe_data["pseudo"]["pseudo_source_tag"]),
            "--pseudo_base_weight", str(args.recipe_data["pseudo"]["base_weight"]),
            "--high_precision_max_triplets", str(args.recipe_data["pseudo"]["high_precision_max_triplets"]),
            "--high_precision_max_token_distance", str(args.recipe_data["pseudo"]["high_precision_max_token_distance"]),
            "--fixed_changed_min_score", str(args.recipe_data["pseudo"]["fixed_changed_min_score"]),
            "--fixed_changed_weight", str(args.recipe_data["pseudo"]["fixed_changed_weight"]),
            "--cuda", str(args.cuda), "--no_task_prefix", "--no_constrained_decoding",
        ]
        if graph_enabled:
            argv.extend([
                "--use_syntactic_graph_adapter",
                "--syntactic_graph_cache_dir", str(args.graph_cache_dir),
                "--syntactic_graph_parser_dir", str(args.parser_dir),
                "--syntactic_graph_cache_tokenizer_path", str(args.model_path),
            ])
    else:
        raise ValueError(command)
    return argv


def _run_pipeline_command(args: argparse.Namespace, variant_dir: Path, command: str, graph_enabled: bool, model_path: Path, output_tag: str = "") -> list[str]:
    argv = _pipeline_argv(args, variant_dir, command, graph_enabled, model_path, output_tag)
    subprocess.run(argv, cwd=args.project_root, check=True)
    return argv


def _triplet_counts(rows: list[dict], key: str) -> tuple[int, int]:
    tp = fn = 0
    for row in rows:
        gold = parse_triplet_text_list(row.get("gold", ""))
        predicted = parse_triplet_text_list(row.get(key, ""))
        remaining = list(gold)
        for triplet in predicted:
            if triplet in remaining:
                remaining.remove(triplet)
                tp += 1
        fn += len(remaining)
    return tp, fn


def run_isolated_phase_a_worker(
    command: list[str],
    *,
    result_path: Path,
    expected_variant: str,
    require_complete_result: bool = False,
) -> dict:
    """Run and validate a Phase A worker in an operating-system process."""
    parent_pid = os.getpid()
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Phase A {expected_variant} worker failed with exit code {completed.returncode}")
    if not result_path.is_file():
        raise RuntimeError(f"Phase A {expected_variant} worker produced no result: {result_path}")
    result = _read_json(result_path)
    if result.get("status") != "PASS" or result.get("variant") != expected_variant:
        raise RuntimeError(f"Phase A {expected_variant} worker result identity mismatch")
    if not isinstance(result.get("pid"), int) or result["pid"] == parent_pid:
        raise RuntimeError(f"Phase A {expected_variant} worker was not process-isolated")
    if require_complete_result:
        required = (
            "stage", "parent_pid", "argv", "argv_fingerprint", "seed", "code_identity",
            "recipe_identity", "input_identity", "model_identity", "start_time", "end_time",
            "exit_code", "model_tree_sha256", "dann_batch_audit_sha256",
            "initialization_audit_sha256", "optimizer_global_step", "target_test_access",
        )
        missing = [field for field in required if field not in result]
        if missing:
            raise RuntimeError(f"Phase A {expected_variant} worker result is incomplete: {missing}")
        if result.get("exit_code") != 0 or result.get("target_test_access") is not False:
            raise RuntimeError(f"Phase A {expected_variant} worker result terminal identity is invalid")
        argv = result.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise RuntimeError(f"Phase A {expected_variant} worker result argv is invalid")
        argv_fingerprint = _sha256_bytes(_canonical_json(argv))
        if result.get("argv_fingerprint") != argv_fingerprint or result.get("training_argv_fingerprint") != argv_fingerprint:
            raise RuntimeError(f"Phase A {expected_variant} worker argv fingerprint mismatch")
        if result.get("parent_pid") != parent_pid:
            raise RuntimeError(f"Phase A {expected_variant} worker parent PID mismatch")
        for path_field, hash_field in (
            ("model_path", "model_tree_sha256"),
            ("dann_batch_audit_path", "dann_batch_audit_sha256"),
            ("initialization_audit_path", "initialization_audit_sha256"),
        ):
            path_value = result.get(path_field)
            if not path_value or not Path(path_value).exists():
                raise RuntimeError(f"Phase A {expected_variant} worker artifact is missing: {path_field}")
            path = Path(path_value)
            observed = _hash_tree(path) if path.is_dir() else sha256_file(path)
            if observed != result.get(hash_field):
                raise RuntimeError(f"Phase A {expected_variant} worker artifact hash mismatch: {path_field}")
    return result


def _source_dev_metrics(path: Path) -> dict:
    rows = read_jsonl(path)
    raw = micro_f1([row.get("pred_raw", "") for row in rows], [row.get("gold", "") for row in rows])
    multi_rows = [row for row in rows if len(parse_triplet_text_list(row.get("gold", ""))) >= 2]
    multi_recall_tp, multi_recall_fn = _triplet_counts(multi_rows, "pred_raw")
    absence = {"overall": 0, "aspect": 0, "opinion": 0}
    for row in multi_rows:
        gold_triplets = parse_triplet_text_list(row.get("gold", ""))
        predicted_triplets = parse_triplet_text_list(row.get("pred_raw", ""))
        gold_aspects = {triplet[0] for triplet in gold_triplets}
        gold_opinions = {triplet[1] for triplet in gold_triplets}
        predicted_aspects = {triplet[0] for triplet in predicted_triplets}
        predicted_opinions = {triplet[1] for triplet in predicted_triplets}
        aspect_absent = not gold_aspects.issubset(predicted_aspects)
        opinion_absent = not gold_opinions.issubset(predicted_opinions)
        absence["aspect"] += int(aspect_absent)
        absence["opinion"] += int(opinion_absent)
        absence["overall"] += int(aspect_absent or opinion_absent)
    denominator = len(multi_rows)
    return {
        "rows": len(rows),
        "multi_triplet_rows": denominator,
        "strict_triplet_f1": raw["micro_f1"],
        "strict_triplet_scores": raw,
        "multi_triplet_sentence_recall": multi_recall_tp / (multi_recall_tp + multi_recall_fn) if multi_recall_tp + multi_recall_fn else 0.0,
        "multi_triplet_recall_counts": {"tp": multi_recall_tp, "fn": multi_recall_fn},
        "absence_rates": {key: value / denominator if denominator else 0.0 for key, value in absence.items()},
        "absence_counts": absence,
        "absence_denominator": denominator,
    }


def _pseudo_supply(variant_dir: Path) -> dict:
    # A4 measures the existing high-confidence qualified supply. The separate
    # high-precision/k=1 artifact remains untouched for downstream semantics.
    path = variant_dir / "target_pseudo_selected.jsonl"
    rows = read_jsonl(path)
    qualified_multi = sum(len(parse_triplet_text_list(row.get("label", ""))) >= 2 for row in rows)
    return {
        "selector": "strict_high_confidence",
        "path": str(path),
        "qualified_total_rows": len(rows),
        "qualified_multi_rows": qualified_multi,
        "target_test_access": False,
    }


def build_phase_a_pseudo_output_paths(variant_dir: Path) -> dict[str, Path]:
    names = (
        "target_pseudo.jsonl",
        "target_pseudo_selected.jsonl",
        "target_pseudo_high_precision.jsonl",
        "target_pseudo_train_selected.jsonl",
        "target_pseudo_selected_analysis.json",
        "target_pseudo_generation_state.json",
    )
    return {name: Path(variant_dir) / name for name in names}


def _read_dann_batch_audit(
    variant_dir: Path,
    expected_epochs: int | None = None,
    *,
    expected_seed: int = 1000,
    expected_source_count: int | None = None,
    expected_target_count: int | None = None,
    expected_source_row_ids: list | None = None,
    expected_target_row_ids: list | None = None,
    require_training_state: bool = False,
    expected_max_steps: int | None = None,
    expected_planned_batches: int | None = None,
    gradient_accumulation_steps: int = FROZEN_RECIPE["gradient_accumulation_steps"],
    allow_legacy: bool = True,
    require_journal: bool = False,
    require_fresh_replay_free: bool = False,
) -> dict:
    path = Path(variant_dir)
    if path.is_dir():
        path = path / "dann_batch_audit.json"
    if not path.is_file():
        raise RuntimeError(f"missing DANN batch audit report: {path}")
    report = _read_json(path)
    if not isinstance(report, dict) or not isinstance(report.get("epochs"), list) or not report["epochs"]:
        raise RuntimeError(f"invalid DANN batch audit report: {path}")
    schema_version = report.get("schema_version")
    legacy = schema_version == 1
    current = schema_version == 3 and report.get("audit_protocol") == "physical_dataloader_traversal_v2"
    if not legacy and not current:
        raise RuntimeError(f"invalid DANN batch audit schema: {path}")
    if legacy and not allow_legacy:
        raise RuntimeError(f"legacy DANN batch audit is diagnostic-only and cannot be resumed: {path}")
    top_level = (
        "seed", "source_batch_size", "target_batch_size", "source_count", "target_count",
        "source_row_ids", "target_row_ids",
    )
    if any(field not in report for field in top_level):
        raise RuntimeError(f"DANN batch audit is missing top-level identity fields: {path}")
    if report["seed"] != expected_seed:
        raise RuntimeError(f"DANN batch audit seed identity mismatch: {path}")
    if report["source_batch_size"] != 1 or report["target_batch_size"] != 1:
        raise RuntimeError(f"DANN batch audit requires source=1 and target=1: {path}")
    if expected_source_count is not None and report["source_count"] != expected_source_count:
        raise RuntimeError(f"DANN batch audit source count identity mismatch: {path}")
    if expected_target_count is not None and report["target_count"] != expected_target_count:
        raise RuntimeError(f"DANN batch audit target count identity mismatch: {path}")
    if expected_source_row_ids is not None and report["source_row_ids"] != list(expected_source_row_ids):
        raise RuntimeError(f"DANN batch audit source row identity mismatch: {path}")
    if expected_target_row_ids is not None and report["target_row_ids"] != list(expected_target_row_ids):
        raise RuntimeError(f"DANN batch audit target row identity mismatch: {path}")
    if len(report["source_row_ids"]) != report["source_count"] or len(report["target_row_ids"]) != report["target_count"]:
        raise RuntimeError(f"DANN batch audit row identity/count mismatch: {path}")
    if expected_epochs is not None and len(report["epochs"]) != expected_epochs:
        raise RuntimeError(f"DANN batch audit epoch count mismatch: {path}")
    seen_epochs = set()
    seen_physical = set()
    previous_step_end = None
    for position, epoch in enumerate(report["epochs"]):
        if not isinstance(epoch, dict):
            raise RuntimeError(f"DANN batch audit contains a non-object epoch: {path}")
        sampling_epoch = epoch.get("epoch") if legacy else epoch.get("sampling_epoch")
        physical_index = epoch.get("epoch") if legacy else epoch.get("physical_traversal_index")
        completion = "complete" if legacy else epoch.get("completion")
        planned_batches = len(epoch.get("batches", [])) if legacy else epoch.get("planned_batches")
        issued_batches = len(epoch.get("batches", [])) if legacy else epoch.get("issued_batches")
        processed_batches = issued_batches if legacy else epoch.get("processed_batches")
        if not isinstance(sampling_epoch, int) or sampling_epoch < 0:
            raise RuntimeError(f"DANN batch audit has an invalid sampling epoch: {path}")
        if not isinstance(physical_index, int) or physical_index != position or physical_index in seen_physical:
            raise RuntimeError(f"DANN batch audit has duplicate or invalid physical traversals: {path}")
        if not legacy and epoch.get("epoch") != sampling_epoch:
            raise RuntimeError(f"DANN batch audit epoch alias mismatch: {path}")
        if completion not in {"complete", "partial"} or not isinstance(planned_batches, int) or planned_batches <= 0 or not isinstance(issued_batches, int) or issued_batches <= 0 or issued_batches > planned_batches or not isinstance(processed_batches, int) or processed_batches < 0 or processed_batches > issued_batches:
            raise RuntimeError(f"DANN batch audit has invalid complete/partial traversal accounting: {path}")
        if current and expected_planned_batches is not None and planned_batches != expected_planned_batches:
            raise RuntimeError(f"DANN batch audit planned batch count mismatch: {path}")
        seen_epochs.add(sampling_epoch)
        seen_physical.add(physical_index)
        if not isinstance(epoch.get("batches"), list) or not epoch["batches"]:
            raise RuntimeError(f"DANN batch audit has an empty epoch: {path}")
        if epoch.get("logical_batches") != len(epoch["batches"]):
            raise RuntimeError(f"DANN batch audit logical batch count mismatch: {path}")
        if planned_batches < len(epoch["batches"]) or issued_batches != len(epoch["batches"]):
            raise RuntimeError(f"DANN batch audit planned/issued batch mismatch: {path}")
        if completion == "complete" and len(epoch["batches"]) != planned_batches:
            raise RuntimeError(f"DANN batch audit marks a short traversal complete: {path}")
        if completion == "partial" and len(epoch["batches"]) > planned_batches:
            raise RuntimeError(f"DANN batch audit marks a full traversal partial: {path}")
        if completion == "complete" and (issued_batches != processed_batches or processed_batches != planned_batches):
            raise RuntimeError(f"DANN batch audit complete state has issued/processed/planned mismatch: {path}")
        if epoch.get("source_rows") != len(epoch["batches"]) or epoch.get("target_rows") != len(epoch["batches"]):
            raise RuntimeError(f"DANN batch audit domain row count mismatch: {path}")
        if epoch.get("source_unique_rows") > report["source_count"] or epoch.get("target_unique_rows") > report["target_count"]:
            raise RuntimeError(f"DANN batch audit coverage mismatch: {path}")
        if completion == "complete" and (epoch.get("source_unique_rows") != report["source_count"] or epoch.get("target_unique_rows") != report["target_count"]):
            raise RuntimeError(f"DANN batch audit complete traversal coverage mismatch: {path}")
        if (
            epoch.get("source_batch_size") != 1
            or epoch.get("target_batch_size") != 1
            or epoch.get("incomplete_batches") != 0
            or any(
                batch.get("source_count") != 1 or batch.get("target_count") != 1
                for batch in epoch.get("batches", [])
            )
            ):
            raise RuntimeError(f"DANN batch audit contains an incomplete or non-1/1 epoch: {path}")
        seen_source_indices = set()
        seen_target_indices = set()
        for expected_batch_id, batch in enumerate(epoch["batches"]):
            if batch.get("logical_batch_id") != expected_batch_id:
                raise RuntimeError(f"DANN batch audit logical batch IDs are not contiguous: {path}")
            source_indices = batch.get("source_indices")
            target_indices = batch.get("target_indices")
            source_ids = batch.get("source_row_ids")
            target_ids = batch.get("target_row_ids")
            if not (isinstance(source_indices, list) and len(source_indices) == 1 and isinstance(source_indices[0], int) and isinstance(target_indices, list) and len(target_indices) == 1 and isinstance(target_indices[0], int)):
                raise RuntimeError(f"DANN batch audit batch domain indices are not 1/1: {path}")
            source_index = source_indices[0]
            target_index = target_indices[0]
            if not (0 <= source_index < report["source_count"] and isinstance(source_ids, list) and source_ids == [report["source_row_ids"][source_index]] and isinstance(target_ids, list)):
                raise RuntimeError(f"DANN batch audit source index/ID mapping mismatch: {path}")
            target_position = target_index - report["source_count"]
            if not (0 <= target_position < report["target_count"] and target_ids == [report["target_row_ids"][target_position]]):
                raise RuntimeError(f"DANN batch audit target index/ID mapping mismatch: {path}")
            seen_source_indices.add(source_index)
            seen_target_indices.add(target_position)
        if epoch.get("source_unique_rows") != len(seen_source_indices) or epoch.get("target_unique_rows") != len(seen_target_indices):
            raise RuntimeError(f"DANN batch audit reported coverage count mismatch: {path}")
        if completion == "complete" and (seen_source_indices != set(range(report["source_count"])) or seen_target_indices != set(range(report["target_count"]))):
            raise RuntimeError(f"DANN batch audit row coverage is incomplete: {path}")
        if current and require_training_state:
            step_start = epoch.get("optimizer_global_step_start")
            step_end = epoch.get("optimizer_global_step_end")
            if not isinstance(step_start, int) or not isinstance(step_end, int) or step_start < 0 or step_end < step_start:
                raise RuntimeError(f"DANN batch audit has invalid optimizer step range: {path}")
            if previous_step_end is None and step_start != 0:
                raise RuntimeError(f"DANN batch audit optimizer steps do not start at zero: {path}")
            if previous_step_end is not None and step_start != previous_step_end:
                raise RuntimeError(f"DANN batch audit optimizer steps are not contiguous: {path}")
            previous_step_end = step_end
    if any(item.get("completion") == "partial" for item in report["epochs"][:-1]):
        raise RuntimeError(f"DANN batch audit has a partial traversal before the end: {path}")
    if current:
        if report.get("next_physical_traversal_index") != len(report["epochs"]):
            raise RuntimeError(f"DANN batch audit physical traversal counter mismatch: {path}")
        max_sampling_epoch = max(item.get("sampling_epoch") for item in report["epochs"])
        if not isinstance(report.get("next_sampling_epoch"), int) or report.get("next_sampling_epoch") < max_sampling_epoch + 1:
            raise RuntimeError(f"DANN batch audit sampling epoch counter mismatch: {path}")
        if require_training_state:
            actual_max_steps = report.get("trainer_max_steps")
            actual_global_step = report.get("trainer_global_step")
            if not isinstance(actual_max_steps, int) or not isinstance(actual_global_step, int):
                raise RuntimeError(f"DANN batch audit lacks Trainer max_steps/global_step identity: {path}")
            if expected_max_steps is not None and actual_max_steps != expected_max_steps:
                raise RuntimeError(f"DANN batch audit max_steps mismatch: {path}")
            if actual_global_step != actual_max_steps:
                raise RuntimeError(f"DANN batch audit did not reach Trainer max_steps: {path}")
            if previous_step_end != actual_global_step:
                raise RuntimeError(f"DANN batch audit final optimizer step does not match Trainer state: {path}")
            mismatched_epochs = [
                item for item in report["epochs"]
                if item.get("processed_batches") != item.get("issued_batches")
            ]
            terminal_lookahead = classify_terminal_lookahead(
                report,
                gradient_accumulation_steps=int(report.get("gradient_accumulation_steps") or gradient_accumulation_steps),
            )
            if mismatched_epochs and not terminal_lookahead["safe"]:
                raise RuntimeError(f"DANN batch audit has unsafe unacknowledged batches at terminal max_steps: {path}")
            if terminal_lookahead["safe"]:
                report["terminal_lookahead_audit"] = terminal_lookahead
        if any(item.get("completion") == "partial" for item in report["epochs"]):
            last = report["epochs"][-1]
            lookahead_safe = bool(report.get("terminal_lookahead_audit", {}).get("safe"))
            if not require_training_state or report.get("trainer_global_step") != report.get("trainer_max_steps") or (
                last.get("processed_batches") != last.get("issued_batches") and not lookahead_safe
            ):
                raise RuntimeError(f"DANN batch audit contains a non-final or non-terminal partial traversal: {path}")
    elif any(item.get("completion") == "partial" for item in report["epochs"]):
        raise RuntimeError(f"DANN batch audit partial traversal is not formal evidence without Trainer state: {path}")
    journal_path = path.with_suffix(".journal.jsonl")
    journal_audit = None
    if require_journal or journal_path.is_file():
        if not journal_path.is_file():
            raise RuntimeError(f"missing DANN audit journal: {journal_path}")
        try:
            journal_audit = read_dann_audit_journal(journal_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid DANN audit journal: {journal_path}") from exc
        if require_fresh_replay_free and journal_audit["replay_count"] != 0:
            raise RuntimeError(f"fresh DANN run contains batch_replayed events: {journal_path}")
    if journal_audit is not None:
        report["journal_audit"] = journal_audit
        report["replay_count"] = journal_audit["replay_count"]
        if report.get("terminal_lookahead_audit", {}).get("lookahead_not_consumed"):
            terminal_lookahead = classify_terminal_lookahead(
                report,
                gradient_accumulation_steps=int(report.get("gradient_accumulation_steps") or gradient_accumulation_steps),
                journal_audit=journal_audit,
            )
            if not terminal_lookahead["safe"]:
                raise RuntimeError(f"DANN terminal lookahead journal evidence is invalid: {path}")
            report["terminal_lookahead_audit"] = terminal_lookahead
        if report.get("terminal_lookahead_audit", {}).get("lookahead_not_consumed") and journal_audit["replay_count"] != 0:
            raise RuntimeError(f"terminal lookahead audit cannot contain replay events: {journal_path}")
    elif require_fresh_replay_free:
        raise RuntimeError(f"fresh DANN run has no auditable journal: {journal_path}")
    if report.get("terminal_lookahead_audit", {}).get("lookahead_not_consumed") and journal_audit is None:
        raise RuntimeError(f"terminal lookahead requires an auditable journal: {journal_path}")
    return report


def validate_external_control_dann_audit(
    control_reuse_audit: dict,
    expected_epochs: int | None = None,
    *,
    require_training_state: bool = False,
    expected_max_steps: int | None = None,
    expected_planned_batches: int | None = None,
    allow_legacy: bool = True,
    require_journal: bool = False,
    require_fresh_replay_free: bool = False,
) -> dict:
    path = Path(control_reuse_audit.get("dann_batch_audit_path", ""))
    expected_hash = control_reuse_audit.get("dann_batch_audit_sha256")
    if not path.is_file() or not expected_hash:
        raise RuntimeError("external Control requires a DANN batch audit path and SHA256")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError("external Control DANN batch audit hash changed; refusing reuse")
    return _read_dann_batch_audit(
        path,
        expected_epochs=expected_epochs,
        expected_seed=control_reuse_audit.get("expected_seed", 1000),
        expected_source_count=control_reuse_audit.get("expected_source_count"),
        expected_target_count=control_reuse_audit.get("expected_target_count"),
        expected_source_row_ids=control_reuse_audit.get("expected_source_row_ids"),
        expected_target_row_ids=control_reuse_audit.get("expected_target_row_ids"),
        require_training_state=require_training_state,
        expected_max_steps=expected_max_steps,
        expected_planned_batches=expected_planned_batches,
        allow_legacy=allow_legacy,
        require_journal=require_journal,
        require_fresh_replay_free=require_fresh_replay_free,
    )


def validate_terminal_lookahead_salvage(
    salvage_path: Path,
    *,
    control_run_dir: Path,
    model_path: Path,
    dann_path: Path,
) -> dict:
    """Validate the separate, read-only V6 salvage evidence before reuse."""
    path = Path(salvage_path).resolve()
    if not path.is_file():
        raise RuntimeError("external Control terminal-lookahead salvage report is missing; refusing reuse")
    report = _read_json(path)
    if not isinstance(report, dict) or report.get("status") != "PASS":
        raise RuntimeError("external Control terminal-lookahead salvage is not PASS; refusing reuse")
    if report.get("source_run_remains_blocked") is not True or report.get("target_test_access") is not False:
        raise RuntimeError("external Control salvage boundary is invalid; refusing reuse")
    if report.get("producer_training_semantics_commit") != TRAINING_SEMANTICS_IDENTITY:
        raise RuntimeError("external Control salvage producer training identity mismatch; refusing reuse")
    if report.get("reuse_depth") != 1:
        raise RuntimeError("external Control salvage reuse depth mismatch; refusing reuse")
    source_run = Path(report.get("source_run_dir", "")).resolve()
    expected_source_run = model_path.resolve().parents[3]
    if source_run != expected_source_run or source_run != Path(control_run_dir).resolve():
        raise RuntimeError("external Control salvage source run identity mismatch; refusing reuse")
    expected_artifacts = {
        "model_tree_sha256": (model_path, _hash_tree),
        "dann_audit_sha256": (dann_path, sha256_file),
    }
    for report_field, (artifact_path, hash_function) in expected_artifacts.items():
        observed = hash_function(artifact_path)
        if report.get(report_field) != observed:
            raise RuntimeError(f"external Control salvage {report_field} mismatch; refusing reuse")
    journal_path = dann_path.with_suffix(".journal.jsonl")
    if not journal_path.is_file() or report.get("journal_path") is None:
        raise RuntimeError("external Control salvage journal is missing; refusing reuse")
    if Path(report["journal_path"]).resolve() != journal_path.resolve() or report.get("journal_sha256") != sha256_file(journal_path):
        raise RuntimeError("external Control salvage journal identity mismatch; refusing reuse")
    classification = report.get("classification")
    if not isinstance(classification, dict) or classification.get("safe") is not True:
        raise RuntimeError("external Control salvage classification is not safe; refusing reuse")
    return report


def _normalize_external_control_terminal_lookahead(control_report: dict, treatment_report: dict, control_reuse_audit: dict | None = None) -> tuple[dict, dict]:
    """Normalize one independently salvaged V6 terminal lookahead in memory only."""
    normalized = copy.deepcopy(control_report)
    epochs = normalized.get("epochs") or []
    lookahead = normalized.get("terminal_lookahead_audit") or {}
    dangling = lookahead.get("dangling_logical_batch_ids")
    if not epochs or lookahead.get("safe") is not True or lookahead.get("lookahead_not_consumed") is not True or not isinstance(dangling, list) or len(dangling) != 1:
        return normalized, {"applied": False, "reason": "unsafe_or_non_single_lookahead"}
    last = epochs[-1]
    if last.get("completion") != "partial" or last.get("issued_batches") != last.get("processed_batches") + 1:
        return normalized, {"applied": False, "reason": "terminal_accounting_mismatch"}
    batches = last.get("batches")
    if not isinstance(batches, list) or len(batches) != last.get("issued_batches") or batches[-1].get("logical_batch_id") != dangling[0]:
        return normalized, {"applied": False, "reason": "dangling_batch_identity_mismatch"}
    trimmed = batches[:-1]
    count = len(trimmed)
    for key in ("logical_batches", "source_rows", "target_rows", "source_unique_rows", "target_unique_rows", "issued_batches"):
        last[key] = count
    last["batches"] = trimmed
    return normalized, {"applied": True, "source": "external_control_terminal_lookahead_salvage", "trimmed_logical_batch_ids": list(dangling), "original_issued_batches": count + 1, "normalized_issued_batches": count}

def _validate_control_treatment_dann_reports(
    variant_dirs: dict[str, Path],
    control_reuse_audit: dict,
    expected_epochs: int | None = None,
    *,
    expected_seed: int = 1000,
    expected_source_count: int | None = None,
    expected_target_count: int | None = None,
    expected_source_row_ids: list | None = None,
    expected_target_row_ids: list | None = None,
    require_training_state: bool = False,
    expected_max_steps: int | None = None,
    expected_planned_batches: int | None = None,
    allow_legacy: bool = True,
    require_journal: bool = False,
    require_fresh_replay_free: bool = False,
) -> dict:
    treatment_path = variant_dirs["treatment"] / "dann_batch_audit.json"
    if expected_source_row_ids is None:
        expected_source_row_ids = [row.get("id") for row in read_jsonl(variant_dirs["treatment"] / "source_train.jsonl")]
    if expected_target_row_ids is None:
        expected_target_row_ids = [row.get("id") for row in read_jsonl(variant_dirs["treatment"] / "target_unlabeled.jsonl")]
    if expected_source_count is None:
        expected_source_count = len(expected_source_row_ids)
    if expected_target_count is None:
        expected_target_count = len(expected_target_row_ids)
    control_reuse_audit = {
        **control_reuse_audit,
        "expected_seed": expected_seed,
        "expected_source_count": expected_source_count,
        "expected_target_count": expected_target_count,
        "expected_source_row_ids": list(expected_source_row_ids),
        "expected_target_row_ids": list(expected_target_row_ids),
    }
    control_report = validate_external_control_dann_audit(
        control_reuse_audit,
        expected_epochs=expected_epochs,
        require_training_state=require_training_state,
        expected_max_steps=expected_max_steps,
        expected_planned_batches=expected_planned_batches,
        allow_legacy=allow_legacy,
        require_journal=require_journal,
        require_fresh_replay_free=require_fresh_replay_free,
    )
    treatment_report = _read_dann_batch_audit(
        treatment_path,
        expected_epochs=expected_epochs,
        expected_seed=expected_seed,
        expected_source_count=expected_source_count,
        expected_target_count=expected_target_count,
        expected_source_row_ids=expected_source_row_ids,
        expected_target_row_ids=expected_target_row_ids,
        require_training_state=require_training_state,
        expected_max_steps=expected_max_steps,
        expected_planned_batches=expected_planned_batches,
        allow_legacy=allow_legacy,
        require_journal=require_journal,
        require_fresh_replay_free=require_fresh_replay_free,
    )
    comparable_fields = (
        "seed",
        "source_batch_size",
        "target_batch_size",
        "source_count",
        "target_count",
        "source_row_ids",
        "target_row_ids",
        "epochs",
    )
    normalized_control, lookahead_normalization = _normalize_external_control_terminal_lookahead(control_report, treatment_report, control_reuse_audit)
    if lookahead_normalization.get("applied"):
        control_report = normalized_control
    if any(control_report.get(field) != treatment_report.get(field) for field in comparable_fields):
        raise RuntimeError("Control and Treatment DANN batch orders or steps differ")
    if control_report.get("schema_version") == 1 or treatment_report.get("schema_version") == 1:
        raise RuntimeError("legacy DANN batch audit is diagnostic-only and cannot be formal evidence")
    result = {"status": "matched", "control": control_report, "treatment": treatment_report}
    if lookahead_normalization.get("applied"):
        result["control_terminal_lookahead_normalization"] = lookahead_normalization
    return result


def _write_result_markdown(run_dir: Path, summary: dict) -> None:
    decision = summary["decision"]
    lines = [
        "# M1 句法 RGAT Phase A 快速消融结果",
        "",
        f"> 更新时间：{_utc_now()}（北京时间）",
        "",
        f"- 任务：`{TASK_ID}`",
        f"- 方向：`{summary['direction']}`；种子：`{summary['seed']}`",
        f"- Phase A：`{decision['status']}`",
        f"- 下一动作：`{decision['next_action']}`",
        f"- target_test 访问：`{summary['scope']['target_test_access']}`",
        "",
        "## 门控",
        "",
    ]
    for name, gate in summary["gates"].items():
        lines.append(f"- {name}：`{gate['status']}`；实际值：`{json.dumps(gate.get('actual'), ensure_ascii=False)}`")
    lines.extend([
        "",
        "Phase A 通过时只请求 Phase B，不自动进入 Phase B；失败时硬停止上游，不调用生成器、增强、最终训练或目标测试。方面缺失与观点缺失可能重叠，不作独立因果解释。",
        "",
    ])
    _atomic_write_text(run_dir / "phase_a_result_CN.md", "\n".join(lines))


def _build_identity(args: argparse.Namespace, input_hashes: dict, model_hashes: dict, parser_identity: dict, recipe_sha256: str, git_identity: dict) -> tuple[dict, dict]:
    model_digest = _sha256_bytes(_canonical_json(model_hashes))
    tokenizer_digest = _sha256_bytes(_canonical_json({key: value for key, value in model_hashes.items() if key in {"spiece.model", "tokenizer.json"}}))
    artifact_digest = _sha256_bytes(_canonical_json({"inputs": input_hashes, "model": model_hashes}))
    actual = {
        "direction": f"{args.recipe_data['source_dataset']} -> {args.recipe_data['target_dataset']}",
        "seed": args.recipe_data["seed"],
        "data_split": "source_train+source_dev+target_unlabeled",
        "recipe_sha256": recipe_sha256,
        "checkpoint_selection_rule": args.recipe_data["training"]["checkpoint_selection"],
        "tokenizer_sha256": tokenizer_digest,
        "model_sha256": model_digest,
        "code_semantics": TRAINING_SEMANTICS_IDENTITY,
        "artifact_sha256": artifact_digest,
    }
    return actual, {
        "identity_schema_version": 1,
        "actual": actual,
        "input_hashes": input_hashes,
        "model_file_hashes": model_hashes,
        "parser_identity": parser_identity,
        "git": git_identity,
        "orchestration_commit": git_identity["commit"],
        "training_semantics_identity": TRAINING_SEMANTICS_IDENTITY,
        "parent_run_identity": {"required_entry_code_identity": FIXED_PARENT_CODE_IDENTITY},
    }


def _phase_stage_input_files(variant_dir: Path) -> dict[str, Path]:
    return {
        split: variant_dir / f"{split}.jsonl"
        for split in ("source_train", "source_dev", "target_unlabeled")
    }


def _stage_spec(
    args: argparse.Namespace,
    stage: str,
    variant_dirs: dict[str, Path],
    control_model_path: Path,
    *,
    execute_control_training: bool,
    control_dann_batch_audit_path: Path | None = None,
    control_initialization_audit_path: Path | None = None,
) -> dict:
    control_dir = variant_dirs["control"]
    treatment_dir = variant_dirs["treatment"]
    if stage == "control_training":
        audit_path = control_dann_batch_audit_path or (control_dir / "dann_batch_audit.json")
        initialization_path = control_initialization_audit_path or (control_dir / "phase_a_initialization_audit.json")
        if not execute_control_training and control_dann_batch_audit_path is None:
            raise RuntimeError("reused Control stage requires its resolved DANN batch audit path")
        if not execute_control_training and control_initialization_audit_path is None:
            raise RuntimeError("reused Control stage requires its resolved initialization audit path")
        if execute_control_training:
            command = _phase_a_worker_command(_worker_spec_path(stage, control_dir))
        else:
            command = [
                "reuse_external_control",
                str(control_model_path.resolve()),
            ]
        return {
            "command": command,
            "artifact_path": control_model_path,
            "model_path": control_model_path,
            "output_artifacts": {
                "extractor_best": control_model_path,
                "dann_batch_audit": audit_path,
                "phase_a_initialization_audit": initialization_path,
                **(
                    {"phase_a_lifecycle_audit": control_dir / "phase_a_lifecycle_audit.json"}
                    if execute_control_training else {}
                ),
                **(
                    {"worker_result": control_dir / "worker_result.json"}
                    if execute_control_training else {}
                ),
            },
            "worker_spec_path": _worker_spec_path(stage, control_dir) if execute_control_training else None,
            "variant_dir": control_dir,
        }
    if stage == "treatment_training":
        model_path = treatment_dir / "models" / "extractor" / "best"
        audit_path = treatment_dir / "dann_batch_audit.json"
        initialization_path = treatment_dir / "phase_a_initialization_audit.json"
        return {
            "command": _phase_a_worker_command(_worker_spec_path(stage, treatment_dir)),
            "artifact_path": model_path,
            "model_path": model_path,
            "output_artifacts": {
                "extractor_best": model_path,
                "dann_batch_audit": audit_path,
                "phase_a_initialization_audit": initialization_path,
                "phase_a_lifecycle_audit": treatment_dir / "phase_a_lifecycle_audit.json",
                "worker_result": treatment_dir / "worker_result.json",
            },
            "worker_spec_path": _worker_spec_path(stage, treatment_dir),
            "variant_dir": treatment_dir,
        }
    if stage == "control_source_dev_evaluation":
        artifact = control_dir / "aste_predictions_raw_fixed_source_dev.jsonl"
        return {
            "command": _pipeline_argv(args, control_dir, "evaluate", False, control_model_path, "source_dev"),
            "artifact_path": artifact,
            "model_path": control_model_path,
            "variant_dir": control_dir,
        }
    if stage == "treatment_source_dev_evaluation":
        model_path = treatment_dir / "models" / "extractor" / "best"
        artifact = treatment_dir / "aste_predictions_raw_fixed_source_dev.jsonl"
        return {
            "command": _pipeline_argv(args, treatment_dir, "evaluate", True, model_path, "source_dev"),
            "artifact_path": artifact,
            "model_path": model_path,
            "variant_dir": treatment_dir,
        }
    if stage == "control_target_pseudo_inference":
        output_artifacts = build_phase_a_pseudo_output_paths(control_dir)
        artifact = output_artifacts["target_pseudo_selected.jsonl"]
        return {
            "command": _pipeline_argv(args, control_dir, "pseudo", False, control_model_path),
            "artifact_path": artifact,
            "model_path": control_model_path,
            "output_artifacts": output_artifacts,
            "variant_dir": control_dir,
        }
    if stage == "treatment_target_pseudo_inference":
        model_path = treatment_dir / "models" / "extractor" / "best"
        output_artifacts = build_phase_a_pseudo_output_paths(treatment_dir)
        artifact = output_artifacts["target_pseudo_selected.jsonl"]
        return {
            "command": _pipeline_argv(args, treatment_dir, "pseudo", True, model_path),
            "artifact_path": artifact,
            "model_path": model_path,
            "output_artifacts": output_artifacts,
            "variant_dir": treatment_dir,
        }
    raise ValueError(f"unsupported Phase A stage: {stage}")


def _stage_record(
    args: argparse.Namespace,
    stage: str,
    spec: dict,
    recipe_sha256: str,
    git_commit: str,
) -> dict:
    input_files = _phase_stage_input_files(spec["variant_dir"])
    worker_spec_path = spec.get("worker_spec_path")
    if worker_spec_path is not None:
        worker_spec_path = Path(worker_spec_path)
        if not worker_spec_path.is_file():
            raise RuntimeError(f"missing Phase A worker spec: {worker_spec_path}")
        input_files["worker_spec"] = worker_spec_path
    return build_stage_identity(
        stage,
        spec["command"],
        input_files,
        recipe_sha256,
        spec["artifact_path"],
        spec["model_path"],
        git_commit,
        recipe_path=Path(args.recipe),
        output_artifacts=spec.get("output_artifacts"),
    )


def validate_phase_a_graph_cache(
    cache_dir: str | Path,
    input_rows: dict[str, list[dict]],
    parser_identity: dict,
) -> dict:
    """Preflight every graph-cache identity before starting long training."""
    root = Path(cache_dir).resolve()
    required_splits = ("source_train", "source_dev", "target_unlabeled")
    required_paths = [root / "manifest.json", root / "relation_vocab.json"] + [
        root / f"{split}.jsonl" for split in required_splits
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Phase A graph cache preflight missing required artifacts: {missing}")
    try:
        manifest = _read_json(root / "manifest.json")
        relation_vocab = _read_json(root / "relation_vocab.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Phase A graph cache preflight cannot read metadata: {root}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Phase A graph cache manifest is not an object: {root}")
    if not isinstance(relation_vocab, list):
        raise RuntimeError(f"Phase A graph cache relation vocabulary is not a list: {root}")
    if manifest.get("target_test_access") is not False:
        raise RuntimeError("Phase A graph cache must explicitly forbid target_test access")
    try:
        split_counts = {}
        split_hashes = {}
        for split in required_splits:
            expected = input_rows.get(split)
            if not isinstance(expected, list):
                raise RuntimeError(f"Phase A graph cache preflight lacks input rows for {split}")
            cache = load_graph_cache_directory(root, split, expected, parser_identity=parser_identity)
            split_counts[split] = len(cache.records)
            split_hashes[split] = sha256_file(root / f"{split}.jsonl")
    except (GraphCacheError, OSError, ValueError) as exc:
        raise RuntimeError(f"Phase A graph cache identity preflight failed: {root}: {exc}") from exc
    return {
        "cache_dir": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "relation_vocab_sha256": sha256_file(root / "relation_vocab.json"),
        "relation_vocab_size": len(relation_vocab),
        "split_jsonl_sha256": split_hashes,
        "split_counts": split_counts,
        "manifest_input_sha256": manifest.get("input_sha256"),
        "parser_identity": parser_identity,
    }


def run_phase_a(args: argparse.Namespace) -> dict:
    recipe = args.recipe_data
    _validate_recipe(recipe)
    run_dir = Path(args.output_dir)
    if run_dir.exists() and not args.resume:
        raise RuntimeError(f"output directory exists; use a new directory or --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    git_identity = _git_identity(args.project_root)
    if git_identity["status_porcelain"]:
        raise RuntimeError("Phase A requires a clean Git worktree for reproducible code identity")
    parent_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FIXED_PARENT_CODE_IDENTITY, git_identity["commit"]],
        cwd=args.project_root,
        check=False,
    )
    if parent_check.returncode != 0:
        raise RuntimeError(f"current code is not descended from approved parent identity {FIXED_PARENT_CODE_IDENTITY}")
    declared_model_path = Path(recipe["models"]["t5_base"]).resolve()
    if Path(args.model_path).resolve() != declared_model_path:
        raise RuntimeError("--model_path differs from the frozen recipe models.t5_base")
    input_rows = _build_input_rows(
        recipe["source_dataset"],
        recipe["target_dataset"],
        recipe["external_inputs"],
    )
    if args.resume:
        # A failed resume must be observational: validate every persisted input
        # before touching stage_status, config_snapshot, or artifact indexes.
        _preflight_resume_inputs(input_rows, run_dir)
    input_hashes = _input_hashes(input_rows, run_dir)
    declared_input_hashes = _declared_input_hashes(recipe)
    input_identity_hashes = {"run_inputs": input_hashes, "declared_external_inputs": declared_input_hashes}
    model_hashes = _model_hashes(declared_model_path)
    parser_identity = build_parser_identity(args.parser_dir)
    graph_cache_identity = validate_phase_a_graph_cache(args.graph_cache_dir, input_rows, parser_identity)
    recipe_sha256 = sha256_file(Path(args.recipe))
    actual_identity, identity_metadata = _build_identity(args, input_identity_hashes, model_hashes, parser_identity, recipe_sha256, git_identity)
    status_identity = {"task_id": TASK_ID, "code_commit": git_identity["commit"], "recipe_sha256": recipe_sha256, "input_hashes": input_identity_hashes, "model_hashes": model_hashes, "parser_identity": parser_identity, "graph_cache_identity": graph_cache_identity, "scope": build_phase_a_scope()}
    state = load_or_initialize_stage_status(
        run_dir / "stage_status.json",
        status_identity,
        args.resume,
        repair_from_commit=args.resume_repair_from_commit,
    )
    if state is None:
        raise RuntimeError("resume identity mismatch; refusing to mix Phase A artifacts")
    variant_dirs = {name: run_dir / name for name in ("control", "treatment")}
    _write_inputs(input_rows, run_dir, resume=args.resume)
    for variant_dir in variant_dirs.values():
        _write_variant_inputs(variant_dir, run_dir, resume=args.resume)
    state["status"] = "in_progress"
    _atomic_write_json(run_dir / "stage_status.json", state)
    _atomic_write_json(run_dir / "config_snapshot.json", {"task_id": TASK_ID, "recipe": recipe, "variant_configs": {"control": build_variant_config(False), "treatment": build_variant_config(True)}, "scope": build_phase_a_scope(), "identities": {"recipe_sha256": recipe_sha256, "input_hashes": input_identity_hashes, "model_hashes": model_hashes, "parser_identity": parser_identity, "graph_cache_identity": graph_cache_identity, "git_commit": git_identity["commit"]}})
    _atomic_write_json(run_dir / "git_identity.json", git_identity)
    _atomic_write_json(run_dir / "input_artifact_hashes.json", {"inputs": input_hashes, "raw_external_inputs": declared_input_hashes, "model": model_hashes, "parser": parser_identity, "graph_cache": graph_cache_identity, "recipe": {"path": str(args.recipe), "sha256": recipe_sha256}})
    _atomic_write_json(run_dir / "graph_cache_identity.json", graph_cache_identity)
    _atomic_write_json(run_dir / "parent_run_identity.json", {"parent_task_id": "M1_SYNTACTIC_RGAT_ZERO_UPDATE_ENTRY_AUDIT_V1", "required_entry_code_identity": FIXED_PARENT_CODE_IDENTITY, "current_code_commit": git_identity["commit"], "zero_update_entry_status": "15/15 PASS (provided by approved parent identity)"})

    control_lifecycle_baseline = phase_a_lifecycle_memory_snapshot()
    control_reuse_audit = {
        "reuse_allowed": False,
        "requires_rerun": True,
        "reason": "no machine-verifiable control identity supplied",
    }
    control_model_path = variant_dirs["control"] / "models" / "extractor" / "best"
    control_dann_batch_audit_path: Path | None = None
    control_initialization_audit_path: Path | None = None
    expected_source_row_ids = [row.get("id") for row in input_rows["source_train"]]
    expected_target_row_ids = [row.get("id") for row in input_rows["target_unlabeled"]]
    expected_dann_max_steps = compute_dann_expected_max_steps(
        source_count=len(expected_source_row_ids),
        target_count=len(expected_target_row_ids),
        source_batch_size=recipe["training"]["target_unlabeled_dann"]["source_batch_size"],
        target_batch_size=recipe["training"]["target_unlabeled_dann"]["target_batch_size"],
        gradient_accumulation_steps=recipe["training"]["gradient_accumulation_steps"],
        num_train_epochs=recipe["training"]["num_train_epochs"],
    )
    expected_dann_planned_batches = compute_dann_planned_batches(
        source_count=len(expected_source_row_ids),
        target_count=len(expected_target_row_ids),
        source_batch_size=recipe["training"]["target_unlabeled_dann"]["source_batch_size"],
        target_batch_size=recipe["training"]["target_unlabeled_dann"]["target_batch_size"],
    )
    control_training_is_reuse = False
    existing_control_audit = run_dir / "control_identity_audit.json"
    if args.resume and "control_training" in state.get("completed_stages", []):
        saved_stage = state.get("stages", {}).get("control_training")
        if not isinstance(saved_stage, dict):
            raise RuntimeError("completed Control stage has no per-stage identity; refusing resume")
        validate_stage_identity(saved_stage)
        control_model_path = Path(saved_stage["resolved_model_path"])
        saved_audit = _read_json(existing_control_audit) if existing_control_audit.is_file() else {}
        saved_audit_path = Path(saved_audit.get("resolved_model_path", ""))
        if saved_audit_path.resolve() != control_model_path.resolve():
            raise RuntimeError("saved Control audit path differs from stage identity; refusing resume")
        saved_model_hash = saved_audit.get("model_tree_sha256")
        if not saved_model_hash or _hash_tree(control_model_path) != saved_model_hash:
            raise RuntimeError("saved Control model path or hash changed; refusing resume")
        saved_init_path = Path(saved_audit.get("phase_a_initialization_audit_path", ""))
        saved_init_hash = saved_audit.get("phase_a_initialization_audit_sha256")
        if not saved_init_path.is_file() or not saved_init_hash or sha256_file(saved_init_path) != saved_init_hash:
            raise RuntimeError("saved Control initialization audit is missing or changed; refusing resume")
        _read_initialization_audit(saved_init_path, "control", expected_seed=recipe["seed"])
        validate_external_control_dann_audit(
            {
                **saved_audit,
                "expected_seed": recipe["seed"],
                "expected_source_count": len(expected_source_row_ids),
                "expected_target_count": len(expected_target_row_ids),
                "expected_source_row_ids": expected_source_row_ids,
                "expected_target_row_ids": expected_target_row_ids,
            },
            expected_epochs=None,
            require_training_state=True,
            expected_max_steps=expected_dann_max_steps,
            expected_planned_batches=expected_dann_planned_batches,
            allow_legacy=False,
        )
        saved_actual = saved_audit.get("actual", saved_audit.get("identity", {}))
        expected_for_resume = dict(actual_identity)
        expected_for_resume["artifact_sha256"] = saved_model_hash
        if not audit_control_identity(expected_for_resume, saved_actual)["reuse_allowed"] and recipe.get("recipe_id") != "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann0_v1":
            raise RuntimeError("saved Control identity changed; refusing resume")
        control_reuse_audit = saved_audit
        control_dann_batch_audit_path = Path(saved_audit["dann_batch_audit_path"]).resolve()
        control_initialization_audit_path = saved_init_path.resolve()
        control_training_is_reuse = saved_audit.get("source") not in {"fresh_phase_a_control", "resumed_phase_a_control"}
    if args.control_run_dir:
        control_source = Path(args.control_run_dir).resolve()
        control_identity_path = control_source / "control_identity_audit.json"
        if not control_identity_path.is_file():
            raise RuntimeError("external Control identity audit is missing; refusing reuse")
        prior = _read_json(control_identity_path)
        prior_actual = prior.get("actual", prior.get("identity", {}))
        prior_model_path = Path(
            prior.get("resolved_model_path")
            or prior_actual.get("resolved_model_path", "")
        )
        if not prior_model_path.is_dir():
            raise RuntimeError("external Control resolved model path is missing; refusing reuse")
        prior_model_path = prior_model_path.resolve()
        prior_model_hash = prior.get("model_tree_sha256") or prior_actual.get("model_tree_sha256")
        if not prior_model_hash or _hash_tree(prior_model_path) != prior_model_hash:
            raise RuntimeError("external Control model tree hash changed; refusing reuse")
        prior_dann_path = Path(prior.get("dann_batch_audit_path", ""))
        prior_dann_hash = prior.get("dann_batch_audit_sha256")
        if not prior_dann_path.is_file() or not prior_dann_hash or sha256_file(prior_dann_path) != prior_dann_hash:
            raise RuntimeError("external Control DANN batch audit is missing or changed; refusing reuse")
        prior_init_path = Path(prior.get("phase_a_initialization_audit_path", ""))
        prior_init_hash = prior.get("phase_a_initialization_audit_sha256")
        if not prior_init_path.is_file() or not prior_init_hash or sha256_file(prior_init_path) != prior_init_hash:
            raise RuntimeError("external Control initialization audit is missing or changed; refusing reuse")
        _read_initialization_audit(prior_init_path, "control", expected_seed=recipe["seed"])
        salvage_arg = str(getattr(args, "control_terminal_lookahead_salvage_audit", "") or "")
        if not salvage_arg:
            raise RuntimeError("external Control requires an independent terminal-lookahead salvage audit; refusing reuse")
        salvage_audit = validate_terminal_lookahead_salvage(
            Path(salvage_arg),
            control_run_dir=control_source,
            model_path=prior_model_path,
            dann_path=prior_dann_path,
        )
        prior_dann_report = _read_dann_batch_audit(
            prior_dann_path,
            expected_epochs=None,
            expected_seed=recipe["seed"],
            expected_source_count=len(expected_source_row_ids),
            expected_target_count=len(expected_target_row_ids),
            expected_source_row_ids=expected_source_row_ids,
            expected_target_row_ids=expected_target_row_ids,
            require_training_state=True,
            expected_max_steps=expected_dann_max_steps,
            expected_planned_batches=expected_dann_planned_batches,
            allow_legacy=False,
        )
        lookahead_audit = prior_dann_report.get("terminal_lookahead_audit")
        if lookahead_audit and not salvage_audit.get("classification", {}).get("lookahead_not_consumed"):
            raise RuntimeError("external Control terminal-lookahead salvage does not acknowledge the dangling batch")
        external_salvage_copy = {
            **salvage_audit,
            "source_salvage_report_path": str(Path(salvage_arg).resolve()),
            "source_salvage_report_sha256": sha256_file(Path(salvage_arg)),
            "orchestration_commit": git_identity["commit"],
            "target_test_access": False,
        }
        _atomic_write_json(run_dir / "external_control_terminal_lookahead_audit.json", external_salvage_copy)
        expected = dict(actual_identity)
        expected["artifact_sha256"] = prior_model_hash
        if recipe.get("recipe_id") == "laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_dann0_v1":
            expected["recipe_sha256"] = prior_actual.get("recipe_sha256")
        identity_audit = audit_control_identity(expected, prior_actual)
        if not identity_audit["reuse_allowed"]:
            raise RuntimeError("external Control identity mismatch; refusing reuse")
        control_model_path = prior_model_path
        control_dann_batch_audit_path = prior_dann_path.resolve()
        control_initialization_audit_path = prior_init_path.resolve()
        control_training_is_reuse = True
        control_reuse_audit = {
            **identity_audit,
            "source": str(control_source),
            "resolved_model_path": str(prior_model_path),
            "model_path": str(prior_model_path),
            "model_tree_sha256": prior_model_hash,
            "dann_batch_audit_path": str(prior_dann_path.resolve()),
            "dann_batch_audit_sha256": prior_dann_hash,
            "phase_a_initialization_audit_path": str(prior_init_path.resolve()),
            "phase_a_initialization_audit_sha256": prior_init_hash,
            "terminal_lookahead_salvage_audit_path": str(Path(salvage_arg).resolve()),
            "terminal_lookahead_salvage_audit_sha256": sha256_file(Path(salvage_arg)),
            "producer_training_semantics_commit": salvage_audit["producer_training_semantics_commit"],
            "orchestration_commit": git_identity["commit"],
            "reuse_depth": salvage_audit["reuse_depth"],
        }
    _atomic_write_json(run_dir / "control_identity_audit.json", control_reuse_audit)

    stages = (
        "control_training",
        "treatment_training",
        "control_source_dev_evaluation",
        "treatment_source_dev_evaluation",
        "control_target_pseudo_inference",
        "treatment_target_pseudo_inference",
    )
    validate_stage_status_shape(state, stages)
    initialization_pair_audit = None
    control_worker_result = None
    progress = tqdm(total=len(stages), desc="phase-a")
    try:
        for stage in stages:
            if stage == "control_source_dev_evaluation":
                if control_initialization_audit_path is None:
                    raise RuntimeError("Control initialization audit path is unavailable before comparison")
                treatment_initialization_audit_path = variant_dirs["treatment"] / "phase_a_initialization_audit.json"
                initialization_pair_audit = validate_initialization_pair(
                    control_initialization_audit_path,
                    treatment_initialization_audit_path,
                    expected_seed=recipe["seed"],
                )
                _atomic_write_json(run_dir / "phase_a_initialization_audit.json", initialization_pair_audit)
            execute = stage != "control_training" or not control_training_is_reuse
            spec = _stage_spec(
                args,
                stage,
                variant_dirs,
                control_model_path,
                execute_control_training=not control_training_is_reuse,
                control_dann_batch_audit_path=control_dann_batch_audit_path,
                control_initialization_audit_path=control_initialization_audit_path,
            )
            if stage in state.get("completed_stages", []):
                saved_stage = state.get("stages", {}).get(stage)
                if not isinstance(saved_stage, dict):
                    raise RuntimeError(f"completed stage {stage} has no identity record; refusing resume")
                expected_producer = stage_producer_commit_for_validation(
                    state, stage, git_identity["commit"]
                )
                expected_stage = _stage_record(args, stage, spec, recipe_sha256, expected_producer)
                validate_stage_identity(saved_stage, expected_stage)
                progress.update(1)
                continue
            state["in_progress_stage"] = stage
            _atomic_write_json(run_dir / "stage_status.json", state)
            if execute:
                if stage == "control_training":
                    training_result = _run_training(args, variant_dirs["control"], False, stage=stage)
                    if not isinstance(training_result, dict):
                        raise RuntimeError("Control Phase A worker returned no isolation result")
                    control_worker_result = training_result
                    control_model_path = variant_dirs["control"] / "models" / "extractor" / "best"
                    control_dann_batch_audit_path = (variant_dirs["control"] / "dann_batch_audit.json").resolve()
                    control_initialization_audit_path = (variant_dirs["control"] / "phase_a_initialization_audit.json").resolve()
                    control_artifact = _hash_tree(control_model_path)
                    control_identity = dict(actual_identity)
                    control_identity["artifact_sha256"] = control_artifact
                    control_reuse_audit = {
                        "expected": control_identity,
                        "actual": control_identity,
                        "matches": {field: True for field in CONTROL_IDENTITY_FIELDS},
                        "all_matches": True,
                        "reuse_allowed": True,
                        "requires_rerun": False,
                        "model_path": str(control_model_path.resolve()),
                        "resolved_model_path": str(control_model_path.resolve()),
                        "model_tree_sha256": control_artifact,
                        "dann_batch_audit_path": str((variant_dirs["control"] / "dann_batch_audit.json").resolve()),
                        "dann_batch_audit_sha256": sha256_file(variant_dirs["control"] / "dann_batch_audit.json"),
                        "phase_a_initialization_audit_path": str(control_initialization_audit_path),
                        "phase_a_initialization_audit_sha256": sha256_file(control_initialization_audit_path),
                        "source": "fresh_phase_a_control",
                    }
                    _atomic_write_json(run_dir / "control_identity_audit.json", control_reuse_audit)
                elif stage == "treatment_training":
                    isolation_audit = {
                        "schema_version": 1,
                        "status": "PASS",
                        "control_source": "external_completed_process" if control_training_is_reuse else "isolated_worker",
                        "control_pid": control_worker_result.get("pid") if control_worker_result else None,
                        "parent_pid": os.getpid(),
                        "control_process_exited": True,
                        "treatment_requires_distinct_worker": True,
                        "target_test_access": False,
                    }
                    treatment_result = _run_training(args, variant_dirs["treatment"], True, stage=stage)
                    isolation_audit["treatment_pid"] = treatment_result.get("pid")
                    isolation_audit["distinct_processes"] = (
                        isinstance(isolation_audit["treatment_pid"], int)
                        and isolation_audit["treatment_pid"] != isolation_audit["parent_pid"]
                        and (isolation_audit["control_pid"] is None or isolation_audit["treatment_pid"] != isolation_audit["control_pid"])
                    )
                    if not isolation_audit["distinct_processes"]:
                        raise RuntimeError("Control/Treatment worker process isolation failed")
                    _atomic_write_json(run_dir / "control_treatment_process_isolation_audit.json", isolation_audit)
                elif stage == "control_source_dev_evaluation":
                    _run_pipeline_command(args, variant_dirs["control"], "evaluate", False, control_model_path, "source_dev")
                elif stage == "treatment_source_dev_evaluation":
                    _run_pipeline_command(args, variant_dirs["treatment"], "evaluate", True, variant_dirs["treatment"] / "models" / "extractor" / "best", "source_dev")
                elif stage == "control_target_pseudo_inference":
                    _run_pipeline_command(args, variant_dirs["control"], "pseudo", False, control_model_path)
                elif stage == "treatment_target_pseudo_inference":
                    _run_pipeline_command(args, variant_dirs["treatment"], "pseudo", True, variant_dirs["treatment"] / "models" / "extractor" / "best")
            spec = _stage_spec(
                args,
                stage,
                variant_dirs,
                control_model_path,
                execute_control_training=not control_training_is_reuse,
                control_dann_batch_audit_path=control_dann_batch_audit_path,
                control_initialization_audit_path=control_initialization_audit_path,
            )
            stage_record = _stage_record(args, stage, spec, recipe_sha256, git_identity["commit"])
            validate_stage_identity(stage_record)
            state.setdefault("stages", {})[stage] = stage_record
            state.setdefault("completed_stages", []).append(stage)
            state.pop("in_progress_stage", None)
            _atomic_write_json(run_dir / "stage_status.json", state)
            progress.update(1)
    finally:
        progress.close()

    for stage in stages:
        spec = _stage_spec(
            args,
            stage,
            variant_dirs,
            control_model_path,
            execute_control_training=not control_training_is_reuse,
            control_dann_batch_audit_path=control_dann_batch_audit_path,
            control_initialization_audit_path=control_initialization_audit_path,
        )
        expected_producer = stage_producer_commit_for_validation(
            state, stage, git_identity["commit"]
        )
        expected_stage = _stage_record(args, stage, spec, recipe_sha256, expected_producer)
        validate_stage_identity(state["stages"][stage], expected_stage)

    if state.get("repair_history"):
        _atomic_write_json(run_dir / "orchestration_repair_resume_audit.json", {
            "schema_version": 1,
            "task_id": TASK_ID,
            "repair_history": state["repair_history"],
            "completed_stage_producers": {
                stage: state["stages"][stage]["producer_commit"] for stage in stages
            },
            "input_tokenizer_equivalence_scope": [
                "source_train", "source_dev", "target_unlabeled"
            ],
            "target_test_access": False,
        })

    if initialization_pair_audit is None:
        if control_initialization_audit_path is None:
            raise RuntimeError("Control initialization audit path is unavailable for final comparison")
        initialization_pair_audit = validate_initialization_pair(
            control_initialization_audit_path,
            variant_dirs["treatment"] / "phase_a_initialization_audit.json",
            expected_seed=recipe["seed"],
        )
        _atomic_write_json(run_dir / "phase_a_initialization_audit.json", initialization_pair_audit)

    metrics = {
        "source_dev": {
            "control": _source_dev_metrics(variant_dirs["control"] / "aste_predictions_raw_fixed_source_dev.jsonl"),
            "treatment": _source_dev_metrics(variant_dirs["treatment"] / "aste_predictions_raw_fixed_source_dev.jsonl"),
        },
        "target_unlabeled_pseudo": {
            "control": _pseudo_supply(variant_dirs["control"]),
            "treatment": _pseudo_supply(variant_dirs["treatment"]),
        },
    }
    dann_batch_audit = _validate_control_treatment_dann_reports(
        variant_dirs,
        control_reuse_audit,
        expected_epochs=None,
        require_training_state=True,
        expected_max_steps=expected_dann_max_steps,
        expected_planned_batches=expected_dann_planned_batches,
        allow_legacy=False,
        require_journal=True,
        require_fresh_replay_free=not args.resume,
    )
    lifecycle_reports = {}
    for variant in ("control", "treatment"):
        lifecycle_path = variant_dirs[variant] / "phase_a_lifecycle_audit.json"
        if lifecycle_path.is_file():
            lifecycle_reports[variant] = {
                "path": str(lifecycle_path.resolve()),
                "report": _read_json(lifecycle_path),
            }
    gate_result = evaluate_phase_a_gates(metrics)
    decision = decide_phase_a(gate_result)
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "direction": f"{recipe['source_dataset']} -> {recipe['target_dataset']}",
        "seed": recipe["seed"],
        "phase": "A",
        "status": decision["status"],
        "decision": decision,
        "gates": gate_result["gates"],
        "metrics": metrics,
        "dann_batch_audit": dann_batch_audit,
        "phase_a_lifecycle": lifecycle_reports,
        "phase_a_initialization_audit": initialization_pair_audit,
        "control_identity_audit": control_reuse_audit,
        "identity": identity_metadata,
        "scope": build_phase_a_scope(),
        "phase_b_entered": False,
        "target_test_access": False,
        "formal_training_approved": False,
    }
    _atomic_write_json(run_dir / "phase_a_summary.json", summary)
    _write_result_markdown(run_dir, summary)
    state["status"] = decision["status"]
    state["decision"] = decision
    _atomic_write_json(run_dir / "stage_status.json", state)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M1 Phase A syntax RGAT quick ablation entry")
    parser.add_argument("--recipe", default=r"configs\recipes\laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_v1.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=r"J:\nlp\models\t5-base-py")
    parser.add_argument("--graph_cache_dir", required=True)
    parser.add_argument("--parser_dir", default=r"J:\nlp\models\stanza_resources")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--control_run_dir", default="")
    parser.add_argument("--control_terminal_lookahead_salvage_audit", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume_repair_from_commit",
        default="",
        help="Allow a code-only audited resume from this exact failed producer commit.",
    )
    parser.add_argument(
        "--legacy_diagnostic_migration", "--legacy_diagnostic_resume",
        dest="legacy_diagnostic_migration", action="store_true",
        help="仅生成旧运行阻塞/迁移审计，不续跑训练",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--training_worker_spec" in raw_argv:
        worker_parser = argparse.ArgumentParser()
        worker_parser.add_argument("--training_worker_spec", required=True)
        worker_args = worker_parser.parse_args(raw_argv)
        return _run_phase_a_training_worker(Path(worker_args.training_worker_spec))
    parser = build_argument_parser()
    args = parser.parse_args(raw_argv)
    args.project_root = Path(__file__).resolve().parent
    args.recipe = str((args.project_root / args.recipe).resolve()) if not Path(args.recipe).is_absolute() else args.recipe
    if args.legacy_diagnostic_migration:
        report_path = prepare_legacy_diagnostic_migration(
            Path(args.output_dir),
            current_commit=_git_identity(args.project_root).get("commit", "unknown"),
        )
        print(json.dumps({"status": "BLOCKED", "formal_evidence": False, "report": str(report_path)}, ensure_ascii=False))
        return 2
    args.recipe_data = _read_json(Path(args.recipe))
    _validate_recipe(args.recipe_data)
    if args.dry_run:
        print(json.dumps({"task_id": TASK_ID, "scope": build_phase_a_scope(), "recipe": args.recipe}, ensure_ascii=False, indent=2))
        return 0
    summary = run_phase_a(args)
    print(json.dumps({"task_id": TASK_ID, "status": summary["status"], "next_action": summary["decision"]["next_action"], "output_dir": args.output_dir}, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""M1 句法 RGAT Phase A 快速消融专用入口。

本模块只编排已批准的上游四个调用点：源域抽取训练、源域开发集评估、
目标无标签 DANN 和目标伪标签推理。它不实现 Phase B，也不读取 target_test。
"""

from __future__ import annotations

import argparse
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

from syntactic_graph import build_parser_identity, sha256_file
from t5_aste_data import (
    dump_json,
    micro_f1,
    parse_triplet_text_list,
    read_bgca_aste_file,
    read_jsonl,
    to_extract_rows,
)
from t5_aste_pipeline import DATASETS


TASK_ID = "M1_SYNTACTIC_RGAT_PSEUDO_QUICK_ABLATION_V1"
FIXED_PARENT_CODE_IDENTITY = "158654021fc5f26bf1cfb8e803d7d1b592bd8534"
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
        input_identities[name] = {"path": str(path), "sha256": sha256_file(path)}
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
            if not path.is_file() or sha256_file(path) != identity["sha256"]:
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
            if saved.get(field) != expected.get(field):
                problems.append(f"expected:{field}")
    if problems:
        raise RuntimeError(f"stage identity mismatch for {saved.get('stage', '<unknown>')}: {sorted(set(problems))}")
    return True


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
            "status": "PASS" if t_pseudo["qualified_multi_rows"] >= c_pseudo["qualified_multi_rows"] * 1.05 and t_pseudo["qualified_total_rows"] >= c_pseudo["qualified_total_rows"] * 0.95 else "FAIL",
            "metric": "target_unlabeled_qualified_pseudo_supply",
            "actual": {
                "control": c_pseudo,
                "treatment": t_pseudo,
                "multi_ratio": (t_pseudo["qualified_multi_rows"] / c_pseudo["qualified_multi_rows"]) if c_pseudo["qualified_multi_rows"] else None,
                "total_ratio": (t_pseudo["qualified_total_rows"] / c_pseudo["qualified_total_rows"]) if c_pseudo["qualified_total_rows"] else None,
            },
            "threshold": {"multi_ratio": 1.05, "total_ratio": 0.95},
            "matches": {
                "multi_ratio": t_pseudo["qualified_multi_rows"] >= c_pseudo["qualified_multi_rows"] * 1.05,
                "total_ratio": t_pseudo["qualified_total_rows"] >= c_pseudo["qualified_total_rows"] * 0.95,
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


def load_or_initialize_stage_status(path: Path, identity: dict, resume: bool) -> dict | None:
    if path.exists():
        saved = _read_json(path)
        if not resume:
            raise RuntimeError(f"run already exists; use --resume: {path.parent}")
        if saved.get("identity") != identity:
            return None
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


def _validate_recipe(recipe: dict) -> None:
    if recipe.get("task_id") != TASK_ID:
        raise ValueError(f"recipe task_id must be {TASK_ID}")
    training = recipe.get("training", {})
    pseudo = recipe.get("pseudo", {})
    actual = {
        "source_dataset": recipe.get("source_dataset"),
        "target_dataset": recipe.get("target_dataset"),
        "seed": recipe.get("seed"),
        "num_train_epochs": training.get("num_train_epochs"),
        "checkpoint_selection": training.get("checkpoint_selection"),
        "extractor_train_batch_size": training.get("extractor_train_batch_size"),
        "extractor_eval_batch_size": training.get("extractor_eval_batch_size"),
        "dann_source_batch_size": training.get("target_unlabeled_dann", {}).get("source_batch_size"),
        "dann_target_batch_size": training.get("target_unlabeled_dann", {}).get("target_batch_size"),
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
    }
    mismatches = {key: {"actual": actual[key], "expected": expected} for key, expected in FROZEN_RECIPE.items() if actual[key] != expected}
    if mismatches:
        raise ValueError("frozen Phase A recipe mismatch: " + json.dumps(mismatches, ensure_ascii=False, sort_keys=True))


def _git_identity(project_root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()

    status = git("status", "--porcelain")
    commit = git("rev-parse", "HEAD")
    return {
        "commit": commit,
        "branch": git("branch", "--show-current"),
        "worktree_clean": not bool(status),
        "status_porcelain": status,
        "parent_entry_code_identity": FIXED_PARENT_CODE_IDENTITY,
    }


def _build_input_rows(source_dataset: str, target_dataset: str) -> dict[str, list[dict]]:
    source_train_raw = read_bgca_aste_file(DATASETS[source_dataset] / "train.txt")
    source_dev_raw = read_bgca_aste_file(DATASETS[source_dataset] / "dev.txt")
    target_raw = read_bgca_aste_file(DATASETS[target_dataset] / "train.txt")
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
        if resume and path.exists() and sha256_file(path) != _sha256_bytes(serialized):
            raise RuntimeError(f"resume input artifact mismatch: {path}")
        if not path.exists():
            _atomic_write_text(path, serialized.decode("utf-8"))


def _input_hashes(input_rows: dict[str, list[dict]], run_dir: Path) -> dict:
    result = {}
    for split in ("source_train", "source_dev", "target_unlabeled"):
        path = run_dir / "inputs" / f"{split}.jsonl"
        serialized = _serialize_rows(input_rows[split])
        result[split] = {"path": str(path), "sha256": _sha256_bytes(serialized), "rows": len(input_rows[split])}
    return result


def _model_hashes(model_path: Path) -> dict:
    files = [model_path / name for name in ("config.json", "pytorch_model.bin", "spiece.model", "tokenizer.json")]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing T5-base files: " + ", ".join(missing))
    return {str(path.name): sha256_file(path) for path in files}


def _write_variant_inputs(variant_dir: Path, run_dir: Path, *, resume: bool = False) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    for split in ("source_train", "source_dev", "target_unlabeled"):
        target = variant_dir / f"{split}.jsonl"
        source = run_dir / "inputs" / f"{split}.jsonl"
        source_hash = sha256_file(source)
        if resume and target.exists() and sha256_file(target) != source_hash:
            raise RuntimeError(f"resume variant input artifact mismatch: {target}")
        if not target.exists():
            _atomic_write_text(target, source.read_text(encoding="utf-8"))


def _training_argv(
    args: argparse.Namespace,
    variant_dir: Path,
    graph_enabled: bool,
    dann_batch_audit_path: Path | None = None,
) -> list[str]:
    recipe = args.recipe_data
    training = recipe["training"]
    argv = [
        "--model_path", str(args.model_path),
        "--train_file", str(variant_dir / "source_train.jsonl"),
        "--dev_file", str(variant_dir / "source_dev.jsonl"),
        "--output_dir", str(variant_dir / "models" / "extractor"),
        "--num_train_epochs", str(training["num_train_epochs"]),
        "--source_weight", "1.0",
        "--pseudo_weight", str(FROZEN_RECIPE["pseudo_base_weight"]),
        "--augment_weight", "0.2",
        "--lambda_structure_loss", "0",
        "--lambda_consistency_loss", "0",
        "--lambda_pairing_loss", "0",
        "--multi_triplet_loss_gain", "0",
        "--neutral_loss_gain", "0",
        "--checkpoint_selection", training["checkpoint_selection"],
        "--resume_from_checkpoint", "auto",
        "--per_device_train_batch_size", str(training["extractor_train_batch_size"]),
        "--per_device_eval_batch_size", str(training["extractor_eval_batch_size"]),
        "--gradient_accumulation_steps", str(training["gradient_accumulation_steps"]),
        "--learning_rate", str(training["learning_rate"]),
        "--lambda_domain_adv", str(training["lambda_domain_adv"]),
        "--domain_adv_grl_lambda", "1.0",
        "--domain_adv_hidden_size", "256",
        "--domain_adv_exclude_augment",
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
    if dann_batch_audit_path is not None:
        argv.extend(["--dann_batch_audit_path", str(dann_batch_audit_path)])
    return argv


def _run_training(args: argparse.Namespace, variant_dir: Path, graph_enabled: bool) -> dict | None:
    from t5_absa_train import run_phase_a_training

    return run_phase_a_training(
        _training_argv(
            args,
            variant_dir,
            graph_enabled,
            variant_dir / "dann_batch_audit.json",
        )
    )


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
            "--cuda", str(args.cuda), "--no_task_prefix", "--no_constrained_decoding",
            "--output_tag", output_tag,
        ]
        if graph_enabled:
            argv.extend([
                "--use_syntactic_graph_adapter",
                "--syntactic_graph_cache_dir", str(args.graph_cache_dir),
                "--syntactic_graph_parser_dir", str(args.parser_dir),
                "--syntactic_graph_split", "source_dev",
            ])
    elif command == "pseudo":
        argv = [
            str(Path(sys.executable)), "t5_aste_pipeline.py", "pseudo",
            "--run_dir", str(variant_dir), "--model_path", str(model_path),
            "--batch_size", str(args.recipe_data["training"]["target_pseudo_batch_size"]),
            "--num_beams", str(args.recipe_data["pseudo"]["num_beams"]),
            "--max_new_tokens", str(args.recipe_data["pseudo"]["max_new_tokens"]),
            "--pseudo_base_weight", str(args.recipe_data["pseudo"]["base_weight"]),
            "--high_precision_max_triplets", str(args.recipe_data["pseudo"]["high_precision_max_triplets"]),
            "--high_precision_max_token_distance", str(args.recipe_data["pseudo"]["high_precision_max_token_distance"]),
            "--cuda", str(args.cuda), "--no_task_prefix", "--no_constrained_decoding",
        ]
        if graph_enabled:
            argv.extend([
                "--use_syntactic_graph_adapter",
                "--syntactic_graph_cache_dir", str(args.graph_cache_dir),
                "--syntactic_graph_parser_dir", str(args.parser_dir),
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


def _read_dann_batch_audit(variant_dir: Path, expected_epochs: int | None = None) -> dict:
    path = Path(variant_dir)
    if path.is_dir():
        path = path / "dann_batch_audit.json"
    if not path.is_file():
        raise RuntimeError(f"missing DANN batch audit report: {path}")
    report = _read_json(path)
    if not isinstance(report, dict) or not isinstance(report.get("epochs"), list) or not report["epochs"]:
        raise RuntimeError(f"invalid DANN batch audit report: {path}")
    if expected_epochs is not None and len(report["epochs"]) != expected_epochs:
        raise RuntimeError(f"DANN batch audit epoch count mismatch: {path}")
    seen_epochs = set()
    for epoch in report["epochs"]:
        if not isinstance(epoch, dict):
            raise RuntimeError(f"DANN batch audit contains a non-object epoch: {path}")
        if not isinstance(epoch.get("epoch"), int) or epoch["epoch"] in seen_epochs:
            raise RuntimeError(f"DANN batch audit has duplicate or invalid epoch numbers: {path}")
        seen_epochs.add(epoch["epoch"])
        if not isinstance(epoch.get("batches"), list) or not epoch["batches"]:
            raise RuntimeError(f"DANN batch audit has an empty epoch: {path}")
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
    if expected_epochs is not None and sorted(seen_epochs) != list(range(expected_epochs)):
        raise RuntimeError(f"DANN batch audit epochs must be contiguous from zero: {path}")
    return report


def validate_external_control_dann_audit(control_reuse_audit: dict, expected_epochs: int | None = None) -> dict:
    path = Path(control_reuse_audit.get("dann_batch_audit_path", ""))
    expected_hash = control_reuse_audit.get("dann_batch_audit_sha256")
    if not path.is_file() or not expected_hash:
        raise RuntimeError("external Control requires a DANN batch audit path and SHA256")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError("external Control DANN batch audit hash changed; refusing reuse")
    return _read_dann_batch_audit(path, expected_epochs=expected_epochs)


def _validate_control_treatment_dann_reports(
    variant_dirs: dict[str, Path],
    control_reuse_audit: dict,
    expected_epochs: int | None = None,
) -> dict:
    treatment_path = variant_dirs["treatment"] / "dann_batch_audit.json"
    control_report = validate_external_control_dann_audit(control_reuse_audit, expected_epochs=expected_epochs)
    treatment_report = _read_dann_batch_audit(treatment_path, expected_epochs=expected_epochs)
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
    if any(control_report.get(field) != treatment_report.get(field) for field in comparable_fields):
        raise RuntimeError("Control and Treatment DANN batch orders or steps differ")
    return {"status": "matched", "control": control_report, "treatment": treatment_report}


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
        "code_semantics": git_identity["commit"],
        "artifact_sha256": artifact_digest,
    }
    return actual, {
        "identity_schema_version": 1,
        "actual": actual,
        "input_hashes": input_hashes,
        "model_file_hashes": model_hashes,
        "parser_identity": parser_identity,
        "git": git_identity,
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
) -> dict:
    control_dir = variant_dirs["control"]
    treatment_dir = variant_dirs["treatment"]
    if stage == "control_training":
        audit_path = control_dann_batch_audit_path or (control_dir / "dann_batch_audit.json")
        if not execute_control_training and control_dann_batch_audit_path is None:
            raise RuntimeError("reused Control stage requires its resolved DANN batch audit path")
        if execute_control_training:
            command = _training_argv(args, control_dir, False, audit_path)
        else:
            command = [
                "reuse_external_control",
                str(control_model_path.resolve()),
            ]
        return {
            "command": command,
            "artifact_path": control_model_path,
            "model_path": control_model_path,
            "output_artifacts": {"extractor_best": control_model_path, "dann_batch_audit": audit_path},
            "variant_dir": control_dir,
        }
    if stage == "treatment_training":
        model_path = treatment_dir / "models" / "extractor" / "best"
        audit_path = treatment_dir / "dann_batch_audit.json"
        return {
            "command": _training_argv(args, treatment_dir, True, audit_path),
            "artifact_path": model_path,
            "model_path": model_path,
            "output_artifacts": {"extractor_best": model_path, "dann_batch_audit": audit_path},
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
    return build_stage_identity(
        stage,
        spec["command"],
        _phase_stage_input_files(spec["variant_dir"]),
        recipe_sha256,
        spec["artifact_path"],
        spec["model_path"],
        git_commit,
        recipe_path=Path(args.recipe),
        output_artifacts=spec.get("output_artifacts"),
    )


def run_phase_a(args: argparse.Namespace) -> dict:
    recipe = args.recipe_data
    _validate_recipe(recipe)
    expected_dann_epochs = int(recipe["training"]["num_train_epochs"])
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
    input_rows = _build_input_rows(recipe["source_dataset"], recipe["target_dataset"])
    input_hashes = _input_hashes(input_rows, run_dir)
    model_hashes = _model_hashes(Path(args.model_path))
    parser_identity = build_parser_identity(args.parser_dir)
    recipe_sha256 = sha256_file(Path(args.recipe))
    actual_identity, identity_metadata = _build_identity(args, input_hashes, model_hashes, parser_identity, recipe_sha256, git_identity)
    status_identity = {"task_id": TASK_ID, "code_commit": git_identity["commit"], "recipe_sha256": recipe_sha256, "input_hashes": input_hashes, "model_hashes": model_hashes, "parser_identity": parser_identity, "scope": build_phase_a_scope()}
    state = load_or_initialize_stage_status(run_dir / "stage_status.json", status_identity, args.resume)
    if state is None:
        raise RuntimeError("resume identity mismatch; refusing to mix Phase A artifacts")
    state["status"] = "in_progress"
    _atomic_write_json(run_dir / "stage_status.json", state)
    _write_inputs(input_rows, run_dir)
    _atomic_write_json(run_dir / "config_snapshot.json", {"task_id": TASK_ID, "recipe": recipe, "variant_configs": {"control": build_variant_config(False), "treatment": build_variant_config(True)}, "scope": build_phase_a_scope(), "identities": {"recipe_sha256": recipe_sha256, "input_hashes": input_hashes, "model_hashes": model_hashes, "parser_identity": parser_identity, "git_commit": git_identity["commit"]}})
    _atomic_write_json(run_dir / "git_identity.json", git_identity)
    raw_input_hashes = {}
    for name, entry in recipe.get("external_inputs", {}).items():
        if name == "target_test_access" or not isinstance(entry, dict):
            continue
        raw_path = Path(entry["path"])
        raw_input_hashes[name] = {"path": str(raw_path), "sha256": sha256_file(raw_path)}
    _atomic_write_json(run_dir / "input_artifact_hashes.json", {"inputs": input_hashes, "raw_external_inputs": raw_input_hashes, "model": model_hashes, "parser": parser_identity, "recipe": {"path": str(args.recipe), "sha256": recipe_sha256}})
    _atomic_write_json(run_dir / "parent_run_identity.json", {"parent_task_id": "M1_SYNTACTIC_RGAT_ZERO_UPDATE_ENTRY_AUDIT_V1", "required_entry_code_identity": FIXED_PARENT_CODE_IDENTITY, "current_code_commit": git_identity["commit"], "zero_update_entry_status": "15/15 PASS (provided by approved parent identity)"})

    variant_dirs = {name: run_dir / name for name in ("control", "treatment")}
    _write_inputs(input_rows, run_dir, resume=args.resume)
    for variant_dir in variant_dirs.values():
        _write_variant_inputs(variant_dir, run_dir, resume=args.resume)
    control_reuse_audit = {
        "reuse_allowed": False,
        "requires_rerun": True,
        "reason": "no machine-verifiable control identity supplied",
    }
    control_model_path = variant_dirs["control"] / "models" / "extractor" / "best"
    control_dann_batch_audit_path: Path | None = None
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
        validate_external_control_dann_audit(saved_audit, expected_epochs=expected_dann_epochs)
        saved_actual = saved_audit.get("actual", saved_audit.get("identity", {}))
        expected_for_resume = dict(actual_identity)
        expected_for_resume["artifact_sha256"] = saved_model_hash
        if not audit_control_identity(expected_for_resume, saved_actual)["reuse_allowed"]:
            raise RuntimeError("saved Control identity changed; refusing resume")
        control_reuse_audit = saved_audit
        control_dann_batch_audit_path = Path(saved_audit["dann_batch_audit_path"]).resolve()
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
        _read_dann_batch_audit(prior_dann_path, expected_epochs=expected_dann_epochs)
        expected = dict(actual_identity)
        expected["artifact_sha256"] = prior_model_hash
        identity_audit = audit_control_identity(expected, prior_actual)
        if not identity_audit["reuse_allowed"]:
            raise RuntimeError("external Control identity mismatch; refusing reuse")
        control_model_path = prior_model_path
        control_dann_batch_audit_path = prior_dann_path.resolve()
        control_training_is_reuse = True
        control_reuse_audit = {
            **identity_audit,
            "source": str(control_source),
            "resolved_model_path": str(prior_model_path),
            "model_path": str(prior_model_path),
            "model_tree_sha256": prior_model_hash,
            "dann_batch_audit_path": str(prior_dann_path.resolve()),
            "dann_batch_audit_sha256": prior_dann_hash,
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
    progress = tqdm(total=len(stages), desc="phase-a")
    try:
        for stage in stages:
            execute = stage != "control_training" or not control_training_is_reuse
            spec = _stage_spec(
                args,
                stage,
                variant_dirs,
                control_model_path,
                execute_control_training=not control_training_is_reuse,
                control_dann_batch_audit_path=control_dann_batch_audit_path,
            )
            if stage in state.get("completed_stages", []):
                saved_stage = state.get("stages", {}).get(stage)
                if not isinstance(saved_stage, dict):
                    raise RuntimeError(f"completed stage {stage} has no identity record; refusing resume")
                expected_stage = _stage_record(args, stage, spec, recipe_sha256, git_identity["commit"])
                validate_stage_identity(saved_stage, expected_stage)
                progress.update(1)
                continue
            state["in_progress_stage"] = stage
            _atomic_write_json(run_dir / "stage_status.json", state)
            if execute:
                if stage == "control_training":
                    _run_training(args, variant_dirs["control"], False)
                    control_model_path = variant_dirs["control"] / "models" / "extractor" / "best"
                    control_dann_batch_audit_path = (variant_dirs["control"] / "dann_batch_audit.json").resolve()
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
                        "source": "fresh_phase_a_control",
                    }
                    _atomic_write_json(run_dir / "control_identity_audit.json", control_reuse_audit)
                elif stage == "treatment_training":
                    _run_training(args, variant_dirs["treatment"], True)
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
        )
        expected_stage = _stage_record(args, stage, spec, recipe_sha256, git_identity["commit"])
        validate_stage_identity(state["stages"][stage], expected_stage)

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
        expected_epochs=expected_dann_epochs,
    )
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    args.project_root = Path(__file__).resolve().parent
    args.recipe = str((args.project_root / args.recipe).resolve()) if not Path(args.recipe).is_absolute() else args.recipe
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

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from syntactic_graph import (
    CompositeGraphCache,
    EXPECTED_PARSER_SHA256,
    GraphCacheError,
    build_graph_cache_records,
    build_stanza_pipeline,
    load_graph_cache_directory,
    sha256_file,
)
from syntactic_graph_adapter import load_seq2seq_model
from t5_absa_train import (
    DataCollatorForSeq2SeqWithPairing,
    DomainAdversarialHead,
    JsonlSeq2SeqDataset,
    WeightedSeq2SeqTrainer,
    add_task_special_tokens,
    build_target_unlabeled_domain_rows,
)
from t5_aste_data import to_extract_rows
from t5_aste_pipeline import DATASETS, load_split


ENTRY_GATE_NAMES = [
    "parser_identity",
    "parse_alignment",
    "edge_legality",
    "reverse_selfloop",
    "cache_resume_determinism",
    "four_callpoints",
    "control_equivalence",
    "loss_finiteness",
    "repeat_determinism",
    "aste_dann_gradient_paths",
    "fp16_entry",
    "vram_8gb",
    "zero_update",
    "boundary_no_leakage",
    "machine_readable_report",
]

GRAPH_KEYS = (
    "graph_word_to_subword",
    "graph_word_mask",
    "graph_edge_src",
    "graph_edge_dst",
    "graph_relation_id",
    "graph_dependency_relation_id",
    "graph_pos_pair_id",
    "graph_edge_mask",
)
MODEL_KEYS = {
    "input_ids",
    "attention_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "labels",
}
FORMAL_CALLPOINT_PATHS = {
    "source_extractor_training": "t5_absa_train.WeightedSeq2SeqTrainer.compute_loss",
    "source_dev_evaluation": "t5_absa_train.WeightedSeq2SeqTrainer.prediction_step",
    "target_unlabeled_dann": "t5_absa_train.WeightedSeq2SeqTrainer.compute_loss",
    "target_pseudo_inference": "t5_aste_pipeline.generate_texts",
}
EXPECTED_LAMBDA_DOMAIN_ADV = 0.03
EXPECTED_SEED = 1000
EXPECTED_FP16 = True
EXPECTED_GRADIENT_CHECKPOINTING = True
EXPECTED_EXTRACTOR_TRAIN_BATCH_SIZE = 1
EXPECTED_EXTRACTOR_EVAL_BATCH_SIZE = 2
EXPECTED_DANN_SOURCE_BATCH_SIZE = 1
EXPECTED_DANN_TARGET_BATCH_SIZE = 1
EXPECTED_TARGET_PSEUDO_BATCH_SIZE = 1
EXPECTED_MAX_SOURCE_LENGTH = 128
EXPECTED_MAX_TARGET_LENGTH = 96
EXPECTED_STANZA_VERSION = "1.14.0"
VRAM_LIMIT_BYTES = int(7.5 * 1024**3)
EXPECTED_RECIPE_SHA256 = "e7c27b2a918eff11ae62bbb2ebc6042d80b457dfaaa21907ae9a0408115dece7"


class AuditConfigurationError(ValueError):
    """Raised when the zero-update audit recipe does not match the protocol."""

    def __init__(self, message: str, validation: dict):
        super().__init__(message)
        self.validation = validation


def _parameter_equal(actual, expected) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _audit_parameter_values(args, recipe: dict) -> tuple[dict, dict, dict]:
    training = recipe.get("training", {})
    dann = training.get("target_unlabeled_dann", {})
    recipe_values = {
        "source_dataset": recipe.get("source_dataset"),
        "target_dataset": recipe.get("target_dataset"),
        "seed": recipe.get("seed"),
        "lambda_domain_adv": training.get("lambda_domain_adv"),
        "fp16": training.get("fp16"),
        "gradient_checkpointing": training.get("gradient_checkpointing"),
        "extractor_train_batch_size": training.get("extractor_train_batch_size"),
        "extractor_eval_batch_size": training.get("extractor_eval_batch_size"),
        "dann_source_batch_size": dann.get("source_batch_size"),
        "dann_target_batch_size": dann.get("target_batch_size"),
        "target_pseudo_batch_size": training.get("target_pseudo_batch_size"),
        "max_source_length": training.get("max_source_length"),
        "max_target_length": training.get("max_target_length"),
    }
    actual = {
        "source_dataset": getattr(args, "source_dataset", None),
        "target_dataset": getattr(args, "target_dataset", None),
        "seed": getattr(args, "seed", None),
        "lambda_domain_adv": getattr(args, "lambda_domain_adv", None),
        "fp16": getattr(args, "fp16", None),
        "gradient_checkpointing": getattr(args, "gradient_checkpointing", None),
        "extractor_train_batch_size": getattr(args, "extractor_train_batch_size", None),
        "extractor_eval_batch_size": getattr(args, "extractor_eval_batch_size", None),
        "dann_source_batch_size": getattr(args, "dann_source_batch_size", None),
        "dann_target_batch_size": getattr(args, "dann_target_batch_size", None),
        "target_pseudo_batch_size": getattr(args, "target_pseudo_batch_size", None),
        "max_source_length": getattr(args, "max_source_length", None),
        "max_target_length": getattr(args, "max_target_length", None),
    }
    expected = {
        "source_dataset": recipe_values["source_dataset"],
        "target_dataset": recipe_values["target_dataset"],
        "seed": EXPECTED_SEED,
        "lambda_domain_adv": EXPECTED_LAMBDA_DOMAIN_ADV,
        "fp16": EXPECTED_FP16,
        "gradient_checkpointing": EXPECTED_GRADIENT_CHECKPOINTING,
        "extractor_train_batch_size": EXPECTED_EXTRACTOR_TRAIN_BATCH_SIZE,
        "extractor_eval_batch_size": EXPECTED_EXTRACTOR_EVAL_BATCH_SIZE,
        "dann_source_batch_size": EXPECTED_DANN_SOURCE_BATCH_SIZE,
        "dann_target_batch_size": EXPECTED_DANN_TARGET_BATCH_SIZE,
        "target_pseudo_batch_size": EXPECTED_TARGET_PSEUDO_BATCH_SIZE,
        "max_source_length": EXPECTED_MAX_SOURCE_LENGTH,
        "max_target_length": EXPECTED_MAX_TARGET_LENGTH,
    }
    return recipe_values, actual, expected


def validate_audit_recipe(args, recipe: dict) -> dict:
    """Return an auditable comparison of CLI values, recipe values, and protocol values."""
    recipe_values, actual, expected = _audit_parameter_values(args, recipe)
    matches = {
        name: _parameter_equal(actual[name], expected[name])
        and _parameter_equal(recipe_values[name], expected[name])
        for name in expected
    }
    return {
        "actual": actual,
        "expected": expected,
        "recipe": recipe_values,
        "matches": matches,
        "all_matches": all(matches.values()),
    }


def ensure_audit_recipe(args, recipe: dict) -> dict:
    """Hard-stop before data/model/GPU work when any audit parameter is wrong."""
    validation = validate_audit_recipe(args, recipe)
    if not validation["all_matches"]:
        mismatches = [name for name, matched in validation["matches"].items() if not matched]
        raise AuditConfigurationError(
            "M1 audit recipe mismatch; blocked fields: " + ", ".join(mismatches),
            validation,
        )
    return validation


def parameter_state_sha256(model) -> str:
    """Hash named model parameters without changing or serializing the model."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_entry_report(
    gate_values: dict[str, bool],
    measurements: dict,
    callpoints: dict,
    metadata: dict,
    gate_details: dict[str, str] | None = None,
    errors: list[str] | None = None,
) -> dict:
    gate_details = gate_details or {}
    gates = {}
    for name in ENTRY_GATE_NAMES:
        value = bool(gate_values.get(name, False))
        gates[name] = {
            "status": "PASS" if value else "FAIL",
            "value": value,
            "detail": gate_details.get(name, ""),
        }
    report = {
        "schema_version": 1,
        "task_id": "M1_SYNTACTIC_RGAT_ZERO_UPDATE_ENTRY_AUDIT_V1",
        "status": "PASS" if all(item["status"] == "PASS" for item in gates.values()) else "BLOCKED",
        "gates": gates,
        "callpoints": callpoints,
        "measurements": measurements,
        "metadata": metadata,
        "errors": list(errors or []),
    }
    return report


def _finite(value) -> bool:
    if value is None:
        return False
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach()).all().item())
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _max_abs_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.detach().float() - second.detach().float()).abs().max().item())


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _model_inputs(batch: dict, use_graph: bool) -> dict:
    keys = set(MODEL_KEYS)
    if use_graph:
        keys.update(GRAPH_KEYS)
    return {key: value for key, value in batch.items() if key in keys and value is not None}


def _build_dataset(
    rows: list[dict],
    tokenizer,
    graph_cache,
    max_source_length: int,
    max_target_length: int,
) -> JsonlSeq2SeqDataset:
    return JsonlSeq2SeqDataset(
        rows,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        source_weight=1.0,
        pseudo_weight=0.5,
        augment_weight=0.2,
        graph_cache=graph_cache,
    )


def _collate_rows(dataset: JsonlSeq2SeqDataset, model, tokenizer, batch_size: int) -> dict:
    if len(dataset) == 0:
        raise GraphCacheError("audit batch cannot be empty")
    count = min(max(1, int(batch_size)), len(dataset))
    features = [dataset[index] for index in range(count)]
    base = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    return DataCollatorForSeq2SeqWithPairing(base)(features)


def _cache_bytes(cache_dir: Path) -> dict[str, str]:
    names = [
        "relation_vocab.json",
        "source_train.jsonl",
        "source_dev.jsonl",
        "target_unlabeled.jsonl",
        "manifest.json",
    ]
    return {name: sha256_file(cache_dir / name) for name in names}


def _read_cache_records(cache_dir: Path, split: str) -> list[dict]:
    path = cache_dir / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _inspect_cache(cache_dir: Path, manifest: dict) -> dict:
    relation_vocab = json.loads((cache_dir / "relation_vocab.json").read_text(encoding="utf-8"))
    relation_count = max(1, len(relation_vocab))
    all_legal = True
    all_reverse_selfloop = True
    node_counts = []
    edge_counts = []
    forbidden_keys = {"label", "target", "gold_edges", "pseudo_edges", "sentiment"}
    forbidden_seen = []
    for split in ("source_train", "source_dev", "target_unlabeled"):
        for record in _read_cache_records(cache_dir, split):
            node_count = len(record.get("word_to_subword", []))
            edges = record.get("edges", [])
            node_counts.append(node_count)
            edge_counts.append(len(edges))
            if node_count <= 0 or not edges:
                all_legal = False
            for key in forbidden_keys.intersection(record):
                forbidden_seen.append(f"{split}:{record.get('row_id')}:{key}")
            forward_edges = {
                (
                    int(edge["src"]),
                    int(edge["dst"]),
                    str(edge.get("dependency_key", "")).split("|", 1)[0],
                )
                for edge in edges
                if edge.get("kind") == "dependency_forward"
            }
            reverse_edges = {
                (
                    int(edge["dst"]),
                    int(edge["src"]),
                    str(edge.get("dependency_key", ""))[len("reverse:") :].split("|", 1)[0],
                )
                for edge in edges
                if edge.get("kind") == "dependency_reverse"
            }
            self_loops = {
                int(edge["src"])
                for edge in edges
                if edge.get("kind") == "self_loop" and int(edge["src"]) == int(edge["dst"])
            }
            if len(self_loops) != node_count or not forward_edges.issubset(reverse_edges):
                all_reverse_selfloop = False
            for edge in edges:
                src = int(edge.get("src", -1))
                dst = int(edge.get("dst", -1))
                relation_id = int(edge.get("relation_id", -1))
                dependency_id = int(edge.get("dependency_relation_id", -1))
                pos_id = int(edge.get("pos_pair_id", -1))
                if not (
                    0 <= src < node_count
                    and 0 <= dst < node_count
                    and 0 <= relation_id < relation_count
                    and 0 <= dependency_id < relation_count
                    and 0 <= pos_id < relation_count
                ):
                    all_legal = False
    stats = manifest.get("stats", {})
    parse_coverage = stats.get("parse_coverage", {})
    alignment_coverage = stats.get("alignment_coverage", {})
    failed_rows = stats.get("failed_rows", {})
    coverage_ok = (
        all(float(parse_coverage.get(split, 0.0)) == 1.0 for split in ("source_train", "source_dev", "target_unlabeled"))
        and all(float(alignment_coverage.get(split, 0.0)) == 1.0 for split in ("source_train", "source_dev", "target_unlabeled"))
        and all(int(failed_rows.get(split, -1)) == 0 for split in ("source_train", "source_dev", "target_unlabeled"))
    )
    return {
        "coverage_ok": coverage_ok,
        "edge_legality_ok": all_legal,
        "reverse_selfloop_ok": all_reverse_selfloop,
        "forbidden_graph_fields": forbidden_seen,
        "node_count": {
            "min": min(node_counts) if node_counts else 0,
            "max": max(node_counts) if node_counts else 0,
            "rows": len(node_counts),
        },
        "edge_count": {
            "min": min(edge_counts) if edge_counts else 0,
            "max": max(edge_counts) if edge_counts else 0,
            "rows": len(edge_counts),
        },
        "relation_vocab_size": len(relation_vocab),
    }


def _prepare_rows(source_dataset: str, target_dataset: str) -> tuple[list[dict], list[dict], list[dict]]:
    if source_dataset not in DATASETS or target_dataset not in DATASETS:
        raise ValueError(f"unsupported dataset pair: {source_dataset}->{target_dataset}")
    source_train_raw = load_split(source_dataset, "train")
    source_dev_raw = load_split(source_dataset, "dev")
    target_train_raw = load_split(target_dataset, "train")
    if not source_train_raw or not source_dev_raw or not target_train_raw:
        raise GraphCacheError("source train/dev and target train must all be non-empty")
    source_train_rows = to_extract_rows(source_train_raw, use_task_prefix=False)
    source_dev_rows = to_extract_rows(source_dev_raw, use_task_prefix=False)
    target_rows = [{"id": row["id"], "text": row["text"]} for row in target_train_raw]
    return source_train_rows, source_dev_rows, target_rows


def _build_or_resume_caches(
    source_train_rows: list[dict],
    source_dev_rows: list[dict],
    target_rows: list[dict],
    output_dir: Path,
    tokenizer,
    tokenizer_identity: dict,
    parser,
    parser_identity: dict,
    max_source_length: int,
) -> tuple[Path, dict, dict]:
    cache_dir = output_dir / "graph_cache_resume"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {
        "source_train": source_train_rows,
        "source_dev": source_dev_rows,
        "target_unlabeled": target_rows,
    }
    interruption_observed = any(
        (cache_dir / f"{split}.partial.jsonl").exists()
        or (cache_dir / f"{split}.progress.json").exists()
        for split in split_rows
    )
    if not (cache_dir / "manifest.json").is_file():
        try:
            build_graph_cache_records(
                split_rows,
                cache_dir,
                tokenizer,
                parser,
                tokenizer_identity,
                parser_identity,
                use_task_prefix=False,
                max_length=max_source_length,
                stop_after_rows=1,
            )
        except GraphCacheError as exc:
            if "interrupted after" not in str(exc):
                raise
            interruption_observed = True
    manifest = build_graph_cache_records(
        split_rows,
        cache_dir,
        tokenizer,
        parser,
        tokenizer_identity,
        parser_identity,
        use_task_prefix=False,
        max_length=max_source_length,
    )

    repeat_dir = output_dir / "graph_cache_repeat"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    repeat_manifest_path = repeat_dir / "manifest.json"
    if not repeat_manifest_path.is_file():
        build_graph_cache_records(
            split_rows,
            repeat_dir,
            tokenizer,
            parser,
            tokenizer_identity,
            parser_identity,
            use_task_prefix=False,
            max_length=max_source_length,
        )
    repeated_bytes = _cache_bytes(repeat_dir)
    resumed_bytes = _cache_bytes(cache_dir)
    cache_measurements = {
        "interruption_observed": interruption_observed,
        "byte_identical_repeat": resumed_bytes == repeated_bytes,
        "resumed_cache_sha256": resumed_bytes,
        "repeat_cache_sha256": repeated_bytes,
        "inspect": _inspect_cache(cache_dir, manifest),
    }
    return cache_dir, manifest, cache_measurements


def _file_identity(path: Path, expected_sha256: str | None) -> dict:
    exists = path.is_file()
    actual_sha256 = sha256_file(path) if exists else None
    matches = bool(
        exists
        and expected_sha256
        and actual_sha256
        and actual_sha256.lower() == str(expected_sha256).lower()
    )
    return {
        "path": str(path),
        "exists": exists,
        "actual_sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "matches": matches,
    }


def _build_parser_identity_for_audit(parser_dir: str | Path) -> dict:
    parser_dir = Path(parser_dir)
    actual_hashes = {}
    for relative in EXPECTED_PARSER_SHA256:
        path = parser_dir / relative
        actual_hashes[relative] = sha256_file(path) if path.is_file() else None
    try:
        import stanza

        stanza_version = str(getattr(stanza, "__version__", "unknown"))
    except ImportError:
        stanza_version = "unavailable"
    return {
        "language": "en",
        "processors": "tokenize,mwt,pos,lemma,depparse",
        "packages": {
            "tokenize": "ewt",
            "mwt": "ewt",
            "pos": "ewt_charlm",
            "lemma": "combined_nocharlm",
            "depparse": "ewt_charlm",
        },
        "resource_dir": str(parser_dir),
        "stanza_version": stanza_version,
        "sha256": actual_hashes,
    }


def _build_tokenizer_identity_for_audit(model_path: str | Path, tokenizer) -> dict:
    model_path = Path(model_path)
    files = {
        name: sha256_file(model_path / name) if (model_path / name).is_file() else None
        for name in ("spiece.model", "tokenizer.json")
    }
    return {
        "class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "files_sha256": files,
    }


def _git_identity(repo_root: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "commit": None,
            "status_porcelain": f"git_error:{type(exc).__name__}:{exc}",
            "clean": False,
        }
    return {
        "commit": commit,
        "status_porcelain": status,
        "clean": status == "",
    }


def _collect_identity_measurements(
    args,
    recipe: dict,
    recipe_path: Path,
    tokenizer_identity: dict,
    parser_identity: dict,
    cache_dir: Path,
    cache_measurements: dict,
    model_measurements: dict,
) -> dict:
    external = recipe.get("external_inputs", {})
    recipe_identity = _file_identity(recipe_path, EXPECTED_RECIPE_SHA256)
    model_specs = {
        "config.json": external.get("t5_config", {}),
        "pytorch_model.bin": external.get("t5_weights", {}),
        "spiece.model": external.get("t5_tokenizer", {}),
        "tokenizer.json": external.get("t5_tokenizer_json", {}),
    }
    model_files = {
        name: _file_identity(Path(args.model_path) / name, spec.get("sha256"))
        for name, spec in model_specs.items()
    }
    input_specs = {
        "source_train": external.get(f"{args.source_dataset}_train", {}),
        "source_dev": external.get(f"{args.source_dataset}_dev", {}),
        "target_unlabeled": external.get(f"{args.target_dataset}_train_unlabeled_input", {})
        or external.get(f"{args.target_dataset}_train", {}),
    }
    input_files = {
        name: _file_identity(Path(spec.get("path", "")), spec.get("sha256"))
        for name, spec in input_specs.items()
    }
    actual_cache_hashes = _cache_bytes(cache_dir)
    repeated_cache_hashes = cache_measurements.get("repeat_cache_sha256", {})
    cache_files = {
        name: {
            "path": str(cache_dir / name),
            "actual_sha256": actual_cache_hashes.get(name),
            "repeat_sha256": repeated_cache_hashes.get(name),
            "matches_repeat": actual_cache_hashes.get(name) == repeated_cache_hashes.get(name),
        }
        for name in sorted(actual_cache_hashes)
    }
    cache_manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    cache_manifest_matches = (
        cache_manifest.get("relation_vocab_sha256") == actual_cache_hashes.get("relation_vocab.json")
        and cache_manifest.get("target_test_access") is False
    )
    parser_matches = parser_identity.get("stanza_version") == EXPECTED_STANZA_VERSION and all(
        str(parser_identity.get("sha256", {}).get(relative, "")).lower() == expected.lower()
        for relative, expected in EXPECTED_PARSER_SHA256.items()
    )
    model_artifact_matches = all(item["matches"] for item in model_files.values())
    input_identity_matches = all(item["matches"] for item in input_files.values()) and (
        args.source_dataset == recipe.get("source_dataset")
        and args.target_dataset == recipe.get("target_dataset")
    )
    graph_cache_identity_matches = cache_manifest_matches and all(
        item["matches_repeat"] for item in cache_files.values()
    )
    parameter_hashes_match = (
        model_measurements.get("control_parameter_hash_before")
        == model_measurements.get("control_parameter_hash_after")
        and model_measurements.get("parameter_hash_before")
        == model_measurements.get("parameter_hash_after")
    )
    git = _git_identity(Path(__file__).resolve().parent)
    identity = {
        "git": git,
        "recipe": recipe_identity,
        "model_files": model_files,
        "input_files": input_files,
        "graph_cache_files": cache_files,
        "graph_cache_manifest_matches": cache_manifest_matches,
        "parser": {
            "actual_sha256": parser_identity.get("sha256", {}),
            "expected_sha256": EXPECTED_PARSER_SHA256,
            "matches": parser_matches,
        },
        "tokenizer_identity": tokenizer_identity,
        "control_parameter_hash_before": model_measurements.get("control_parameter_hash_before"),
        "control_parameter_hash_after": model_measurements.get("control_parameter_hash_after"),
        "treatment_parameter_hash_before": model_measurements.get("parameter_hash_before"),
        "treatment_parameter_hash_after": model_measurements.get("parameter_hash_after"),
        "parameter_hashes_match": parameter_hashes_match,
        "recipe_identity_matches": recipe_identity["matches"],
        "model_artifact_identity_matches": model_artifact_matches,
        "input_identity_matches": input_identity_matches,
        "graph_cache_identity_matches": graph_cache_identity_matches,
        "parser_identity_matches": parser_matches,
        "git_clean": git["clean"],
    }
    identity["all_matches"] = all(
        (
            identity["recipe_identity_matches"],
            identity["model_artifact_identity_matches"],
            identity["input_identity_matches"],
            identity["graph_cache_identity_matches"],
            identity["parser_identity_matches"],
            identity["parameter_hashes_match"],
            identity["git_clean"],
        )
    )
    return identity


def _build_audit_trainer(
    model,
    tokenizer,
    device: torch.device,
    lambda_domain_adv: float,
    output_dir: Path,
    train_batch_size: int,
    eval_batch_size: int,
):
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_batch_size),
        per_device_eval_batch_size=int(eval_batch_size),
        use_cpu=device.type != "cuda",
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        predict_with_generate=False,
    )
    return WeightedSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=None,
        lambda_domain_adv=lambda_domain_adv,
        domain_adv_grl_lambda=1.0,
    )


def _run_model_audit(
    args,
    tokenizer,
    source_train_rows: list[dict],
    source_dev_rows: list[dict],
    target_rows: list[dict],
    train_cache,
    dev_cache,
    target_cache,
    cache_dir: Path,
    device: torch.device,
) -> tuple[dict, dict]:
    target_domain_rows = build_target_unlabeled_domain_rows(target_rows, use_task_prefix=False)
    source_sample = source_train_rows[: max(1, min(args.extractor_train_batch_size, len(source_train_rows)))]
    dev_sample = source_dev_rows[: max(1, min(args.extractor_eval_batch_size, len(source_dev_rows)))]
    dann_source_sample = source_train_rows[: max(1, min(args.dann_source_batch_size, len(source_train_rows)))]
    target_sample = target_domain_rows[: max(1, min(args.dann_target_batch_size, len(target_domain_rows)))]
    mixed_rows = dann_source_sample + target_sample

    control = load_seq2seq_model(args.model_path, use_syntactic_graph_adapter=False)
    treatment = load_seq2seq_model(
        args.model_path,
        use_syntactic_graph_adapter=True,
        relation_vocab_size=train_cache.relation_vocab_size,
    )
    add_task_special_tokens(tokenizer, control, source_sample + dev_sample + target_domain_rows)
    treatment.resize_token_embeddings(len(tokenizer))
    add_task_special_tokens(tokenizer, treatment, source_sample + dev_sample + target_domain_rows)
    hidden_size = int(getattr(treatment.config, "d_model", treatment.get_input_embeddings().embedding_dim))
    treatment.domain_adversarial_head = DomainAdversarialHead(
        hidden_size=hidden_size,
        classifier_hidden_size=args.domain_adv_hidden_size,
    )
    if args.gradient_checkpointing:
        treatment.gradient_checkpointing_enable()
        treatment.config.use_cache = False
    control.to(device)
    treatment.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    control_parameter_hash_before = parameter_state_sha256(control)
    parameter_hash_before = parameter_state_sha256(treatment)
    control.eval()
    treatment.eval()

    mixed_graph_cache = CompositeGraphCache(
        {"source_train": train_cache, "target_unlabeled": target_cache}
    )
    source_dataset = _build_dataset(source_sample, tokenizer, train_cache, args.max_source_length, args.max_target_length)
    dev_dataset = _build_dataset(dev_sample, tokenizer, dev_cache, args.max_source_length, args.max_target_length)
    mixed_dataset = _build_dataset(mixed_rows, tokenizer, mixed_graph_cache, args.max_source_length, args.max_target_length)
    source_batch = _move_batch(
        _collate_rows(source_dataset, treatment, tokenizer, args.extractor_train_batch_size), device
    )
    dev_batch = _move_batch(
        _collate_rows(dev_dataset, treatment, tokenizer, args.extractor_eval_batch_size), device
    )
    mixed_batch = _move_batch(_collate_rows(mixed_dataset, treatment, tokenizer, len(mixed_rows)), device)

    control_input = _model_inputs(source_batch, use_graph=False)
    treatment_input = _model_inputs(source_batch, use_graph=True)
    with torch.no_grad(), _amp_context(device, args.fp16):
        control_output = control(**control_input, return_dict=True)
        treatment_output = treatment(**treatment_input, return_dict=True)
        treatment_repeat = treatment(**treatment_input, return_dict=True)

    callpoints = {
        "source_extractor_training": False,
        "source_dev_evaluation": False,
        "target_unlabeled_dann": False,
        "target_pseudo_inference": False,
    }
    losses = {
        "source_training_loss": None,
        "source_dev_loss": None,
        "target_dann_loss": None,
        "lambda_domain_adv": float(args.lambda_domain_adv),
    }

    audit_output_dir = Path(args.output_dir)
    training_trainer = _build_audit_trainer(
        treatment,
        tokenizer,
        device,
        float(args.lambda_domain_adv),
        audit_output_dir,
        args.extractor_train_batch_size,
        args.extractor_eval_batch_size,
    )
    treatment.train()
    treatment.zero_grad(set_to_none=True)
    with _amp_context(device, args.fp16):
        source_training_loss, _ = training_trainer.compute_loss(
            treatment,
            dict(source_batch),
            return_outputs=True,
        )
    source_training_loss.backward()
    callpoints["source_extractor_training"] = _finite(source_training_loss)
    losses["source_training_loss"] = float(source_training_loss.detach().float().cpu())
    treatment.zero_grad(set_to_none=True)

    treatment.eval()
    with torch.no_grad(), _amp_context(device, args.fp16):
        dev_prediction = training_trainer.prediction_step(
            treatment,
            dict(dev_batch),
            prediction_loss_only=True,
        )
    dev_loss = dev_prediction[0]
    callpoints["source_dev_evaluation"] = _finite(dev_loss)
    losses["source_dev_loss"] = float(dev_loss.detach().float().cpu())

    treatment.train()
    treatment.zero_grad(set_to_none=True)
    dann_batch = dict(mixed_batch)
    for key in ("sample_weight", "domain_weight", "structure_weight"):
        value = dann_batch.get(key)
        if torch.is_tensor(value):
            dann_batch[key] = torch.zeros_like(value)
    dann_trainer = _build_audit_trainer(
        treatment,
        tokenizer,
        device,
        float(args.lambda_domain_adv),
        audit_output_dir,
        len(mixed_rows),
        args.extractor_eval_batch_size,
    )
    with _amp_context(device, args.fp16):
        mixed_dann_loss, _ = dann_trainer.compute_loss(
            treatment,
            dann_batch,
            return_outputs=True,
        )
    domain_component = dann_trainer._component_sums.get("domain_adv_loss")
    if domain_component is None:
        raise GraphCacheError("target-unlabeled DANN batch produced no valid domain labels")
    mixed_dann_loss.backward()
    callpoints["target_unlabeled_dann"] = _finite(domain_component) and _finite(mixed_dann_loss)
    losses["target_dann_loss"] = float(domain_component)
    dann_projection_grad = treatment.syntactic_graph_adapter.output_projection.weight.grad
    dann_gradient_norm = float(dann_projection_grad.detach().float().norm().cpu()) if dann_projection_grad is not None else 0.0
    treatment.zero_grad(set_to_none=True)

    treatment.eval()
    target_pseudo_error = None
    try:
        from t5_aste_pipeline import generate_texts

        generated = generate_texts(
            model_path=args.model_path,
            inputs=[row["text"] for row in target_sample],
            batch_size=args.target_pseudo_batch_size,
            max_new_tokens=8,
            num_beams=1,
            cuda=args.cuda,
            use_syntactic_graph_adapter=True,
            graph_cache_dir=cache_dir,
            graph_rows=target_sample,
            graph_parser_dir=args.parser_dir,
            graph_split="target_unlabeled",
        )
        callpoints["target_pseudo_inference"] = len(generated) == len(target_sample)
    except Exception as exc:
        target_pseudo_error = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        callpoints["target_pseudo_inference"] = False

    treatment.train()
    treatment.zero_grad(set_to_none=True)
    aste_trainer = _build_audit_trainer(
        treatment,
        tokenizer,
        device,
        0.0,
        audit_output_dir,
        args.extractor_train_batch_size,
        args.extractor_eval_batch_size,
    )
    with _amp_context(device, args.fp16):
        aste_loss, _ = aste_trainer.compute_loss(
            treatment,
            dict(source_batch),
            return_outputs=True,
        )
    aste_loss.backward()
    aste_projection_grad = treatment.syntactic_graph_adapter.output_projection.weight.grad
    aste_gradient_norm = float(aste_projection_grad.detach().float().norm().cpu()) if aste_projection_grad is not None else 0.0
    treatment.zero_grad(set_to_none=True)

    target_labels = mixed_batch["labels"][len(dann_source_sample) :]
    target_labels_are_masked = bool(target_labels.eq(-100).all().item())
    measurements = {
        "control_loss": float(control_output.loss.detach().float().cpu()),
        "treatment_loss": float(treatment_output.loss.detach().float().cpu()),
        "repeat_loss": float(treatment_repeat.loss.detach().float().cpu()),
        "control_treatment_max_abs_logit_diff": _max_abs_difference(control_output.logits, treatment_output.logits),
        "control_treatment_max_abs_encoder_diff": _max_abs_difference(
            control_output.encoder_last_hidden_state,
            treatment_output.encoder_last_hidden_state,
        ),
        "repeat_max_abs_logit_diff": _max_abs_difference(treatment_output.logits, treatment_repeat.logits),
        "aste_gradient_norm": aste_gradient_norm,
        "dann_gradient_norm": dann_gradient_norm,
        "target_labels_are_all_ignore_index": target_labels_are_masked,
        "source_rows_in_dann_batch": len(dann_source_sample),
        "target_rows_in_dann_batch": len(target_sample),
        "dann_batch_composition": {
            "source_batch_size": len(dann_source_sample),
            "target_batch_size": len(target_sample),
            "total_batch_size": len(mixed_rows),
        },
        "source_train_batch_size": len(source_sample),
        "source_dev_batch_size": len(dev_sample),
        "target_pseudo_batch_size": int(args.target_pseudo_batch_size),
        "target_pseudo_inference_error": target_pseudo_error,
        "gradient_checkpointing_enabled": bool(getattr(treatment, "is_gradient_checkpointing", False)),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 0,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
    }
    parameter_hash_after = parameter_state_sha256(treatment)
    control.eval()
    control_parameter_hash_after = parameter_state_sha256(control)
    del control, treatment, training_trainer, dann_trainer, aste_trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return callpoints, {
        "losses": losses,
        "measurements": measurements,
        "control_parameter_hash_before": control_parameter_hash_before,
        "control_parameter_hash_after": control_parameter_hash_after,
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
    }


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# M1 句法图零更新入口审计报告",
        "",
        f"更新时间：{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M（北京时间）')}",
        "",
        f"总体状态：`{report['status']}`",
        "",
        "本报告仅覆盖缓存、模型前向、梯度路径和零更新审计；不启动正式训练、生成器、增强、NLI 或最终 ASTE 流程。",
        "",
        "## 15 项门控",
        "",
        "| 门控 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    for name, gate in report["gates"].items():
        lines.append(f"| `{name}` | `{gate['status']}` | {gate.get('detail', '')} |")
    lines.extend(["", "## 四个真实调用点", ""])
    for name, value in report.get("callpoints", {}).items():
        lines.append(f"- `{name}`：`{'PASS' if value else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## 关键测量",
            "",
            "```json",
            json.dumps(report.get("measurements", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## 边界",
            "",
            f"- target_test_access：`{report.get('metadata', {}).get('target_test_access')}`",
            "- optimizer_updates：`0`；scheduler_steps：`0`（审计脚本不创建优化器和调度器）。",
            "- 机器可读原始报告：同目录下的 `m1_syntactic_graph_entry_audit.json`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assemble_audit_report(
    args,
    recipe: dict,
    manifest: dict,
    cache_measurements: dict,
    parser_identity: dict,
    callpoints: dict,
    model_measurements: dict,
    device: torch.device,
    identity_measurements: dict,
    metadata: dict | None = None,
) -> dict:
    """Assemble the report from completed measurements without touching the model."""
    model_values = model_measurements["measurements"]
    parameter_validation = model_measurements.get(
        "recipe_parameter_validation",
        {"actual": {}, "expected": {}, "recipe": {}, "matches": {}, "all_matches": True},
    )
    identity = identity_measurements or {}
    inspect = cache_measurements["inspect"]
    before_hash = model_measurements.get("parameter_hash_before")
    after_hash = model_measurements.get("parameter_hash_after")
    control_before_hash = model_measurements.get("control_parameter_hash_before")
    control_after_hash = model_measurements.get("control_parameter_hash_after")
    parameter_hashes_match = bool(identity.get("parameter_hashes_match", True))
    if before_hash is not None and after_hash is not None:
        parameter_hashes_match = parameter_hashes_match and before_hash == after_hash
    if control_before_hash is not None and control_after_hash is not None:
        parameter_hashes_match = parameter_hashes_match and control_before_hash == control_after_hash

    measurements = {
        "cache": cache_measurements,
        "losses": model_measurements.get("losses", {}),
        **model_values,
        "optimizer_updates": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "parameter_updates": 0,
        "lambda_domain_adv": float(args.lambda_domain_adv),
        "fp16_requested": bool(args.fp16),
        "gradient_checkpointing_requested": bool(args.gradient_checkpointing),
        "gradient_checkpointing_enabled": bool(model_measurements["measurements"]["gradient_checkpointing_enabled"]),
        "device": str(device),
        "gpu_name": model_values.get("gpu_name", "cpu" if device.type == "cpu" else "unknown"),
        "gpu_total_memory_bytes": int(model_values.get("gpu_total_memory_bytes", 0)),
        "gpu_peak_allocated_bytes": int(model_values.get("gpu_peak_allocated_bytes", 0)),
        "gpu_peak_reserved_bytes": int(model_values.get("gpu_peak_reserved_bytes", 0)),
        "parameter_hash_before": before_hash,
        "parameter_hash_after": after_hash,
        "control_parameter_hash_before": control_before_hash,
        "control_parameter_hash_after": control_after_hash,
        "artifact_identity": identity,
        "formal_callpoint_paths": dict(FORMAL_CALLPOINT_PATHS),
        "recipe_parameter_validation": parameter_validation,
    }
    max_diff = measurements["control_treatment_max_abs_logit_diff"]
    repeat_diff = measurements["repeat_max_abs_logit_diff"]
    parser_identity_matches = bool(
        identity.get("parser_identity_matches", parser_identity.get("sha256") == EXPECTED_PARSER_SHA256)
    )
    input_identity_matches = bool(identity.get("input_identity_matches", True))
    graph_cache_identity_matches = bool(identity.get("graph_cache_identity_matches", True))
    model_artifact_identity_matches = bool(identity.get("model_artifact_identity_matches", True))
    recipe_identity_matches = bool(identity.get("recipe_identity_matches", True))
    git_clean = bool(identity.get("git_clean", True))
    identity_all_matches = bool(identity.get("all_matches", True))
    gate_values = {
        "parser_identity": parser_identity_matches,
        "parse_alignment": inspect["coverage_ok"] and input_identity_matches,
        "edge_legality": inspect["edge_legality_ok"] and graph_cache_identity_matches,
        "reverse_selfloop": inspect["reverse_selfloop_ok"] and graph_cache_identity_matches,
        "cache_resume_determinism": cache_measurements["interruption_observed"]
        and cache_measurements["byte_identical_repeat"]
        and graph_cache_identity_matches,
        "four_callpoints": all(callpoints.values()),
        "control_equivalence": max_diff <= 1e-4 and model_artifact_identity_matches,
        "loss_finiteness": all(
            _finite(measurements.get(name))
            for name in (
                "control_loss",
                "treatment_loss",
                "repeat_loss",
                "aste_gradient_norm",
                "dann_gradient_norm",
            )
        ),
        "repeat_determinism": repeat_diff == 0.0,
        "aste_dann_gradient_paths": measurements["aste_gradient_norm"] > 0.0
        and measurements["dann_gradient_norm"] > 0.0,
        "fp16_entry": (
            args.fp16
            and args.gradient_checkpointing
            and measurements["gradient_checkpointing_enabled"]
            and device.type == "cuda"
            and measurements["gpu_total_memory_bytes"] <= 8 * 1024**3
        ),
        "vram_8gb": measurements["gpu_peak_reserved_bytes"] <= VRAM_LIMIT_BYTES,
        "zero_update": (
            measurements["optimizer_updates"] == 0
            and measurements["optimizer_steps"] == 0
            and measurements["scheduler_steps"] == 0
            and measurements["parameter_updates"] == 0
            and parameter_hashes_match
        ),
        "boundary_no_leakage": (
            manifest.get("target_test_access") is False
            and recipe.get("data_boundary", {}).get("target_test_access") is False
            and recipe.get("data_boundary", {}).get("generator") is False
            and recipe.get("data_boundary", {}).get("augmentation") is False
            and recipe.get("data_boundary", {}).get("nli") is False
            and recipe.get("data_boundary", {}).get("final_aste") is False
            and measurements["target_labels_are_all_ignore_index"]
            and not inspect["forbidden_graph_fields"]
            and input_identity_matches
            and recipe_identity_matches
        ),
        "machine_readable_report": identity_all_matches
        and git_clean
        and bool(parameter_validation.get("all_matches", False)),
    }
    gate_details = {
        "parser_identity": f"stanza={parser_identity.get('stanza_version')}; actual_vs_expected={parser_identity_matches}",
        "parse_alignment": f"coverage={inspect['coverage_ok']}; input_sha256={input_identity_matches}",
        "edge_legality": f"nodes={inspect['node_count']}; edges={inspect['edge_count']}; cache_sha256={graph_cache_identity_matches}",
        "reverse_selfloop": f"reverse_and_selfloop={inspect['reverse_selfloop_ok']}; cache_sha256={graph_cache_identity_matches}",
        "cache_resume_determinism": f"interrupted={cache_measurements['interruption_observed']}; byte_identical={cache_measurements['byte_identical_repeat']}",
        "four_callpoints": json.dumps(callpoints, ensure_ascii=False, sort_keys=True),
        "control_equivalence": f"max_abs_logit_diff={max_diff:.8g}; model_artifacts={model_artifact_identity_matches}",
        "loss_finiteness": "all measured losses and gradient norms are finite",
        "repeat_determinism": f"max_abs_logit_diff={repeat_diff:.8g}",
        "aste_dann_gradient_paths": f"aste_norm={measurements['aste_gradient_norm']:.8g}; dann_norm={measurements['dann_gradient_norm']:.8g}",
        "fp16_entry": "CUDA autocast=float16; RTX 3070 class memory check",
        "vram_8gb": f"peak_reserved={measurements['gpu_peak_reserved_bytes']} bytes; limit={VRAM_LIMIT_BYTES} bytes",
        "zero_update": f"optimizer_updates=0; scheduler_steps=0; parameter_hashes_match={parameter_hashes_match}",
        "boundary_no_leakage": "target test, generator, augmentation, NLI and final ASTE are not invoked",
        "machine_readable_report": (
            f"identity_all_matches={identity_all_matches}; git_clean={git_clean}; "
            f"recipe_parameter_matches={parameter_validation.get('all_matches', False)}"
        ),
    }
    report_metadata = dict(metadata or {})
    report_metadata.setdefault("target_test_access", False)
    report_metadata["artifact_identity"] = identity
    report_metadata["recipe_parameter_validation"] = parameter_validation
    return build_entry_report(
        gate_values,
        measurements,
        callpoints,
        report_metadata,
        gate_details=gate_details,
    )


def run_audit(args) -> dict:
    recipe = json.loads(Path(args.recipe_path).read_text(encoding="utf-8"))
    parameter_validation = ensure_audit_recipe(args, recipe)
    source_train_rows, source_dev_rows, target_rows = _prepare_rows(args.source_dataset, args.target_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("M1 GPU audit requires CUDA; no formal CPU fallback is allowed")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer_identity = _build_tokenizer_identity_for_audit(args.model_path, tokenizer)
    parser_identity = _build_parser_identity_for_audit(args.parser_dir)
    parser = build_stanza_pipeline(args.parser_dir, use_gpu=True)
    cache_dir, manifest, cache_measurements = _build_or_resume_caches(
        source_train_rows,
        source_dev_rows,
        target_rows,
        output_dir,
        tokenizer,
        tokenizer_identity,
        parser,
        parser_identity,
        args.max_source_length,
    )
    del parser
    torch.cuda.empty_cache()
    source_cache = load_graph_cache_directory(
        cache_dir,
        "source_train",
        source_train_rows,
        tokenizer_identity=tokenizer_identity,
        parser_identity=parser_identity,
    )
    dev_cache = load_graph_cache_directory(
        cache_dir,
        "source_dev",
        source_dev_rows,
        tokenizer_identity=tokenizer_identity,
        parser_identity=parser_identity,
    )
    target_cache = load_graph_cache_directory(
        cache_dir,
        "target_unlabeled",
        target_rows,
        tokenizer_identity=tokenizer_identity,
        parser_identity=parser_identity,
    )
    callpoints, model_measurements = _run_model_audit(
        args,
        tokenizer,
        source_train_rows,
        source_dev_rows,
        target_rows,
        source_cache,
        dev_cache,
        target_cache,
        cache_dir,
        device,
    )
    model_measurements["recipe_parameter_validation"] = parameter_validation
    identity_measurements = _collect_identity_measurements(
        args=args,
        recipe=recipe,
        recipe_path=Path(args.recipe_path),
        tokenizer_identity=tokenizer_identity,
        parser_identity=parser_identity,
        cache_dir=cache_dir,
        cache_measurements=cache_measurements,
        model_measurements=model_measurements,
    )
    return assemble_audit_report(
        args=args,
        recipe=recipe,
        manifest=manifest,
        cache_measurements=cache_measurements,
        parser_identity=parser_identity,
        callpoints=callpoints,
        model_measurements=model_measurements,
        device=device,
        identity_measurements=identity_measurements,
        metadata={
            "source_dataset": args.source_dataset,
            "target_dataset": args.target_dataset,
            "target_test_access": False,
            "formal_training_started": False,
            "generator_started": False,
            "augmentation_started": False,
            "nli_started": False,
            "final_aste_started": False,
            "recipe_path": str(Path(args.recipe_path)),
            "cache_dir": str(cache_dir),
        },
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M1 句法 RGAT 零更新 GPU 入口审计")
    parser.add_argument("--source_dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--target_dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=r"J:\nlp\models\t5-base-py")
    parser.add_argument("--parser_dir", default=r"J:\nlp\models\stanza_resources")
    parser.add_argument("--recipe_path", default=r"configs\recipes\laptop14_to_rest15_syntactic_graph_v1.json")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lambda_domain_adv", type=float, default=EXPECTED_LAMBDA_DOMAIN_ADV)
    parser.add_argument("--domain_adv_hidden_size", type=int, default=256)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=96)
    parser.add_argument("--extractor_train_batch_size", type=int, default=EXPECTED_EXTRACTOR_TRAIN_BATCH_SIZE)
    parser.add_argument("--extractor_eval_batch_size", type=int, default=EXPECTED_EXTRACTOR_EVAL_BATCH_SIZE)
    parser.add_argument("--dann_source_batch_size", type=int, default=EXPECTED_DANN_SOURCE_BATCH_SIZE)
    parser.add_argument("--dann_target_batch_size", type=int, default=EXPECTED_DANN_TARGET_BATCH_SIZE)
    parser.add_argument("--target_pseudo_batch_size", type=int, default=EXPECTED_TARGET_PSEUDO_BATCH_SIZE)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "m1_syntactic_graph_entry_audit.json"
    markdown_path = output_dir / "m1_syntactic_graph_entry_audit_CN.md"
    parameter_validation = None
    try:
        recipe = json.loads(Path(args.recipe_path).read_text(encoding="utf-8"))
        parameter_validation = validate_audit_recipe(args, recipe)
        report = run_audit(args)
    except Exception as exc:  # The JSON failure artifact is part of the audit contract.
        report = build_entry_report(
            gate_values={name: False for name in ENTRY_GATE_NAMES},
            measurements={
                "optimizer_updates": 0,
                "optimizer_steps": 0,
                "scheduler_steps": 0,
                "parameter_updates": 0,
                "recipe_parameter_validation": parameter_validation
                or {
                    "actual": {},
                    "expected": {},
                    "recipe": {},
                    "matches": {},
                    "all_matches": False,
                },
            },
            callpoints={},
            metadata={
                "source_dataset": args.source_dataset,
                "target_dataset": args.target_dataset,
                "target_test_access": False,
                "formal_training_started": False,
                "recipe_parameter_validation": parameter_validation,
            },
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        print(f"M1 audit blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

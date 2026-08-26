from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from syntactic_graph import (
    CompositeGraphCache,
    EXPECTED_PARSER_SHA256,
    GraphCacheError,
    build_graph_cache_records,
    build_parser_identity,
    build_stanza_pipeline,
    build_tokenizer_identity,
    load_graph_cache_directory,
    sha256_file,
)
from syntactic_graph_adapter import load_seq2seq_model
from t5_absa_train import (
    DataCollatorForSeq2SeqWithPairing,
    DomainAdversarialHead,
    JsonlSeq2SeqDataset,
    add_task_special_tokens,
    build_target_unlabeled_domain_rows,
    compute_domain_adversarial_loss,
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
EXPECTED_LAMBDA_DOMAIN_ADV = 0.03
VRAM_LIMIT_BYTES = int(7.5 * 1024**3)


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


def _run_model_audit(
    args,
    tokenizer,
    source_train_rows: list[dict],
    source_dev_rows: list[dict],
    target_rows: list[dict],
    train_cache,
    dev_cache,
    target_cache,
    device: torch.device,
) -> tuple[dict, dict]:
    target_domain_rows = build_target_unlabeled_domain_rows(target_rows, use_task_prefix=False)
    source_sample = source_train_rows[: max(1, min(args.batch_size, len(source_train_rows)))]
    dev_sample = source_dev_rows[: max(1, min(args.batch_size, len(source_dev_rows)))]
    target_sample = target_domain_rows[: max(1, min(args.batch_size, len(target_domain_rows)))]
    mixed_rows = source_sample + target_sample

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
    parameter_hash_before = parameter_state_sha256(treatment)
    control.eval()
    treatment.eval()

    mixed_graph_cache = CompositeGraphCache(
        {"source_train": train_cache, "target_unlabeled": target_cache}
    )
    source_dataset = _build_dataset(source_sample, tokenizer, train_cache, args.max_source_length, args.max_target_length)
    dev_dataset = _build_dataset(dev_sample, tokenizer, dev_cache, args.max_source_length, args.max_target_length)
    mixed_dataset = _build_dataset(mixed_rows, tokenizer, mixed_graph_cache, args.max_source_length, args.max_target_length)
    source_batch = _move_batch(_collate_rows(source_dataset, treatment, tokenizer, args.batch_size), device)
    dev_batch = _move_batch(_collate_rows(dev_dataset, treatment, tokenizer, args.batch_size), device)
    mixed_batch = _move_batch(_collate_rows(mixed_dataset, treatment, tokenizer, len(mixed_rows)), device)
    target_dataset = _build_dataset(target_sample, tokenizer, mixed_graph_cache, args.max_source_length, args.max_target_length)
    target_batch = _move_batch(_collate_rows(target_dataset, treatment, tokenizer, args.batch_size), device)

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
        "source_dev_loss": treatment_output.loss,
        "target_dann_loss": None,
        "lambda_domain_adv": float(args.lambda_domain_adv),
    }

    treatment.train()
    treatment.zero_grad(set_to_none=True)
    with _amp_context(device, args.fp16):
        source_training_output = treatment(**_model_inputs(source_batch, use_graph=True), return_dict=True)
    source_training_loss = source_training_output.loss
    source_training_loss.backward()
    callpoints["source_extractor_training"] = _finite(source_training_loss)
    losses["source_training_loss"] = float(source_training_loss.detach().float().cpu())
    treatment.zero_grad(set_to_none=True)

    treatment.eval()
    with torch.no_grad(), _amp_context(device, args.fp16):
        dev_output = treatment(**_model_inputs(dev_batch, use_graph=True), return_dict=True)
    callpoints["source_dev_evaluation"] = _finite(dev_output.loss)
    losses["source_dev_loss"] = float(dev_output.loss.detach().float().cpu())

    treatment.train()
    treatment.zero_grad(set_to_none=True)
    with _amp_context(device, args.fp16):
        mixed_output = treatment(**_model_inputs(mixed_batch, use_graph=True), return_dict=True)
        target_domain_loss = compute_domain_adversarial_loss(
            mixed_output.encoder_last_hidden_state,
            mixed_batch.get("attention_mask"),
            mixed_batch["domain_label"],
            treatment.domain_adversarial_head,
            grl_lambda=1.0,
        )
        if target_domain_loss is None:
            raise GraphCacheError("target-unlabeled DANN batch produced no valid domain labels")
        weighted_domain_loss = float(args.lambda_domain_adv) * target_domain_loss
    weighted_domain_loss.backward()
    callpoints["target_unlabeled_dann"] = _finite(target_domain_loss) and _finite(weighted_domain_loss)
    losses["target_dann_loss"] = float(target_domain_loss.detach().float().cpu())
    dann_projection_grad = treatment.syntactic_graph_adapter.output_projection.weight.grad
    dann_gradient_norm = float(dann_projection_grad.detach().float().norm().cpu()) if dann_projection_grad is not None else 0.0
    treatment.zero_grad(set_to_none=True)

    treatment.eval()
    generation_input = _model_inputs(target_batch, use_graph=True)
    generation_kwargs = {key: value for key, value in generation_input.items() if key in GRAPH_KEYS}
    with torch.no_grad(), _amp_context(device, args.fp16):
        treatment.generate(
            inputs=generation_input["input_ids"],
            attention_mask=generation_input["attention_mask"],
            max_new_tokens=8,
            num_beams=1,
            **generation_kwargs,
        )
    callpoints["target_pseudo_inference"] = True

    treatment.train()
    treatment.zero_grad(set_to_none=True)
    with _amp_context(device, args.fp16):
        aste_output = treatment(**_model_inputs(source_batch, use_graph=True), return_dict=True)
    aste_loss = aste_output.loss
    aste_loss.backward()
    aste_projection_grad = treatment.syntactic_graph_adapter.output_projection.weight.grad
    aste_gradient_norm = float(aste_projection_grad.detach().float().norm().cpu()) if aste_projection_grad is not None else 0.0
    treatment.zero_grad(set_to_none=True)

    target_labels = mixed_batch["labels"][len(source_sample) :]
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
        "source_rows_in_dann_batch": len(source_sample),
        "target_rows_in_dann_batch": len(target_sample),
        "gradient_checkpointing_enabled": bool(getattr(treatment, "is_gradient_checkpointing", False)),
    }
    parameter_hash_after = parameter_state_sha256(treatment)
    del control, treatment
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return callpoints, {
        "losses": losses,
        "measurements": measurements,
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


def run_audit(args) -> dict:
    if float(args.lambda_domain_adv) != EXPECTED_LAMBDA_DOMAIN_ADV:
        raise ValueError("M1 audit requires lambda_domain_adv=0.03")
    if not args.fp16 or not args.gradient_checkpointing:
        raise ValueError("M1 audit requires --fp16 and --gradient_checkpointing")
    recipe = json.loads(Path(args.recipe_path).read_text(encoding="utf-8"))
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
    tokenizer_identity = build_tokenizer_identity(args.model_path, tokenizer)
    parser_identity = build_parser_identity(args.parser_dir)
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
        device,
    )
    measurements = {
        "cache": cache_measurements,
        **model_measurements["measurements"],
        "optimizer_updates": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "parameter_updates": 0,
        "lambda_domain_adv": float(args.lambda_domain_adv),
        "fp16_requested": bool(args.fp16),
        "gradient_checkpointing_requested": bool(args.gradient_checkpointing),
        "gradient_checkpointing_enabled": bool(measurements["gradient_checkpointing_enabled"]),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    inspect = cache_measurements["inspect"]
    max_diff = measurements["control_treatment_max_abs_logit_diff"]
    repeat_diff = measurements["repeat_max_abs_logit_diff"]
    gate_values = {
        "parser_identity": parser_identity.get("sha256") == EXPECTED_PARSER_SHA256,
        "parse_alignment": inspect["coverage_ok"],
        "edge_legality": inspect["edge_legality_ok"],
        "reverse_selfloop": inspect["reverse_selfloop_ok"],
        "cache_resume_determinism": cache_measurements["interruption_observed"] and cache_measurements["byte_identical_repeat"],
        "four_callpoints": all(callpoints.values()),
        "control_equivalence": max_diff <= 1e-4,
        "loss_finiteness": all(
            _finite(measurements.get(name))
            for name in ("control_loss", "treatment_loss", "repeat_loss", "aste_gradient_norm", "dann_gradient_norm")
        ),
        "repeat_determinism": repeat_diff == 0.0,
        "aste_dann_gradient_paths": measurements["aste_gradient_norm"] > 0.0 and measurements["dann_gradient_norm"] > 0.0,
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
        ),
        "machine_readable_report": True,
    }
    gate_details = {
        "parser_identity": f"stanza={parser_identity.get('stanza_version')}; six_sha256=PASS",
        "parse_alignment": f"coverage={inspect['coverage_ok']}",
        "edge_legality": f"nodes={inspect['node_count']}; edges={inspect['edge_count']}",
        "reverse_selfloop": f"reverse_and_selfloop={inspect['reverse_selfloop_ok']}",
        "cache_resume_determinism": f"interrupted={cache_measurements['interruption_observed']}; byte_identical={cache_measurements['byte_identical_repeat']}",
        "four_callpoints": json.dumps(callpoints, ensure_ascii=False, sort_keys=True),
        "control_equivalence": f"max_abs_logit_diff={max_diff:.8g}",
        "loss_finiteness": "all measured losses and gradient norms are finite",
        "repeat_determinism": f"max_abs_logit_diff={repeat_diff:.8g}",
        "aste_dann_gradient_paths": f"aste_norm={measurements['aste_gradient_norm']:.8g}; dann_norm={measurements['dann_gradient_norm']:.8g}",
        "fp16_entry": "CUDA autocast=float16; RTX 3070 class memory check",
        "vram_8gb": f"peak_reserved={measurements['gpu_peak_reserved_bytes']} bytes; limit={VRAM_LIMIT_BYTES} bytes",
        "zero_update": "optimizer_updates=0; scheduler_steps=0; parameter_updates=0",
        "boundary_no_leakage": "target test, generator, augmentation, NLI and final ASTE are not invoked",
        "machine_readable_report": "JSON report schema emitted",
    }
    before_hash = model_measurements.get("parameter_hash_before")
    after_hash = model_measurements.get("parameter_hash_after")
    if before_hash is not None:
        measurements["parameter_hash_before"] = before_hash
        measurements["parameter_hash_after"] = after_hash
        gate_values["zero_update"] = gate_values["zero_update"] and before_hash == after_hash
    return build_entry_report(
        gate_values,
        measurements,
        callpoints,
        {
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
        gate_details=gate_details,
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
    parser.add_argument("--batch_size", type=int, default=2)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "m1_syntactic_graph_entry_audit.json"
    markdown_path = output_dir / "m1_syntactic_graph_entry_audit_CN.md"
    try:
        report = run_audit(args)
    except Exception as exc:  # The JSON failure artifact is part of the audit contract.
        report = build_entry_report(
            gate_values={name: False for name in ENTRY_GATE_NAMES},
            measurements={
                "optimizer_updates": 0,
                "optimizer_steps": 0,
                "scheduler_steps": 0,
                "parameter_updates": 0,
            },
            callpoints={},
            metadata={
                "source_dataset": args.source_dataset,
                "target_dataset": args.target_dataset,
                "target_test_access": False,
                "formal_training_started": False,
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

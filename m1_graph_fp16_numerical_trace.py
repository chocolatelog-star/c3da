"""Read-only numerical tracing for the M1 syntactic graph adapter.

This module intentionally contains no optimizer, scheduler, checkpoint, or
parameter-update operation.  The GPU entry point is only invoked by the user;
the CPU tests exercise the recorder and synthetic adapter path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch


TRACE_STAGE_NAMES = (
    "t5_encoder_last_hidden_state",
    "pooled_word_hidden",
    "node_projection",
    "query_projection",
    "key_projection",
    "value_projection",
    "edge_query",
    "edge_key",
    "edge_value",
    "query_key_product",
    "attention_logits_before_scaling",
    "attention_logits_scaled",
    "dependency_bias",
    "pos_pair_bias",
    "final_attention_logits",
    "softmax_input_float32_logits",
    "attention_probabilities",
    "relation_embeddings",
    "edge_messages",
    "aggregated_messages",
    "graph_hidden",
    "dropout_graph_hidden",
    "output_projection_input",
    "output_projection_output",
    "residual",
    "gate",
    "word_delta",
    "fused_hidden",
    "decoder_logits",
    "final_loss",
    "output_projection_gradient",
    "dann_domain_loss",
    "dann_weighted_loss",
    "dann_gradient",
)


def _json_number(value):
    value = float(value)
    return value if torch.isfinite(torch.tensor(value)) else None


def summarize_tensor_stats(
    stage: str,
    value: torch.Tensor,
    axes: tuple[str, ...] | list[str] = (),
    context: dict | None = None,
) -> dict:
    """Return JSON-safe statistics without replacing non-finite values."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    nan_mask = torch.isnan(tensor)
    posinf_mask = torch.isposinf(tensor)
    neginf_mask = torch.isneginf(tensor)
    first_nonfinite = None
    if not bool(finite.all().item()):
        first_index = torch.nonzero(~finite, as_tuple=False)[0].tolist()
        first_nonfinite = {
            str(axis): int(index)
            for axis, index in zip(tuple(axes), first_index)
        }
        if context:
            first_nonfinite.update(context)

    finite_values = tensor[finite].to(dtype=torch.float32)
    if finite_values.numel():
        minimum = _json_number(finite_values.min().item())
        maximum = _json_number(finite_values.max().item())
        max_abs = _json_number(finite_values.abs().max().item())
        mean = _json_number(finite_values.mean().item())
        std = _json_number(finite_values.std(unbiased=False).item())
    else:
        minimum = maximum = max_abs = mean = std = None

    return {
        "stage": str(stage),
        "dtype": str(tensor.dtype),
        "shape": [int(size) for size in tensor.shape],
        "finite_count": int(finite.sum().item()),
        "total_count": int(tensor.numel()),
        "nan_count": int(nan_mask.sum().item()),
        "posinf_count": int(posinf_mask.sum().item()),
        "neginf_count": int(neginf_mask.sum().item()),
        "min": minimum,
        "max": maximum,
        "max_abs": max_abs,
        "mean": mean,
        "std": std,
        "stats_over_finite_values": True,
        "first_nonfinite": first_nonfinite,
    }


class NumericalTrace:
    """Collect first-observation statistics and first non-finite locations."""

    def __init__(self, mode: str, row_ids: list[str] | None = None):
        self.mode = str(mode)
        self.row_ids = list(row_ids or [])
        self.stages: dict[str, dict] = {}
        self.exceptions: list[dict] = []
        self.first_nonfinite_stage: str | None = None
        self._context: dict = {}

    def set_context(
        self,
        split: str,
        row_ids: list[str],
        batch: dict | None = None,
        relation_vocab=None,
        dependency_vocab=None,
        pos_vocab=None,
    ):
        self._context = {
            "split": str(split),
            "row_ids": [str(row_id) for row_id in row_ids],
            "batch": batch,
            "relation_vocab": list(relation_vocab or []),
            "dependency_vocab": list(dependency_vocab or []),
            "pos_vocab": list(pos_vocab or []),
        }

    def _decorate_first_nonfinite(self, location: dict | None) -> dict | None:
        if location is None:
            return None
        decorated = dict(location)
        split = self._context.get("split")
        row_ids = self._context.get("row_ids", [])
        batch_index = location.get("batch")
        if split is not None:
            decorated["split"] = split
        if batch_index is not None and int(batch_index) < len(row_ids):
            decorated["row_id"] = row_ids[int(batch_index)]
        batch = self._context.get("batch") or {}
        edge_index = location.get("edge")
        if batch_index is not None and edge_index is not None:
            batch_index = int(batch_index)
            edge_index = int(edge_index)
            edge_keys = (
                "graph_edge_src",
                "graph_edge_dst",
                "graph_relation_id",
                "graph_dependency_relation_id",
                "graph_pos_pair_id",
            )
            for key in edge_keys:
                value = batch.get(key)
                if torch.is_tensor(value) and batch_index < value.size(0) and edge_index < value.size(1):
                    decorated[key] = int(value[batch_index, edge_index].detach().cpu().item())
            destination = decorated.get("graph_edge_dst")
            edge_dst = batch.get("graph_edge_dst")
            edge_mask = batch.get("graph_edge_mask")
            if destination is not None and torch.is_tensor(edge_dst):
                valid = edge_dst[batch_index].eq(destination)
                if torch.is_tensor(edge_mask):
                    valid = valid & edge_mask[batch_index].bool()
                decorated["current_node_incoming_edge_count"] = int(valid.sum().item())
            relation_vocab = self._context.get("relation_vocab", [])
            relation_id = decorated.get("graph_relation_id")
            if relation_id is not None and relation_id < len(relation_vocab):
                decorated["relation_type"] = relation_vocab[relation_id]
            dependency_vocab = self._context.get("dependency_vocab", [])
            dependency_id = decorated.get("graph_dependency_relation_id")
            if dependency_id is not None and dependency_id < len(dependency_vocab):
                decorated["dependency_relation_type"] = dependency_vocab[dependency_id]
            pos_vocab = self._context.get("pos_vocab", [])
            pos_id = decorated.get("graph_pos_pair_id")
            if pos_id is not None and pos_id < len(pos_vocab):
                decorated["pos_pair_type"] = pos_vocab[pos_id]
        return decorated

    def record(
        self,
        stage: str,
        value: torch.Tensor,
        axes: tuple[str, ...] | list[str] = (),
        context: dict | None = None,
    ) -> dict:
        stats = summarize_tensor_stats(stage, value, axes=axes, context=context)
        stats["first_nonfinite"] = self._decorate_first_nonfinite(stats["first_nonfinite"])
        existing = self.stages.get(stage)
        if existing is None:
            stats["observation_count"] = 1
            stats["nonfinite_observation_count"] = int(stats["first_nonfinite"] is not None)
            self.stages[stage] = stats
        else:
            existing["observation_count"] += 1
            if stats["first_nonfinite"] is not None:
                existing["nonfinite_observation_count"] += 1
                if existing["first_nonfinite"] is None:
                    existing["first_nonfinite"] = stats["first_nonfinite"]
                    for key in ("nan_count", "posinf_count", "neginf_count"):
                        existing[key] = stats[key]
        if stats["first_nonfinite"] is not None and self.first_nonfinite_stage is None:
            self.first_nonfinite_stage = str(stage)
        return stats

    def record_exception(self, stage: str, exc: BaseException) -> dict:
        result = {
            "stage": str(stage),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        self.exceptions.append(result)
        return result

    def finalize(self, require_all: bool = False) -> dict:
        missing_stages = [stage for stage in TRACE_STAGE_NAMES if stage not in self.stages]
        pass_value = (
            self.first_nonfinite_stage is None
            and not self.exceptions
            and (not require_all or not missing_stages)
        )
        return {
            "mode": self.mode,
            "pass": bool(pass_value),
            "first_nonfinite_stage": self.first_nonfinite_stage,
            "missing_stages": missing_stages,
            "stages": self.stages,
            "exceptions": list(self.exceptions),
            "row_ids": list(self.row_ids),
        }


def record_target_pseudo_result(operation) -> dict:
    """Run a pseudo-inference callable and retain exception type/message."""
    try:
        result = operation()
    except Exception as exc:  # The structured error is part of the diagnostic contract.
        return {
            "status": "ERROR",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "status": "PASS",
        "output_count": len(result) if hasattr(result, "__len__") else None,
    }


def build_trace_report(
    fp32: dict,
    fp16: dict,
    model_hash_before: str | None = None,
    model_hash_after: str | None = None,
    target_test_access: bool = False,
    target_pseudo_inference: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Assemble a machine-readable report for both numerical modes."""
    model_hashes_match = (
        model_hash_before is not None
        and model_hash_after is not None
        and model_hash_before == model_hash_after
    )
    fp32_first = fp32.get("first_nonfinite_stage")
    fp16_first = fp16.get("first_nonfinite_stage")
    pseudo = target_pseudo_inference or {"status": "NOT_RUN"}
    report_status = (
        fp32.get("pass")
        and fp16.get("pass")
        and model_hashes_match
        and not target_test_access
        and pseudo.get("status") == "PASS"
        and pseudo.get("numerical_pass", True)
    )
    return {
        "schema_version": 1,
        "diagnostic_id": "M1_SYNTACTIC_RGAT_FP16_NUMERICAL_TRACE_V1",
        "status": "PASS" if report_status else "BLOCKED",
        "fp32": fp32,
        "fp16": fp16,
        "fp32_pass": bool(fp32.get("pass", False)),
        "fp16_pass": bool(fp16.get("pass", False)),
        "fp32_first_nonfinite_stage": fp32_first,
        "fp16_first_nonfinite_stage": fp16_first,
        "first_nonfinite_stage": fp32_first or fp16_first,
        "model_parameter_hash_before": model_hash_before,
        "model_parameter_hash_after": model_hash_after,
        "model_parameter_hashes_match": bool(model_hashes_match),
        "optimizer_updates": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "parameter_updates": 0,
        "target_test_access": bool(target_test_access),
        "target_pseudo_inference": pseudo,
        "metadata": dict(metadata or {}),
    }


def parameter_state_sha256(model) -> str:
    """Hash model parameters without moving or mutating model state."""
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


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _edge_context(batch: dict, trace: NumericalTrace, stage: str) -> dict | None:
    location = trace.stages.get(stage, {}).get("first_nonfinite")
    if not location or "batch" not in location or "edge" not in location:
        return None
    batch_index = int(location["batch"])
    edge_index = int(location["edge"])
    result = {
        "batch": batch_index,
        "edge": edge_index,
    }
    for key in ("graph_edge_src", "graph_edge_dst", "graph_relation_id", "graph_dependency_relation_id", "graph_pos_pair_id"):
        value = batch.get(key)
        if torch.is_tensor(value) and batch_index < value.size(0) and edge_index < value.size(1):
            result[key] = int(value[batch_index, edge_index].detach().cpu().item())
    return result


def _trace_model_batch(model, batch: dict, trace: NumericalTrace, labels: bool):
    from m1_syntactic_graph_entry_audit import _model_inputs

    inputs = _model_inputs(batch, use_graph=True)
    if not labels:
        inputs.pop("labels", None)
    return model(**inputs, return_dict=True, output_hidden_states=False, graph_trace=trace)


def _record_gradient(model, trace: NumericalTrace, stage: str):
    gradient = model.syntactic_graph_adapter.output_projection.weight.grad
    if gradient is None:
        trace.record(stage, torch.tensor(float("nan")), axes=())
    else:
        trace.record(stage, gradient, axes=("output_feature", "graph_feature"))
    return gradient


def _run_mode(
    model,
    source_batch: dict,
    dev_batch: dict,
    mixed_batch: dict,
    mode: str,
    fp16: bool,
    args,
    row_contexts: dict,
    relation_vocab: list[str],
    dependency_vocab: list[str],
    pos_vocab: list[str],
):
    """Run source train/dev and target DANN forward paths without updates."""
    trace = NumericalTrace(mode, row_ids=list(args.trace_row_ids))
    model.zero_grad(set_to_none=True)
    model.train()
    try:
        trace.set_context(
            "source_train",
            row_contexts["source_train"],
            source_batch,
            relation_vocab,
            dependency_vocab,
            pos_vocab,
        )
        with _amp_context(model.device, fp16):
            source_output = _trace_model_batch(model, source_batch, trace, labels=True)
        if source_output.loss is not None:
            trace.record("final_loss", source_output.loss, axes=())
        source_output.loss.backward()
        _record_gradient(model, trace, "output_projection_gradient")
    except Exception as exc:
        trace.record_exception("source_train_backward", exc)
    model.zero_grad(set_to_none=True)

    model.eval()
    try:
        trace.set_context(
            "source_dev",
            row_contexts["source_dev"],
            dev_batch,
            relation_vocab,
            dependency_vocab,
            pos_vocab,
        )
        with torch.no_grad(), _amp_context(model.device, fp16):
            dev_output = _trace_model_batch(model, dev_batch, trace, labels=True)
        if dev_output.loss is not None:
            trace.record("final_loss", dev_output.loss, axes=())
    except Exception as exc:
        trace.record_exception("source_dev_evaluation", exc)
    model.train()
    model.zero_grad(set_to_none=True)

    try:
        from t5_absa_train import compute_domain_adversarial_loss

        trace.set_context(
            "target_unlabeled_dann",
            row_contexts["dann"],
            mixed_batch,
            relation_vocab,
            dependency_vocab,
            pos_vocab,
        )
        with _amp_context(model.device, fp16):
            dann_output = _trace_model_batch(model, mixed_batch, trace, labels=False)
            domain_loss = compute_domain_adversarial_loss(
                dann_output.encoder_last_hidden_state,
                mixed_batch.get("attention_mask"),
                mixed_batch["domain_label"],
                model.domain_adversarial_head,
            )
        if domain_loss is None:
            raise RuntimeError("target-unlabeled DANN produced no valid domain labels")
        trace.record("dann_domain_loss", domain_loss, axes=())
        weighted_domain_loss = args.lambda_domain_adv * domain_loss
        trace.record("dann_weighted_loss", weighted_domain_loss, axes=())
        weighted_domain_loss.backward()
        _record_gradient(model, trace, "dann_gradient")
    except Exception as exc:
        trace.record_exception("target_unlabeled_dann", exc)
    model.zero_grad(set_to_none=True)
    return trace.finalize(require_all=True)


def _first_bad_details(result: dict) -> dict:
    for stage, stats in result.get("stages", {}).items():
        location = stats.get("first_nonfinite")
        if location is not None:
            return {
                "stage": stage,
                "location": location,
            }
    return {"stage": None, "location": None}


def _write_trace_markdown(path: Path, report: dict) -> None:
    fp32_bad = _first_bad_details(report["fp32"])
    fp16_bad = _first_bad_details(report["fp16"])
    lines = [
        "# M1 句法图 FP16 数值追踪报告",
        "",
        f"总体状态：`{report['status']}`",
        "",
        "本报告只记录固定审计样本的 FP32/FP16 前向、反向数值和目标无标签 DANN（领域对抗网络）诊断；没有创建优化器、调度器或模型检查点。",
        "",
        f"- FP32：`{'PASS' if report['fp32_pass'] else 'BLOCKED'}`；首个非有限阶段：`{report['fp32_first_nonfinite_stage']}`",
        f"- FP16：`{'PASS' if report['fp16_pass'] else 'BLOCKED'}`；首个非有限阶段：`{report['fp16_first_nonfinite_stage']}`",
        f"- 全局首个非有限阶段：`{report['first_nonfinite_stage']}`",
        f"- 首个异常行：`{report.get('first_bad_row_id')}`",
        f"- 首个异常边：`{json.dumps(report.get('first_bad_edge'), ensure_ascii=False, sort_keys=True)}`",
        f"- 当前节点入边数：`{report.get('current_node_incoming_edge_count')}`",
        f"- target pseudo inference（目标伪标签推理）：`{report.get('target_pseudo_inference', {}).get('status')}`；异常：`{report.get('target_pseudo_inference', {}).get('exception_type')}`；消息：`{report.get('target_pseudo_inference', {}).get('message')}`",
        "",
        "## 首个非有限位置",
        "",
        f"- FP32：`{json.dumps(fp32_bad, ensure_ascii=False, sort_keys=True)}`",
        f"- FP16：`{json.dumps(fp16_bad, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## 边界",
        "",
        "- target_test_access（目标测试集访问）：`false`",
        "- optimizer_updates（优化器更新）：`0`；scheduler_steps（调度器步数）：`0`；parameter_updates（参数更新）：`0`",
        "- GPU（图形处理器）诊断命令由用户运行；本次实现验证未启动 GPU 诊断。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gpu_trace(args) -> dict:
    """Run the user-invoked GPU trace against existing read-only graph caches."""
    from tqdm import tqdm
    from transformers import AutoTokenizer

    import m1_syntactic_graph_entry_audit as audit
    from syntactic_graph import CompositeGraphCache, load_graph_cache_directory
    from syntactic_graph_adapter import load_seq2seq_model
    from t5_absa_train import DomainAdversarialHead, add_task_special_tokens, build_target_unlabeled_domain_rows

    recipe = json.loads(Path(args.recipe_path).read_text(encoding="utf-8"))
    recipe_validation = audit.ensure_audit_recipe(args, recipe)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("M1 numerical trace requires CUDA")
    requested_cuda_index = str(args.cuda)
    actual_cuda_index = int(torch.cuda.current_device())
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    source_train_rows, source_dev_rows, target_rows = audit._prepare_rows(
        args.source_dataset,
        args.target_dataset,
    )
    target_domain_rows = build_target_unlabeled_domain_rows(target_rows, use_task_prefix=False)
    source_sample = source_train_rows[: args.extractor_train_batch_size]
    dev_sample = source_dev_rows[: args.extractor_eval_batch_size]
    dann_source_sample = source_train_rows[: args.dann_source_batch_size]
    target_sample = target_domain_rows[: args.dann_target_batch_size]
    mixed_rows = dann_source_sample + target_sample
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer_identity = audit._build_tokenizer_identity_for_audit(args.model_path, tokenizer)
    parser_identity = audit._build_parser_identity_for_audit(args.parser_dir)
    source_cache = load_graph_cache_directory(
        args.graph_cache_dir,
        "source_train",
        source_train_rows,
        tokenizer_identity=tokenizer_identity,
        parser_identity=parser_identity,
    )
    dev_cache = load_graph_cache_directory(
        args.graph_cache_dir,
        "source_dev",
        source_dev_rows,
        tokenizer_identity=tokenizer_identity,
        parser_identity=parser_identity,
    )
    target_cache = load_graph_cache_directory(
        args.graph_cache_dir,
        "target_unlabeled",
        target_rows,
        tokenizer_identity=tokenizer_identity,
        parser_identity=parser_identity,
    )
    mixed_cache = CompositeGraphCache({"source_train": source_cache, "target_unlabeled": target_cache})

    model = load_seq2seq_model(
        args.model_path,
        use_syntactic_graph_adapter=True,
        relation_vocab_size=source_cache.relation_vocab_size,
    )
    add_task_special_tokens(tokenizer, model, source_sample + dev_sample + target_domain_rows)
    hidden_size = int(getattr(model.config, "d_model", model.get_input_embeddings().embedding_dim))
    model.domain_adversarial_head = DomainAdversarialHead(
        hidden_size=hidden_size,
        classifier_hidden_size=args.domain_adv_hidden_size,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)
    model_parameter_hash_before = parameter_state_sha256(model)

    source_dataset = audit._build_dataset(
        source_sample,
        tokenizer,
        source_cache,
        args.max_source_length,
        args.max_target_length,
    )
    dev_dataset = audit._build_dataset(
        dev_sample,
        tokenizer,
        dev_cache,
        args.max_source_length,
        args.max_target_length,
    )
    mixed_dataset = audit._build_dataset(
        mixed_rows,
        tokenizer,
        mixed_cache,
        args.max_source_length,
        args.max_target_length,
    )
    target_dataset = audit._build_dataset(
        target_sample,
        tokenizer,
        target_cache,
        args.max_source_length,
        args.max_target_length,
    )
    source_batch = audit._move_batch(
        audit._collate_rows(source_dataset, model, tokenizer, args.extractor_train_batch_size),
        device,
    )
    dev_batch = audit._move_batch(
        audit._collate_rows(dev_dataset, model, tokenizer, args.extractor_eval_batch_size),
        device,
    )
    mixed_batch = audit._move_batch(
        audit._collate_rows(mixed_dataset, model, tokenizer, len(mixed_rows)),
        device,
    )
    target_batch = audit._move_batch(
        audit._collate_rows(target_dataset, model, tokenizer, args.target_pseudo_batch_size),
        device,
    )
    relation_vocab = list(source_cache.relation_vocab)
    cache_records = []
    for split in ("source_train", "source_dev", "target_unlabeled"):
        cache_records.extend(audit._read_cache_records(Path(args.graph_cache_dir), split))
    dependency_vocab = sorted(
        {str(edge.get("dependency_key", "self")) for record in cache_records for edge in record.get("edges", [])}
    )
    pos_vocab = sorted(
        {str(edge.get("pos_pair_key", "self")) for record in cache_records for edge in record.get("edges", [])}
    )
    row_contexts = {
        "source_train": [str(row["id"]) for row in source_sample],
        "source_dev": [str(row["id"]) for row in dev_sample],
        "dann": [str(row["id"]) for row in dann_source_sample + target_sample],
    }
    args.trace_row_ids = row_contexts["source_train"]
    torch.cuda.reset_peak_memory_stats(device)
    mode_results = {}
    for mode, use_fp16 in tqdm(
        (("fp32", False), ("fp16", True)),
        desc="m1-numerical-trace",
    ):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        mode_results[mode] = _run_mode(
            model,
            source_batch,
            dev_batch,
            mixed_batch,
            mode,
            use_fp16,
            args,
            row_contexts,
            relation_vocab,
            dependency_vocab,
            pos_vocab,
        )
    fp32 = mode_results["fp32"]
    fp16 = mode_results["fp16"]
    pseudo_trace = NumericalTrace("target_pseudo_inference", row_ids=row_contexts["dann"][-len(target_sample):])
    pseudo_trace.set_context(
        "target_pseudo_inference",
        [str(row["id"]) for row in target_sample],
        target_batch,
        relation_vocab,
        dependency_vocab,
        pos_vocab,
    )

    def _run_target_pseudo_inference():
        model.eval()
        inputs = audit._model_inputs(target_batch, use_graph=True)
        inputs.pop("labels", None)
        return model.generate(
            **inputs,
            max_new_tokens=8,
            num_beams=1,
            graph_trace=pseudo_trace,
        )

    target_pseudo_inference = record_target_pseudo_result(_run_target_pseudo_inference)
    target_pseudo_inference["entry"] = "SyntacticGraphT5ForConditionalGeneration.generate"
    target_pseudo_trace = pseudo_trace.finalize()
    target_pseudo_inference["trace"] = target_pseudo_trace
    target_pseudo_inference["numerical_pass"] = bool(target_pseudo_trace["pass"])
    model_parameter_hash_after = parameter_state_sha256(model)
    report = build_trace_report(
        fp32=fp32,
        fp16=fp16,
        model_hash_before=model_parameter_hash_before,
        model_hash_after=model_parameter_hash_after,
        target_test_access=False,
        target_pseudo_inference=target_pseudo_inference,
        metadata={
            "requested_cuda_index": int(requested_cuda_index),
            "actual_cuda_index": actual_cuda_index,
            "parser_device": "cache_reuse",
            "model_device": str(device),
            "recipe_parameter_validation": recipe_validation,
            "fixed_samples": row_contexts,
            "batch_parameters": {
                "source_train": args.extractor_train_batch_size,
                "source_dev": args.extractor_eval_batch_size,
                "dann_source": args.dann_source_batch_size,
                "dann_target": args.dann_target_batch_size,
                "target_pseudo": args.target_pseudo_batch_size,
            },
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
            "lambda_domain_adv": args.lambda_domain_adv,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "fp16_requested": bool(args.fp16),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "parser_identity": parser_identity,
            "tokenizer_identity": tokenizer_identity,
            "graph_cache_dir": str(Path(args.graph_cache_dir)),
            "target_test_access": False,
            "target_pseudo_inference_entry": "SyntacticGraphT5ForConditionalGeneration.generate",
        },
    )
    bad = _first_bad_details(fp32 if fp32.get("first_nonfinite_stage") else fp16)
    location = bad.get("location") or {}
    report["first_bad_row_id"] = location.get("row_id")
    report["first_bad_edge"] = {
        key: location.get(key)
        for key in (
            "graph_edge_src",
            "graph_edge_dst",
            "graph_relation_id",
            "relation_type",
            "graph_dependency_relation_id",
            "dependency_relation_type",
            "graph_pos_pair_id",
            "pos_pair_type",
        )
        if key in location
    } or None
    report["current_node_incoming_edge_count"] = location.get("current_node_incoming_edge_count")
    report["target_pseudo_inference_exception"] = {
        "exception_type": target_pseudo_inference.get("exception_type"),
        "message": target_pseudo_inference.get("message"),
    }
    return report


def _build_parser():
    parser = argparse.ArgumentParser(description="M1 句法图 FP32/FP16 只读数值追踪")
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--target_dataset", required=True)
    parser.add_argument("--graph_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=r"J:\nlp\models\t5-base-py")
    parser.add_argument("--parser_dir", default=r"J:\nlp\models\stanza_resources")
    parser.add_argument("--recipe_path", default=r"configs\recipes\laptop14_to_rest15_syntactic_graph_v1.json")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=96)
    parser.add_argument("--lambda_domain_adv", type=float, default=0.03)
    parser.add_argument("--domain_adv_hidden_size", type=int, default=256)
    parser.add_argument("--extractor_train_batch_size", type=int, default=1)
    parser.add_argument("--extractor_eval_batch_size", type=int, default=2)
    parser.add_argument("--dann_source_batch_size", type=int, default=1)
    parser.add_argument("--dann_target_batch_size", type=int, default=1)
    parser.add_argument("--target_pseudo_batch_size", type=int, default=1)
    parser.add_argument("--trace_row_ids", nargs="*", default=[])
    return parser


def main():
    """GPU entry point; never called by the CPU test suite."""
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "m1_syntactic_graph_fp16_numerical_trace.json"
    markdown_path = output_dir / "m1_syntactic_graph_fp16_numerical_trace_CN.md"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    try:
        report = run_gpu_trace(args)
    except Exception as exc:  # Keep the failure machine-readable and retain the exception identity.
        fp32 = NumericalTrace("fp32")
        fp16 = NumericalTrace("fp16")
        fp32.record_exception("entry", exc)
        fp16.record_exception("entry", exc)
        report = build_trace_report(
            fp32=fp32.finalize(require_all=True),
            fp16=fp16.finalize(require_all=True),
            target_test_access=False,
            target_pseudo_inference={
                "status": "ERROR",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            metadata={
                "requested_cuda_index": int(str(args.cuda).split(",", 1)[0]),
                "actual_cuda_index": None,
                "parser_device": "not_run",
                "model_device": "not_run",
                "target_test_access": False,
            },
        )
        report["errors"] = [{"exception_type": type(exc).__name__, "message": str(exc)}]
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_trace_markdown(markdown_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

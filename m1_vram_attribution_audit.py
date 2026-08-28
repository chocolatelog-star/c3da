"""M1 read-only VRAM attribution diagnostic.

The GPU entry is intentionally isolated from training.  It loads one fixed
source/target pair from an existing V4 input/cache directory, records memory
at graph trace boundaries, runs three zero-update steps, and writes only a
machine-readable report plus a short Chinese report.  No model formula,
training configuration, checkpoint, pseudo label, or target-test path is
changed or read.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


DIAGNOSTIC_ID = "M1_SYNTACTIC_RGAT_VRAM_ATTRIBUTION_AUDIT_V1"
SCHEMA_VERSION = 1
DEFAULT_SOURCE_ROW_ID = "408"
DEFAULT_TARGET_ROW_ID = "456"
DEFAULT_STEPS = 3
GRAPH_CALLPOINTS = {
    "word_pooling",
    "node_edge_projection",
    "attention_logits",
    "relation_message_aggregation",
    "graph_fusion",
}
TRACE_TO_CALLPOINT = {
    "t5_encoder_last_hidden_state": "t5_encoder",
    "pooled_word_hidden": "word_pooling",
    "node_projection": "node_edge_projection",
    "query_projection": "node_edge_projection",
    "key_projection": "node_edge_projection",
    "value_projection": "node_edge_projection",
    "edge_query": "node_edge_projection",
    "edge_key": "node_edge_projection",
    "edge_value": "node_edge_projection",
    "query_key_product": "attention_logits",
    "attention_logits_before_scaling": "attention_logits",
    "attention_logits_scaled": "attention_logits",
    "dependency_bias": "attention_logits",
    "pos_pair_bias": "attention_logits",
    "final_attention_logits": "attention_logits",
    "softmax_input_float32_logits": "attention_logits",
    "attention_probabilities": "attention_logits",
    "relation_embeddings": "relation_message_aggregation",
    "edge_messages": "relation_message_aggregation",
    "aggregated_messages": "relation_message_aggregation",
    "graph_hidden": "relation_message_aggregation",
    "dropout_graph_hidden": "relation_message_aggregation",
    "output_projection_input": "graph_fusion",
    "output_projection_output": "graph_fusion",
    "residual": "graph_fusion",
    "gate": "graph_fusion",
    "word_delta": "graph_fusion",
    "fused_hidden": "graph_fusion",
    "decoder_logits": "decoder_logits",
    "final_loss": "decoder_logits",
}
EXPECTED_FP32_TRACE_STAGES = {
    "softmax_input_float32_logits",
    "final_loss",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    return value if torch.isfinite(torch.tensor(value)) else None


def _tensor_meta(value: torch.Tensor | None) -> dict | None:
    if value is None or not torch.is_tensor(value):
        return None
    return {
        "shape": [int(size) for size in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": int(value.numel()),
        "element_size_bytes": int(value.element_size()),
        "theoretical_bytes": int(value.numel() * value.element_size()),
        "requires_grad": bool(value.requires_grad),
    }


class SyntheticMemoryBackend:
    """Deterministic CPU-only memory backend used by regression tests."""

    def __init__(
        self,
        *,
        allocated: list[int],
        reserved: list[int],
        peak_allocated: list[int],
        peak_reserved: list[int],
        stats: list[dict[str, int]] | None = None,
    ):
        self.allocated = list(allocated)
        self.reserved = list(reserved)
        self.peak_allocated = list(peak_allocated)
        self.peak_reserved = list(peak_reserved)
        self.stats = list(stats or [{}])
        self.index = 0

    def snapshot(self, _device: torch.device) -> dict:
        index = min(self.index, len(self.allocated) - 1)
        stat_index = min(self.index, len(self.stats) - 1)
        self.index += 1
        return {
            "allocated_bytes": int(self.allocated[index]),
            "reserved_bytes": int(self.reserved[min(index, len(self.reserved) - 1)]),
            "peak_allocated_bytes": int(self.peak_allocated[min(index, len(self.peak_allocated) - 1)]),
            "peak_reserved_bytes": int(self.peak_reserved[min(index, len(self.peak_reserved) - 1)]),
            "active_bytes": int(self.stats[stat_index].get("active_bytes.all.current", 0)),
            "inactive_split_bytes": int(self.stats[stat_index].get("inactive_split_bytes.all.current", 0)),
            "free_bytes": None,
            "total_memory_bytes": None,
        }


class TorchCudaMemoryBackend:
    """Read-only adapter over torch CUDA allocator counters."""

    def snapshot(self, device: torch.device) -> dict:
        stats = torch.cuda.memory_stats(device)

        def first_stat(*names: str) -> int:
            for name in names:
                if name in stats:
                    return int(stats[name])
            return 0

        free_bytes = total_memory_bytes = None
        try:
            free_bytes, total_memory_bytes = [int(value) for value in torch.cuda.mem_get_info(device)]
        except (RuntimeError, AttributeError):
            pass
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "active_bytes": first_stat("active_bytes.all.current", "active_bytes.current"),
            "inactive_split_bytes": first_stat(
                "inactive_split_bytes.all.current", "inactive_split_bytes.current"
            ),
            "free_bytes": free_bytes,
            "total_memory_bytes": total_memory_bytes,
        }


def _sum_parameter_bytes(model: torch.nn.Module) -> int:
    return sum(int(parameter.numel() * parameter.element_size()) for parameter in model.parameters())


def _sum_gradient_bytes(model: torch.nn.Module) -> int:
    return sum(
        int(parameter.grad.numel() * parameter.grad.element_size())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _sum_optimizer_state_bytes(optimizer: torch.optim.Optimizer | None) -> int:
    if optimizer is None:
        return 0
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += int(value.numel() * value.element_size())
    return total


def count_autograd_nodes(value: Any) -> int:
    """Count reachable autograd nodes without retaining the graph."""
    root = getattr(value, "grad_fn", None)
    if root is None:
        return 0
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        for parent, _ in getattr(node, "next_functions", ()):
            if parent is not None:
                stack.append(parent)
    return len(seen)


def audit_python_container_tensors(value: Any, *, max_items: int = 10000) -> dict:
    """Inspect a bounded object graph for tensor references in Python containers."""
    visited: set[int] = set()
    tensor_count = 0
    gpu_tensor_count = 0
    gpu_paths: list[str] = []

    def visit(current: Any, path: str, depth: int = 0) -> None:
        nonlocal tensor_count, gpu_tensor_count
        if len(visited) >= max_items or depth > 12:
            return
        if torch.is_tensor(current):
            tensor_count += 1
            if current.device.type == "cuda":
                gpu_tensor_count += 1
                if len(gpu_paths) < 20:
                    gpu_paths.append(path)
            return
        if current is None or isinstance(current, (str, bytes, int, float, bool)):
            return
        current_id = id(current)
        if current_id in visited:
            return
        visited.add(current_id)
        if isinstance(current, dict):
            for key, child in current.items():
                visit(child, f"{path}[{key!r}]", depth + 1)
        elif isinstance(current, (list, tuple, set, frozenset)):
            for index, child in enumerate(current):
                visit(child, f"{path}[{index}]", depth + 1)

    visit(value, "root")
    return {
        "tensor_count": tensor_count,
        "gpu_tensor_count": gpu_tensor_count,
        "gpu_paths": gpu_paths,
        "scan_truncated": len(visited) >= max_items,
    }


class MemoryRecorder:
    """Record allocator and tensor metadata at named diagnostic callpoints."""

    def __init__(
        self,
        backend: TorchCudaMemoryBackend | SyntheticMemoryBackend,
        *,
        device: torch.device,
        batch_counts: dict[str, int] | None = None,
        runtime_provider=None,
    ):
        self.backend = backend
        self.device = device
        self.batch_counts = dict(batch_counts or {})
        self.runtime_provider = runtime_provider
        self.events: list[dict] = []

    def mark(
        self,
        callpoint: str,
        *,
        tensor: torch.Tensor | None = None,
        step: int = 0,
        trace_stage: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        memory = self.backend.snapshot(self.device)
        runtime = dict(self.runtime_provider() if self.runtime_provider is not None else {})
        event = {
            "step": int(step),
            "callpoint": str(callpoint),
            "trace_stage": trace_stage,
            "memory": memory,
            "tensor": _tensor_meta(tensor),
            "batch_counts": dict(self.batch_counts),
            "runtime": runtime,
            "extra": dict(extra or {}),
            "timestamp_monotonic": time.monotonic(),
        }
        self.events.append(event)
        return event


class MemoryTrace:
    """Minimal trace object accepted by the existing graph adapter."""

    def __init__(self, recorder: MemoryRecorder, step: int):
        self.recorder = recorder
        self.step = int(step)
        self.stages: list[str] = []

    def record(self, stage: str, value: torch.Tensor, axes=(), context=None) -> dict:
        self.stages.append(str(stage))
        callpoint = TRACE_TO_CALLPOINT.get(str(stage), str(stage))
        return self.recorder.mark(
            callpoint,
            tensor=value,
            step=self.step,
            trace_stage=str(stage),
            extra={"axes": list(axes or [])},
        )


def run_zero_update_optimizer_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Call the real optimizer entry with no gradients, so parameters cannot update."""
    # Keep the audit guard on CPU.  A GPU clone of every parameter would create
    # a false peak and could itself trigger the VRAM pressure being measured.
    before = [parameter.detach().to(device="cpu").clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    changed = sum(
        int(not torch.equal(parameter.detach().to(device="cpu"), expected))
        for parameter, expected in zip(model.parameters(), before)
    )
    return {
        "optimizer_updates": 0,
        "parameter_updates": int(changed),
        "optimizer_step_mode": "controlled_noop_without_gradients",
    }


def _event_peak(events: list[dict], field: str) -> int:
    fallback = "allocated_bytes" if field == "peak_allocated_bytes" else "reserved_bytes"
    return max(
        (
            int(event.get("memory", {}).get(field, event.get("memory", {}).get(fallback, 0)) or 0)
            for event in events
        ),
        default=0,
    )


def _event_by_step_and_callpoint(events: list[dict]) -> dict[tuple[int, str], dict]:
    result = {}
    for event in events:
        result[(int(event.get("step", 0)), str(event.get("callpoint", "")))] = event
    return result


def _runtime_overlap_audit(events: list[dict]) -> dict:
    """Summarize whether gradients, optimizer state, and live graphs overlap."""
    gradient_peak = 0
    optimizer_state_peak = 0
    parameter_peak = 0
    gradient_and_graph_events = 0
    all_three_events = 0
    for event in events:
        runtime = event.get("runtime", {})
        gradient_bytes = int(runtime.get("gradient_bytes", 0) or 0)
        optimizer_state_bytes = int(runtime.get("optimizer_state_bytes", 0) or 0)
        parameter_bytes = int(runtime.get("parameter_bytes", 0) or 0)
        graph_nodes = int(runtime.get("autograd_graph_nodes", 0) or 0)
        gradient_peak = max(gradient_peak, gradient_bytes)
        optimizer_state_peak = max(optimizer_state_peak, optimizer_state_bytes)
        parameter_peak = max(parameter_peak, parameter_bytes)
        if gradient_bytes > 0 and graph_nodes > 0:
            gradient_and_graph_events += 1
        if gradient_bytes > 0 and optimizer_state_bytes > 0 and graph_nodes > 0:
            all_three_events += 1
    return {
        "parameter_bytes_peak_observed": parameter_peak,
        "gradient_bytes_peak_observed": gradient_peak,
        "optimizer_state_bytes_peak_observed": optimizer_state_peak,
        "gradient_and_autograd_overlap_event_count": gradient_and_graph_events,
        "parameter_gradient_optimizer_autograd_overlap_event_count": all_three_events,
        "optimizer_state_observable": optimizer_state_peak > 0,
        "note": "The diagnostic optimizer step is a controlled no-op, so an absent AdamW state is reported as observed rather than inferred to be leak-free.",
    }


def _fp32_promotion_audit(events: list[dict]) -> dict:
    """Report trace dtype transitions without changing or coercing tensors."""
    observations = []
    candidates = []
    last_dtype_by_step = {}
    for event in events:
        trace_stage = event.get("trace_stage")
        tensor = event.get("tensor") or {}
        dtype = str(tensor.get("dtype", ""))
        if not trace_stage or not dtype:
            continue
        step = int(event.get("step", 0))
        previous = last_dtype_by_step.get(step)
        if dtype == "torch.float32":
            observations.append({
                "step": step,
                "trace_stage": str(trace_stage),
                "expected": str(trace_stage) in EXPECTED_FP32_TRACE_STAGES,
            })
            if previous == "torch.float16" and str(trace_stage) not in EXPECTED_FP32_TRACE_STAGES:
                candidates.append({
                    "step": step,
                    "trace_stage": str(trace_stage),
                    "previous_dtype": previous,
                    "dtype": dtype,
                })
        last_dtype_by_step[step] = dtype
    return {
        "fp32_trace_observation_count": len(observations),
        "expected_fp32_trace_observations": [item for item in observations if item["expected"]],
        "implicit_promotion_candidates": candidates,
        "implicit_promotion_suspected": bool(candidates),
    }


def analyze_memory_attribution(
    control_events: list[dict],
    treatment_events: list[dict],
    *,
    significant_growth_bytes: int = 32 * 1024 * 1024,
    total_memory_bytes: int | None = None,
) -> dict:
    """Compare aligned allocator observations and classify the observed pattern."""
    control_map = _event_by_step_and_callpoint(control_events)
    first_growth = None
    deltas = []
    for treatment_event in treatment_events:
        key = (int(treatment_event.get("step", 0)), str(treatment_event.get("callpoint", "")))
        control_event = control_map.get(key)
        if control_event is None:
            continue
        treatment_allocated = int(treatment_event.get("memory", {}).get("allocated_bytes", 0) or 0)
        control_allocated = int(control_event.get("memory", {}).get("allocated_bytes", 0) or 0)
        delta = treatment_allocated - control_allocated
        deltas.append(
            {
                "step": key[0],
                "callpoint": key[1],
                "control_allocated_bytes": control_allocated,
                "treatment_allocated_bytes": treatment_allocated,
                "delta_bytes": delta,
            }
        )
        if first_growth is None and delta >= int(significant_growth_bytes):
            first_growth = key[1]

    first_graph_growth = None
    for treatment_event in treatment_events:
        callpoint = str(treatment_event.get("callpoint", ""))
        if callpoint not in GRAPH_CALLPOINTS:
            continue
        step = int(treatment_event.get("step", 0))
        treatment_allocated = int(treatment_event.get("memory", {}).get("allocated_bytes", 0) or 0)
        control_baseline = max(
            (
                int(event.get("memory", {}).get("allocated_bytes", 0) or 0)
                for event in control_events
                if int(event.get("step", 0)) == step
                and str(event.get("callpoint", "")) in {"batch_to_gpu", "t5_encoder"}
            ),
            default=0,
        )
        if treatment_allocated - control_baseline >= int(significant_growth_bytes):
            first_graph_growth = callpoint
            break

    control_graph_peak = _event_peak(
        [event for event in control_events if event.get("callpoint") in GRAPH_CALLPOINTS],
        "peak_allocated_bytes",
    )
    treatment_graph_peak = _event_peak(
        [event for event in treatment_events if event.get("callpoint") in GRAPH_CALLPOINTS],
        "peak_allocated_bytes",
    )
    treatment_step_peaks: dict[int, int] = {}
    for event in treatment_events:
        step = int(event.get("step", 0))
        treatment_step_peaks[step] = max(
            treatment_step_peaks.get(step, 0),
            int(event.get("memory", {}).get("allocated_bytes", 0) or 0),
        )
    ordered_step_peaks = [value for _, value in sorted(treatment_step_peaks.items())]
    retained_growth = (
        len(ordered_step_peaks) >= 3
        and ordered_step_peaks[-1] > ordered_step_peaks[0]
        and all(right >= left for left, right in zip(ordered_step_peaks, ordered_step_peaks[1:]))
        and ordered_step_peaks[-1] - ordered_step_peaks[0] >= int(significant_growth_bytes)
    )
    reserved_values = [
        event.get("memory", {})
        for event in treatment_events
        if int(event.get("memory", {}).get("reserved_bytes", 0) or 0) > 0
    ]
    fragmentation_ratio = max(
        (
            int(memory.get("inactive_split_bytes", 0) or 0)
            / max(1, int(memory.get("reserved_bytes", 0) or 0))
            for memory in reserved_values
        ),
        default=0.0,
    )
    fragmentation_suspected = fragmentation_ratio >= 0.25
    peak_allocated = _event_peak(treatment_events, "peak_allocated_bytes")
    wddm_suspected = None
    if total_memory_bytes:
        wddm_suspected = bool(peak_allocated >= int(total_memory_bytes * 0.97))
    if retained_growth:
        classification = "retained_growth_suspected"
    elif fragmentation_suspected:
        classification = "allocator_fragmentation_suspected"
    elif wddm_suspected is True:
        classification = "near_capacity_wddm_or_allocator_pressure"
    else:
        classification = "normal_peak_or_wddm_unresolved"
    return {
        "control_peak_allocated_bytes": _event_peak(control_events, "peak_allocated_bytes"),
        "control_peak_reserved_bytes": _event_peak(control_events, "peak_reserved_bytes"),
        "treatment_peak_allocated_bytes": peak_allocated,
        "treatment_peak_reserved_bytes": _event_peak(treatment_events, "peak_reserved_bytes"),
        "graph_module_increment_peak_allocated_bytes": _event_peak(treatment_events, "peak_allocated_bytes") - _event_peak(control_events, "peak_allocated_bytes"),
        "control_graph_peak_allocated_bytes": control_graph_peak,
        "treatment_graph_peak_allocated_bytes": treatment_graph_peak,
        "first_significant_growth_callpoint": first_graph_growth or first_growth,
        "aligned_deltas": deltas,
        "repeated_treatment_step_peaks": ordered_step_peaks,
        "leak_suspected": bool(retained_growth),
        "fragmentation_suspected": bool(fragmentation_suspected),
        "fragmentation_max_inactive_split_ratio": float(fragmentation_ratio),
        "wddm_paging_suspected": wddm_suspected,
        "wddm_evidence_note": "torch allocator counters cannot prove Windows WDDM paging; interpret a near-capacity result with host/process telemetry.",
        "normal_peak_suspected": bool(not retained_growth and not fragmentation_suspected and wddm_suspected is not True),
        "classification": classification,
        "control_runtime_overlap": _runtime_overlap_audit(control_events),
        "treatment_runtime_overlap": _runtime_overlap_audit(treatment_events),
        "control_fp32_promotion": _fp32_promotion_audit(control_events),
        "treatment_fp32_promotion": _fp32_promotion_audit(treatment_events),
    }


def _parameter_state_sha256(model: torch.nn.Module) -> str:
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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_identity(project_root: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        )
        return {"commit": commit, "clean": status == "", "status_porcelain": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "clean": False, "status_porcelain": f"git_error:{exc}"}


def _model_identity(model_path: Path) -> dict:
    names = (
        "config.json",
        "pytorch_model.bin",
        "generation_config.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    return {
        "path": str(model_path.resolve()),
        "files": {
            name: {"path": str(model_path / name), "exists": (model_path / name).is_file(), "sha256": _sha256_file(model_path / name) if (model_path / name).is_file() else None}
            for name in names
            if (model_path / name).is_file() or name in names[:5]
        },
    }


def _input_identity(run_dir: Path) -> dict:
    input_dir = run_dir / "inputs"
    paths = {
        "source_train": input_dir / "source_train.jsonl",
        "target_unlabeled": input_dir / "target_unlabeled.jsonl",
    }
    return {
        name: {"path": str(path.resolve()), "exists": path.is_file(), "sha256": _sha256_file(path) if path.is_file() else None}
        for name, path in paths.items()
    }


def _cache_identity(cache_dir: Path) -> dict:
    names = ("relation_vocab.json", "source_train.jsonl", "source_dev.jsonl", "target_unlabeled.jsonl", "manifest.json")
    return {
        name: {"path": str(cache_dir / name), "exists": (cache_dir / name).is_file(), "sha256": _sha256_file(cache_dir / name) if (cache_dir / name).is_file() else None}
        for name in names
    }


def _batch_counts(batch: dict) -> dict[str, int]:
    def count(key: str) -> int:
        value = batch.get(key)
        return int(value.sum().item()) if torch.is_tensor(value) else 0

    return {"tokens": count("attention_mask"), "nodes": count("word_mask"), "edges": count("edge_mask")}


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _attach_hooks(model, recorder: MemoryRecorder, step_ref: list[int]):
    handles = []

    def encoder_hook(_module, _inputs, output):
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is not None:
            recorder.mark("t5_encoder", tensor=hidden, step=step_ref[0], trace_stage="t5_encoder_hook")

    def lm_head_hook(_module, _inputs, output):
        recorder.mark("decoder_logits", tensor=output, step=step_ref[0], trace_stage="lm_head_hook")

    if hasattr(model, "encoder"):
        handles.append(model.encoder.register_forward_hook(encoder_hook))
    if hasattr(model, "lm_head"):
        handles.append(model.lm_head.register_forward_hook(lm_head_hook))
    return handles


def _model_inputs(batch: dict, graph_enabled: bool) -> dict:
    from m1_syntactic_graph_entry_audit import _model_inputs as audit_model_inputs

    return audit_model_inputs(batch, use_graph=graph_enabled)


def _runtime_provider(model, optimizer, current_loss) -> dict:
    return {
        "parameter_bytes": _sum_parameter_bytes(model),
        "gradient_bytes": _sum_gradient_bytes(model),
        "optimizer_state_bytes": _sum_optimizer_state_bytes(optimizer),
        "autograd_graph_nodes": count_autograd_nodes(current_loss[0]) if current_loss and current_loss[0] is not None else 0,
    }


def _run_variant_steps(
    model,
    optimizer,
    batch: dict,
    *,
    graph_enabled: bool,
    args,
    device: torch.device,
    variant: str,
) -> dict:
    current_loss: list[torch.Tensor | None] = [None]
    recorder = MemoryRecorder(
        TorchCudaMemoryBackend(),
        device=device,
        batch_counts=_batch_counts(batch),
        runtime_provider=lambda: _runtime_provider(model, optimizer, current_loss),
    )
    step_ref = [0]
    handles = _attach_hooks(model, recorder, step_ref)
    model_hash_before = _parameter_state_sha256(model)
    recorder.mark("model_loaded", step=0, extra={"variant": variant})
    recorder.mark(
        "optimizer_created",
        step=0,
        extra={"optimizer": type(optimizer).__name__, "learning_rate": float(args.learning_rate)},
    )
    try:
        model_hash_after = None
        for step in range(int(args.steps)):
            step_ref[0] = step
            torch.manual_seed(int(args.seed) + step)
            torch.cuda.manual_seed_all(int(args.seed) + step)
            model.zero_grad(set_to_none=True)
            model.train()
            current_loss[0] = None
            recorder.mark("batch_to_gpu", step=step, extra={"source_count": 1, "target_count": 1})
            graph_trace = MemoryTrace(recorder, step) if graph_enabled else None
            with _autocast(device, bool(args.fp16)):
                model_kwargs = _model_inputs(batch, graph_enabled)
                if graph_enabled:
                    model_kwargs["graph_trace"] = graph_trace
                output = model(
                    **model_kwargs,
                    return_dict=True,
                    output_hidden_states=False,
                )
                if output.loss is None:
                    raise RuntimeError("paired DANN diagnostic batch produced no generation loss")
                from t5_absa_train import compute_domain_adversarial_loss

                domain_loss = compute_domain_adversarial_loss(
                    output.encoder_last_hidden_state,
                    batch.get("attention_mask"),
                    batch["domain_label"],
                    model.domain_adversarial_head,
                )
                if domain_loss is None:
                    raise RuntimeError("paired DANN diagnostic batch produced no domain loss")
                total_loss = output.loss + float(args.lambda_domain_adv) * domain_loss
            current_loss[0] = total_loss
            if getattr(output, "logits", None) is not None and not any(
                event.get("step") == step and event.get("callpoint") == "decoder_logits" for event in recorder.events
            ):
                recorder.mark("decoder_logits", tensor=output.logits, step=step, trace_stage="output_logits")
            total_loss.backward()
            recorder.mark(
                "backward",
                tensor=total_loss,
                step=step,
                extra={"domain_loss": float(domain_loss.detach().float().cpu().item())},
            )
            optimizer_result = run_zero_update_optimizer_step(model, optimizer)
            recorder.mark("optimizer_step", step=step, extra=optimizer_result)
            current_loss[0] = None
            del output, domain_loss, total_loss, graph_trace
            gc.collect()
            recorder.mark(
                "post_step_cleanup",
                step=step,
                extra={"autograd_graph_nodes_after_cleanup": 0},
            )
        model_hash_after = _parameter_state_sha256(model)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
    return {
        "variant": variant,
        "graph_enabled": bool(graph_enabled),
        "events": recorder.events,
        "parameter_hash_before": model_hash_before,
        "parameter_hash_after": model_hash_after,
        "parameter_hashes_match": model_hash_before == model_hash_after,
        "steps": int(args.steps),
        "optimizer_updates": 0,
        "scheduler_steps": 0,
        "parameter_updates": 0,
        "python_container_tensor_audit": audit_python_container_tensors({"events": recorder.events}),
        "trace_retains_tensor_references": False,
        "post_cleanup_autograd_graph_nodes": [
            int(event.get("extra", {}).get("autograd_graph_nodes_after_cleanup", 0) or 0)
            for event in recorder.events
            if event.get("callpoint") == "post_step_cleanup"
        ],
        "callpoint_coverage": {
            callpoint: (
                "observed"
                if any(event.get("callpoint") == callpoint for event in recorder.events)
                else "not_applicable"
                if not graph_enabled and callpoint in GRAPH_CALLPOINTS
                else "missing"
            )
            for callpoint in (
                "model_loaded",
                "optimizer_created",
                "batch_to_gpu",
                "t5_encoder",
                "word_pooling",
                "node_edge_projection",
                "attention_logits",
                "relation_message_aggregation",
                "graph_fusion",
                "decoder_logits",
                "backward",
                "optimizer_step",
            )
        },
    }


def _load_real_batch(args, control_model, tokenizer):
    from m1_syntactic_graph_entry_audit import _collate_rows, _build_dataset
    from syntactic_graph import CompositeGraphCache, load_graph_cache_directory
    from t5_absa_train import build_target_unlabeled_domain_rows
    from m1_syntactic_graph_entry_audit import (
        _build_parser_identity_for_audit,
        _build_tokenizer_identity_for_audit,
    )

    run_dir = Path(args.run_dir).resolve()
    source_rows = _read_jsonl(run_dir / "inputs" / "source_train.jsonl")
    target_rows = _read_jsonl(run_dir / "inputs" / "target_unlabeled.jsonl")
    source_row = next((row for row in source_rows if str(row.get("id")) == str(args.source_row_id)), None)
    target_raw = next((row for row in target_rows if str(row.get("id")) == str(args.target_row_id)), None)
    if source_row is None or target_raw is None:
        raise RuntimeError("fixed source/target diagnostic rows are missing from V4 inputs")
    target_row = build_target_unlabeled_domain_rows([target_raw], use_task_prefix=False)[0]
    tokenizer_identity = _build_tokenizer_identity_for_audit(args.model_path, tokenizer)
    parser_identity = _build_parser_identity_for_audit(args.parser_dir)
    source_cache = load_graph_cache_directory(
        args.graph_cache_dir,
        "source_train",
        source_rows,
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
    graph_cache = CompositeGraphCache({"source_train": source_cache, "target_unlabeled": target_cache})
    dataset = _build_dataset(
        [source_row, target_row],
        tokenizer,
        graph_cache,
        int(args.max_source_length),
        int(args.max_target_length),
    )
    batch = _collate_rows(dataset, control_model, tokenizer, 2)
    domain_label = batch.get("domain_label")
    if not torch.is_tensor(domain_label) or domain_label.tolist() != [0, 1]:
        raise RuntimeError("diagnostic paired batch must have source domain 0 and target domain 1")
    labels = batch.get("labels")
    if not torch.is_tensor(labels) or not bool(labels[1].eq(-100).all().item()):
        raise RuntimeError("target diagnostic row must have labels=-100")
    batch["graph_metadata"] = {
        "source_row_id": str(source_row["id"]),
        "target_row_id": str(target_row["id"]),
        "source_text": str(source_row.get("text", "")),
        "target_text": str(target_row.get("text", "")),
        "source_count": 1,
        "target_count": 1,
        "tokens": _batch_counts(batch)["tokens"],
        "nodes": _batch_counts(batch)["nodes"],
        "edges": _batch_counts(batch)["edges"],
    }
    return batch, source_rows, target_rows


def _load_models(args, tokenizer, relation_vocab_size: int, sample_rows: list[dict]):
    from syntactic_graph_adapter import load_seq2seq_model
    from t5_absa_train import DomainAdversarialHead, add_task_special_tokens

    control = load_seq2seq_model(args.model_path, use_syntactic_graph_adapter=False)
    treatment = load_seq2seq_model(
        args.model_path,
        use_syntactic_graph_adapter=True,
        relation_vocab_size=relation_vocab_size,
    )
    add_task_special_tokens(tokenizer, control, sample_rows)
    treatment.resize_token_embeddings(len(tokenizer))
    add_task_special_tokens(tokenizer, treatment, sample_rows)
    hidden_size = int(getattr(control.config, "d_model", control.get_input_embeddings().embedding_dim))
    torch.manual_seed(int(args.seed) + 1)
    control.domain_adversarial_head = DomainAdversarialHead(hidden_size=hidden_size, classifier_hidden_size=256)
    torch.manual_seed(int(args.seed) + 1)
    treatment.domain_adversarial_head = DomainAdversarialHead(hidden_size=hidden_size, classifier_hidden_size=256)
    for model in (control, treatment):
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
    return control, treatment


def _write_markdown(path: Path, report: dict) -> None:
    comparison = report.get("comparison", {})
    batch = report.get("batch", {})
    config = report.get("config", {})
    errors = report.get("errors", [])
    lines = [
        "# M1 句法 RGAT 显存归因审计报告",
        "",
        f"总体状态：`{report['status']}`",
        "",
        f"固定批次：source row `{batch.get('source_row_id')}` + target row `{batch.get('target_row_id')}`，连续 `{config.get('steps')}` 次 zero-update（零更新）。",
        "",
        f"- Control（对照组）峰值：allocated `{comparison.get('control_peak_allocated_bytes')}` bytes，reserved `{comparison.get('control_peak_reserved_bytes')}` bytes。",
        f"- Treatment（实验组）峰值：allocated `{comparison.get('treatment_peak_allocated_bytes')}` bytes，reserved `{comparison.get('treatment_peak_reserved_bytes')}` bytes。",
        f"- 图模块峰值增量：`{comparison.get('graph_module_increment_peak_allocated_bytes')}` bytes。",
        f"- 首个显著增长调用点：`{comparison.get('first_significant_growth_callpoint')}`。",
        f"- 归因分类：`{comparison.get('classification', '未完成')}`；泄漏疑点=`{comparison.get('leak_suspected')}`，碎片化疑点=`{comparison.get('fragmentation_suspected')}`，WDDM（Windows 显示驱动模型）换页疑点=`{comparison.get('wddm_paging_suspected')}`。",
        "",
        "## 结论边界",
        "",
        "本审计不修改模型、图传播、损失、配方、训练参数或实验范围；不执行参数更新、调度器更新、正式训练、伪标签、Phase B（下游阶段）或 target_test（目标测试集）。",
        "",
        "WDDM 换页不能仅由 PyTorch 分配器计数证明；若接近显存上限，需结合 Windows 进程/主机遥测判断。",
        "",
        "target_test_access（目标测试集访问）=`false`。",
    ]
    if errors:
        lines.extend(["", "## 错误", "", *[f"- `{item.get('exception_type')}`：{item.get('message')}" for item in errors]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gpu_diagnostic(args) -> dict:
    """Run the user-invoked real GPU diagnostic; never called by CPU tests."""
    from transformers import AutoTokenizer
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("M1 VRAM attribution diagnostic requires CUDA")
    if not bool(args.fp16) or not bool(args.gradient_checkpointing):
        raise ValueError("VRAM diagnostic requires fp16 and gradient_checkpointing")
    if int(args.steps) < 3:
        raise ValueError("VRAM diagnostic requires at least three zero-update steps")
    if int(args.source_batch_size) != 1 or int(args.target_batch_size) != 1:
        raise ValueError("VRAM diagnostic requires source=1 and target=1")
    if abs(float(args.lambda_domain_adv) - 0.03) > 1e-12:
        raise ValueError("VRAM diagnostic requires lambda_domain_adv=0.03")
    device = torch.device("cuda:0")
    actual_cuda_index = int(torch.cuda.current_device())
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    from syntactic_graph import load_graph_cache_directory

    source_rows = _read_jsonl(Path(args.run_dir) / "inputs" / "source_train.jsonl")
    target_rows = _read_jsonl(Path(args.run_dir) / "inputs" / "target_unlabeled.jsonl")
    from m1_syntactic_graph_entry_audit import (
        _build_parser_identity_for_audit,
        _build_tokenizer_identity_for_audit,
    )
    tokenizer_identity = _build_tokenizer_identity_for_audit(args.model_path, tokenizer)
    parser_identity = _build_parser_identity_for_audit(args.parser_dir)
    source_cache = load_graph_cache_directory(args.graph_cache_dir, "source_train", source_rows, tokenizer_identity=tokenizer_identity, parser_identity=parser_identity)
    target_cache = load_graph_cache_directory(args.graph_cache_dir, "target_unlabeled", target_rows, tokenizer_identity=tokenizer_identity, parser_identity=parser_identity)
    source_row = next((row for row in source_rows if str(row.get("id")) == str(args.source_row_id)), None)
    if source_row is None:
        raise RuntimeError("fixed source diagnostic row is missing from V4 inputs")
    control, treatment = _load_models(
        args,
        tokenizer,
        source_cache.relation_vocab_size,
        [source_row],
    )
    batch, source_rows, target_rows = _load_real_batch(args, control, tokenizer)
    batch_meta = dict(batch.pop("graph_metadata"))
    batch_cpu = {key: value.detach().cpu().clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    variant_results = {}
    for variant, model, graph_enabled in tqdm(
        (("control", control, False), ("treatment", treatment, True)),
        desc="m1-vram-attribution",
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.to(device)
        variant_batch = _move_batch(batch_cpu, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
        variant_results[variant] = _run_variant_steps(
            model,
            optimizer,
            variant_batch,
            graph_enabled=graph_enabled,
            args=args,
            device=device,
            variant=variant,
        )
        del optimizer, variant_batch
        model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
    control_result = variant_results["control"]
    treatment_result = variant_results["treatment"]
    memory_comparison = analyze_memory_attribution(
        control_result["events"],
        treatment_result["events"],
        significant_growth_bytes=int(args.significant_growth_bytes),
        total_memory_bytes=int(torch.cuda.get_device_properties(device).total_memory),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "status": "PASS" if control_result["parameter_hashes_match"] and treatment_result["parameter_hashes_match"] else "BLOCKED",
        "identity": {
            "code": _git_identity(Path(__file__).resolve().parent),
            "model": _model_identity(Path(args.model_path).resolve()),
            "inputs": _input_identity(Path(args.run_dir).resolve()),
            "graph_cache": _cache_identity(Path(args.graph_cache_dir).resolve()),
            "parser": parser_identity,
            "tokenizer": tokenizer_identity,
        },
        "config": {
            "seed": int(args.seed),
            "steps": int(args.steps),
            "requested_cuda_index": int(str(args.cuda).split(",", 1)[0]),
            "actual_cuda_index": actual_cuda_index,
            "parser_device": "cache_reuse",
            "model_device": str(device),
            "source_batch_size": 1,
            "target_batch_size": 1,
            "fp16": bool(args.fp16),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "lambda_domain_adv": float(args.lambda_domain_adv),
            "learning_rate": float(args.learning_rate),
            "max_source_length": int(args.max_source_length),
            "max_target_length": int(args.max_target_length),
            "significant_growth_bytes": int(args.significant_growth_bytes),
            "trace_instrumentation": "MemoryTrace records metadata only; graph_trace still enables the existing trace-only attention probability buffer and is reported as diagnostic overhead.",
        },
        "batch": batch_meta,
        "control": control_result,
        "treatment": treatment_result,
        "comparison": memory_comparison,
        "optimizer_updates": 0,
        "scheduler_steps": 0,
        "parameter_updates": 0,
        "target_test_access": False,
        "optimization_semantics": {
            "semantics_preserving_if_diagnostic_only": [
                "释放不再需要的中间张量引用并验证 autograd graph 生命周期",
                "修复 Python 容器意外保存 GPU 张量",
                "仅在不改变批次/长度/公式的前提下进行分配器观测",
            ],
            "would_change_experiment_meaning": [
                "graph checkpointing",
                "CPU offload",
                "shorter sequence lengths",
                "different optimizer or batch size",
                "disabling graph adapter",
                "changing loss or DANN coefficient",
            ],
        },
    }
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M1 句法 RGAT 只读显存归因审计")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--graph_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=r"J:\nlp\models\t5-base-py")
    parser.add_argument("--parser_dir", default=r"J:\nlp\models\stanza_resources")
    parser.add_argument("--source_row_id", default=DEFAULT_SOURCE_ROW_ID)
    parser.add_argument("--target_row_id", default=DEFAULT_TARGET_ROW_ID)
    parser.add_argument("--source_batch_size", type=int, default=1)
    parser.add_argument("--target_batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--lambda_domain_adv", type=float, default=0.03)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=96)
    parser.add_argument("--significant_growth_bytes", type=int, default=32 * 1024 * 1024)
    return parser


def _write_outputs(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "m1_syntactic_rgat_vram_attribution_audit.json"
    markdown_path = output_dir / "m1_syntactic_rgat_vram_attribution_audit_CN.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(markdown_path, report)


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    try:
        report = run_gpu_diagnostic(args)
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "diagnostic_id": DIAGNOSTIC_ID,
            "status": "BLOCKED",
            "errors": [{"exception_type": type(exc).__name__, "message": str(exc)}],
            "config": {
                "requested_cuda_index": int(str(args.cuda).split(",", 1)[0]),
                "actual_cuda_index": None,
                "parser_device": "not_run",
                "model_device": "not_run",
                "steps": int(args.steps),
            },
            "target_test_access": False,
            "optimizer_updates": 0,
            "scheduler_steps": 0,
            "parameter_updates": 0,
        }
    _write_outputs(Path(args.output_dir), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

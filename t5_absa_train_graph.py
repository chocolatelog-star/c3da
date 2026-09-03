from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import sys
import tempfile
import weakref
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from t5_absa_data import read_jsonl
from t5_aste_data import (
    micro_f1,
    micro_f1_by_triplet_count,
    parse_triplet_text_list,
    triplet_count_diagnostics,
)
from syntactic_graph import (
    CompositeGraphCache,
    GraphCacheError,
    build_parser_identity,
    build_tokenizer_identity,
    load_graph_cache_directory,
)
from syntactic_graph_adapter import load_seq2seq_model
from element_aware_rgat import multi_element_coverage_loss


_PHASE_A_GRAPH_TRAINING_AUTHORIZED = False
_PHASE_A_LIFECYCLE_CLEANUP_REQUESTED = False


def _phase_a_state_sha256(value: object) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def phase_a_rng_state_hashes(*, include_cuda: bool | None = None) -> dict[str, str]:
    """Hash all RNG streams touched by the Phase A training process."""
    if include_cuda is None:
        include_cuda = torch.cuda.is_available()
    cuda_state: object = "unavailable"
    if include_cuda and torch.cuda.is_available():
        cuda_state = tuple(
            bytes(state.detach().cpu().contiguous().numpy().tobytes())
            for state in torch.cuda.get_rng_state_all()
        )
    return {
        "python_rng_sha256": _phase_a_state_sha256(random.getstate()),
        "numpy_rng_sha256": _phase_a_state_sha256(np.random.get_state()),
        "torch_cpu_rng_sha256": _phase_a_state_sha256(
            bytes(torch.get_rng_state().contiguous().numpy().tobytes())
        ),
        "torch_cuda_rng_sha256": _phase_a_state_sha256(cuda_state),
    }


def _phase_a_live_cuda_tensor_stats() -> dict[str, int]:
    count = 0
    total_bytes = 0
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj) or not obj.is_cuda:
                continue
            count += 1
            total_bytes += int(obj.numel() * obj.element_size())
        except (ReferenceError, RuntimeError, AttributeError):
            continue
    return {
        "live_cuda_tensor_count": count,
        "live_cuda_tensor_bytes": total_bytes,
    }


def phase_a_lifecycle_memory_snapshot(*, include_cuda: bool | None = None) -> dict[str, int]:
    """Return allocator and Python-owned CUDA tensor counters without allocating tensors."""
    if include_cuda is None:
        include_cuda = torch.cuda.is_available()
    if include_cuda and torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    else:
        memory = {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    live = (
        _phase_a_live_cuda_tensor_stats()
        if include_cuda and torch.cuda.is_available()
        else {"live_cuda_tensor_count": 0, "live_cuda_tensor_bytes": 0}
    )
    return {**memory, **live}


def _phase_a_reference_flags(runtime_refs: dict[str, object]) -> dict[str, bool]:
    return {name: value is not None for name, value in runtime_refs.items()}


def _phase_a_make_weakref_metadata(runtime_refs: dict[str, object]) -> dict[str, object]:
    """Capture weak references before cleanup while retaining no strong ownership."""
    weak_refs: dict[str, weakref.ReferenceType] = {}
    weakref_supported: dict[str, bool] = {}
    top_names: list[str] = []
    for name, value in runtime_refs.items():
        if value is None:
            continue
        top_names.append(name)
        try:
            weak_refs[name] = weakref.ref(value)
            weakref_supported[name] = True
            continue
        except TypeError:
            weakref_supported[name] = False
        if isinstance(value, dict):
            children = list(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = list(value)
        else:
            children = []
        for index, child in enumerate(children):
            if child is None:
                continue
            child_name = f"{name}[{index}]"
            try:
                weak_refs[child_name] = weakref.ref(child)
                weakref_supported[child_name] = True
            except TypeError:
                weakref_supported[child_name] = False
    return {
        "weak_refs": weak_refs,
        "weakref_supported": weakref_supported,
        "top_names": top_names,
    }


def _phase_a_weakref_evidence(metadata: dict[str, object]) -> dict[str, object]:
    weak_refs = metadata.get("weak_refs", {})
    weakref_supported = metadata.get("weakref_supported", {})
    top_names = metadata.get("top_names", [])
    weakref_alive = {
        name: bool(reference() is not None)
        for name, reference in weak_refs.items()
    }
    logical_flags: dict[str, bool] = {}
    for name in top_names:
        if name in weakref_alive:
            logical_flags[name] = weakref_alive[name]
        else:
            logical_flags[name] = any(
                alive for child_name, alive in weakref_alive.items()
                if child_name.startswith(f"{name}[")
            )
    return {
        **logical_flags,
        "weakref_alive": weakref_alive,
        "weakref_supported": dict(weakref_supported),
    }


def _phase_a_lifecycle_event(
    callpoint: str,
    *,
    runtime_refs: dict[str, object],
    include_cuda: bool | None = None,
    extra: dict[str, object] | None = None,
) -> dict:
    event = {
        "callpoint": callpoint,
        "memory": phase_a_lifecycle_memory_snapshot(include_cuda=include_cuda),
        "references": _phase_a_reference_flags(runtime_refs),
    }
    if extra:
        event.update(extra)
    return event


def cleanup_phase_a_training_runtime(
    runtime_refs: dict[str, object],
    *,
    cuda: bool | None = None,
    variant: str = "phase_a",
    defer_finalization: bool = False,
) -> dict:
    """Release Phase A runtime objects after artifacts have been durably saved.

    ``runtime_refs`` is deliberately a mutable ownership registry.  Clearing it
    makes the release contract testable and prevents a returned report from
    retaining a Trainer, model, optimizer, dataloader, callback, or batch.
    This function does not call any random operation and never changes an
    artifact on disk.
    """
    use_cuda = torch.cuda.is_available() if cuda is None else bool(cuda)
    trainer = runtime_refs.get("trainer")
    if runtime_refs.get("optimizer") is None and trainer is not None:
        runtime_refs["optimizer"] = getattr(trainer, "optimizer", None)
    if runtime_refs.get("dataloader") is None and trainer is not None:
        runtime_refs["dataloader"] = getattr(trainer, "_train_dataloader", None)
    if runtime_refs.get("callbacks") is None and trainer is not None:
        runtime_refs["callbacks"] = getattr(getattr(trainer, "callback_handler", None), "callbacks", None)
    if runtime_refs.get("accelerator") is None and trainer is not None:
        runtime_refs["accelerator"] = getattr(trainer, "accelerator", None)
    if runtime_refs.get("scheduler") is None and trainer is not None:
        runtime_refs["scheduler"] = getattr(trainer, "lr_scheduler", None)
    if runtime_refs.get("scaler") is None and trainer is not None:
        runtime_refs["scaler"] = getattr(trainer, "scaler", None)
    before_rng = phase_a_rng_state_hashes(include_cuda=use_cuda)
    events = [
        _phase_a_lifecycle_event(
            f"{variant}_return_before_cleanup",
            runtime_refs=runtime_refs,
            include_cuda=use_cuda,
            extra={"rng_state": before_rng},
        )
    ]
    weakref_metadata = _phase_a_make_weakref_metadata(runtime_refs)
    model = runtime_refs.get("model")
    optimizer = runtime_refs.get("optimizer")
    dataloader = runtime_refs.get("dataloader")
    callbacks = runtime_refs.get("callbacks")
    accelerator = runtime_refs.get("accelerator")

    if optimizer is not None:
        try:
            optimizer.zero_grad(set_to_none=True)
        except (AttributeError, TypeError):
            pass
        try:
            optimizer.state.clear()
        except AttributeError:
            pass
    if model is not None and isinstance(model, nn.Module):
        model.zero_grad(set_to_none=True)
        if model is not None and next(model.parameters(), None) is not None:
            model.cpu()
    if trainer is not None:
        for name in (
            "optimizer",
            "lr_scheduler",
            "scaler",
            "model",
            "train_dataset",
            "eval_dataset",
            "data_collator",
            "_train_dataloader",
            "_past",
            "dann_batch_sampler",
            "_models",
            "_optimizers",
            "_dataloaders",
        ):
            if hasattr(trainer, name):
                holder = getattr(trainer, name)
                if hasattr(holder, "clear"):
                    holder.clear()
                setattr(trainer, name, None)
        callback_handler = getattr(trainer, "callback_handler", None)
        if callback_handler is not None and hasattr(callback_handler, "callbacks"):
            callback_handler.callbacks.clear()
        if hasattr(trainer, "callback_handler"):
            trainer.callback_handler = None
        if hasattr(trainer, "accelerator"):
            trainer.accelerator = None
    if accelerator is not None:
        for name in (
            "_models",
            "_optimizers",
            "_dataloaders",
            "_schedulers",
            "_scalers",
        ):
            if hasattr(accelerator, name):
                holder = getattr(accelerator, name)
                if hasattr(holder, "clear"):
                    holder.clear()
                setattr(accelerator, name, None)
        for name in ("optimizer", "lr_scheduler", "scaler"):
            if hasattr(accelerator, name):
                setattr(accelerator, name, None)
    if callbacks is not None and hasattr(callbacks, "clear"):
        callbacks.clear()
    if dataloader is not None:
        del dataloader
    runtime_refs.clear()
    trainer = None
    model = None
    optimizer = None
    callbacks = None
    dataloader = None
    accelerator = None
    callback_handler = None
    gc.collect()
    events.append(
        {
            "callpoint": f"{variant}_gc_after",
            "memory": phase_a_lifecycle_memory_snapshot(include_cuda=use_cuda),
            "references": _phase_a_weakref_evidence(weakref_metadata),
        }
    )
    if use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    after_rng = phase_a_rng_state_hashes(include_cuda=use_cuda)
    events.append(
        {
            "callpoint": f"{variant}_cuda_empty_cache_after",
            "memory": phase_a_lifecycle_memory_snapshot(include_cuda=use_cuda),
            "references": _phase_a_weakref_evidence(weakref_metadata),
            "rng_state": after_rng,
        }
    )
    lifecycle = {
        "schema_version": 1,
        "cleanup_performed": True,
        "events": events,
        "rng_state_before_cleanup": before_rng,
        "rng_state_after_cleanup": after_rng,
        "rng_state_unchanged": before_rng == after_rng,
        "references_after_cleanup": events[-1]["references"],
        "_weakref_metadata": weakref_metadata,
        "_include_cuda": use_cuda,
    }
    if defer_finalization:
        return lifecycle
    return finalize_phase_a_training_runtime(lifecycle, include_cuda=use_cuda, variant=variant)


def finalize_phase_a_training_runtime(
    lifecycle: dict,
    *,
    include_cuda: bool | None = None,
    variant: str | None = None,
) -> dict:
    """Record final weakref evidence after the caller releases its own locals."""
    use_cuda = bool(lifecycle.get("_include_cuda", False)) if include_cuda is None else bool(include_cuda)
    gc.collect()
    if use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    after_rng = phase_a_rng_state_hashes(include_cuda=use_cuda)
    before_rng = lifecycle.get("rng_state_before_cleanup", {})
    metadata = lifecycle.get("_weakref_metadata", {})
    final_references = _phase_a_weakref_evidence(metadata)
    final_event = {
        "callpoint": "phase_a_return_after_local_release",
        "variant": variant,
        "memory": phase_a_lifecycle_memory_snapshot(include_cuda=use_cuda),
        "references": final_references,
        "rng_state_before_cleanup": before_rng,
        "rng_state_after_cleanup": after_rng,
        "rng_state": after_rng,
    }
    lifecycle.setdefault("events", []).append(final_event)
    lifecycle["rng_state_after_finalization"] = after_rng
    lifecycle["rng_state_unchanged"] = before_rng == after_rng
    lifecycle["references_after_finalization"] = final_references
    lifecycle["references_after_cleanup"] = final_references
    lifecycle.pop("_weakref_metadata", None)
    lifecycle.pop("_include_cuda", None)
    return lifecycle


def _phase_a_cpu_copy(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _phase_a_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_phase_a_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_phase_a_cpu_copy(item) for item in value)
    return value


def reproducibility_training_args(seed: int, mode: str) -> dict:
    if mode == "legacy":
        return {"seed": seed}
    if mode not in {"seeded", "deterministic"}:
        raise ValueError(f"unsupported reproducibility mode: {mode}")
    return {
        "seed": seed,
        "data_seed": seed,
        "full_determinism": mode == "deterministic",
        "dataloader_num_workers": 0,
    }


def configure_reproducibility(seed: int, mode: str) -> dict:
    deterministic = mode == "deterministic"
    if mode == "legacy":
        os.environ.pop("PYTHONHASHSEED", None)
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "mode": mode,
        "deterministic": deterministic,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def initialize_domain_adversarial_head(
    model: nn.Module,
    *,
    hidden_size: int,
    classifier_hidden_size: int,
    seed: int,
) -> nn.Module:
    """Create the shared DANN head without consuming the training RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        head = DomainAdversarialHead(
            hidden_size=hidden_size,
            classifier_hidden_size=classifier_hidden_size,
        )
    model.domain_adversarial_head = head
    return head


def _parameter_digest(parameters: list[tuple[str, torch.Tensor]]) -> tuple[str, list[str], list[dict]]:
    digest = hashlib.sha256()
    names = []
    stats = []
    for name, parameter in sorted(parameters, key=lambda item: item[0]):
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
        finite = torch.isfinite(value)
        finite_count = int(finite.sum().item())
        total_count = int(value.numel())
        names.append(name)
        stats.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "finite": finite_count == total_count,
                "finite_count": finite_count,
                "total_count": total_count,
                "min": float(value[finite].min().item()) if finite_count else None,
                "max": float(value[finite].max().item()) if finite_count else None,
                "max_abs": float(value[finite].abs().max().item()) if finite_count else None,
            }
        )
    return digest.hexdigest(), names, stats


def build_initialization_audit(model: nn.Module, *, variant: str, seed: int) -> dict:
    named_parameters = list(model.named_parameters())
    graph_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("syntactic_graph_adapter.")
    ]
    dann_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("domain_adversarial_head.")
    ]
    shared_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if not name.startswith("syntactic_graph_adapter.")
        and not name.startswith("domain_adversarial_head.")
    ]
    shared_hash, shared_names, shared_stats = _parameter_digest(shared_parameters)
    dann_hash, dann_names, dann_stats = _parameter_digest(dann_parameters)
    graph_hash, graph_names, graph_stats = _parameter_digest(graph_parameters)
    graph_init = dict(getattr(model, "graph_parameter_initialization", {}))
    return {
        "schema_version": 1,
        "variant": variant,
        "seed": int(seed),
        "shared_t5_parameter_sha256": shared_hash,
        "dann_head_parameter_sha256": dann_hash,
        "graph_parameter_sha256": graph_hash,
        "parameter_groups": {
            "shared_t5": {"parameter_names": shared_names, "sha256": shared_hash, "parameter_stats": shared_stats},
            "domain_adversarial_head": {"parameter_names": dann_names, "sha256": dann_hash, "parameter_stats": dann_stats},
            "syntactic_graph_adapter": {"parameter_names": graph_names, "sha256": graph_hash, "parameter_stats": graph_stats},
        },
        "graph_parameter_initialization": graph_init,
        "graph_parameter_stats": graph_stats,
    }


def _utf8_lf_bytes(value: str) -> bytes:
    """Encode durable text as UTF-8 without BOM and with fixed LF endings."""
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
    _atomic_write_bytes(
        path,
        _utf8_lf_bytes(json.dumps(value, ensure_ascii=False, indent=2) + "\n"),
    )


def _json_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_json_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_dann_expected_max_steps(
    *,
    source_count: int,
    target_count: int,
    source_batch_size: int,
    target_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int | float,
) -> int:
    """Independently mirror Trainer's paired-loader max-step calculation."""
    planned_batches = compute_dann_planned_batches(
        source_count=source_count,
        target_count=target_count,
        source_batch_size=source_batch_size,
        target_batch_size=target_batch_size,
    )
    updates_per_epoch = max(1, planned_batches // int(gradient_accumulation_steps))
    return math.ceil(float(num_train_epochs) * updates_per_epoch)


def compute_dann_planned_batches(
    *, source_count: int, target_count: int, source_batch_size: int, target_batch_size: int
) -> int:
    return max(
        math.ceil(int(source_count) / int(source_batch_size)),
        math.ceil(int(target_count) / int(target_batch_size)),
    )


def classify_terminal_lookahead(
    report: dict,
    *,
    gradient_accumulation_steps: int,
    journal_audit: dict | None = None,
) -> dict:
    """Classify a terminal DataLoader lookahead without rewriting the journal.

    ``issued`` is durable loader intent while ``processed`` is an acknowledged
    training batch.  A single final dangling issue is acceptable only when the
    Trainer has already reached ``max_steps`` and the acknowledged prefix is a
    complete optimizer-step boundary.  This helper is shared by the runtime
    audit and the Phase A parent, so a terminal lookahead cannot be silently
    counted as processed in one path and rejected in another.
    """
    result = {
        "safe": False,
        "classification": "invalid_terminal_accounting",
        "dangling_logical_batch_ids": [],
        "processed_batches": None,
        "journal_chain_validated": journal_audit is not None,
        "reasons": [],
        "lookahead_not_consumed": False,
    }
    if not isinstance(report, dict) or not isinstance(report.get("epochs"), list) or not report["epochs"]:
        result["reasons"].append("missing_epochs")
        return result
    if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps <= 0:
        result["reasons"].append("invalid_gradient_accumulation_steps")
        return result

    epochs = report["epochs"]
    last = epochs[-1]
    if not isinstance(last, dict):
        result["reasons"].append("invalid_last_epoch")
        return result
    issued = last.get("issued_batches")
    processed = last.get("processed_batches")
    planned = last.get("planned_batches")
    batches = last.get("batches")
    result["processed_batches"] = processed
    if not all(isinstance(value, int) for value in (issued, processed, planned)):
        result["reasons"].append("non_integer_accounting")
        return result
    if not isinstance(batches, list) or len(batches) != issued or not 0 <= processed <= issued <= planned:
        result["reasons"].append("invalid_issued_processed_planned_accounting")
        return result
    if any(
        not isinstance(batch, dict) or batch.get("logical_batch_id") != index
        for index, batch in enumerate(batches)
    ):
        result["reasons"].append("logical_batch_ids_not_contiguous")
        return result

    partial_before_last = any(
        isinstance(epoch, dict) and epoch.get("completion") == "partial"
        for epoch in epochs[:-1]
    )
    if partial_before_last:
        result["reasons"].append("partial_traversal_before_last")
    if last.get("completion") not in {"complete", "partial"}:
        result["reasons"].append("invalid_completion")

    trainer_global_step = report.get("trainer_global_step")
    trainer_max_steps = report.get("trainer_max_steps")
    is_terminal = (
        isinstance(trainer_global_step, int)
        and isinstance(trainer_max_steps, int)
        and trainer_global_step == trainer_max_steps
        and trainer_max_steps > 0
    )

    if issued == processed:
        if last.get("completion") == "complete" and issued == planned and not result["reasons"]:
            result.update({"safe": True, "classification": "normal_complete"})
            return result
        if is_terminal and last.get("completion") == "partial" and not result["reasons"]:
            result.update({"safe": True, "classification": "terminal_partial_consumed"})
            return result
        result["reasons"].append("issued_processed_mismatch_for_nonterminal_or_partial")
        return result

    dangling = list(range(processed, issued))
    result["dangling_logical_batch_ids"] = dangling
    result["reasons"].extend([
        reason for reason, condition in (
            ("not_at_trainer_max_steps", not is_terminal),
            ("last_traversal_not_partial", last.get("completion") != "partial"),
            ("multiple_dangling_issued_batches", issued != processed + 1),
            ("dangling_batch_is_not_last_logical_id", dangling != [issued - 1]),
            ("processed_prefix_not_optimizer_boundary", processed % gradient_accumulation_steps != 0),
        ) if condition
    ])
    step_start = last.get("optimizer_global_step_start")
    step_end = last.get("optimizer_global_step_end")
    if not isinstance(step_start, int) or not isinstance(step_end, int) or step_end < step_start:
        result["reasons"].append("invalid_optimizer_step_range")
    elif step_end != trainer_global_step or step_end - step_start != processed // gradient_accumulation_steps:
        result["reasons"].append("optimizer_step_range_does_not_match_processed_prefix")

    for field, expected in (
        ("accumulation_remainder", 0),
        ("uncommitted_gradient_count", 0),
    ):
        if field in report and report.get(field) != expected:
            result["reasons"].append(f"{field}_is_not_zero")
        if field in last and last.get(field) != expected:
            result["reasons"].append(f"last_{field}_is_not_zero")
    for field in ("uncommitted_gradients", "pending_gradient_update", "optimizer_update_after_terminal", "checkpoint_after_terminal"):
        if bool(report.get(field, False)) or bool(last.get(field, False)):
            result["reasons"].append(f"{field}_present")
    for field in ("later_checkpoints", "optimizer_updates_after_terminal"):
        value = report.get(field, 0)
        if value not in (0, [], None):
            result["reasons"].append(f"{field}_present")

    if journal_audit is not None:
        if journal_audit.get("chain_valid") is False:
            result["reasons"].append("journal_chain_invalid")
        processed_ids = {
            (item[0], item[1])
            for item in journal_audit.get("processed_batch_ids", [])
            if isinstance(item, (list, tuple)) and len(item) == 2
        }
        physical = last.get("physical_traversal_index")
        if physical is not None and any((physical, batch_id) in processed_ids for batch_id in dangling):
            result["reasons"].append("dangling_batch_has_processed_event")
        issued_ids = {
            (item[0], item[1])
            for item in journal_audit.get("issued_batch_ids", [])
            if isinstance(item, (list, tuple)) and len(item) == 2
        }
        if physical is not None and any((physical, batch_id) not in issued_ids for batch_id in dangling):
            result["reasons"].append("dangling_batch_missing_issued_event")

    if not result["reasons"]:
        result.update({
            "safe": True,
            "classification": "terminal_lookahead_not_consumed",
            "lookahead_not_consumed": True,
        })
    return result


def _read_dann_audit_journal_records(path: Path) -> list[dict]:
    records = []
    expected_previous = ""
    raw_lines = path.read_bytes().splitlines()
    for line_number, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if line_number == len(raw_lines) - 1:
                break
            raise ValueError(f"invalid DANN audit journal line {line_number + 1}") from exc
        body = {
            "event": record.get("event"),
            "payload": record.get("payload"),
            "previous_hash": record.get("previous_hash"),
        }
        if record.get("previous_hash") != expected_previous or record.get("hash") != hashlib.sha256(_canonical_json_bytes(body)).hexdigest():
            raise ValueError(f"DANN audit journal hash-chain mismatch at line {line_number + 1}")
        records.append(record)
        expected_previous = record["hash"]
    return records


def recover_dann_audit_journal(path: str | Path) -> dict:
    """Replay complete journal lines, tolerating one torn final line."""
    records = _read_dann_audit_journal_records(Path(path))
    if not records or records[0].get("event") != "identity":
        raise ValueError("DANN audit journal has no identity record")
    identity = dict(records[0]["payload"])
    epochs = []
    by_physical = {}
    for record in records[1:]:
        event = record.get("event")
        payload = record.get("payload") or {}
        if event == "traversal_started":
            report = dict(payload["report"])
            report["batches"] = list(report.get("batches", []))
            epochs.append(report)
            by_physical[report["physical_traversal_index"]] = report
        elif event == "batch_issued":
            report = by_physical[payload["physical_traversal_index"]]
            report["batches"].append(dict(payload["batch"]))
            report["issued_batches"] += 1
            report["logical_batches"] = report["issued_batches"]
            report["source_rows"] += report["batches"][-1]["source_count"]
            report["target_rows"] += report["batches"][-1]["target_count"]
            report["source_unique_rows"] = len({index for batch in report["batches"] for index in batch["source_indices"]})
            report["target_unique_rows"] = len({index - identity["source_count"] for batch in report["batches"] for index in batch["target_indices"]})
        elif event == "batch_processed":
            by_physical[payload["physical_traversal_index"]]["processed_batches"] += 1
        elif event == "traversal_completed":
            report = by_physical[payload["physical_traversal_index"]]
            report["completion"] = "complete"
            report["optimizer_global_step_end"] = payload.get("optimizer_global_step_end")
    last_sampling = epochs[-1]["sampling_epoch"] if epochs else -1
    return {
        **identity,
        "current_epoch": last_sampling,
        "next_physical_traversal_index": len(epochs),
        "next_sampling_epoch": max((item["sampling_epoch"] for item in epochs), default=-1) + 1,
        "trainer_global_step": epochs[-1].get("optimizer_global_step_end") if epochs else None,
        "trainer_max_steps": None,
        "epochs": epochs,
    }


def read_dann_audit_journal(path: str | Path) -> dict:
    """Read journal replay events for the formal fresh/resume audit gate."""
    journal_path = Path(path)
    records = _read_dann_audit_journal_records(journal_path)
    replay_events = []
    issued_batch_ids = []
    processed_batch_ids = []
    for record in records:
        payload = record.get("payload") or {}
        if record.get("event") in {"batch_issued", "batch_reissued"}:
            batch = payload.get("batch") or {}
            physical = payload.get("physical_traversal_index")
            logical = batch.get("logical_batch_id")
            if isinstance(physical, int) and isinstance(logical, int):
                issued_batch_ids.append([physical, logical])
        elif record.get("event") == "batch_processed":
            physical = payload.get("physical_traversal_index")
            logical = payload.get("logical_batch_id")
            if isinstance(physical, int) and isinstance(logical, int):
                processed_batch_ids.append([physical, logical])
        if record.get("event") != "batch_replayed":
            continue
        if not isinstance(payload.get("checkpoint_path"), str) or not payload.get("checkpoint_path"):
            raise ValueError("DANN replay event is missing checkpoint_path")
        if not isinstance(payload.get("checkpoint_sha256"), str) or not payload.get("checkpoint_sha256"):
            raise ValueError("DANN replay event is missing checkpoint_sha256")
        batch_identity = payload.get("recovery_batch_identity")
        if (
            not isinstance(batch_identity, dict)
            or not isinstance(batch_identity.get("physical_traversal_index"), int)
            or not isinstance(batch_identity.get("logical_batch_id"), int)
        ):
            raise ValueError("DANN replay event is missing recovery batch identity")
        replay_events.append(
            {
                "checkpoint_path": payload["checkpoint_path"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "recovery_batch_identity": dict(batch_identity),
            }
        )
    return {
        "journal_path": str(journal_path),
        "record_count": len(records),
        "chain_valid": True,
        "issued_batch_ids": issued_batch_ids,
        "processed_batch_ids": processed_batch_ids,
        "replay_count": len(replay_events),
        "replay_events": replay_events,
    }


TASK_SPECIAL_TOKENS = ["<pos>", "<neg>", "<neu>", "<opinion>", "<aspect>"]
CSA_AUGMENT_CHANNELS = {
    "aspect_channel",
    "opinion_sentiment_channel",
    "masked_aspect_channel",
    "masked_opinion_sentiment_channel",
    "label_composition_channel",
    "label_to_text_channel",
    "sentence_fusion_composition_channel",
}
TAG_INIT_WORDS = {
    "<pos>": "positive",
    "<neg>": "negative",
    "<neu>": "neutral",
    "<opinion>": "opinion",
    "<aspect>": "aspect",
}
SENTIMENT_LABEL_IDS = {"pos": 0, "neg": 1, "neu": 2}


def build_target_unlabeled_domain_rows(
    rows: list[dict],
    use_task_prefix: bool = True,
) -> list[dict]:
    """Build target rows which contribute only to the existing DANN loss."""
    domain_rows = []
    for row in rows:
        text = str(row.get("text", ""))
        if not text:
            raise ValueError(f"target-unlabeled row has empty text: {row.get('id')}")
        input_text = f"extract aste: {text}" if use_task_prefix else text
        domain_rows.append(
            {
                "id": row["id"],
                "text": text,
                "input": input_text,
                "target": "",
                "augmentation": "target_unlabeled",
                "sample_weight": 0.0,
                "domain_weight": 0.0,
                "structure_weight": 0.0,
                "domain_label": 1,
            }
        )
    return domain_rows


def decode_keep_aste_task_tokens(tokenizer, token_ids) -> str:
    token_ids = [tokenizer.pad_token_id if int(token) < 0 else int(token) for token in token_ids]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    for token in (
        tokenizer.pad_token,
        tokenizer.eos_token,
        tokenizer.unk_token,
        "<s>",
    ):
        if token:
            text = text.replace(token, " ")
    return " ".join(text.split())


def _metric_input_to_numpy(value, name: str) -> np.ndarray:
    if isinstance(value, tuple):
        if not value:
            raise ValueError(f"{name} tuple must not be empty")
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a numpy array") from exc


def build_aste_compute_metrics(tokenizer):
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be defined for ASTE metrics")

    def compute_metrics(eval_prediction):
        predictions = _metric_input_to_numpy(eval_prediction.predictions, "predictions")
        labels = _metric_input_to_numpy(eval_prediction.label_ids, "labels")
        if predictions.ndim == 3:
            if predictions.shape[-1] == 0:
                raise ValueError("predictions logits dimension must not be empty")
            predictions = predictions.argmax(axis=-1)
        if predictions.ndim != 2:
            raise ValueError(
                f"predictions dimension must be 2 for token ids or 3 for logits; got {predictions.ndim}"
            )
        if labels.ndim != 2:
            raise ValueError(f"labels dimension must be 2; got {labels.ndim}")
        if predictions.shape[0] != labels.shape[0]:
            raise ValueError(
                "predictions and labels batch lengths must match; "
                f"got {predictions.shape[0]} and {labels.shape[0]}"
            )
        labels = labels.copy()
        labels[labels == -100] = tokenizer.pad_token_id

        prediction_texts = [decode_keep_aste_task_tokens(tokenizer, row) for row in predictions]
        gold_texts = [decode_keep_aste_task_tokens(tokenizer, row) for row in labels]
        overall = micro_f1(prediction_texts, gold_texts)
        grouped = micro_f1_by_triplet_count(prediction_texts, gold_texts)
        diagnostics = triplet_count_diagnostics(prediction_texts, gold_texts)

        multi_predictions = []
        multi_golds = []
        for prediction, gold in zip(prediction_texts, gold_texts):
            if len(parse_triplet_text_list(gold)) >= 2:
                multi_predictions.append(prediction)
                multi_golds.append(gold)
        multi = micro_f1(multi_predictions, multi_golds)

        metrics = {
            "micro_f1": overall["micro_f1"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "multi_micro_f1": multi["micro_f1"],
            "exact_count_accuracy": diagnostics["exact_count_accuracy"],
            "under_generated_rows": diagnostics["under_generated_rows"],
            "over_generated_rows": diagnostics["over_generated_rows"],
        }
        for bucket in ("count1", "count2", "count3", "count4plus"):
            metrics[f"{bucket}_micro_f1"] = grouped[bucket]["micro_f1"]
        metrics["selection_score"] = metrics["micro_f1"] + 0.001 * metrics["multi_micro_f1"]
        return metrics

    return compute_metrics


def build_checkpoint_selection_config(checkpoint_selection: str) -> dict:
    if checkpoint_selection not in {"last", "best", "aste_f1"}:
        raise ValueError(f"unsupported checkpoint selection: {checkpoint_selection}")
    if checkpoint_selection == "aste_f1":
        return {
            "predict_with_generate": True,
            "generation_num_beams": 1,
            "generation_max_length": 128,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_selection_score",
            "greater_is_better": True,
        }
    return {
        "predict_with_generate": True,
        "load_best_model_at_end": checkpoint_selection == "best",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }


def cleanup_training_checkpoints(output_dir: Path) -> list[str]:
    """Remove resumable checkpoints after the final model is materialized."""
    removed: list[str] = []
    for checkpoint_dir in sorted(output_dir.glob("checkpoint-*")):
        if not checkpoint_dir.is_dir():
            continue
        removed.append(checkpoint_dir.name)
        shutil.rmtree(checkpoint_dir)
    return removed


class JsonlSeq2SeqDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        max_source_length: int,
        max_target_length: int,
        source_weight: float,
        pseudo_weight: float,
        augment_weight: float,
        multi_triplet_loss_gain: float = 0.0,
        neutral_loss_gain: float = 0.0,
        max_effective_weight: float = 1.0,
        neutral_generation_loss_gain: float = 0.0,
        neutral_generation_max_effective_weight: float | None = None,
        force_domain_weights: bool = False,
        max_pairing_triplets: int = 4,
        min_pairing_triplets: int = 2,
        min_pairing_sample_weight: float = 0.65,
        pairing_source_only: bool = False,
        domain_adv_exclude_augment: bool = False,
        sentiment_contrastive_min_weight: float = 0.65,
        sentiment_contrastive_exclude_augment: bool = False,
        sentiment_contrastive_source_only: bool = False,
        graph_cache=None,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_weight = source_weight
        self.pseudo_weight = pseudo_weight
        self.augment_weight = augment_weight
        self.multi_triplet_loss_gain = multi_triplet_loss_gain
        self.neutral_loss_gain = neutral_loss_gain
        self.max_effective_weight = max_effective_weight
        self.neutral_generation_loss_gain = neutral_generation_loss_gain
        self.neutral_generation_max_effective_weight = (
            1.0
            if neutral_generation_max_effective_weight is None or neutral_generation_max_effective_weight <= 0
            else neutral_generation_max_effective_weight
        )
        self.force_domain_weights = force_domain_weights
        self.max_pairing_triplets = max_pairing_triplets
        self.min_pairing_triplets = min_pairing_triplets
        self.min_pairing_sample_weight = min_pairing_sample_weight
        self.pairing_source_only = pairing_source_only
        self.domain_adv_exclude_augment = domain_adv_exclude_augment
        self.sentiment_contrastive_min_weight = sentiment_contrastive_min_weight
        self.sentiment_contrastive_exclude_augment = sentiment_contrastive_exclude_augment
        self.sentiment_contrastive_source_only = sentiment_contrastive_source_only
        self.graph_cache = graph_cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        model_inputs = self.tokenizer(
            row["input"],
            max_length=self.max_source_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=row["target"],
            max_length=self.max_target_length,
            truncation=True,
        )
        if row.get("augmentation") == "target_unlabeled":
            labels["input_ids"] = [-100] * len(labels["input_ids"])
        model_inputs["labels"] = labels["input_ids"]
        sample_weight = self.sample_weight(row)
        model_inputs["sample_weight"] = sample_weight
        model_inputs["domain_weight"] = self.generation_weight(row, sample_weight)
        model_inputs["domain_label"] = self.domain_label(row)
        model_inputs["structure_weight"] = self.structure_weight(row, sample_weight)
        model_inputs["consistency_group"] = self.consistency_group(row, idx)
        model_inputs.update(self.pairing_features(row, model_inputs["input_ids"], sample_weight))
        model_inputs.update(self.sentiment_contrastive_features(row, model_inputs["input_ids"], sample_weight))
        if self.graph_cache is not None:
            model_inputs.update(self.graph_cache.get(row))
        return model_inputs

    def sample_weight(self, row: dict) -> float:
        if row.get("augmentation") == "target_unlabeled":
            return 0.0
        if "sample_weight" in row and not self.force_domain_weights:
            return float(row["sample_weight"])
        augmentation = row.get("augmentation")
        if augmentation == "target_pseudo":
            return self.pseudo_weight
        if augmentation in CSA_AUGMENT_CHANNELS:
            return self.augment_weight
        return self.source_weight

    def domain_label(self, row: dict) -> int:
        augmentation = row.get("augmentation")
        if augmentation == "target_unlabeled":
            return 1
        if self.domain_adv_exclude_augment and augmentation in CSA_AUGMENT_CHANNELS:
            return -100
        if augmentation == "target_pseudo" or augmentation in CSA_AUGMENT_CHANNELS:
            return 1
        return 0

    def structure_weight(self, row: dict, domain_weight: float) -> float:
        triplets = parse_triplet_text_list(row.get("target", ""))
        multiplier = 1.0
        if len(triplets) >= 2:
            multiplier += self.multi_triplet_loss_gain * min(len(triplets) - 1, 2)
        if any(sentiment == "neu" for _aspect, _opinion, sentiment in triplets):
            multiplier += self.neutral_loss_gain
        return min(domain_weight * multiplier, self.max_effective_weight)

    def generation_weight(self, row: dict, sample_weight: float) -> float:
        triplets = parse_triplet_text_list(row.get("target", ""))
        if not any(sentiment == "neu" for _aspect, _opinion, sentiment in triplets):
            return sample_weight
        return min(
            sample_weight * (1.0 + self.neutral_generation_loss_gain),
            self.neutral_generation_max_effective_weight,
        )

    def consistency_group(self, row: dict, idx: int) -> int:
        if row.get("base_id") is not None:
            return stable_group_id(row["base_id"])
        if row.get("id") is not None:
            return stable_group_id(row["id"])
        return int(idx)

    def pairing_features(self, row: dict, input_ids: list[int], sample_weight: float) -> dict:
        target = row.get("target", "")
        triplets = parse_triplet_text_list(target)
        augmentation = row.get("augmentation")
        if self.pairing_source_only and (
            augmentation == "target_pseudo" or augmentation in CSA_AUGMENT_CHANNELS
        ):
            return {
                "pairing_aspect_spans": [],
                "pairing_opinion_spans": [],
                "pairing_mask": [],
            }
        if len(triplets) < self.min_pairing_triplets:
            return {
                "pairing_aspect_spans": [],
                "pairing_opinion_spans": [],
                "pairing_mask": [],
            }
        if sample_weight < self.min_pairing_sample_weight and row.get("augmentation") != "target_pseudo":
            return {
                "pairing_aspect_spans": [],
                "pairing_opinion_spans": [],
                "pairing_mask": [],
            }
        aspect_spans: list[list[int]] = []
        opinion_spans: list[list[int]] = []
        mask: list[int] = []
        for aspect, opinion, _sentiment in triplets[: self.max_pairing_triplets]:
            aspect_span = find_fragment_span_in_input(
                self.tokenizer, row.get("input", ""), input_ids, aspect
            )
            opinion_span = find_fragment_span_in_input(
                self.tokenizer, row.get("input", ""), input_ids, opinion
            )
            if aspect_span is None or opinion_span is None:
                continue
            aspect_spans.append(list(aspect_span))
            opinion_spans.append(list(opinion_span))
            mask.append(1)
        if len(mask) < self.min_pairing_triplets:
            return {
                "pairing_aspect_spans": [],
                "pairing_opinion_spans": [],
                "pairing_mask": [],
            }
        return {
            "pairing_aspect_spans": aspect_spans,
            "pairing_opinion_spans": opinion_spans,
            "pairing_mask": mask,
        }

    def sentiment_contrastive_features(self, row: dict, input_ids: list[int], domain_weight: float) -> dict:
        augmentation = row.get("augmentation")
        if domain_weight < self.sentiment_contrastive_min_weight:
            return self.empty_sentiment_contrastive_features()
        if self.sentiment_contrastive_exclude_augment and augmentation in CSA_AUGMENT_CHANNELS:
            return self.empty_sentiment_contrastive_features()
        if self.sentiment_contrastive_source_only and (
            augmentation == "target_pseudo" or augmentation in CSA_AUGMENT_CHANNELS
        ):
            return self.empty_sentiment_contrastive_features()
        spans = []
        labels = []
        for _aspect, opinion, sentiment in parse_triplet_text_list(row.get("target", "")):
            sentiment_id = SENTIMENT_LABEL_IDS.get(sentiment)
            span = find_opinion_span_in_input(self.tokenizer, row.get("input", ""), input_ids, opinion)
            if sentiment_id is None or span is None:
                continue
            spans.append(list(span))
            labels.append(sentiment_id)
        return {
            "sentiment_contrastive_spans": spans,
            "sentiment_contrastive_labels": labels,
            "sentiment_contrastive_mask": [1] * len(labels),
            "sentiment_contrastive_weights": [float(domain_weight)] * len(labels),
        }

    @staticmethod
    def empty_sentiment_contrastive_features() -> dict:
        return {
            "sentiment_contrastive_spans": [],
            "sentiment_contrastive_labels": [],
            "sentiment_contrastive_mask": [],
            "sentiment_contrastive_weights": [],
        }


def stable_group_id(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
        return int(digest[:12], 16)


def find_token_subsequence_span(sequence: list[int], subsequence: list[int]) -> tuple[int, int] | None:
    if not sequence or not subsequence or len(subsequence) > len(sequence):
        return None
    width = len(subsequence)
    for start in range(0, len(sequence) - width + 1):
        if sequence[start : start + width] == subsequence:
            return start, start + width
    return None


def find_fragment_span_in_input(
    tokenizer,
    text: str,
    input_ids: list[int],
    fragment: str,
) -> tuple[int, int] | None:
    candidates = [fragment]
    lower_text = text.lower()
    lower_fragment = fragment.lower()
    start = 0
    while lower_fragment and (match_start := lower_text.find(lower_fragment, start)) >= 0:
        candidates.append(text[match_start : match_start + len(fragment)])
        start = match_start + max(1, len(fragment))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        span = find_token_subsequence_span(input_ids, tokenizer.encode(candidate, add_special_tokens=False))
        if span is not None:
            return span
    return None


def find_opinion_span_in_input(tokenizer, text: str, input_ids: list[int], opinion: str) -> tuple[int, int] | None:
    return find_fragment_span_in_input(tokenizer, text, input_ids, opinion)


class DataCollatorForSeq2SeqWithPairing:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features: list[dict]) -> dict:
        graph_keys = (
            "word_to_subword",
            "word_mask",
            "edge_src",
            "edge_dst",
            "relation_id",
            "dependency_relation_id",
            "pos_pair_id",
            "compositional_dependency_id",
            "compositional_direction_id",
            "compositional_src_pos_id",
            "compositional_dst_pos_id",
            "edge_mask",
        )
        core_graph_keys = tuple(key for key in graph_keys if not key.startswith("compositional_"))
        graph_present = [key in feature for feature in features for key in core_graph_keys]
        if any(graph_present) and not all(graph_present):
            raise ValueError("graph fields must be present for every feature or for none of them")
        graph_values = {key: [feature.pop(key, None) for feature in features] for key in graph_keys}
        # Legacy caches may not contain compositional fields; use neutral IDs
        # so the adapter can fall back to the legacy relation embedding.
        for key in (
            "compositional_dependency_id",
            "compositional_direction_id",
            "compositional_src_pos_id",
            "compositional_dst_pos_id",
        ):
            graph_values[key] = [
                value if value is not None else [0] * len(graph_values["edge_src"][index] or [])
                for index, value in enumerate(graph_values[key])
            ]
        pairing_aspect_spans = [feature.pop("pairing_aspect_spans", []) for feature in features]
        pairing_opinion_spans = [feature.pop("pairing_opinion_spans", []) for feature in features]
        pairing_masks = [feature.pop("pairing_mask", []) for feature in features]
        sentiment_spans = [feature.pop("sentiment_contrastive_spans", []) for feature in features]
        sentiment_labels = [feature.pop("sentiment_contrastive_labels", []) for feature in features]
        sentiment_masks = [feature.pop("sentiment_contrastive_mask", []) for feature in features]
        sentiment_weights = [feature.pop("sentiment_contrastive_weights", []) for feature in features]
        batch = self.base_collator(features)
        max_pairs = max([len(mask) for mask in pairing_masks] + [0])
        if max_pairs == 0:
            batch["pairing_aspect_spans"] = torch.zeros((len(features), 0, 2), dtype=torch.long)
            batch["pairing_opinion_spans"] = torch.zeros((len(features), 0, 2), dtype=torch.long)
            batch["pairing_mask"] = torch.zeros((len(features), 0), dtype=torch.long)
        else:
            aspect_tensor = torch.zeros((len(features), max_pairs, 2), dtype=torch.long)
            opinion_tensor = torch.zeros((len(features), max_pairs, 2), dtype=torch.long)
            mask_tensor = torch.zeros((len(features), max_pairs), dtype=torch.long)
            for row_idx, (aspect_spans, opinion_spans, mask) in enumerate(
                zip(pairing_aspect_spans, pairing_opinion_spans, pairing_masks)
            ):
                for pair_idx, (aspect_span, opinion_span, active) in enumerate(zip(aspect_spans, opinion_spans, mask)):
                    if pair_idx >= max_pairs:
                        break
                    aspect_tensor[row_idx, pair_idx] = torch.tensor(aspect_span, dtype=torch.long)
                    opinion_tensor[row_idx, pair_idx] = torch.tensor(opinion_span, dtype=torch.long)
                    mask_tensor[row_idx, pair_idx] = int(active)
            batch["pairing_aspect_spans"] = aspect_tensor
            batch["pairing_opinion_spans"] = opinion_tensor
            batch["pairing_mask"] = mask_tensor

        max_sentiments = max([len(mask) for mask in sentiment_masks] + [0])
        sentiment_span_tensor = torch.zeros((len(features), max_sentiments, 2), dtype=torch.long)
        sentiment_label_tensor = torch.full((len(features), max_sentiments), -100, dtype=torch.long)
        sentiment_mask_tensor = torch.zeros((len(features), max_sentiments), dtype=torch.long)
        sentiment_weight_tensor = torch.zeros((len(features), max_sentiments), dtype=torch.float)
        for row_idx, (spans, labels, mask, weights) in enumerate(
            zip(sentiment_spans, sentiment_labels, sentiment_masks, sentiment_weights)
        ):
            for item_idx, (span, label, active, weight) in enumerate(zip(spans, labels, mask, weights)):
                sentiment_span_tensor[row_idx, item_idx] = torch.tensor(span, dtype=torch.long)
                sentiment_label_tensor[row_idx, item_idx] = int(label)
                sentiment_mask_tensor[row_idx, item_idx] = int(active)
                sentiment_weight_tensor[row_idx, item_idx] = float(weight)
        batch["sentiment_contrastive_spans"] = sentiment_span_tensor
        batch["sentiment_contrastive_labels"] = sentiment_label_tensor
        batch["sentiment_contrastive_mask"] = sentiment_mask_tensor
        batch["sentiment_contrastive_weights"] = sentiment_weight_tensor
        if any(graph_present):
            max_words = max(len(value) for value in graph_values["word_to_subword"])
            max_subwords = max(
                max((len(indices) for indices in value), default=0)
                for value in graph_values["word_to_subword"]
            )
            max_edges = max(len(value) for value in graph_values["edge_src"])
            word_to_subword = torch.full(
                (len(features), max_words, max(1, max_subwords)),
                -1,
                dtype=torch.long,
            )
            word_mask = torch.zeros((len(features), max_words), dtype=torch.bool)
            edge_tensors = {
                key: torch.zeros((len(features), max_edges), dtype=torch.long)
                for key in (
                    "edge_src",
                    "edge_dst",
                    "relation_id",
                    "dependency_relation_id",
                    "pos_pair_id",
                    "compositional_dependency_id",
                    "compositional_direction_id",
                    "compositional_src_pos_id",
                    "compositional_dst_pos_id",
                )
            }
            edge_mask = torch.zeros((len(features), max_edges), dtype=torch.bool)
            for row_index, values in enumerate(zip(*graph_values.values())):
                row_word_to_subword, row_word_mask, row_src, row_dst, row_relation, row_dependency, row_pos, row_comp_dep, row_comp_dir, row_comp_src, row_comp_dst, row_edge_mask = values
                word_mask[row_index, : len(row_word_mask)] = torch.tensor(row_word_mask, dtype=torch.bool)
                for word_index, indices in enumerate(row_word_to_subword):
                    word_to_subword[row_index, word_index, : len(indices)] = torch.tensor(indices, dtype=torch.long)
                edge_count = len(row_src)
                edge_mask[row_index, :edge_count] = torch.tensor(row_edge_mask, dtype=torch.bool)
                edge_tensors["edge_src"][row_index, :edge_count] = torch.tensor(row_src, dtype=torch.long)
                edge_tensors["edge_dst"][row_index, :edge_count] = torch.tensor(row_dst, dtype=torch.long)
                edge_tensors["relation_id"][row_index, :edge_count] = torch.tensor(row_relation, dtype=torch.long)
                edge_tensors["dependency_relation_id"][row_index, :edge_count] = torch.tensor(row_dependency, dtype=torch.long)
                edge_tensors["pos_pair_id"][row_index, :edge_count] = torch.tensor(row_pos, dtype=torch.long)
                edge_tensors["compositional_dependency_id"][row_index, :edge_count] = torch.tensor(row_comp_dep, dtype=torch.long)
                edge_tensors["compositional_direction_id"][row_index, :edge_count] = torch.tensor(row_comp_dir, dtype=torch.long)
                edge_tensors["compositional_src_pos_id"][row_index, :edge_count] = torch.tensor(row_comp_src, dtype=torch.long)
                edge_tensors["compositional_dst_pos_id"][row_index, :edge_count] = torch.tensor(row_comp_dst, dtype=torch.long)
            batch["graph_word_to_subword"] = word_to_subword
            batch["graph_word_mask"] = word_mask
            batch["graph_edge_src"] = edge_tensors["edge_src"]
            batch["graph_edge_dst"] = edge_tensors["edge_dst"]
            batch["graph_relation_id"] = edge_tensors["relation_id"]
            batch["graph_dependency_relation_id"] = edge_tensors["dependency_relation_id"]
            batch["graph_pos_pair_id"] = edge_tensors["pos_pair_id"]
            batch["graph_compositional_dependency_id"] = edge_tensors["compositional_dependency_id"]
            batch["graph_compositional_direction_id"] = edge_tensors["compositional_direction_id"]
            batch["graph_compositional_src_pos_id"] = edge_tensors["compositional_src_pos_id"]
            batch["graph_compositional_dst_pos_id"] = edge_tensors["compositional_dst_pos_id"]
            batch["graph_edge_mask"] = edge_mask
        return batch


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        ctx.grl_lambda = grl_lambda
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.grl_lambda * grad_output, None


def gradient_reverse(inputs: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(inputs, grl_lambda)


class DomainAdversarialHead(nn.Module):
    def __init__(self, hidden_size: int, classifier_hidden_size: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden_size, 2),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled_hidden)


class SentimentPrototypeHead(nn.Module):
    def __init__(self, hidden_size: int, num_sentiments: int = 3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_sentiments, hidden_size))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, p=2, dim=-1)


def build_sentiment_prototype_centroids(
    vectors: torch.Tensor,
    labels: torch.Tensor,
    num_sentiments: int = 3,
) -> tuple[torch.Tensor, list[int]]:
    if vectors.ndim != 2 or labels.ndim != 1 or vectors.size(0) != labels.size(0):
        raise ValueError("vectors and labels must have aligned [N, H] and [N] shapes")
    centroids = []
    counts = []
    for sentiment_id in range(num_sentiments):
        class_vectors = vectors[labels == sentiment_id]
        counts.append(int(class_vectors.size(0)))
        if class_vectors.size(0) == 0:
            raise ValueError(f"cannot initialize sentiment prototype {sentiment_id}: no examples")
        centroids.append(F.normalize(class_vectors.mean(dim=0), p=2, dim=0))
    return torch.stack(centroids), counts


def initialize_sentiment_prototypes_from_context(
    model,
    tokenizer,
    rows: list[dict],
    batch_size: int,
    max_source_length: int,
) -> dict:
    source_rows = [
        row for row in rows
        if row.get("augmentation") != "target_pseudo"
        and row.get("augmentation") not in CSA_AUGMENT_CHANNELS
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    was_training = model.training
    model.eval()
    collected_vectors = []
    collected_labels = []
    for start in tqdm(range(0, len(source_rows), batch_size), desc="init-sentiment-prototypes"):
        batch_rows = source_rows[start : start + batch_size]
        encoded = tokenizer(
            [row["input"] for row in batch_rows],
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            encoder_hidden = model.get_encoder()(**encoded, return_dict=True).last_hidden_state
        for row_idx, row in enumerate(batch_rows):
            row_input_ids = encoded["input_ids"][row_idx].tolist()
            for _aspect, opinion, sentiment in parse_triplet_text_list(row.get("target", "")):
                sentiment_id = SENTIMENT_LABEL_IDS.get(sentiment)
                span = find_opinion_span_in_input(
                    tokenizer,
                    row.get("input", ""),
                    row_input_ids,
                    opinion,
                )
                if sentiment_id is None or span is None:
                    continue
                collected_vectors.append(encoder_hidden[row_idx, span[0] : span[1]].mean(dim=0).float().cpu())
                collected_labels.append(sentiment_id)
    if was_training:
        model.train()
    if not collected_vectors:
        raise ValueError("no opinion context vectors were collected for sentiment prototype initialization")
    vectors = torch.stack(collected_vectors)
    labels = torch.tensor(collected_labels, dtype=torch.long)
    centroids, counts = build_sentiment_prototype_centroids(vectors, labels)
    with torch.no_grad():
        model.sentiment_prototype_head.prototypes.copy_(
            centroids.to(
                model.sentiment_prototype_head.prototypes.device,
                dtype=model.sentiment_prototype_head.prototypes.dtype,
            )
        )
    return {
        "source_rows": len(source_rows),
        "embedded_triplets": len(collected_vectors),
        "sentiment_counts": dict(zip(("pos", "neg", "neu"), counts)),
        "prototype_norms": [round(float(value), 6) for value in centroids.norm(dim=-1)],
        "device": str(device),
    }


def mean_pool_encoder_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def compute_domain_adversarial_loss(
    encoder_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    domain_labels: torch.Tensor,
    domain_adversarial_head: nn.Module,
    grl_lambda: float = 1.0,
) -> torch.Tensor | None:
    """Compute the existing DANN loss on post-graph encoder states."""
    pooled_hidden = mean_pool_encoder_hidden(encoder_hidden, attention_mask)
    reversed_hidden = gradient_reverse(pooled_hidden, grl_lambda)
    domain_logits = domain_adversarial_head(reversed_hidden)
    domain_targets = domain_labels.to(domain_logits.device, dtype=torch.long).view(-1)
    domain_valid_mask = domain_targets.ne(-100)
    if not domain_valid_mask.any():
        return None
    return F.cross_entropy(domain_logits[domain_valid_mask], domain_targets[domain_valid_mask])


class PairedDomainBatchSampler:
    """Deterministically emit one source and one target row per DANN batch.

    This sampler is opt-in for the approved Phase A entry.  Legacy training
    paths which do not request paired batches keep Trainer's normal sampler.
    The smaller domain is cycled so every logical batch is complete while the
    order remains identical for Control and Treatment with the same seed.
    """

    AUDIT_SCHEMA_VERSION = 3
    AUDIT_PROTOCOL = "physical_dataloader_traversal_v2"

    def __init__(
        self,
        source_count: int,
        target_count: int,
        *,
        source_batch_size: int,
        target_batch_size: int,
        seed: int,
        source_row_ids: list | None = None,
        target_row_ids: list | None = None,
        audit_path: str | Path | None = None,
    ):
        if source_count <= 0 or target_count <= 0:
            raise ValueError("DANN paired batches require both source and target rows")
        if source_batch_size <= 0 or target_batch_size <= 0:
            raise ValueError("DANN source and target batch sizes must be positive")
        self.source_count = int(source_count)
        self.target_count = int(target_count)
        self.source_batch_size = int(source_batch_size)
        self.target_batch_size = int(target_batch_size)
        self.seed = int(seed)
        self.source_row_ids = list(source_row_ids or range(self.source_count))
        self.target_row_ids = list(target_row_ids or range(self.target_count))
        if len(self.source_row_ids) != self.source_count or len(self.target_row_ids) != self.target_count:
            raise ValueError("DANN row-id lists must match their domain counts")
        self._epoch = 0
        self._explicit_epoch = False
        self._next_physical_traversal_index = 0
        self._next_sampling_epoch = 0
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.audit_journal_path = self.audit_path.with_suffix(".journal.jsonl") if self.audit_path is not None else None
        self.epoch_reports: list[dict] = []
        self._training_state_provider = None
        self._sampling_epoch_provider = None
        self._acknowledgement_required = False
        self._pending_acks: list[tuple[int, int, bool]] = []
        self._resume_replay_batch_ids: list[tuple[int, int]] = []
        self._resume_reissue_batch_ids: list[tuple[int, int]] = []
        self._resume_checkpoint_identity: dict | None = None
        self._journal_last_hash = ""
        self._journal_write_count = 0
        self._journal_bytes = 0
        self._snapshot_write_count = 0
        self._snapshot_bytes = 0
        self._initialize_audit_journal()

    def __len__(self) -> int:
        source_batches = math.ceil(self.source_count / self.source_batch_size)
        target_batches = math.ceil(self.target_count / self.target_batch_size)
        return max(source_batches, target_batches)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("DANN sampler epoch must be non-negative")
        self._epoch = epoch
        self._explicit_epoch = True

    def bind_training_state_provider(self, provider) -> None:
        """Bind Trainer state for metadata and post-training-step acknowledgement."""
        self._training_state_provider = provider

        self._acknowledgement_required = True

    def bind_sampling_epoch_provider(self, provider) -> None:
        """Preserve the historical ``int(Trainer.state.epoch)`` shuffle identity."""
        self._sampling_epoch_provider = provider

    def audit_io_stats(self) -> dict:
        return {
            "journal_write_count": self._journal_write_count,
            "journal_bytes": self._journal_bytes,
            "snapshot_write_count": self._snapshot_write_count,
            "snapshot_bytes": self._snapshot_bytes,
        }

    def _journal_identity(self) -> dict:
        return {
            "schema_version": self.AUDIT_SCHEMA_VERSION,
            "audit_protocol": self.AUDIT_PROTOCOL,
            "source_batch_size": self.source_batch_size,
            "target_batch_size": self.target_batch_size,
            "seed": self.seed,
            "source_count": self.source_count,
            "target_count": self.target_count,
            "source_row_ids": list(self.source_row_ids),
            "target_row_ids": list(self.target_row_ids),
        }

    def _initialize_audit_journal(self) -> None:
        if self.audit_journal_path is None:
            return
        self.audit_journal_path.parent.mkdir(parents=True, exist_ok=True)
        if self.audit_journal_path.is_file() and self.audit_journal_path.stat().st_size:
            records = _read_dann_audit_journal_records(self.audit_journal_path)
            if not records or records[0].get("event") != "identity" or records[0].get("payload") != self._journal_identity():
                raise ValueError("DANN audit journal identity mismatch")
            self._journal_last_hash = records[-1]["hash"]
            self._journal_write_count = len(records)
            self._journal_bytes = self.audit_journal_path.stat().st_size
        else:
            self._append_journal_event("identity", self._journal_identity())

    def _append_journal_event(self, event: str, payload: dict) -> None:
        if self.audit_journal_path is None:
            return
        body = {"event": event, "payload": payload, "previous_hash": self._journal_last_hash}
        digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        record = {**body, "hash": digest}
        line = _utf8_lf_bytes(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        with self.audit_journal_path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._journal_last_hash = digest
        self._journal_write_count += 1
        self._journal_bytes += len(line)

    def _training_state(self) -> dict:
        if self._training_state_provider is None:
            return {"global_step": None, "max_steps": None, "gradient_accumulation_steps": None}
        state = self._training_state_provider()
        if not isinstance(state, dict):
            return {"global_step": None, "max_steps": None, "gradient_accumulation_steps": None}
        return {
            "global_step": state.get("global_step"),
            "max_steps": state.get("max_steps"),
            "gradient_accumulation_steps": state.get("gradient_accumulation_steps"),
        }

    def _training_has_reached_max_steps(self) -> bool:
        state = self._training_state()
        global_step = state.get("global_step")
        max_steps = state.get("max_steps")
        if isinstance(global_step, int) and isinstance(max_steps, int) and max_steps > 0 and global_step >= max_steps:
            return True
        grad_accumulation = state.get("gradient_accumulation_steps")
        if not isinstance(max_steps, int) or max_steps <= 0 or not isinstance(grad_accumulation, int) or grad_accumulation <= 0:
            return False
        report = self.epoch_reports[-1] if self.epoch_reports else None
        if report is None or report.get("completion") == "complete":
            return False
        in_flight = sum(1 for _, _, is_replay in self._pending_acks if not is_replay)
        step_start = report.get("optimizer_global_step_start")
        processed = report.get("processed_batches")
        if not isinstance(step_start, int) or not isinstance(processed, int):
            return False
        completed_updates_in_traversal = (processed + in_flight) // grad_accumulation
        return step_start + completed_updates_in_traversal >= max_steps

    def flush_audit_snapshot(self) -> None:
        if self.audit_path is not None:
            _atomic_write_json(self.audit_path, self.audit_report())
            self._snapshot_write_count += 1
            self._snapshot_bytes += self.audit_path.stat().st_size

    def _persist_audit(self) -> None:
        self.flush_audit_snapshot()

    def state_dict(self) -> dict:
        return {
            "schema_version": 3,
            "seed": self.seed,
            "epoch": self._epoch,
            "next_physical_traversal_index": self._next_physical_traversal_index,
            "next_sampling_epoch": self._next_sampling_epoch,
            "source_count": self.source_count,
            "target_count": self.target_count,
            "source_batch_size": self.source_batch_size,
            "target_batch_size": self.target_batch_size,
            "source_row_ids": list(self.source_row_ids),
            "target_row_ids": list(self.target_row_ids),
            "resume_replay_batch_ids": [list(item) for item in self._resume_replay_batch_ids],
            "resume_reissue_batch_ids": [list(item) for item in self._resume_reissue_batch_ids],
        }

    def build_checkpoint_state(self, *, accumulation_remainder: int) -> dict:
        """Build future-resume state without changing the live sampler."""
        if not isinstance(accumulation_remainder, int) or accumulation_remainder < 0:
            raise ValueError("DANN checkpoint accumulation remainder must be a non-negative integer")
        state = self.state_dict()
        state["resume_replay_batch_ids"] = []
        state["resume_reissue_batch_ids"] = []
        if not self.epoch_reports:
            return state
        report = self.epoch_reports[-1]
        batches = report.get("batches", [])
        processed_batches = report.get("processed_batches", 0)
        issued_batches = report.get("issued_batches", 0)
        if not isinstance(processed_batches, int) or not isinstance(issued_batches, int) or not 0 <= processed_batches <= issued_batches:
            raise ValueError("DANN checkpoint cannot serialize invalid issued/processed accounting")
        if accumulation_remainder > processed_batches:
            raise ValueError("DANN checkpoint accumulation remainder exceeds processed batches")
        if accumulation_remainder:
            state["resume_replay_batch_ids"] = [
                [report["physical_traversal_index"], batch["logical_batch_id"]]
                for batch in batches[-accumulation_remainder:]
            ]
        state["resume_reissue_batch_ids"] = [
            [report["physical_traversal_index"], batch["logical_batch_id"]]
            for batch in batches[processed_batches:issued_batches]
        ]
        return state

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("DANN sampler state must be a mapping")
        expected = self.state_dict()
        for field in (
            "schema_version",
            "seed",
            "source_count",
            "target_count",
            "source_batch_size",
            "target_batch_size",
            "source_row_ids",
            "target_row_ids",
        ):
            if state.get(field) != expected[field]:
                raise ValueError(f"DANN sampler state mismatch: {field}")
        if state.get("schema_version") != 3:
            raise ValueError("DANN sampler state mismatch: schema_version")
        epoch = state.get("epoch")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("DANN sampler state has an invalid epoch")
        self._epoch = epoch
        next_physical = state.get("next_physical_traversal_index")
        next_sampling = state.get("next_sampling_epoch")
        if not isinstance(next_physical, int) or next_physical < 0:
            raise ValueError("DANN sampler state has an invalid physical traversal index")
        if not isinstance(next_sampling, int) or next_sampling < 0:
            raise ValueError("DANN sampler state has an invalid sampling epoch")
        self._next_physical_traversal_index = next_physical
        self._next_sampling_epoch = next_sampling
        replay_ids = state.get("resume_replay_batch_ids", [])
        if not isinstance(replay_ids, list) or any(not isinstance(item, list) or len(item) != 2 or not all(isinstance(value, int) and value >= 0 for value in item) for item in replay_ids):
            raise ValueError("DANN sampler state has invalid replay batch identities")
        self._resume_replay_batch_ids = [tuple(item) for item in replay_ids]
        reissue_ids = state.get("resume_reissue_batch_ids", [])
        if not isinstance(reissue_ids, list) or any(not isinstance(item, list) or len(item) != 2 or not all(isinstance(value, int) and value >= 0 for value in item) for item in reissue_ids):
            raise ValueError("DANN sampler state has invalid reissue batch identities")
        self._resume_reissue_batch_ids = [tuple(item) for item in reissue_ids]
        self._resume_checkpoint_identity = None
        # Replay/reissue is enabled only after the complete checkpoint loader
        # supplies the checkpoint path and hash.  A raw state-only restore is
        # intentionally not sufficient to consume recovery batches.
        self._explicit_epoch = True

    def set_resume_checkpoint_identity(self, checkpoint_dir: str | Path, checkpoint_sha256: str) -> None:
        checkpoint_path = Path(checkpoint_dir).resolve()
        if not checkpoint_path.is_dir() or not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
            raise ValueError("DANN resume checkpoint identity is invalid")
        self._resume_checkpoint_identity = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
        }

    def load_audit_report(self, report: dict) -> None:
        if not isinstance(report, dict) or report.get("schema_version") != self.AUDIT_SCHEMA_VERSION or report.get("audit_protocol") != self.AUDIT_PROTOCOL or not isinstance(report.get("epochs"), list):
            raise ValueError("DANN sampler audit report is invalid")
        if report.get("seed") != self.seed:
            raise ValueError("DANN sampler audit report mismatch: seed")
        if report.get("source_batch_size") != self.source_batch_size:
            raise ValueError("DANN sampler audit report mismatch: source_batch_size")
        if report.get("target_batch_size") != self.target_batch_size:
            raise ValueError("DANN sampler audit report mismatch: target_batch_size")
        if report.get("source_count") != self.source_count:
            raise ValueError("DANN sampler audit report mismatch: source_count")
        if report.get("target_count") != self.target_count:
            raise ValueError("DANN sampler audit report mismatch: target_count")
        if report.get("source_row_ids") != self.source_row_ids:
            raise ValueError("DANN sampler audit report mismatch: source_row_ids")
        if report.get("target_row_ids") != self.target_row_ids:
            raise ValueError("DANN sampler audit report mismatch: target_row_ids")
        epochs = report["epochs"]
        seen = set()
        for expected_physical_index, epoch_report in enumerate(epochs):
            physical_index = epoch_report.get("physical_traversal_index") if isinstance(epoch_report, dict) else None
            sampling_epoch = epoch_report.get("sampling_epoch") if isinstance(epoch_report, dict) else None
            if not isinstance(physical_index, int) or physical_index != expected_physical_index or physical_index in seen:
                raise ValueError("DANN sampler audit report has duplicate or invalid physical traversals")
            if not isinstance(sampling_epoch, int) or sampling_epoch < 0:
                raise ValueError("DANN sampler audit report has an invalid sampling epoch")
            if epoch_report.get("epoch") != sampling_epoch:
                raise ValueError("DANN sampler audit report epoch alias mismatch")
            seen.add(physical_index)
            batches = epoch_report.get("batches")
            if (
                epoch_report.get("source_batch_size") != self.source_batch_size
                or epoch_report.get("target_batch_size") != self.target_batch_size
                or epoch_report.get("incomplete_batches") != 0
                or not isinstance(batches, list)
                or epoch_report.get("logical_batches") != len(batches)
                or epoch_report.get("planned_batches") != len(self)
                or epoch_report.get("issued_batches") != len(batches)
                or not isinstance(epoch_report.get("processed_batches"), int)
                or epoch_report.get("processed_batches") < 0
                or epoch_report.get("processed_batches") > epoch_report.get("issued_batches")
                or epoch_report.get("source_rows") != sum(batch.get("source_count", 0) for batch in batches if isinstance(batch, dict))
                or epoch_report.get("target_rows") != sum(batch.get("target_count", 0) for batch in batches if isinstance(batch, dict))
                or epoch_report.get("completion") not in {"complete", "partial"}
            ):
                raise ValueError("DANN sampler audit report has incomplete epoch accounting")
            if epoch_report["completion"] == "complete" and (
                epoch_report["issued_batches"] != epoch_report["processed_batches"]
                or epoch_report["issued_batches"] != epoch_report["planned_batches"]
            ):
                raise ValueError("DANN sampler audit report marks a non-terminal traversal complete")
            if epoch_report["completion"] == "partial" and len(batches) > len(self):
                raise ValueError("DANN sampler audit report marks a full traversal partial")
            if epoch_report["source_unique_rows"] > self.source_count or epoch_report["target_unique_rows"] > self.target_count:
                raise ValueError("DANN sampler audit report has invalid partial coverage")
            seen_source = set()
            seen_target = set()
            for batch_id, batch in enumerate(batches):
                source_indices = batch.get("source_indices") if isinstance(batch, dict) else None
                target_indices = batch.get("target_indices") if isinstance(batch, dict) else None
                if (
                    not isinstance(batch, dict)
                    or batch.get("logical_batch_id") != batch_id
                    or batch.get("source_count") != self.source_batch_size
                    or batch.get("target_count") != self.target_batch_size
                    or not isinstance(source_indices, list)
                    or len(source_indices) != self.source_batch_size
                    or not isinstance(target_indices, list)
                    or len(target_indices) != self.target_batch_size
                ):
                    raise ValueError("DANN sampler audit report has an invalid 1/1 batch")
                if any(not isinstance(index, int) or not 0 <= index < self.source_count for index in source_indices) or any(not isinstance(index, int) or not self.source_count <= index < self.source_count + self.target_count for index in target_indices):
                    raise ValueError("DANN sampler audit report index/row identity mismatch")
                if batch.get("source_row_ids") != [self.source_row_ids[index] for index in source_indices] or batch.get("target_row_ids") != [self.target_row_ids[index - self.source_count] for index in target_indices]:
                    raise ValueError("DANN sampler audit report index/row identity mismatch")
                seen_source.update(source_indices)
                seen_target.update(index - self.source_count for index in target_indices)
            if seen_source != set(range(self.source_count)) or seen_target != set(range(self.target_count)):
                if epoch_report["completion"] == "complete":
                    raise ValueError("DANN sampler audit report does not cover both domains")
            if epoch_report["source_unique_rows"] != len(seen_source) or epoch_report["target_unique_rows"] != len(seen_target):
                raise ValueError("DANN sampler audit report partial coverage count mismatch")
        if any(item.get("completion") == "partial" for item in epochs[:-1]):
            raise ValueError("DANN sampler audit report has a partial traversal before the end")
        for physical_index, batch_id in self._resume_replay_batch_ids:
            if physical_index >= len(epochs) or batch_id >= epochs[physical_index].get("issued_batches", 0):
                raise ValueError("DANN sampler audit report has an invalid replay identity")
        for physical_index, batch_id in self._resume_reissue_batch_ids:
            if physical_index >= len(epochs) or batch_id >= epochs[physical_index].get("issued_batches", 0):
                raise ValueError("DANN sampler audit report has an invalid reissue identity")
        self.epoch_reports = list(epochs)
        self._next_physical_traversal_index = len(epochs)
        self._next_sampling_epoch = max((item.get("sampling_epoch", -1) for item in epochs), default=-1) + 1
        self._explicit_epoch = False
        if epochs:
            self._epoch = epochs[-1].get("sampling_epoch", self._epoch)

    def __iter__(self):
        last = self.epoch_reports[-1] if self.epoch_reports else None
        terminal_lookahead_allowed = bool(
            self._training_has_reached_max_steps()
            and last
            and last.get("completion") == "partial"
            and last.get("issued_batches") == last.get("processed_batches")
            and last.get("issued_batches", 0) < last.get("planned_batches", 0)
        )
        if self._training_has_reached_max_steps() and not terminal_lookahead_allowed:
            return
        if self._resume_reissue_batch_ids and self._resume_checkpoint_identity is None:
            raise RuntimeError("DANN reissue requires an explicit checkpoint load")
        if self._resume_replay_batch_ids:
            if self._resume_checkpoint_identity is None:
                raise RuntimeError("DANN replay requires an explicit checkpoint load")
            replay_ids = list(self._resume_replay_batch_ids)
            self._resume_replay_batch_ids.clear()
            for physical_index, batch_id in replay_ids:
                if self._training_has_reached_max_steps():
                    break
                report = next((item for item in self.epoch_reports if item.get("physical_traversal_index") == physical_index), None)
                if report is None or batch_id >= report.get("issued_batches", 0):
                    raise RuntimeError("DANN checkpoint replay identity mismatch")
                batch = report["batches"][batch_id]
                self._pending_acks.append((physical_index, batch_id, True))
                self._append_journal_event(
                    "batch_replayed",
                    {
                        "checkpoint_path": self._resume_checkpoint_identity["checkpoint_path"],
                        "checkpoint_sha256": self._resume_checkpoint_identity["checkpoint_sha256"],
                        "recovery_batch_identity": {
                            "physical_traversal_index": physical_index,
                            "logical_batch_id": batch_id,
                        },
                        "physical_traversal_index": physical_index,
                        "batch": batch,
                    },
                )
                yield batch["source_indices"] + batch["target_indices"]
                if not self._acknowledgement_required:
                    self.acknowledge_next_batch()
            yield from self.__iter__()
            return
        can_resume_traversal = bool(
            last
            and last.get("completion") == "partial"
            and last.get("processed_batches", 0) < last.get("planned_batches", 0)
            and (
                self._training_state_provider is None
                or self._training_state().get("global_step") != self._training_state().get("max_steps")
            )
        )
        if can_resume_traversal:
            report = last
            physical_traversal_index = report["physical_traversal_index"]
            sampling_epoch = report["sampling_epoch"]
            # Issued is durable intent, not proof that Trainer processed the
            # batch.  Reissue the unprocessed suffix through training_step.
            start_batch = report["processed_batches"]
        else:
            physical_traversal_index = self._next_physical_traversal_index
            if self._sampling_epoch_provider is not None:
                sampling_epoch = int(self._sampling_epoch_provider())
            elif self._explicit_epoch:
                sampling_epoch = self._epoch
                self._explicit_epoch = False
            else:
                sampling_epoch = self._next_sampling_epoch
            self._epoch = sampling_epoch
            self._next_physical_traversal_index += 1
            self._next_sampling_epoch = max(self._next_sampling_epoch, sampling_epoch + 1)
            training_state = self._training_state()
            report = {
                "epoch": sampling_epoch,
                "physical_traversal_index": physical_traversal_index,
                "sampling_epoch": sampling_epoch,
                "source_batch_size": self.source_batch_size,
                "target_batch_size": self.target_batch_size,
                "source_rows": 0,
                "target_rows": 0,
                "source_unique_rows": 0,
                "target_unique_rows": 0,
                "logical_batches": 0,
                "planned_batches": len(self),
                "issued_batches": 0,
                "processed_batches": 0,
                "incomplete_batches": 0,
                "completion": "partial",
                "optimizer_global_step_start": training_state["global_step"],
                "optimizer_global_step_end": training_state["global_step"],
                "batches": [],
            }
            self.epoch_reports.append(report)
            self._append_journal_event("traversal_started", {"report": report})
            start_batch = report["processed_batches"]
        self._epoch = sampling_epoch
        source_order = list(range(self.source_count))
        target_order = list(range(self.target_count))
        random.Random(self.seed + sampling_epoch).shuffle(source_order)
        random.Random(self.seed + sampling_epoch).shuffle(target_order)
        seen_source = set()
        seen_target = set()
        for existing_batch in report.get("batches", []):
            seen_source.update(existing_batch.get("source_indices", []))
            seen_target.update(index - self.source_count for index in existing_batch.get("target_indices", []))
        for batch_id in range(start_batch, len(self)):
            if self._training_has_reached_max_steps() and not terminal_lookahead_allowed:
                break
            if terminal_lookahead_allowed and batch_id > start_batch:
                break
            source_positions = [
                source_order[(batch_id * self.source_batch_size + offset) % self.source_count]
                for offset in range(self.source_batch_size)
            ]
            target_positions = [
                target_order[(batch_id * self.target_batch_size + offset) % self.target_count]
                for offset in range(self.target_batch_size)
            ]
            if len(source_positions) != self.source_batch_size or len(target_positions) != self.target_batch_size:
                report["incomplete_batches"] += 1
                raise RuntimeError("DANN paired sampler produced an incomplete domain batch")
            seen_source.update(source_positions)
            seen_target.update(target_positions)
            is_reissue = batch_id < report["issued_batches"]
            if not is_reissue:
                report["source_rows"] += len(source_positions)
                report["target_rows"] += len(target_positions)
                report["logical_batches"] += 1
            batch_record = {
                    "logical_batch_id": batch_id,
                    "source_indices": list(source_positions),
                    "target_indices": [self.source_count + index for index in target_positions],
                    "source_row_ids": [self.source_row_ids[index] for index in source_positions],
                    "target_row_ids": [self.target_row_ids[index] for index in target_positions],
                    "source_count": len(source_positions),
                    "target_count": len(target_positions),
                }
            if is_reissue:
                batch_record = report["batches"][batch_id]
            else:
                report["issued_batches"] += 1
                report["batches"].append(batch_record)
            report["source_unique_rows"] = len(seen_source)
            report["target_unique_rows"] = len(seen_target)
            report["optimizer_global_step_end"] = self._training_state()["global_step"]
            self._pending_acks.append((physical_traversal_index, batch_id, False))
            self._append_journal_event(
                "batch_reissued" if is_reissue else "batch_issued",
                {"physical_traversal_index": physical_traversal_index, "batch": batch_record},
            )
            yield source_positions + [self.source_count + index for index in target_positions]
            if not self._acknowledgement_required:
                self.acknowledge_next_batch()
            terminal_lookahead_allowed = False
        report["source_unique_rows"] = len(seen_source)
        report["target_unique_rows"] = len(seen_target)
        report["optimizer_global_step_end"] = self._training_state()["global_step"]
        self._finalize_report_if_complete(report)
        self.flush_audit_snapshot()

    def _finalize_report_if_complete(self, report: dict) -> None:
        if (
            report["issued_batches"] == report["processed_batches"]
            and report["issued_batches"] == report["planned_batches"]
        ):
            if report.get("completion") != "complete":
                report["completion"] = "complete"
                self._append_journal_event(
                    "traversal_completed",
                    {
                        "physical_traversal_index": report["physical_traversal_index"],
                        "optimizer_global_step_end": report.get("optimizer_global_step_end"),
                    },
                )

    def acknowledge_next_batch(self) -> None:
        if not self._pending_acks:
            raise RuntimeError("DANN batch acknowledge has no issued batch")
        physical_index, batch_id, is_replay = self._pending_acks.pop(0)
        report = next((item for item in reversed(self.epoch_reports) if item.get("physical_traversal_index") == physical_index), None)
        if report is None or batch_id >= report.get("issued_batches", 0):
            raise RuntimeError("DANN batch acknowledge identity mismatch")
        if is_replay:
            return
        report["processed_batches"] += 1
        report["optimizer_global_step_end"] = self._training_state()["global_step"]
        self._append_journal_event(
            "batch_processed",
            {"physical_traversal_index": physical_index, "logical_batch_id": batch_id},
        )
        self._finalize_report_if_complete(report)

    def prepare_resume_replay(
        self,
        gradient_accumulation_steps: int,
        *,
        accumulation_remainder: int | None = None,
    ) -> dict:
        """Return checkpoint-only replay state; never mutate the live sampler."""
        if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps <= 0:
            raise ValueError("DANN gradient accumulation steps must be positive")
        if accumulation_remainder is None:
            raise ValueError("DANN checkpoint replay requires the actual accumulation remainder")
        return self.build_checkpoint_state(accumulation_remainder=accumulation_remainder)

    def audit_report(self) -> dict:
        training_state = self._training_state()
        if self.epoch_reports and training_state["global_step"] is not None:
            self.epoch_reports[-1]["optimizer_global_step_end"] = training_state["global_step"]
        report = {
            "schema_version": self.AUDIT_SCHEMA_VERSION,
            "audit_protocol": self.AUDIT_PROTOCOL,
            "source_batch_size": self.source_batch_size,
            "target_batch_size": self.target_batch_size,
            "seed": self.seed,
            "current_epoch": self._epoch,
            "next_physical_traversal_index": self._next_physical_traversal_index,
            "next_sampling_epoch": self._next_sampling_epoch,
            "trainer_global_step": training_state["global_step"],
            "trainer_max_steps": training_state["max_steps"],
            "gradient_accumulation_steps": training_state["gradient_accumulation_steps"],
            "source_count": self.source_count,
            "target_count": self.target_count,
            "source_row_ids": list(self.source_row_ids),
            "target_row_ids": list(self.target_row_ids),
            "epochs": list(self.epoch_reports),
        }
        if training_state["global_step"] is not None and training_state["max_steps"] is not None:
            report["terminal_lookahead"] = classify_terminal_lookahead(
                report,
                gradient_accumulation_steps=int(training_state.get("gradient_accumulation_steps") or 1),
            )
        return report


def _checkpoint_step(path: Path) -> int | None:
    try:
        return int(Path(path).name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None


def _read_complete_dann_checkpoint(checkpoint_dir: Path, sampler: PairedDomainBatchSampler) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "dann_checkpoint_state.json"
    sampler_path = checkpoint_dir / "dann_batch_sampler_state.json"
    audit_path = checkpoint_dir / "dann_batch_audit.json"
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    gradient_state_path = checkpoint_dir / "dann_gradient_state.pt"
    model_files = [checkpoint_dir / "pytorch_model.bin", checkpoint_dir / "model.safetensors"]
    required = (manifest_path, sampler_path, audit_path, trainer_state_path, gradient_state_path)
    missing = [str(path) for path in required if not path.is_file()]
    if not any(path.is_file() for path in model_files):
        missing.append(f"{checkpoint_dir}/pytorch_model.bin|model.safetensors")
    if missing:
        raise RuntimeError(f"incomplete DANN checkpoint {checkpoint_dir}: missing {missing}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sampler_state = json.loads(sampler_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        gradient_state = torch.load(gradient_state_path, map_location="cpu")
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid DANN checkpoint JSON in {checkpoint_dir}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise RuntimeError(f"DANN checkpoint is not marked complete: {checkpoint_dir}")
    if manifest.get("schema_version") != 3 or manifest.get("audit_protocol") != PairedDomainBatchSampler.AUDIT_PROTOCOL:
        raise RuntimeError(f"unsupported DANN checkpoint schema: {checkpoint_dir}")
    if manifest.get("sampler_state_sha256") != _json_sha256(sampler_path):
        raise RuntimeError(f"DANN checkpoint sampler state hash mismatch: {checkpoint_dir}")
    if manifest.get("audit_sha256") != _json_sha256(audit_path):
        raise RuntimeError(f"DANN checkpoint audit hash mismatch: {checkpoint_dir}")
    if manifest.get("trainer_state_sha256") != _json_sha256(trainer_state_path):
        raise RuntimeError(f"DANN checkpoint trainer state hash mismatch: {checkpoint_dir}")
    if manifest.get("gradient_state_artifact") != gradient_state_path.name or manifest.get("gradient_state_sha256") != _json_sha256(gradient_state_path):
        raise RuntimeError(f"DANN checkpoint gradient state hash mismatch: {checkpoint_dir}")
    if not isinstance(gradient_state, dict) or gradient_state.get("schema_version") != 1 or not isinstance(gradient_state.get("gradients"), dict) or not isinstance(gradient_state.get("accumulation_remainder"), int) or gradient_state.get("accumulation_remainder") < 0:
        raise RuntimeError(f"DANN checkpoint gradient state is invalid: {checkpoint_dir}")
    model_path = next(path for path in model_files if path.is_file())
    if manifest.get("model_artifact") != model_path.name or manifest.get("model_artifact_sha256") != _json_sha256(model_path):
        raise RuntimeError(f"DANN checkpoint model hash mismatch: {checkpoint_dir}")
    sampler.load_state_dict(sampler_state)
    sampler.load_audit_report(audit)
    epochs = audit.get("epochs") if isinstance(audit, dict) else None
    expected_epochs = [item.get("epoch") for item in epochs] if isinstance(epochs, list) else []
    complete_epochs = [item.get("epoch") for item in epochs if item.get("completion") == "complete"]
    complete_physical = [item.get("physical_traversal_index") for item in epochs if item.get("completion") == "complete"]
    terminal_partial = bool(epochs and epochs[-1].get("completion") == "partial")
    trainer_global_step = trainer_state.get("global_step") if isinstance(trainer_state, dict) else None
    trainer_max_steps = audit.get("trainer_max_steps")
    terminal_decision = classify_terminal_lookahead(
        audit,
        gradient_accumulation_steps=int(audit.get("gradient_accumulation_steps") or 1),
    )
    terminal_consumed = terminal_partial and isinstance(trainer_global_step, int) and isinstance(trainer_max_steps, int) and trainer_global_step == trainer_max_steps and audit.get("trainer_global_step") == trainer_global_step and epochs[-1].get("processed_batches") == epochs[-1].get("issued_batches")
    terminal_is_valid = terminal_consumed or (terminal_partial and terminal_decision.get("safe") and terminal_decision.get("lookahead_not_consumed"))
    resumable_partial = terminal_partial and isinstance(trainer_global_step, int) and isinstance(trainer_max_steps, int) and trainer_global_step < trainer_max_steps
    terminal_processing_gap = isinstance(trainer_global_step, int) and isinstance(trainer_max_steps, int) and trainer_global_step == trainer_max_steps and any(item.get("processed_batches") != item.get("issued_batches") for item in epochs) and not terminal_decision.get("safe")
    if not epochs or terminal_processing_gap or (terminal_partial and not terminal_is_valid and not resumable_partial) or (not terminal_partial and any(item.get("completion") != "complete" for item in epochs)):
        raise RuntimeError(f"DANN checkpoint audit is not a valid resumable terminal state: {checkpoint_dir}")
    if manifest.get("resume_complete") is not True:
        raise RuntimeError(f"DANN checkpoint is not resume-complete: {checkpoint_dir}")
    if manifest.get("training_terminal_partial") is not terminal_is_valid:
        raise RuntimeError(f"DANN checkpoint terminal-partial identity mismatch: {checkpoint_dir}")
    if manifest.get("completed_epochs") != complete_epochs:
        raise RuntimeError(f"DANN checkpoint completed epoch identity mismatch: {checkpoint_dir}")
    if manifest.get("completed_epoch_count") != len(complete_epochs):
        raise RuntimeError(f"DANN checkpoint completed epoch count mismatch: {checkpoint_dir}")
    if manifest.get("completed_physical_traversals") != complete_physical:
        raise RuntimeError(f"DANN checkpoint physical traversal identity mismatch: {checkpoint_dir}")
    if manifest.get("audit_traversal_count") != len(epochs):
        raise RuntimeError(f"DANN checkpoint audit traversal count mismatch: {checkpoint_dir}")
    if manifest.get("trainer_global_step") != audit.get("trainer_global_step") or manifest.get("trainer_max_steps") != audit.get("trainer_max_steps"):
        raise RuntimeError(f"DANN checkpoint Trainer step identity mismatch: {checkpoint_dir}")
    if manifest.get("seed") != sampler.seed:
        raise RuntimeError(f"DANN checkpoint seed mismatch: {checkpoint_dir}")
    for field, expected in (
        ("source_count", sampler.source_count),
        ("target_count", sampler.target_count),
        ("source_batch_size", sampler.source_batch_size),
        ("target_batch_size", sampler.target_batch_size),
        ("source_row_ids", sampler.source_row_ids),
        ("target_row_ids", sampler.target_row_ids),
    ):
        if manifest.get(field) != expected:
            raise RuntimeError(f"DANN checkpoint {field} mismatch: {checkpoint_dir}")
    if not isinstance(trainer_state, dict) or "global_step" not in trainer_state:
        raise RuntimeError(f"DANN checkpoint trainer state is incomplete: {checkpoint_dir}")
    return {
        "checkpoint_dir": checkpoint_dir,
        "manifest": manifest,
        "sampler_state": sampler_state,
        "audit": audit,
        "trainer_state": trainer_state,
        "gradient_state": gradient_state,
    }


def find_latest_complete_dann_checkpoint(
    output_dir: str | Path,
    sampler: PairedDomainBatchSampler,
) -> Path:
    candidates = []
    for path in Path(output_dir).glob("checkpoint-*"):
        if path.is_dir() and _checkpoint_step(path) is not None:
            candidates.append(path)
    candidates.sort(key=lambda path: _checkpoint_step(path) or -1, reverse=True)
    errors = []
    for candidate in candidates:
        try:
            _read_complete_dann_checkpoint(candidate, sampler)
            return candidate
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate.name}: {exc}")
    detail = "; ".join(errors) if errors else "no checkpoint candidates"
    raise RuntimeError(f"no complete identity-valid paired DANN checkpoint found: {detail}")


class _DANNGradientRestoreCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        gradient_state = getattr(self.trainer, "_pending_dann_gradient_state", None)
        if gradient_state is None:
            return control
        gradients = gradient_state.get("gradients", {})
        parameters = dict(self.trainer.model.named_parameters())
        if set(gradients) - set(parameters):
            raise RuntimeError("DANN checkpoint gradient identity mismatch")
        for name, parameter in parameters.items():
            gradient = gradients.get(name)
            parameter.grad = None if gradient is None else gradient.to(device=parameter.device, dtype=parameter.dtype).clone()
        self.trainer._dann_resume_gradient_pending = bool(gradients)
        self.trainer._dann_resume_gradient_accumulation_steps = int(args.gradient_accumulation_steps)
        self.trainer._dann_microbatches_since_optimizer_step = int(gradient_state.get("accumulation_remainder", 0))
        self.trainer.accelerator.step = int(gradient_state.get("accumulation_remainder", 0))
        self.trainer._pending_dann_gradient_state = None
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if getattr(self.trainer, "_dann_resume_gradient_pending", False):
            self.trainer.args.gradient_accumulation_steps = self.trainer._dann_resume_gradient_accumulation_steps
            self.trainer._dann_resume_gradient_pending = False
        return control


class _DANNAccumulationCounterCallback(TrainerCallback):
    """Reset the independent microbatch counter after a real optimizer step."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        restore_steps = getattr(self.trainer, "_dann_restore_gradient_accumulation_steps", None)
        if restore_steps is not None:
            args.gradient_accumulation_steps = int(restore_steps)
            self.trainer._dann_restore_gradient_accumulation_steps = None
        self.trainer._dann_microbatches_since_optimizer_step = 0
        return control


class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    _generation_only_input_keys = {
        "sample_weight",
        "domain_weight",
        "domain_label",
        "structure_weight",
        "consistency_group",
        "pairing_aspect_spans",
        "pairing_opinion_spans",
        "pairing_mask",
        "sentiment_contrastive_spans",
        "sentiment_contrastive_labels",
        "sentiment_contrastive_mask",
        "sentiment_contrastive_weights",
    }
    _graph_input_keys = {
        "graph_word_to_subword",
        "graph_word_mask",
        "graph_edge_src",
        "graph_edge_dst",
        "graph_relation_id",
        "graph_dependency_relation_id",
        "graph_pos_pair_id",
        "graph_edge_mask",
    }

    def __init__(
        self,
        *args,
        lambda_structure_loss: float = 0.0,
        lambda_consistency_loss: float = 0.0,
        lambda_pairing_loss: float = 0.0,
        pairing_temperature: float = 0.1,
        lambda_domain_adv: float = 0.0,
        domain_adv_grl_lambda: float = 1.0,
        lambda_sentiment_contrastive: float = 0.0,
        element_aware_attention: bool = False,
        element_focus_weight: float = 0.0,
        element_coverage_weight: float = 0.0,
        multi_element_coverage_loss_enabled: bool = False,
        sentiment_contrastive_temperature: float = 0.1,
        sentiment_contrastive_class_weights: list[float] | None = None,
        dann_batch_sampler: PairedDomainBatchSampler | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_structure_loss = lambda_structure_loss
        self.lambda_consistency_loss = lambda_consistency_loss
        self.lambda_pairing_loss = lambda_pairing_loss
        self.pairing_temperature = pairing_temperature
        self.lambda_domain_adv = lambda_domain_adv
        self.domain_adv_grl_lambda = domain_adv_grl_lambda
        self.lambda_sentiment_contrastive = lambda_sentiment_contrastive
        self.sentiment_contrastive_temperature = sentiment_contrastive_temperature
        self.sentiment_contrastive_class_weights = sentiment_contrastive_class_weights
        self.element_aware_attention = bool(element_aware_attention)
        self.element_focus_weight = float(element_focus_weight)
        self.element_coverage_weight = float(element_coverage_weight)
        self.multi_element_coverage_loss_enabled = bool(multi_element_coverage_loss_enabled)
        self.dann_batch_sampler = dann_batch_sampler
        self._dann_microbatches_since_optimizer_step = 0
        if self.dann_batch_sampler is not None:
            # Paired checkpoints are written only at epoch boundaries.  A
            # checkpoint restored at such a boundary must not trigger
            # Transformers' data-skip replay traversal, which is a physical
            # DataLoader pass but not a training epoch and would pollute audit
            # identity.
            self.args.ignore_data_skip = True
            self.dann_batch_sampler.bind_training_state_provider(
                lambda: {
                    "global_step": int(self.state.global_step),
                    "max_steps": int(self.state.max_steps),
                    "gradient_accumulation_steps": int(self.args.gradient_accumulation_steps),
                }
            )
            self.dann_batch_sampler.bind_sampling_epoch_provider(
                lambda: int(self.state.epoch or 0)
            )
            self.add_callback(_DANNAccumulationCounterCallback(self))
        self._component_sums: dict[str, float] = {}
        self._component_counts: dict[str, int] = {}
        self._component_reductions: dict[str, str] = {}

    def get_train_dataloader(self):
        if self.dann_batch_sampler is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader = DataLoader(
            train_dataset,
            batch_sampler=self.dann_batch_sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
        prepared = self.accelerator.prepare(dataloader)
        if hasattr(prepared, "batch_sampler") and hasattr(prepared.batch_sampler, "batch_sampler"):
            # Accelerate wraps a custom batch sampler in BatchSamplerShard and
            # only forwards set_epoch through its ``sampler`` attribute.
            prepared.batch_sampler.sampler = self.dann_batch_sampler
        return prepared

    def get_dann_batch_audit(self) -> dict | None:
        return self.dann_batch_sampler.audit_report() if self.dann_batch_sampler is not None else None

    def training_step(self, model, inputs, *args, **kwargs):
        force_tail_update = False
        if self.dann_batch_sampler is not None:
            accumulation_steps = int(self.args.gradient_accumulation_steps)
            is_dataloader_tail = bool(
                getattr(getattr(self.accelerator, "gradient_state", None), "end_of_dataloader", False)
            )
            if (
                is_dataloader_tail
                and (self._dann_microbatches_since_optimizer_step + 1) % accumulation_steps != 0
            ):
                # The Accelerate dataloader state is the authoritative signal
                # for a non-divisible epoch tail.  Set the sync flag before
                # backward and make the outer Trainer commit this real tail
                # update; do not infer it from global_step or a physical-pass
                # counter after the fact.
                force_tail_update = True
                self._dann_restore_gradient_accumulation_steps = accumulation_steps
                self.accelerator.gradient_state._set_sync_gradients(True)
                self.args.gradient_accumulation_steps = 1
        result = super().training_step(model, inputs, *args, **kwargs)
        if self.dann_batch_sampler is not None:
            self._dann_microbatches_since_optimizer_step += 1
            accumulation_steps = int(self.args.gradient_accumulation_steps)
            if (
                not force_tail_update
                and self.accelerator.sync_gradients
                and self._dann_microbatches_since_optimizer_step % accumulation_steps != 0
            ):
                # Accelerate marks the final physical batch of a dataloader,
                # including a non-divisible tail, as synchronized.  Tell the
                # outer Trainer loop to commit that real tail update so an
                # epoch checkpoint has no in-flight accumulation remainder.
                self._dann_restore_gradient_accumulation_steps = accumulation_steps
                self.args.gradient_accumulation_steps = 1
        if getattr(self, "_dann_resume_gradient_pending", False):
            # Trainer's local batch counter restarts at zero on resume, while
            # the restored gradient already represents the prior partial
            # accumulation.  Force this first batch to close that accumulation
            # after computing its loss with the original accumulation factor.
            self.accelerator.gradient_state._set_sync_gradients(True)
            self.args.gradient_accumulation_steps = 1
        if self.dann_batch_sampler is not None:
            self.dann_batch_sampler.acknowledge_next_batch()
        return result

    def checkpoint_accumulation_remainder(self) -> int:
        """Return the independently tracked in-flight microbatch count."""
        if self.dann_batch_sampler is None:
            return 0
        remainder = getattr(self, "_dann_microbatches_since_optimizer_step", 0)
        if not isinstance(remainder, int) or remainder < 0:
            raise RuntimeError("DANN accumulation counter is invalid")
        return remainder

    def load_dann_batch_sampler_state(self, checkpoint_dir: str | Path) -> None:
        if self.dann_batch_sampler is None:
            return
        checkpoint = _read_complete_dann_checkpoint(Path(checkpoint_dir), self.dann_batch_sampler)
        checkpoint_state_path = Path(checkpoint["checkpoint_dir"]) / "dann_checkpoint_state.json"
        self.dann_batch_sampler.set_resume_checkpoint_identity(
            checkpoint["checkpoint_dir"],
            _json_sha256(checkpoint_state_path),
        )
        self._pending_dann_gradient_state = checkpoint["gradient_state"]
        self.add_callback(_DANNGradientRestoreCallback(self))

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        if self.dann_batch_sampler is None or not self.args.should_save:
            return
        checkpoint_dir = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        accumulation_remainder = self.checkpoint_accumulation_remainder()
        save_strategy = getattr(self.args.save_strategy, "value", self.args.save_strategy)
        if save_strategy == "epoch" and accumulation_remainder != 0:
            raise RuntimeError("epoch checkpoint must have zero DANN accumulation remainder")
        gradient_state = {
            "schema_version": 1,
            "accumulation_remainder": accumulation_remainder,
            "gradients": {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            },
        }
        gradient_state_path = checkpoint_dir / "dann_gradient_state.pt"
        _atomic_torch_save(gradient_state_path, gradient_state)
        state_path = checkpoint_dir / "dann_batch_sampler_state.json"
        audit_path = checkpoint_dir / "dann_batch_audit.json"
        sampler_state = self.dann_batch_sampler.build_checkpoint_state(
            accumulation_remainder=accumulation_remainder,
        )
        audit = self.dann_batch_sampler.audit_report()
        self.dann_batch_sampler.flush_audit_snapshot()
        audit = self.dann_batch_sampler.audit_report()
        epochs = audit["epochs"]
        terminal_partial = bool(epochs and epochs[-1].get("completion") == "partial")
        journal_audit = None
        journal_path = self.dann_batch_sampler.audit_journal_path
        if journal_path is not None and Path(journal_path).is_file():
            journal_audit = read_dann_audit_journal(journal_path)
        terminal_decision = classify_terminal_lookahead(
            audit,
            gradient_accumulation_steps=int(audit.get("gradient_accumulation_steps") or 1),
            journal_audit=journal_audit,
        )
        terminal_is_valid = bool(
            terminal_partial
            and audit.get("trainer_global_step") == audit.get("trainer_max_steps")
            and terminal_decision.get("safe")
        )
        resumable_partial = terminal_partial and isinstance(audit.get("trainer_global_step"), int) and isinstance(audit.get("trainer_max_steps"), int) and audit.get("trainer_global_step") < audit.get("trainer_max_steps")
        terminal_processing_gap = bool(
            audit.get("trainer_global_step") == audit.get("trainer_max_steps")
            and any(item.get("processed_batches") != item.get("issued_batches") for item in epochs)
            and not terminal_decision.get("safe")
        )
        resume_complete = bool(epochs) and (
            not terminal_processing_gap and (all(item.get("completion") == "complete" for item in epochs) or terminal_is_valid or resumable_partial)
        )
        _atomic_write_json(state_path, sampler_state)
        _atomic_write_json(audit_path, audit)
        _atomic_write_json(
            checkpoint_dir / "dann_checkpoint_state.json",
            {
                "schema_version": 3,
                "complete": resume_complete,
                "resume_complete": resume_complete,
                "training_terminal_partial": terminal_is_valid,
                "audit_protocol": audit["audit_protocol"],
                "seed": self.dann_batch_sampler.seed,
                "source_count": self.dann_batch_sampler.source_count,
                "target_count": self.dann_batch_sampler.target_count,
                "source_batch_size": self.dann_batch_sampler.source_batch_size,
                "target_batch_size": self.dann_batch_sampler.target_batch_size,
                "source_row_ids": list(self.dann_batch_sampler.source_row_ids),
                "target_row_ids": list(self.dann_batch_sampler.target_row_ids),
                "completed_epochs": [item["epoch"] for item in epochs if item.get("completion") == "complete"],
                "completed_epoch_count": sum(item.get("completion") == "complete" for item in epochs),
                "completed_physical_traversals": [item["physical_traversal_index"] for item in epochs if item.get("completion") == "complete"],
                "audit_traversal_count": len(audit["epochs"]),
                "trainer_global_step": audit.get("trainer_global_step"),
                "trainer_max_steps": audit.get("trainer_max_steps"),
                "last_traversal_completion": audit["epochs"][-1].get("completion") if audit["epochs"] else None,
                "sampler_state_sha256": _json_sha256(state_path),
                "audit_sha256": _json_sha256(audit_path),
                "trainer_state_sha256": _json_sha256(checkpoint_dir / "trainer_state.json"),
                "gradient_state_artifact": gradient_state_path.name,
                "gradient_state_sha256": _json_sha256(gradient_state_path),
                "model_artifact": next(
                    path.name
                    for path in (checkpoint_dir / "pytorch_model.bin", checkpoint_dir / "model.safetensors")
                    if path.is_file()
                ),
                "model_artifact_sha256": _json_sha256(
                    next(
                        path
                        for path in (checkpoint_dir / "pytorch_model.bin", checkpoint_dir / "model.safetensors")
                        if path.is_file()
                    )
                ),
            },
        )

    @classmethod
    def _strip_generation_only_inputs(cls, inputs: dict, keep_graph: bool = False) -> dict:
        cleaned = dict(inputs)
        for key in cls._generation_only_input_keys:
            cleaned.pop(key, None)
        if not keep_graph:
            for key in cls._graph_input_keys:
                cleaned.pop(key, None)
        return cleaned

    def _track_component(self, name: str, value: torch.Tensor | float, reduction: str = "mean") -> None:
        numeric = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        self._component_sums[name] = self._component_sums.get(name, 0.0) + numeric
        self._component_counts[name] = self._component_counts.get(name, 0) + 1
        self._component_reductions[name] = reduction

    def log(self, logs: dict, *args, **kwargs) -> None:
        if "loss" in logs and self._component_sums:
            for name, total in self._component_sums.items():
                if self._component_reductions.get(name) == "sum":
                    logs[name] = round(total, 6)
                else:
                    logs[name] = round(total / max(1, self._component_counts.get(name, 1)), 6)
            self._component_sums.clear()
            self._component_counts.clear()
            self._component_reductions.clear()
        super().log(logs, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        cleaned_inputs = self._strip_generation_only_inputs(
            inputs,
            keep_graph=bool(getattr(model, "use_syntactic_graph_adapter", False)),
        )
        return super().prediction_step(
            model,
            cleaned_inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
            **kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weight = inputs.pop("sample_weight", None)
        domain_weight = inputs.pop("domain_weight", sample_weight)
        domain_label = inputs.pop("domain_label", None)
        structure_weight = inputs.pop("structure_weight", None)
        consistency_group = inputs.pop("consistency_group", None)
        pairing_aspect_spans = inputs.pop("pairing_aspect_spans", None)
        pairing_opinion_spans = inputs.pop("pairing_opinion_spans", None)
        pairing_mask = inputs.pop("pairing_mask", None)
        sentiment_contrastive_spans = inputs.pop("sentiment_contrastive_spans", None)
        sentiment_contrastive_labels = inputs.pop("sentiment_contrastive_labels", None)
        sentiment_contrastive_mask = inputs.pop("sentiment_contrastive_mask", None)
        sentiment_contrastive_weights = inputs.pop("sentiment_contrastive_weights", None)
        graph_inputs = {
            key: inputs.pop(key, None)
            for key in self._graph_input_keys
        }
        if getattr(model, "use_syntactic_graph_adapter", False):
            if any(value is None for value in graph_inputs.values()):
                missing = [key for key, value in graph_inputs.items() if value is None]
                raise ValueError(f"syntactic graph model received incomplete graph batch: {missing}")
            inputs.update(graph_inputs)
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        outputs = model(**inputs, return_dict=True, output_hidden_states=False)
        logits = outputs.logits
        token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.size())
        token_mask = labels.ne(-100)
        per_sample_loss = token_loss.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        if domain_weight is not None:
            domain_weights = domain_weight.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
            paired_dann_normalization = getattr(self, "dann_batch_sampler", None) is not None
            if structure_weight is not None:
                structure_weights = structure_weight.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
                loss = joint_weighted_loss(
                    per_sample_loss,
                    domain_weights,
                    structure_weights,
                    self.lambda_structure_loss,
                    normalize_by_active_weight=paired_dann_normalization,
                )
            else:
                loss = weighted_loss_mean(
                    per_sample_loss,
                    domain_weights,
                    normalize_by_active_weight=paired_dann_normalization,
                )
        else:
            loss = per_sample_loss.mean()
        generation_loss = loss
        if self.multi_element_coverage_loss_enabled and self.element_coverage_weight > 0 and pairing_aspect_spans is not None:
            hidden = getattr(outputs, "encoder_last_hidden_state", None)
            if hidden is not None and pairing_aspect_spans.size(1) > 0:
                spans = torch.cat([pairing_aspect_spans, pairing_opinion_spans], dim=1)
                masks = torch.cat([pairing_mask, pairing_mask], dim=1).bool()
                token_scores = torch.sigmoid(hidden.float().norm(dim=-1))
                source_mask = labels.ne(-100).any(dim=1)
                triplet_count = pairing_mask.sum(dim=1)
                coverage_loss, coverage_stats = multi_element_coverage_loss(
                    token_scores, spans, masks, source_mask, triplet_count
                )
                loss = loss + self.element_coverage_weight * coverage_loss
                if model.training:
                    self._track_component("element_coverage_loss", coverage_loss)
                    for name, value in coverage_stats.items():
                        self._track_component(name, value, reduction="sum" if name.endswith("count") else "mean")
        if consistency_group is not None and self.lambda_consistency_loss > 0:
            consistency_loss = grouped_representation_consistency_loss(
                outputs.encoder_last_hidden_state,
                attention_mask,
                consistency_group,
            )
            loss = loss + self.lambda_consistency_loss * consistency_loss
        if self.lambda_pairing_loss > 0 and pairing_aspect_spans is not None and pairing_opinion_spans is not None:
            encoder_hidden = outputs.encoder_last_hidden_state
            if encoder_hidden is not None:
                pair_loss, pairing_stats = encoder_pairing_contrastive_loss(
                    encoder_hidden,
                    pairing_aspect_spans,
                    pairing_opinion_spans,
                    pairing_mask,
                    temperature=self.pairing_temperature,
                    return_stats=True,
                )
                loss = loss + self.lambda_pairing_loss * pair_loss
                if model.training:
                    self._track_component("pairing_loss", pair_loss)
                    for name, value in pairing_stats.items():
                        reduction = "sum" if name in {"pairing_active_rows", "pairing_active_pairs"} else "mean"
                        self._track_component(name, value, reduction=reduction)
        if (
            self.lambda_sentiment_contrastive > 0
            and sentiment_contrastive_spans is not None
            and hasattr(model, "sentiment_prototype_head")
        ):
            encoder_hidden = outputs.encoder_last_hidden_state
            if encoder_hidden is not None:
                sentiment_loss, sentiment_stats = sentiment_prototype_contrastive_loss(
                    encoder_hidden,
                    sentiment_contrastive_spans,
                    sentiment_contrastive_labels,
                    sentiment_contrastive_mask,
                    model.sentiment_prototype_head,
                    temperature=self.sentiment_contrastive_temperature,
                    sample_weights=sentiment_contrastive_weights,
                    class_weights=(
                        torch.tensor(self.sentiment_contrastive_class_weights, device=encoder_hidden.device)
                        if self.sentiment_contrastive_class_weights else None
                    ),
                    return_stats=True,
                )
                loss = loss + self.lambda_sentiment_contrastive * sentiment_loss
                if model.training:
                    self._track_component("sentiment_contrastive_loss", sentiment_loss)
                    for name, value in sentiment_stats.items():
                        self._track_component(name, value)
        if (
            model.training
            and self.lambda_domain_adv > 0
            and domain_label is not None
            and hasattr(model, "domain_adversarial_head")
            and outputs.encoder_last_hidden_state is not None
        ):
            domain_adv_loss = compute_domain_adversarial_loss(
                outputs.encoder_last_hidden_state,
                attention_mask,
                domain_label,
                model.domain_adversarial_head,
                grl_lambda=self.domain_adv_grl_lambda,
            )
            if domain_adv_loss is not None:
                loss = loss + self.lambda_domain_adv * domain_adv_loss
                self._track_component("domain_adv_loss", domain_adv_loss)
        if model.training:
            self._track_component("generation_loss", generation_loss)
            self._track_component("joint_total_loss", loss)
        return (loss, outputs) if return_outputs else loss


def weighted_loss_mean(
    per_sample_loss: torch.Tensor,
    weights: torch.Tensor,
    *,
    normalize_by_active_weight: bool = False,
) -> torch.Tensor:
    weighted = per_sample_loss * weights
    if normalize_by_active_weight:
        return weighted.sum() / weights.sum().clamp_min(1.0)
    return weighted.mean()


def joint_weighted_loss(
    per_sample_loss: torch.Tensor,
    domain_weights: torch.Tensor,
    structure_weights: torch.Tensor,
    lambda_structure: float,
    *,
    normalize_by_active_weight: bool = False,
) -> torch.Tensor:
    domain_loss = weighted_loss_mean(
        per_sample_loss,
        domain_weights,
        normalize_by_active_weight=normalize_by_active_weight,
    )
    if lambda_structure <= 0:
        return domain_loss
    structure_loss = weighted_loss_mean(
        per_sample_loss,
        structure_weights,
        normalize_by_active_weight=normalize_by_active_weight,
    )
    return domain_loss + lambda_structure * structure_loss


def grouped_representation_consistency_loss(
    representations: torch.Tensor,
    attention_mask: torch.Tensor | None,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    if representations is None or group_ids is None:
        return torch.tensor(0.0, device=representations.device if representations is not None else None)
    if representations.size(0) <= 1:
        return representations.new_tensor(0.0)
    pooled = representations
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(pooled.dtype)
        pooled = (pooled * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    group_ids = group_ids.to(pooled.device).view(-1)
    unique_group_ids = torch.unique(group_ids)
    losses = []
    for group_id in unique_group_ids:
        member_idx = torch.nonzero(group_ids == group_id, as_tuple=False).view(-1)
        if member_idx.numel() < 2:
            continue
        group_repr = F.normalize(pooled.index_select(0, member_idx), p=2, dim=-1)
        center = F.normalize(group_repr.mean(dim=0, keepdim=True), p=2, dim=-1)
        losses.append(1.0 - F.cosine_similarity(group_repr, center.expand_as(group_repr), dim=-1).mean())
    if not losses:
        return representations.new_tensor(0.0)
    return torch.stack(losses).mean()


def span_mean(hidden: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
    vectors = []
    seq_len = hidden.size(1)
    for batch_idx, batch_spans in enumerate(spans):
        row_vectors = []
        for start, end in batch_spans.tolist():
            start = max(0, min(int(start), seq_len - 1))
            end = max(start + 1, min(int(end), seq_len))
            row_vectors.append(hidden[batch_idx, start:end].mean(dim=0))
        vectors.append(torch.stack(row_vectors, dim=0) if row_vectors else hidden.new_zeros((0, hidden.size(-1))))
    if not vectors:
        return hidden.new_zeros((0, 0, hidden.size(-1)))
    return torch.stack(vectors, dim=0)


def pairing_contrastive_loss(
    decoder_hidden: torch.Tensor,
    aspect_spans: torch.Tensor,
    opinion_spans: torch.Tensor,
    pairing_mask: torch.Tensor | None,
    temperature: float = 0.1,
) -> torch.Tensor:
    if decoder_hidden is None or aspect_spans is None or opinion_spans is None:
        return decoder_hidden.new_tensor(0.0) if decoder_hidden is not None else torch.tensor(0.0)
    if aspect_spans.numel() == 0 or opinion_spans.numel() == 0:
        return decoder_hidden.new_tensor(0.0)
    aspect_spans = aspect_spans.to(decoder_hidden.device)
    opinion_spans = opinion_spans.to(decoder_hidden.device)
    if pairing_mask is None:
        pairing_mask = torch.ones(aspect_spans.shape[:2], device=decoder_hidden.device, dtype=torch.bool)
    else:
        pairing_mask = pairing_mask.to(decoder_hidden.device).bool()
    aspect_repr = F.normalize(span_mean(decoder_hidden, aspect_spans), p=2, dim=-1)
    opinion_repr = F.normalize(span_mean(decoder_hidden, opinion_spans), p=2, dim=-1)
    losses = []
    for batch_idx in range(aspect_repr.size(0)):
        active_idx = torch.nonzero(pairing_mask[batch_idx], as_tuple=False).view(-1)
        if active_idx.numel() < 2:
            continue
        aspects = aspect_repr[batch_idx].index_select(0, active_idx)
        opinions = opinion_repr[batch_idx].index_select(0, active_idx)
        logits = aspects @ opinions.transpose(0, 1) / temperature
        targets = torch.arange(active_idx.numel(), device=decoder_hidden.device)
        losses.append(F.cross_entropy(logits, targets))
    if not losses:
        return decoder_hidden.new_tensor(0.0)
    return torch.stack(losses).mean()


def _multi_positive_direction_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, float, int]:
    losses = []
    correct = 0
    active_anchors = 0
    for anchor_idx in range(logits.size(0)):
        positives = positive_mask[anchor_idx]
        if not positives.any() or positives.all():
            continue
        anchor_logits = logits[anchor_idx]
        losses.append(torch.logsumexp(anchor_logits, dim=0) - torch.logsumexp(anchor_logits[positives], dim=0))
        predicted_idx = int(anchor_logits.argmax().item())
        correct += int(bool(positives[predicted_idx]))
        active_anchors += 1
    if not losses:
        return logits.new_tensor(0.0), 0.0, 0
    return torch.stack(losses).mean(), correct / active_anchors, active_anchors


def encoder_pairing_contrastive_loss(
    encoder_hidden: torch.Tensor,
    aspect_spans: torch.Tensor,
    opinion_spans: torch.Tensor,
    pairing_mask: torch.Tensor | None,
    temperature: float = 0.1,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    zero = encoder_hidden.new_tensor(0.0)
    empty_stats = {
        "pairing_aspect_accuracy": 0.0,
        "pairing_opinion_accuracy": 0.0,
        "pairing_active_rows": 0.0,
        "pairing_active_pairs": 0.0,
    }
    if aspect_spans is None or opinion_spans is None or aspect_spans.numel() == 0 or opinion_spans.numel() == 0:
        return (zero, empty_stats) if return_stats else zero
    aspect_spans = aspect_spans.to(encoder_hidden.device)
    opinion_spans = opinion_spans.to(encoder_hidden.device)
    if pairing_mask is None:
        pairing_mask = torch.ones(aspect_spans.shape[:2], device=encoder_hidden.device, dtype=torch.bool)
    else:
        pairing_mask = pairing_mask.to(encoder_hidden.device).bool()
    aspect_repr = F.normalize(span_mean(encoder_hidden, aspect_spans), p=2, dim=-1)
    opinion_repr = F.normalize(span_mean(encoder_hidden, opinion_spans), p=2, dim=-1)
    losses = []
    aspect_correct_weighted = 0.0
    opinion_correct_weighted = 0.0
    aspect_anchor_count = 0
    opinion_anchor_count = 0
    active_rows = 0
    active_pairs = 0
    for batch_idx in range(aspect_repr.size(0)):
        active_idx = torch.nonzero(pairing_mask[batch_idx], as_tuple=False).view(-1)
        if active_idx.numel() < 2:
            continue
        aspects = aspect_repr[batch_idx].index_select(0, active_idx)
        opinions = opinion_repr[batch_idx].index_select(0, active_idx)
        active_aspect_spans = aspect_spans[batch_idx].index_select(0, active_idx)
        active_opinion_spans = opinion_spans[batch_idx].index_select(0, active_idx)
        aspect_same = (active_aspect_spans[:, None, :] == active_aspect_spans[None, :, :]).all(dim=-1)
        opinion_same = (active_opinion_spans[:, None, :] == active_opinion_spans[None, :, :]).all(dim=-1)
        positive_mask = (aspect_same[:, None, :] & opinion_same[None, :, :]).any(dim=-1)
        logits = aspects @ opinions.transpose(0, 1) / max(float(temperature), 1e-6)
        aspect_loss, aspect_accuracy, aspect_anchors = _multi_positive_direction_loss(logits, positive_mask)
        opinion_loss, opinion_accuracy, opinion_anchors = _multi_positive_direction_loss(
            logits.transpose(0, 1), positive_mask.transpose(0, 1)
        )
        row_losses = []
        if aspect_anchors:
            row_losses.append(aspect_loss)
            aspect_correct_weighted += aspect_accuracy * aspect_anchors
            aspect_anchor_count += aspect_anchors
        if opinion_anchors:
            row_losses.append(opinion_loss)
            opinion_correct_weighted += opinion_accuracy * opinion_anchors
            opinion_anchor_count += opinion_anchors
        if row_losses:
            losses.append(torch.stack(row_losses).mean())
            active_rows += 1
            active_pairs += int(active_idx.numel())
    loss = torch.stack(losses).mean() if losses else zero
    stats = {
        "pairing_aspect_accuracy": aspect_correct_weighted / max(1, aspect_anchor_count),
        "pairing_opinion_accuracy": opinion_correct_weighted / max(1, opinion_anchor_count),
        "pairing_active_rows": float(active_rows),
        "pairing_active_pairs": float(active_pairs),
    }
    return (loss, stats) if return_stats else loss


def sentiment_prototype_contrastive_loss(
    contextual_hidden: torch.Tensor,
    opinion_spans: torch.Tensor,
    sentiment_labels: torch.Tensor,
    sentiment_mask: torch.Tensor,
    prototype_head: SentimentPrototypeHead,
    temperature: float = 0.1,
    sample_weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    if opinion_spans is None or opinion_spans.numel() == 0:
        zero = contextual_hidden.new_tensor(0.0)
        return (zero, {}) if return_stats else zero
    opinion_spans = opinion_spans.to(contextual_hidden.device)
    sentiment_labels = sentiment_labels.to(contextual_hidden.device, dtype=torch.long)
    valid_mask = sentiment_mask.to(contextual_hidden.device).bool() & sentiment_labels.ne(-100)
    if not valid_mask.any():
        zero = contextual_hidden.new_tensor(0.0)
        return (zero, {}) if return_stats else zero
    opinion_repr = F.normalize(span_mean(contextual_hidden, opinion_spans), p=2, dim=-1)
    logits = opinion_repr[valid_mask] @ prototype_head.normalized_prototypes().transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    targets = sentiment_labels[valid_mask]
    per_item_loss = F.cross_entropy(
        logits,
        targets,
        weight=class_weights.to(logits.device, dtype=logits.dtype) if class_weights is not None else None,
        reduction="none",
    )
    valid_weights = (
        sample_weights.to(logits.device, dtype=logits.dtype)[valid_mask]
        if sample_weights is not None else torch.ones_like(per_item_loss)
    )
    loss = (per_item_loss * valid_weights).sum() / valid_weights.sum().clamp_min(1e-6)
    if not return_stats:
        return loss
    predictions = logits.argmax(dim=-1)
    stats = {}
    for sentiment_id, sentiment_name in enumerate(("pos", "neg", "neu")):
        class_mask = targets.eq(sentiment_id)
        if class_mask.any():
            stats[f"sentiment_{sentiment_name}_accuracy"] = float(predictions[class_mask].eq(targets[class_mask]).float().mean())
    stats["sentiment_prototype_accuracy"] = float(predictions.eq(targets).float().mean())
    return loss, stats


def summarize_sample_weights(
    rows: list[dict],
    source_weight: float,
    pseudo_weight: float,
    augment_weight: float,
    force_domain_weights: bool = False,
) -> dict:
    counts = {"source_gold": 0, "target_pseudo": 0, "c3da_augment": 0, "target_unlabeled": 0}
    weights = []
    for row in rows:
        augmentation = row.get("augmentation")
        if augmentation == "target_unlabeled":
            counts["target_unlabeled"] += 1
            fallback_weight = 0.0
        elif augmentation == "target_pseudo":
            counts["target_pseudo"] += 1
            fallback_weight = pseudo_weight
        elif augmentation in CSA_AUGMENT_CHANNELS:
            counts["c3da_augment"] += 1
            fallback_weight = augment_weight
        else:
            counts["source_gold"] += 1
            fallback_weight = source_weight
        weights.append(float(fallback_weight if force_domain_weights else row.get("sample_weight", fallback_weight)))
    by_source = {}
    for name, predicate, fallback_weight in [
        ("source_gold", lambda row: row.get("augmentation") not in {"target_pseudo", "target_unlabeled", *CSA_AUGMENT_CHANNELS}, source_weight),
        ("target_pseudo", lambda row: row.get("augmentation") == "target_pseudo", pseudo_weight),
        ("c3da_augment", lambda row: row.get("augmentation") in CSA_AUGMENT_CHANNELS, augment_weight),
    ]:
        source_weights = [
            float(fallback_weight if force_domain_weights else row.get("sample_weight", fallback_weight))
            for row in rows
            if predicate(row)
        ]
        if source_weights:
            by_source[f"{name}_weight_mean"] = sum(source_weights) / len(source_weights)
            by_source[f"{name}_weight_min"] = min(source_weights)
            by_source[f"{name}_weight_max"] = max(source_weights)
    return {
        **counts,
        "source_weight": source_weight,
        "pseudo_weight": pseudo_weight,
        "augment_weight": augment_weight,
        "force_domain_weights": force_domain_weights,
        "sample_weight_min": min(weights) if weights else None,
        "sample_weight_max": max(weights) if weights else None,
        "sample_weight_mean": sum(weights) / len(weights) if weights else None,
        **by_source,
    }


def summarize_generation_weights(dataset: JsonlSeq2SeqDataset) -> dict:
    neutral_weights = []
    non_neutral_weights = []
    for row in dataset.rows:
        if row.get("augmentation") == "target_unlabeled":
            continue
        domain_weight = dataset.sample_weight(row)
        effective_weight = dataset.generation_weight(row, domain_weight)
        triplets = parse_triplet_text_list(row.get("target", ""))
        target = (
            neutral_weights
            if any(sentiment == "neu" for _aspect, _opinion, sentiment in triplets)
            else non_neutral_weights
        )
        target.append(effective_weight)

    def weight_stats(name: str, weights: list[float]) -> dict:
        if not weights:
            return {
                f"{name}_rows": 0,
                f"{name}_weight_mean": None,
                f"{name}_weight_min": None,
                f"{name}_weight_max": None,
            }
        return {
            f"{name}_rows": len(weights),
            f"{name}_weight_mean": sum(weights) / len(weights),
            f"{name}_weight_min": min(weights),
            f"{name}_weight_max": max(weights),
        }

    return {
        **weight_stats("neutral", neutral_weights),
        **weight_stats("non_neutral", non_neutral_weights),
    }


def summarize_sentiment_contrastive_rows(
    rows: list[dict],
    min_weight: float,
    exclude_augment: bool,
    source_only: bool = False,
) -> dict:
    counts = {"pos": 0, "neg": 0, "neu": 0}
    eligible_rows = 0
    for row in rows:
        augmentation = row.get("augmentation")
        fallback_weight = 0.65 if augmentation == "target_pseudo" else (0.2 if augmentation in CSA_AUGMENT_CHANNELS else 1.0)
        weight = float(row.get("sample_weight", fallback_weight) or fallback_weight)
        if weight < min_weight or (exclude_augment and augmentation in CSA_AUGMENT_CHANNELS):
            continue
        if source_only and (augmentation == "target_pseudo" or augmentation in CSA_AUGMENT_CHANNELS):
            continue
        eligible_rows += 1
        for _aspect, _opinion, sentiment in parse_triplet_text_list(row.get("target", "")):
            if sentiment in counts:
                counts[sentiment] += 1
    return {"eligible_rows": eligible_rows, "triplets": sum(counts.values()), **counts}


def build_sentiment_class_weights(counts: dict[str, int]) -> list[float]:
    raw = [1.0 / math.sqrt(max(1, int(counts.get(name, 0)))) for name in ("pos", "neg", "neu")]
    mean_weight = sum(raw) / len(raw)
    return [value / mean_weight for value in raw]


def add_task_special_tokens(tokenizer, model, rows: list[dict]) -> None:
    text = "\n".join(f"{row.get('input', '')}\n{row.get('target', '')}" for row in rows[:2000])
    needed = [tok for tok in TASK_SPECIAL_TOKENS if tok in text]
    if not needed:
        return
    added = tokenizer.add_special_tokens({"additional_special_tokens": needed})
    if added:
        model.resize_token_embeddings(len(tokenizer))
        print(f"added special tokens: {needed}")
    for token in needed:
        init_word = TAG_INIT_WORDS.get(token)
        if not init_word:
            continue
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        init_ids = tokenizer.encode(init_word, add_special_tokens=False)
        if len(token_ids) != 1 or not init_ids:
            continue
        with torch.no_grad():
            model.shared.weight[token_ids[0]] = model.shared.weight[init_ids[0]].clone()
        print(f"initialized {token} from {init_word}")


def enforce_graph_training_boundary(use_syntactic_graph_adapter: bool) -> None:
    """Keep direct graph training closed; only the Phase A API may authorize it."""
    if use_syntactic_graph_adapter and not (_PHASE_A_GRAPH_TRAINING_AUTHORIZED or os.environ.get("C3DA_ALLOW_GRAPH_TRAINING") == "1"):
        raise RuntimeError(
            "syntactic graph training is not approved; run "
            "m1_syntactic_graph_entry_audit.py for zero-update audit only, or use the approved "
            "m1_syntactic_rgat_pseudo_quick_ablation.py Phase A entry"
        )


def run_phase_a_training(argv: list[str]) -> dict | None:
    """Run the existing trainer through a narrow in-process Phase A entry.

    The direct ``t5_absa_train.py`` command never sets the private authorization
    flag, so its graph-training hard stop remains active. The dedicated Phase A
    runner calls this API only after validating its frozen recipe and identities.
    """
    global _PHASE_A_GRAPH_TRAINING_AUTHORIZED, _PHASE_A_LIFECYCLE_CLEANUP_REQUESTED
    previous_argv = sys.argv
    previous_authorization = _PHASE_A_GRAPH_TRAINING_AUTHORIZED
    previous_cleanup_request = _PHASE_A_LIFECYCLE_CLEANUP_REQUESTED
    sys.argv = ["t5_absa_train.py", *argv]
    _PHASE_A_GRAPH_TRAINING_AUTHORIZED = True
    _PHASE_A_LIFECYCLE_CLEANUP_REQUESTED = True
    try:
        return main()
    finally:
        sys.argv = previous_argv
        _PHASE_A_GRAPH_TRAINING_AUTHORIZED = previous_authorization
        _PHASE_A_LIFECYCLE_CLEANUP_REQUESTED = previous_cleanup_request


def main() -> dict | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=r"J:\nlp\models\t5-base-py")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=float, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=96)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", choices=["none", "auto"], default="none")
    parser.add_argument("--seed", type=int, default=1000)
    reproducibility_group = parser.add_mutually_exclusive_group()
    reproducibility_group.add_argument("--deterministic", action="store_true")
    reproducibility_group.add_argument("--legacy_stochastic", action="store_true")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_syntactic_graph_adapter", action="store_true")
    parser.add_argument("--graph_focus_enabled", action="store_true")
    parser.add_argument("--syntactic_graph_cache_dir", default="")
    parser.add_argument("--syntactic_graph_parser_dir", default=r"J:\nlp\models\stanza_resources")
    parser.add_argument("--element_aware_attention", action="store_true")
    parser.add_argument("--element_focus_weight", type=float, default=0.0)
    parser.add_argument("--element_coverage_weight", type=float, default=0.0)
    parser.add_argument("--multi_element_coverage_loss", action="store_true")
    parser.add_argument("--target_unlabeled_file", default="")
    parser.add_argument("--paired_domain_batches", action="store_true")
    parser.add_argument("--dann_source_batch_size", type=int, default=1)
    parser.add_argument("--dann_target_batch_size", type=int, default=1)
    parser.add_argument("--dann_batch_audit_path", default="")
    parser.add_argument("--initialization_audit_path", default="")
    parser.add_argument("--source_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_weight", type=float, default=0.5)
    parser.add_argument("--augment_weight", type=float, default=0.2)
    parser.add_argument("--force_domain_weights", action="store_true")
    parser.add_argument("--lambda_structure_loss", type=float, default=0.15)
    parser.add_argument("--lambda_consistency_loss", type=float, default=0.0)
    parser.add_argument("--lambda_pairing_loss", type=float, default=0.0)
    parser.add_argument("--pairing_temperature", type=float, default=0.1)
    parser.add_argument("--pairing_source_only", action="store_true")
    parser.add_argument("--lambda_domain_adv", type=float, default=0.0)
    parser.add_argument("--domain_adv_hidden_size", type=int, default=256)
    parser.add_argument("--domain_adv_grl_lambda", type=float, default=1.0)
    parser.add_argument("--domain_adv_exclude_augment", action="store_true")
    parser.add_argument("--lambda_sentiment_contrastive", type=float, default=0.0)
    parser.add_argument("--sentiment_contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--sentiment_contrastive_min_weight", type=float, default=0.65)
    parser.add_argument("--sentiment_contrastive_exclude_augment", action="store_true")
    parser.add_argument("--sentiment_contrastive_source_only", action="store_true")
    parser.add_argument("--sentiment_contrastive_class_balanced", action="store_true")
    parser.add_argument("--sentiment_prototype_initialize_from_context", action="store_true")
    parser.add_argument("--sentiment_prototype_init_batch_size", type=int, default=2)
    parser.add_argument("--max_pairing_triplets", type=int, default=4)
    parser.add_argument("--min_pairing_triplets", type=int, default=2)
    parser.add_argument("--min_pairing_sample_weight", type=float, default=0.65)
    parser.add_argument("--multi_triplet_loss_gain", type=float, default=0.1)
    parser.add_argument("--neutral_loss_gain", type=float, default=0.15)
    parser.add_argument("--max_effective_weight", type=float, default=1.0)
    parser.add_argument("--neutral_generation_loss_gain", type=float, default=0.0)
    parser.add_argument("--neutral_generation_max_effective_weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint_selection",
        choices=["last", "best", "aste_f1"],
        default="last",
        help=(
            "last saves the final training step; best selects the lowest dev eval_loss; "
            "aste_f1 selects the highest dev ASTE micro-F1 with multi-triplet F1 as a near-tie breaker."
        ),
    )
    args = parser.parse_args()

    enforce_graph_training_boundary(args.use_syntactic_graph_adapter)
    if args.paired_domain_batches and args.lambda_domain_adv <= 0:
        raise ValueError("paired DANN batches require positive --lambda_domain_adv")
    if args.paired_domain_batches and (args.dann_source_batch_size != 1 or args.dann_target_batch_size != 1):
        raise ValueError("Phase A paired DANN batches require source=1 and target=1")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    reproducibility_mode = "deterministic" if args.deterministic else "legacy"
    reproducibility_config = configure_reproducibility(args.seed, reproducibility_mode)
    initialization_rng_state = torch.get_rng_state()
    print("reproducibility:", reproducibility_config)
    output_dir = Path(args.output_dir)
    checkpoint_dirs = (
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
        if output_dir.exists()
        else []
    )
    latest_checkpoint = max(
        checkpoint_dirs,
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
        default=None,
    )
    resume_from_checkpoint = args.resume_from_checkpoint == "auto" and bool(checkpoint_dirs)

    if args.use_syntactic_graph_adapter:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file)
    train_graph_cache = None
    dev_graph_cache = None
    target_graph_cache = None
    target_domain_rows = []
    source_train_rows = None
    dann_batch_sampler = None
    graph_relation_vocab_size = 1
    if args.use_syntactic_graph_adapter:
        if not args.syntactic_graph_cache_dir:
            raise GraphCacheError("--syntactic_graph_cache_dir is required when graph adapter is enabled")
        if not args.target_unlabeled_file:
            raise GraphCacheError("--target_unlabeled_file is required when graph adapter is enabled")
        tokenizer_identity = build_tokenizer_identity(args.model_path, tokenizer)
        parser_identity = build_parser_identity(args.syntactic_graph_parser_dir)
        train_graph_cache = load_graph_cache_directory(
            args.syntactic_graph_cache_dir,
            "source_train",
            train_rows,
            tokenizer_identity=tokenizer_identity,
            parser_identity=parser_identity,
        )
        dev_graph_cache = load_graph_cache_directory(
            args.syntactic_graph_cache_dir,
            "source_dev",
            dev_rows,
            tokenizer_identity=tokenizer_identity,
            parser_identity=parser_identity,
        )
        target_rows = read_jsonl(args.target_unlabeled_file)
        target_graph_cache = load_graph_cache_directory(
            args.syntactic_graph_cache_dir,
            "target_unlabeled",
            target_rows,
            tokenizer_identity=tokenizer_identity,
            parser_identity=parser_identity,
        )
        if train_graph_cache.relation_vocab != dev_graph_cache.relation_vocab or train_graph_cache.relation_vocab != target_graph_cache.relation_vocab:
            raise GraphCacheError("graph relation vocabulary mismatch across required cache splits")
        graph_relation_vocab_size = train_graph_cache.relation_vocab_size
        if args.lambda_domain_adv > 0:
            target_domain_rows = build_target_unlabeled_domain_rows(
                target_rows,
                use_task_prefix=train_graph_cache.use_task_prefix,
            )
            source_train_rows = train_rows
            train_rows = train_rows + target_domain_rows
            train_graph_cache = CompositeGraphCache(
                {
                    "source_train": train_graph_cache,
                    "target_unlabeled": target_graph_cache,
                }
            )
            if args.paired_domain_batches:
                dann_batch_sampler = PairedDomainBatchSampler(
                    len(source_train_rows),
                    len(target_domain_rows),
                    source_batch_size=args.dann_source_batch_size,
                    target_batch_size=args.dann_target_batch_size,
                    seed=args.seed,
                    source_row_ids=[row.get("id") for row in source_train_rows],
                    target_row_ids=[row.get("id") for row in target_domain_rows],
                    audit_path=args.dann_batch_audit_path or None,
                )
    elif args.target_unlabeled_file and args.lambda_domain_adv > 0:
        target_rows = read_jsonl(args.target_unlabeled_file)
        target_domain_rows = build_target_unlabeled_domain_rows(target_rows, use_task_prefix=False)
        source_train_rows = train_rows
        train_rows = train_rows + target_domain_rows
        if args.paired_domain_batches:
            dann_batch_sampler = PairedDomainBatchSampler(
                len(source_train_rows),
                len(target_domain_rows),
                source_batch_size=args.dann_source_batch_size,
                target_batch_size=args.dann_target_batch_size,
                seed=args.seed,
                source_row_ids=[row.get("id") for row in source_train_rows],
                target_row_ids=[row.get("id") for row in target_domain_rows],
                audit_path=args.dann_batch_audit_path or None,
            )
    elif args.paired_domain_batches:
        raise ValueError("paired DANN batches require --target_unlabeled_file and positive lambda_domain_adv")
    if resume_from_checkpoint and dann_batch_sampler is not None:
        latest_checkpoint = find_latest_complete_dann_checkpoint(output_dir, dann_batch_sampler)
    model = load_seq2seq_model(
        args.model_path,
        use_syntactic_graph_adapter=args.use_syntactic_graph_adapter,
        relation_vocab_size=graph_relation_vocab_size,
        focus_enabled=args.graph_focus_enabled,
    )
    # Graph-only construction is deliberately outside the shared training RNG
    # stream.  Restore the stream before common token initialization so
    # Control and Treatment receive identical shared T5 state and training
    # randomness under the same seed.
    torch.set_rng_state(initialization_rng_state)
    add_task_special_tokens(tokenizer, model, train_rows + dev_rows)
    if args.lambda_domain_adv > 0:
        hidden_size = int(getattr(model.config, "d_model", model.get_input_embeddings().embedding_dim))
        initialize_domain_adversarial_head(
            model,
            hidden_size=hidden_size,
            classifier_hidden_size=args.domain_adv_hidden_size,
            seed=args.seed + 1,
        )
    if args.lambda_sentiment_contrastive > 0:
        hidden_size = int(getattr(model.config, "d_model", model.get_input_embeddings().embedding_dim))
        model.sentiment_prototype_head = SentimentPrototypeHead(hidden_size=hidden_size)
        if args.sentiment_prototype_initialize_from_context and not resume_from_checkpoint:
            prototype_init_stats = initialize_sentiment_prototypes_from_context(
                model,
                tokenizer,
                train_rows,
                batch_size=args.sentiment_prototype_init_batch_size,
                max_source_length=args.max_source_length,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            init_path = output_dir / "sentiment_prototype_init.json"
            init_path.write_text(json.dumps(prototype_init_stats, ensure_ascii=False, indent=2), encoding="utf-8")
            print("sentiment prototype initialization:", {"path": str(init_path), **prototype_init_stats})
        elif args.sentiment_prototype_initialize_from_context:
            print("sentiment prototype initialization: skipped because training will resume from checkpoint")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    training_rng_state = torch.get_rng_state()
    torch.set_rng_state(training_rng_state)

    if args.initialization_audit_path:
        variant = "treatment" if args.use_syntactic_graph_adapter else "control"
        _atomic_write_json(
            Path(args.initialization_audit_path),
            build_initialization_audit(model, variant=variant, seed=args.seed),
        )

    print(
        "sample weights:",
        summarize_sample_weights(
            train_rows,
            args.source_weight,
            args.pseudo_weight,
            args.augment_weight,
            force_domain_weights=args.force_domain_weights,
        ),
    )
    print(
        "joint loss:",
        {
            "lambda_structure_loss": args.lambda_structure_loss,
            "lambda_consistency_loss": args.lambda_consistency_loss,
            "lambda_pairing_loss": args.lambda_pairing_loss,
            "pairing_temperature": args.pairing_temperature,
            "pairing_source_only": args.pairing_source_only,
            "lambda_domain_adv": args.lambda_domain_adv,
            "domain_adv_hidden_size": args.domain_adv_hidden_size,
            "domain_adv_grl_lambda": args.domain_adv_grl_lambda,
            "domain_adv_exclude_augment": args.domain_adv_exclude_augment,
            "lambda_sentiment_contrastive": args.lambda_sentiment_contrastive,
            "sentiment_contrastive_temperature": args.sentiment_contrastive_temperature,
            "sentiment_contrastive_min_weight": args.sentiment_contrastive_min_weight,
            "sentiment_contrastive_exclude_augment": args.sentiment_contrastive_exclude_augment,
            "sentiment_contrastive_source_only": args.sentiment_contrastive_source_only,
            "sentiment_contrastive_class_balanced": args.sentiment_contrastive_class_balanced,
            "sentiment_prototype_initialize_from_context": args.sentiment_prototype_initialize_from_context,
            "sentiment_prototype_init_batch_size": args.sentiment_prototype_init_batch_size,
            "max_pairing_triplets": args.max_pairing_triplets,
            "min_pairing_triplets": args.min_pairing_triplets,
            "min_pairing_sample_weight": args.min_pairing_sample_weight,
            "multi_triplet_loss_gain": args.multi_triplet_loss_gain,
            "neutral_loss_gain": args.neutral_loss_gain,
            "max_effective_weight": args.max_effective_weight,
            "neutral_generation_loss_gain": args.neutral_generation_loss_gain,
            "neutral_generation_max_effective_weight": args.neutral_generation_max_effective_weight,
            "use_syntactic_graph_adapter": args.use_syntactic_graph_adapter,
            "syntactic_graph_cache_dir": args.syntactic_graph_cache_dir,
            "syntactic_graph_parser_dir": args.syntactic_graph_parser_dir,
            "graph_layers": 1 if args.use_syntactic_graph_adapter else 0,
            "graph_hidden_size": 256 if args.use_syntactic_graph_adapter else 0,
            "graph_attention_heads": 4 if args.use_syntactic_graph_adapter else 0,
            "graph_head_size": 64 if args.use_syntactic_graph_adapter else 0,
        },
    )
    if args.lambda_sentiment_contrastive > 0:
        sentiment_summary = summarize_sentiment_contrastive_rows(
            train_rows,
            args.sentiment_contrastive_min_weight,
            args.sentiment_contrastive_exclude_augment,
            args.sentiment_contrastive_source_only,
        )
        sentiment_class_weights = (
            build_sentiment_class_weights(sentiment_summary)
            if args.sentiment_contrastive_class_balanced else None
        )
        print("sentiment contrastive samples:", sentiment_summary)
        print("sentiment contrastive class weights:", sentiment_class_weights)
    else:
        sentiment_class_weights = None
    train_data = JsonlSeq2SeqDataset(
        train_rows,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        args.source_weight,
        args.pseudo_weight,
        args.augment_weight,
        multi_triplet_loss_gain=args.multi_triplet_loss_gain,
        neutral_loss_gain=args.neutral_loss_gain,
        max_effective_weight=args.max_effective_weight,
        neutral_generation_loss_gain=args.neutral_generation_loss_gain,
        neutral_generation_max_effective_weight=args.neutral_generation_max_effective_weight,
        force_domain_weights=args.force_domain_weights,
        max_pairing_triplets=args.max_pairing_triplets,
        min_pairing_triplets=args.min_pairing_triplets,
        min_pairing_sample_weight=args.min_pairing_sample_weight,
        pairing_source_only=args.pairing_source_only,
        domain_adv_exclude_augment=args.domain_adv_exclude_augment,
        sentiment_contrastive_min_weight=args.sentiment_contrastive_min_weight,
        sentiment_contrastive_exclude_augment=args.sentiment_contrastive_exclude_augment,
        sentiment_contrastive_source_only=args.sentiment_contrastive_source_only,
        graph_cache=train_graph_cache,
    )
    print("effective generation weights:", summarize_generation_weights(train_data))
    dev_data = JsonlSeq2SeqDataset(
        dev_rows,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        1.0,
        1.0,
        1.0,
        max_pairing_triplets=args.max_pairing_triplets,
        min_pairing_triplets=args.min_pairing_triplets,
        min_pairing_sample_weight=args.min_pairing_sample_weight,
        pairing_source_only=args.pairing_source_only,
        graph_cache=dev_graph_cache,
    )

    checkpoint_selection_config = build_checkpoint_selection_config(args.checkpoint_selection)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        report_to=[],
        **reproducibility_training_args(args.seed, reproducibility_mode),
        **checkpoint_selection_config,
    )
    collator = DataCollatorForSeq2SeqWithPairing(DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model))
    trainer = WeightedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        data_collator=collator,
        lambda_structure_loss=args.lambda_structure_loss,
        lambda_consistency_loss=args.lambda_consistency_loss,
        lambda_pairing_loss=args.lambda_pairing_loss,
        pairing_temperature=args.pairing_temperature,
        lambda_domain_adv=args.lambda_domain_adv,
        domain_adv_grl_lambda=args.domain_adv_grl_lambda,
        lambda_sentiment_contrastive=args.lambda_sentiment_contrastive,
        sentiment_contrastive_temperature=args.sentiment_contrastive_temperature,
        sentiment_contrastive_class_weights=sentiment_class_weights,
        element_aware_attention=args.element_aware_attention,
        element_focus_weight=args.element_focus_weight,
        element_coverage_weight=args.element_coverage_weight,
        multi_element_coverage_loss_enabled=args.multi_element_coverage_loss,
        dann_batch_sampler=dann_batch_sampler,
        compute_metrics=(
            build_aste_compute_metrics(tokenizer)
            if args.checkpoint_selection == "aste_f1"
            else None
        ),
    )
    if resume_from_checkpoint and dann_batch_sampler is not None:
        if latest_checkpoint is None:
            raise RuntimeError("paired DANN resume requested without a checkpoint")
        trainer.load_dann_batch_sampler_state(latest_checkpoint)
    if resume_from_checkpoint:
        print(f"resuming from latest checkpoint in {output_dir}")
    trainer.train(
        resume_from_checkpoint=str(latest_checkpoint) if resume_from_checkpoint else None
    )
    if args.dann_batch_audit_path:
        if dann_batch_sampler is None:
            raise RuntimeError("DANN batch audit requested without a paired DANN sampler")
        audit_path = Path(args.dann_batch_audit_path)
        dann_batch_sampler.flush_audit_snapshot()
    best_dir = output_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    removed_checkpoints = cleanup_training_checkpoints(output_dir)
    if removed_checkpoints:
        print(f"removed resumable checkpoints after final save: {removed_checkpoints}")
    print(f"saved {args.checkpoint_selection} model to {best_dir}")
    dann_audit = _phase_a_cpu_copy(trainer.get_dann_batch_audit())
    if not _PHASE_A_LIFECYCLE_CLEANUP_REQUESTED:
        return None

    runtime_refs = {
        "trainer": trainer,
        "model": model,
        "optimizer": getattr(trainer, "optimizer", None),
        "scheduler": getattr(trainer, "lr_scheduler", None),
        "scaler": getattr(trainer, "scaler", None),
        "dataloader": getattr(trainer, "_train_dataloader", None),
        "callbacks": getattr(getattr(trainer, "callback_handler", None), "callbacks", None),
        "callback_handler": getattr(trainer, "callback_handler", None),
        "accelerator": getattr(trainer, "accelerator", None),
        "training_batch_refs": getattr(trainer, "_past", None),
        "train_data": train_data,
        "dev_data": dev_data,
        "collator": collator,
        "dann_batch_sampler": dann_batch_sampler,
        "tokenizer": tokenizer,
        "train_graph_cache": train_graph_cache,
        "dev_graph_cache": dev_graph_cache,
        "target_graph_cache": target_graph_cache,
    }
    lifecycle = cleanup_phase_a_training_runtime(
        runtime_refs,
        variant="treatment" if args.use_syntactic_graph_adapter else "control",
        defer_finalization=True,
    )
    # These assignments are intentional: the function's own frame must not
    # keep a strong reference after the lifecycle audit has been constructed.
    trainer = None
    model = None
    train_data = None
    dev_data = None
    collator = None
    dann_batch_sampler = None
    tokenizer = None
    train_rows = None
    dev_rows = None
    target_domain_rows = None
    source_train_rows = None
    train_graph_cache = None
    dev_graph_cache = None
    target_graph_cache = None
    lifecycle = finalize_phase_a_training_runtime(lifecycle)
    return {"dann_batch_audit": dann_audit, "lifecycle": _phase_a_cpu_copy(lifecycle)}


if __name__ == "__main__":
    main()

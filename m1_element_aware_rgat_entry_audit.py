from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from element_aware_rgat import (
    align_gold_elements_to_graph_words,
    balanced_element_focus_loss,
    multi_element_coverage_loss,
)
from syntactic_graph import load_graph_cache_directory
from syntactic_graph_adapter import load_seq2seq_model
from t5_absa_data import read_jsonl
from t5_absa_train import DataCollatorForSeq2SeqWithPairing, JsonlSeq2SeqDataset
from t5_aste_data import parse_triplet_text_list


TASK_ID = "M1_ELEMENT_AWARE_MULTI_TRIPLET_RGAT_IMPLEMENTATION_V1"


class _DiscardTrace:
    def record(self, *_args, **_kwargs) -> None:
        return None


def _hash_parameters(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _finite(value: torch.Tensor | None) -> bool:
    return value is not None and bool(torch.isfinite(value).all())


def build_alignment_report(rows: list[dict], graph_cache) -> tuple[dict, list[dict]]:
    totals = {
        "gold_aspects": 0,
        "aligned_aspects": 0,
        "unmatched_aspects": 0,
        "ambiguous_aspects": 0,
        "gold_opinions": 0,
        "aligned_opinions": 0,
        "unmatched_opinions": 0,
        "ambiguous_opinions": 0,
    }
    representative: list[dict] = []
    for row in rows:
        aligned = align_gold_elements_to_graph_words(
            text=str(row.get("text", "")),
            parser_tokens=graph_cache.get_parser_tokens(row),
            triplets=parse_triplet_text_list(row.get("target", "")),
        )
        for name in totals:
            totals[name] += int(aligned["stats"][name])
        if len(representative) < 10 and any(
            item["status"] != "aligned" for item in aligned["examples"]
        ):
            representative.append(
                {
                    "row_id": str(row.get("id")),
                    "text": row.get("text", ""),
                    "examples": aligned["examples"],
                }
            )
    total_elements = totals["gold_aspects"] + totals["gold_opinions"]
    aligned_elements = totals["aligned_aspects"] + totals["aligned_opinions"]
    report = {
        "rows": len(rows),
        "gold_triplets": sum(len(parse_triplet_text_list(row.get("target", ""))) for row in rows),
        **totals,
        "total_elements": total_elements,
        "aligned_elements": aligned_elements,
        "unmatched_elements": totals["unmatched_aspects"] + totals["unmatched_opinions"],
        "ambiguous_elements": totals["ambiguous_aspects"] + totals["ambiguous_opinions"],
        "overall_alignment_rate": aligned_elements / max(1, total_elements),
    }
    return report, representative


def _select_rows(rows: list[dict]) -> tuple[dict, dict]:
    single = next(row for row in rows if len(parse_triplet_text_list(row.get("target", ""))) == 1)
    multi = next(row for row in rows if len(parse_triplet_text_list(row.get("target", ""))) >= 2)
    return single, multi


def _make_batch(rows, tokenizer, model, graph_cache, element_aware: bool):
    dataset = JsonlSeq2SeqDataset(
        rows,
        tokenizer,
        128,
        96,
        1.0,
        0.75,
        0.2,
        graph_cache=graph_cache,
        element_aware_enabled=element_aware,
    )
    collator = DataCollatorForSeq2SeqWithPairing(
        DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    )
    return collator([dataset[index] for index in range(len(dataset))])


def _move(batch: dict, device: torch.device) -> dict:
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}


def _graph_fields(batch: dict) -> dict:
    return {name: value for name, value in batch.items() if name.startswith("graph_")}


def run_numerical_audit(
    model_path: str,
    relation_vocab_size: int,
    tokenizer,
    graph_cache,
    single,
    multi,
    device,
    *,
    use_fp16: bool = False,
):
    """Run Control then Treatment so two full T5 models never coexist."""
    torch.manual_seed(1000)
    control = load_seq2seq_model(
        model_path, True, relation_vocab_size, low_cpu_mem_usage=True
    ).to(device).eval()
    if use_fp16:
        control.half()
    batch = _move(_make_batch([multi], tokenizer, control, graph_cache, True), device)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    with torch.no_grad():
        encoded = control.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        adapter_fields = {
            name.removeprefix("graph_"): value for name, value in _graph_fields(batch).items()
        }
        control_adapter = control.syntactic_graph_adapter(
            encoded.last_hidden_state,
            attention_mask=attention_mask,
            **adapter_fields,
            trace=_DiscardTrace(),
        )
        common_forward = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": batch["labels"],
            **_graph_fields(batch),
        }
        control_output = control(**common_forward, return_dict=True)
    control_snapshot = {
        "attention": control_adapter.attention_probabilities.detach().float().cpu(),
        "graph": control_adapter.graph_hidden.detach().float().cpu(),
        "fused": control_adapter.fused_hidden.detach().float().cpu(),
        "logits": control_output.logits.detach().float().cpu(),
    }
    control_graph_state = {
        name.removeprefix("syntactic_graph_adapter."): value.detach().cpu().clone()
        for name, value in control.state_dict().items()
        if name.startswith("syntactic_graph_adapter.")
    }
    del control_output, control_adapter, encoded, control, batch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(1000)
    treatment = load_seq2seq_model(
        model_path,
        True,
        relation_vocab_size,
        element_aware_enabled=True,
        focus_enabled=True,
        coverage_enabled=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    missing, unexpected = treatment.syntactic_graph_adapter.load_state_dict(
        control_graph_state, strict=False
    )
    if unexpected or any(not name.startswith("salience_head.") for name in missing):
        raise RuntimeError(
            f"Control/Treatment graph state mismatch: missing={missing}, unexpected={unexpected}"
        )
    if use_fp16:
        treatment.half()
    parameter_hash_before = _hash_parameters(treatment)
    batch = _move(_make_batch([multi], tokenizer, treatment, graph_cache, True), device)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    adapter_fields = {
        name.removeprefix("graph_"): value for name, value in _graph_fields(batch).items()
    }
    common_forward = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": batch["labels"],
        **_graph_fields(batch),
    }
    with torch.no_grad():
        encoded = treatment.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        treatment_adapter = treatment.syntactic_graph_adapter(
            encoded.last_hidden_state,
            attention_mask=attention_mask,
            **adapter_fields,
        )
        treatment_output = treatment(**common_forward, return_dict=True)
    zero_update = {
        "max_attention_diff": float((control_snapshot["attention"] - treatment_adapter.attention_probabilities.detach().float().cpu()).abs().max()),
        "max_graph_output_diff": float((control_snapshot["graph"] - treatment_adapter.graph_hidden.detach().float().cpu()).abs().max()),
        "max_fused_output_diff": float((control_snapshot["fused"] - treatment_adapter.fused_hidden.detach().float().cpu()).abs().max()),
        "max_decoder_logit_diff": float((control_snapshot["logits"] - treatment_output.logits.detach().float().cpu()).abs().max()),
    }
    zero_update["pass"] = all(value <= 1.0e-6 for value in zero_update.values())

    treatment.train()
    treatment.zero_grad(set_to_none=True)
    outputs = treatment(**common_forward, return_dict=True)
    focus, focus_stats = balanced_element_focus_loss(
        outputs.element_salience_scores,
        batch["element_node_labels"],
        batch["element_node_loss_mask"] & batch["element_source_row"][:, None],
    )
    coverage, coverage_stats = multi_element_coverage_loss(
        outputs.element_salience_scores,
        batch["element_spans"],
        batch["element_span_mask"],
        batch["element_source_row"],
        batch["element_triplet_count"],
    )
    total = outputs.loss + 0.05 * focus + 0.05 * coverage
    total.backward()
    salience_grad = sum(
        float(parameter.grad.detach().float().norm())
        for parameter in treatment.syntactic_graph_adapter.salience_head.parameters()
        if parameter.grad is not None
    )
    graph_grad = sum(
        float(parameter.grad.detach().float().norm())
        for name, parameter in treatment.syntactic_graph_adapter.named_parameters()
        if not name.startswith("salience_head.") and parameter.grad is not None
    )
    t5_grad = sum(
        float(parameter.grad.detach().float().norm())
        for name, parameter in treatment.named_parameters()
        if not name.startswith("syntactic_graph_adapter.") and parameter.grad is not None
    )
    gradient = {
        "loss_aste": float(outputs.loss.detach().cpu()),
        "loss_focus": float(focus.detach().cpu()),
        "loss_coverage": float(coverage.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
        "salience_grad": salience_grad,
        "rgat_grad": graph_grad,
        "t5_grad": t5_grad,
        "focus_stats": focus_stats,
        "coverage_stats": coverage_stats,
    }
    gradient["pass"] = all(
        np.isfinite(value) and value > 0
        for value in (salience_grad, graph_grad, t5_grad, gradient["loss_coverage"])
    )

    single_batch = _move(_make_batch([single], tokenizer, treatment, graph_cache, True), device)
    with torch.no_grad():
        single_outputs = treatment(
            input_ids=single_batch["input_ids"],
            attention_mask=single_batch["attention_mask"],
            labels=single_batch["labels"],
            **_graph_fields(single_batch),
            return_dict=True,
        )
        single_coverage, _ = multi_element_coverage_loss(
            single_outputs.element_salience_scores,
            single_batch["element_spans"],
            single_batch["element_span_mask"],
            single_batch["element_source_row"],
            single_batch["element_triplet_count"],
        )
        target_focus, _ = balanced_element_focus_loss(
            single_outputs.element_salience_scores,
            single_batch["element_node_labels"],
            torch.zeros_like(single_batch["element_node_loss_mask"]),
        )
        target_coverage, _ = multi_element_coverage_loss(
            single_outputs.element_salience_scores,
            single_batch["element_spans"],
            single_batch["element_span_mask"],
            torch.zeros_like(single_batch["element_source_row"]),
            torch.full_like(single_batch["element_triplet_count"], 2),
        )
    finite = {
        "salience": _finite(outputs.element_salience_scores),
        "gate": _finite(0.5 + 0.5 * outputs.element_salience_scores),
        "log_gate": _finite(torch.log(0.5 + 0.5 * outputs.element_salience_scores.float())),
        "attention": _finite(outputs.graph_attention_probabilities),
        "logits": _finite(outputs.logits),
        "loss_focus": _finite(focus),
        "loss_coverage": _finite(coverage),
        "loss_total": _finite(total),
        "single_coverage_zero": float(single_coverage) == 0.0,
        "target_focus_zero": float(target_focus) == 0.0,
        "target_coverage_zero": float(target_coverage) == 0.0,
    }
    finite["pass"] = all(finite.values())
    parameter_hash_after = _hash_parameters(treatment)
    finite["parameter_hash_unchanged"] = parameter_hash_before == parameter_hash_after
    finite["pass"] = bool(finite["pass"] and finite["parameter_hash_unchanged"])
    del treatment
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return zero_update, gradient, finite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_train_file", required=True)
    parser.add_argument("--graph_cache_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if "target_test" in args.source_train_file.lower():
        raise RuntimeError("target_test is forbidden in the engineering entry audit")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.source_train_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    graph_cache = load_graph_cache_directory(args.graph_cache_dir, "source_train", rows)
    alignment, examples = build_alignment_report(rows, graph_cache)
    single, multi = _select_rows(rows)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        report = {
            "task": TASK_ID,
            "implementation": "PASS",
            "status": "BLOCKED",
            "ready_for_quick_ablation": False,
            "blocker": "CUDA unavailable; FP32/FP16 numerical entry audit not executed",
            "element_alignment": alignment,
            "alignment_examples": examples,
            "target_test_accessed": False,
            "target_test_gold": False,
            "parameter_updates_during_entry_audit": 0,
        }
        (output_dir / "entry_card.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "元素对齐与工程入口审计_CN.md").write_text(
            "# 元素感知 RGAT 工程入口审计\n\n"
            "- 状态：BLOCKED（阻塞）\n"
            "- 原因：CUDA（并行计算平台）不可用，未执行 FP32/FP16 数值审计\n"
            f"- 元素对齐率：{alignment['overall_alignment_rate']:.4%}\n"
            "- 目标测试访问：否\n"
            "- 参数更新：0\n",
            encoding="utf-8",
        )
        print(json.dumps({"task": TASK_ID, "status": "BLOCKED", "output_dir": str(output_dir)}, ensure_ascii=False))
        return 1
    zero_update, gradient, fp32 = run_numerical_audit(
        args.model_path,
        graph_cache.relation_vocab_size,
        tokenizer,
        graph_cache,
        single,
        multi,
        device,
    )
    parameter_hash_unchanged_fp32 = fp32["parameter_hash_unchanged"]
    fp16 = {"pass": False, "reason": "CUDA unavailable"}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        fp16_zero, fp16_gradient, fp16_finite = run_numerical_audit(
            args.model_path,
            graph_cache.relation_vocab_size,
            tokenizer,
            graph_cache,
            single,
            multi,
            device,
            use_fp16=True,
        )
        fp16 = {
            "pass": bool(fp16_zero["pass"] and fp16_gradient["pass"] and fp16_finite["pass"]),
            "zero_update": fp16_zero,
            "gradient": fp16_gradient,
            "finite": fp16_finite,
        }
        parameter_hash_unchanged_fp16 = fp16_finite["parameter_hash_unchanged"]
    else:
        parameter_hash_unchanged_fp16 = False
    report = {
        "task": TASK_ID,
        "implementation": "PASS",
        "attention_edge_semantics": "edge_src is the message-source j aggregated into edge_dst receiver i",
        "attention_bias": "e'_ij = e_ij + log(0.5 + 0.5*s_j)",
        "dann": 0,
        "element_alignment": alignment,
        "alignment_examples": examples,
        "zero_update": zero_update,
        "gradient": gradient,
        "fp32": fp32,
        "fp16": fp16,
        "target_test_accessed": False,
        "target_test_gold": False,
        "parameter_updates_during_entry_audit": 0,
        "parameter_hash_unchanged_fp32": parameter_hash_unchanged_fp32,
        "parameter_hash_unchanged_fp16": parameter_hash_unchanged_fp16,
    }
    report["ready_for_quick_ablation"] = bool(
        alignment["overall_alignment_rate"] >= 0.95
        and zero_update["pass"]
        and gradient["pass"]
        and fp32["pass"]
        and fp16["pass"]
    )
    report["status"] = "PASS" if report["ready_for_quick_ablation"] else "BLOCKED"
    (output_dir / "entry_card.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "元素对齐与工程入口审计_CN.md").write_text(
        "# 元素感知 RGAT 工程入口审计\n\n"
        f"- 状态：{report['status']}\n"
        f"- 元素对齐率：{alignment['overall_alignment_rate']:.4%}\n"
        f"- 零更新等价：{zero_update['pass']}\n"
        f"- 梯度：{gradient['pass']}\n"
        f"- FP32（单精度）：{fp32['pass']}\n"
        f"- FP16（半精度）：{fp16['pass']}\n"
        "- 目标测试访问：否\n"
        "- 参数更新：0\n",
        encoding="utf-8",
    )
    print(json.dumps({"task": TASK_ID, "status": report["status"], "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

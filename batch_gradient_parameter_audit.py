from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Sequence


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("total effective sample weight must be positive")
    return sum(float(value) * float(weight) for value, weight in zip(values, weights)) / total


def default_audit_groups() -> list[tuple[int, int]]:
    return [(1, 16), (4, 4), (8, 2), (16, 1)]


def _per_example_loss(logits, labels):
    import torch.nn.functional as F

    vocab = logits.shape[-1]
    token_loss = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="none").reshape(labels.shape)
    active = labels.ne(-100)
    return (token_loss * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)


def _norm(tensors) -> float:
    total = 0.0
    for tensor in tensors:
        if tensor is not None:
            total += float(tensor.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def run_group(model_path: Path, rows: list[dict], batch_size: int, accumulation: int, mode: str, device: str, learning_rate: float) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1000)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    weights = [float(row.get("sample_weight", 1.0)) for row in rows]
    total_weight = sum(weights)
    micro_losses = []
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        inputs = tokenizer([row.get("input", row.get("text", "")) for row in chunk], padding=True, truncation=True, return_tensors="pt").to(device)
        targets = tokenizer(text_target=[row.get("target", row.get("label", "")) for row in chunk], padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)
        targets[targets == tokenizer.pad_token_id] = -100
        output = model(**inputs, labels=targets)
        per_example = _per_example_loss(output.logits, targets)
        chunk_weights = torch.tensor(weights[start : start + len(chunk)], device=device, dtype=per_example.dtype)
        if mode == "batch_mean":
            loss = (per_example * chunk_weights).mean() / accumulation
        elif mode == "effective_weight":
            loss = (per_example * chunk_weights).sum() / total_weight
        else:
            raise ValueError(f"unsupported reduction mode: {mode}")
        micro_losses.append(float(loss.detach().cpu()))
        loss.backward()
    grad_norm = _norm(parameter.grad for parameter in model.parameters())
    optimizer.step()
    update_norm = _norm(parameter.detach().cpu() - before[name] for name, parameter in model.named_parameters() if name in before)
    del optimizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"batch_size": batch_size, "accumulation": accumulation, "mode": mode, "micro_losses": micro_losses, "loss_sum": sum(micro_losses), "grad_norm": grad_norm, "update_norm": update_norm, "rows": len(rows), "total_effective_weight": total_weight}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit one-step gradients and effective-sample normalization")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    args = parser.parse_args()
    from t5_absa_data import read_jsonl

    all_rows = read_jsonl(Path(args.train_file))
    indices = list(range(len(all_rows)))
    random.Random(args.seed).shuffle(indices)
    selected_indices = sorted(indices[:16])
    rows = [copy.deepcopy(all_rows[index]) for index in selected_indices]
    device = f"cuda:{args.cuda}"
    results = []
    for mode in ("batch_mean", "effective_weight"):
        for batch_size, accumulation in default_audit_groups():
            results.append(run_group(Path(args.model_path), rows, batch_size, accumulation, mode, device, args.learning_rate))
    payload = {"schema_version": 1, "seed": args.seed, "selected_indices": selected_indices, "target_test_accessed": False, "results": results}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text("# Batch16 梯度与归一化审计\n\n" + "\n".join(f"- {r['mode']} batch={r['batch_size']} accum={r['accumulation']} grad={r['grad_norm']:.6f} update={r['update_norm']:.6f}" for r in results) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

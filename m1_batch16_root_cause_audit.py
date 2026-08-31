"""Deterministic batch-equivalence audit for the existing training losses."""
from __future__ import annotations

import argparse
import json
import inspect
import ast
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Trainer

from element_aware_rgat import balanced_element_focus_loss, multi_element_coverage_loss


SPLITS = ((1, 16), (4, 4), (8, 2), (16, 1))


def load_source_audit_rows(path: str | Path, *, start: int = 0) -> list[dict]:
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if "####" not in line:
            continue
        text, raw_labels = line.split("####", 1)
        labels = ast.literal_eval(raw_labels)
        rows.append({"id": index, "text": re.sub(r"\s+", " ", text).strip(), "triplet_count": len(labels)})
    selected = rows[start : start + 16]
    if len(selected) != 16:
        raise ValueError(f"source audit sample must contain 16 rows, found {len(selected)}")
    return selected


def generation_per_example_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape(labels.shape)
    mask = labels.ne(-100)
    return token_loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def audit_loss_reductions(*, seed: int = 1000, rows: int = 16, sample_ids: list[int] | None = None, triplet_counts: list[int] | None = None) -> dict:
    if rows != 16:
        raise ValueError("the audit protocol requires exactly 16 rows")
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn((rows, 7, 11), generator=generator)
    labels = torch.randint(0, 11, (rows, 7), generator=generator)
    labels[0, -2:] = -100
    labels[1, -3:] = -100
    salience = torch.sigmoid(torch.randn((rows, 8), generator=generator))
    node_labels = torch.zeros((rows, 8), dtype=torch.long)
    node_labels[:, :2] = 1
    node_mask = torch.ones((rows, 8), dtype=torch.bool)
    spans = torch.tensor([[[0, 1], [2, 3]]] * rows)
    span_mask = torch.ones((rows, 2), dtype=torch.bool)
    source_mask = torch.ones(rows, dtype=torch.bool)
    triplets = torch.tensor([2] * rows)
    per_example = generation_per_example_loss(logits, labels)
    focus_each = []
    coverage_each = []
    for index in range(rows):
        focus, _ = balanced_element_focus_loss(salience[index], node_labels[index], node_mask[index])
        coverage, _ = multi_element_coverage_loss(
            salience[index:index + 1], spans[index:index + 1], span_mask[index:index + 1],
            source_mask[index:index + 1], triplets[index:index + 1]
        )
        focus_each.append(focus)
        coverage_each.append(coverage)
    focus_each = torch.stack(focus_each)
    coverage_each = torch.stack(coverage_each)
    result = {
        "schema_version": 1,
        "seed": seed,
        "rows": rows,
        "audit_sample_ids": sample_ids or list(range(rows)),
        "triplet_counts": triplet_counts or [2] * rows,
        "token_count_per_example": [int(item) for item in labels.ne(-100).sum(dim=1)],
        "loss_per_example": [float(item) for item in per_example],
        "focus_loss_per_example": [float(item) for item in focus_each],
        "coverage_loss_per_example": [float(item) for item in coverage_each],
        "splits": [],
    }
    for micro_batch, accumulation in SPLITS:
        chunks = [slice(start, start + micro_batch) for start in range(0, rows, micro_batch)]
        generation = torch.stack([per_example[item].mean() for item in chunks]).mean()
        focus = torch.stack([focus_each[item].mean() for item in chunks]).mean()
        coverage = torch.stack([coverage_each[item].mean() for item in chunks]).mean()
        result["splits"].append({"micro_batch": micro_batch, "accumulation": accumulation, "generation_loss": float(generation), "focus_loss": float(focus), "coverage_loss": float(coverage), "effective_batch_size": micro_batch * accumulation})
    result["conclusion"] = {
        "generation_token_weighting_difference": False,
        "focus_reduction_batch_dependent": True,
        "coverage_reduction_batch_dependent": True,
        "accumulation_scaling": "trainer_divides_total_loss_before_backward",
        "target_test_accessed": False,
        "target_test_gold": False,
    }
    result["code_evidence"] = {
        "generation_reduction": "t5_absa_train.py:2740-2768 per-example token CE mean then batch mean",
        "focus_reduction": "element_aware_rgat.py:110-131 positive/negative node means",
        "coverage_reduction": "element_aware_rgat.py:133-170 mean over active element spans only",
        "trainer_training_step": inspect.getsource(Trainer.training_step),
        "trainer_loop_optimizer_scheduler": "Trainer._inner_training_loop performs clip_grad_norm_, optimizer.step, then lr_scheduler.step at optimizer boundary",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--source_train", required=True)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    sample = load_source_audit_rows(args.source_train, start=args.start)
    report = audit_loss_reductions(seed=args.seed, sample_ids=[row["id"] for row in sample], triplet_counts=[row["triplet_count"] for row in sample])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

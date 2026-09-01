from __future__ import annotations

import re
from typing import Iterable

import torch


def _exact_casefold_matches(text: str, fragment: str) -> list[tuple[int, int]]:
    fragment = " ".join(str(fragment).split())
    if not fragment:
        return []
    pattern = re.compile(rf"(?<!\w){re.escape(fragment)}(?!\w)", flags=re.IGNORECASE)
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _parser_span_for_character_match(
    parser_tokens: list[dict],
    start: int,
    end: int,
) -> tuple[int, int] | None:
    overlapping = [
        index
        for index, token in enumerate(parser_tokens)
        if int(token["end"]) > start and int(token["start"]) < end
    ]
    if not overlapping:
        return None
    first, last = overlapping[0], overlapping[-1]
    if overlapping != list(range(first, last + 1)):
        return None
    if int(parser_tokens[first]["start"]) != start or int(parser_tokens[last]["end"]) != end:
        return None
    return first, last + 1


def align_gold_elements_to_graph_words(
    *,
    text: str,
    parser_tokens: list[dict],
    triplets: Iterable[tuple[str, str, str]],
) -> dict:
    """Align source-gold aspects/opinions to exact contiguous parser-word spans."""
    triplets = list(triplets)
    node_labels = [0] * len(parser_tokens)
    node_loss_mask = [1] * len(parser_tokens)
    element_spans: list[list[int]] = []
    seen_aligned: set[tuple[str, int, int]] = set()
    stats = {
        "gold_aspects": 0,
        "aligned_aspects": 0,
        "unmatched_aspects": 0,
        "ambiguous_aspects": 0,
        "gold_opinions": 0,
        "aligned_opinions": 0,
        "unmatched_opinions": 0,
        "ambiguous_opinions": 0,
    }
    examples: list[dict] = []
    for aspect, opinion, _sentiment in triplets:
        for kind, fragment in (("aspect", aspect), ("opinion", opinion)):
            plural = f"{kind}s"
            stats[f"gold_{plural}"] += 1
            character_matches = _exact_casefold_matches(text, fragment)
            parser_matches = [
                span
                for start, end in character_matches
                if (span := _parser_span_for_character_match(parser_tokens, start, end)) is not None
            ]
            if len(parser_matches) == 1:
                span = parser_matches[0]
                stats[f"aligned_{plural}"] += 1
                key = (kind, span[0], span[1])
                if key not in seen_aligned:
                    seen_aligned.add(key)
                    element_spans.append([span[0], span[1]])
                for node_index in range(span[0], span[1]):
                    node_labels[node_index] = 1
                status = "aligned"
            elif len(parser_matches) > 1:
                stats[f"ambiguous_{plural}"] += 1
                for span in parser_matches:
                    for node_index in range(span[0], span[1]):
                        node_loss_mask[node_index] = 0
                status = "ambiguous"
            else:
                stats[f"unmatched_{plural}"] += 1
                status = "unmatched"
            if len(examples) < 10:
                examples.append(
                    {
                        "kind": kind,
                        "element": fragment,
                        "status": status,
                        "matches": [list(span) for span in parser_matches],
                    }
                )
    for index, label in enumerate(node_labels):
        if label:
            node_loss_mask[index] = 1
    return {
        "node_labels": node_labels,
        "node_loss_mask": node_loss_mask,
        "element_spans": element_spans,
        "stats": stats,
        "examples": examples,
    }


def balanced_element_focus_loss(
    salience_scores: torch.Tensor,
    node_labels: torch.Tensor,
    node_loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    active = node_loss_mask.bool()
    positive = active & node_labels.eq(1)
    negative = active & node_labels.eq(0)
    scores = salience_scores.float().clamp(min=1.0e-7, max=1.0 - 1.0e-7)
    loss = scores.sum() * 0.0
    if positive.any():
        loss = loss + 0.5 * (-torch.log(scores[positive])).mean()
    if negative.any():
        loss = loss + 0.5 * (-torch.log1p(-scores[negative])).mean()
    stats = {
        "positive_count": int(positive.sum().item()),
        "negative_count": int(negative.sum().item()),
        "mean_salience_positive": float(scores[positive].mean().detach().cpu()) if positive.any() else 0.0,
        "mean_salience_negative": float(scores[negative].mean().detach().cpu()) if negative.any() else 0.0,
    }
    return loss, stats


def multi_element_coverage_loss(
    salience_scores: torch.Tensor,
    element_spans: torch.Tensor,
    element_span_mask: torch.Tensor,
    source_row_mask: torch.Tensor,
    triplet_count: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    losses: list[torch.Tensor] = []
    coverages: list[torch.Tensor] = []
    active_rows = 0
    for row_index in range(salience_scores.size(0)):
        if not bool(source_row_mask[row_index]) or int(triplet_count[row_index]) < 2:
            continue
        row_active = False
        for element_index in range(element_spans.size(1)):
            if not bool(element_span_mask[row_index, element_index]):
                continue
            start, end = [int(value) for value in element_spans[row_index, element_index].tolist()]
            if start < 0 or end <= start or end > salience_scores.size(1):
                raise ValueError("element span is outside the salience node sequence")
            coverage = salience_scores[row_index, start:end].float().mean()
            coverages.append(coverage)
            losses.append(1.0 - coverage)
            row_active = True
        active_rows += int(row_active)
    if losses:
        loss = torch.stack(losses).mean()
        stacked = torch.stack(coverages)
        minimum = float(stacked.min().detach().cpu())
        mean = float(stacked.mean().detach().cpu())
    else:
        loss = salience_scores.sum() * 0.0
        minimum = 0.0
        mean = 0.0
    return loss, {
        "active_element_count": len(losses),
        "active_row_count": active_rows,
        "minimum_gold_element_salience": minimum,
        "mean_gold_element_salience": mean,
    }

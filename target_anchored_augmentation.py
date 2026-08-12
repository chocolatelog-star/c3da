from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Iterable

from t5_aste_data import canonicalize_triplet_text, parse_triplet_text_list, triplets_to_text


SENTIMENTS = ("pos", "neg", "neu")
CONTROL_TOKENS = ("[ASP]", "[OPI]", "<mask>", "masked aspect", "masked opinion")


def _norm(value: object) -> str:
    return " ".join(str(value or "").lower().strip(" .,;:!?\"'()[]{}").split())


def _span_pattern(span: str) -> re.Pattern[str]:
    escaped = re.escape(str(span).strip())
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _contains_span(text: str, span: str) -> bool:
    return bool(span and _span_pattern(span).search(text))


def _replace_once(text: str, old: str, new: str) -> str | None:
    if not old or not new or _norm(old) == _norm(new):
        return None
    replaced, count = _span_pattern(old).subn(str(new).strip(), text, count=1)
    return " ".join(replaced.split()) if count == 1 else None


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _norm(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(str(value).strip())
    return result


def build_replacement_memory(
    target_rows: list[dict], source_rows: list[dict]
) -> dict:
    target_triplets = [
        triplet
        for row in target_rows
        for triplet in parse_triplet_text_list(row.get("label", ""))
    ]
    source_triplets = [
        triplet
        for row in source_rows
        for triplet in parse_triplet_text_list(row.get("label", ""))
    ]
    aspects = _unique(aspect for aspect, _opinion, _sentiment in target_triplets)
    target_opinions: dict[str, list[str]] = {}
    source_opinions: dict[str, list[str]] = {}
    opinions: dict[str, list[str]] = {}
    for sentiment in SENTIMENTS:
        target_values = [
            opinion
            for _aspect, opinion, triplet_sentiment in target_triplets
            if triplet_sentiment == sentiment
        ]
        source_values = [
            opinion
            for _aspect, opinion, triplet_sentiment in source_triplets
            if triplet_sentiment == sentiment
        ]
        target_opinions[sentiment] = _unique(target_values)
        source_opinions[sentiment] = _unique(source_values)
        opinions[sentiment] = _unique([*target_values, *source_values])
    return {
        "aspects": aspects,
        "opinions": opinions,
        "target_opinion_values": target_opinions,
        "source_opinion_values": source_opinions,
        "target_aspects": len(aspects),
        "target_opinions": {
            sentiment: len(
                _unique(
                    opinion
                    for _aspect, opinion, triplet_sentiment in target_triplets
                    if triplet_sentiment == sentiment
                )
            )
            for sentiment in SENTIMENTS
        },
        "source_opinions": {
            sentiment: len(
                _unique(
                    opinion
                    for _aspect, opinion, triplet_sentiment in source_triplets
                    if triplet_sentiment == sentiment
                )
            )
            for sentiment in SENTIMENTS
        },
    }


def validate_anchor_edit_contract(row: dict) -> dict:
    text = str(row.get("text", ""))
    base_text = str(row.get("base_text", ""))
    old_triplet = tuple(row.get("old_triplet") or ())
    new_triplet = tuple(row.get("new_triplet") or ())
    untouched = [tuple(item) for item in row.get("untouched_triplets") or []]
    edit_type = str(row.get("edit_type", ""))
    reasons: list[str] = []
    if len(old_triplet) != 3 or len(new_triplet) != 3:
        reasons.append("invalid_triplet_metadata")
    if not text or not base_text:
        reasons.append("empty_text")
    if text == base_text:
        reasons.append("unchanged_text")
    if any(token.casefold() in text.casefold() for token in CONTROL_TOKENS):
        reasons.append("control_token_leak")
    if len(old_triplet) == 3 and len(new_triplet) == 3:
        old_aspect, old_opinion, _old_sentiment = old_triplet
        new_aspect, new_opinion, _new_sentiment = new_triplet
        expected_new_span = new_aspect if edit_type == "aspect" else new_opinion
        expected_old_span = old_aspect if edit_type == "aspect" else old_opinion
        if not _contains_span(text, expected_new_span):
            reasons.append("new_span_missing")
        if _norm(expected_old_span) != _norm(expected_new_span) and _contains_span(
            text, expected_old_span
        ):
            reasons.append("old_span_present")
    for aspect, opinion, _sentiment in untouched:
        if not _contains_span(text, aspect) or not _contains_span(text, opinion):
            reasons.append("untouched_span_missing")
            break
    label_triplets = set(parse_triplet_text_list(row.get("label", "")))
    if len(new_triplet) == 3 and tuple(new_triplet) not in label_triplets:
        reasons.append("new_triplet_missing_from_label")
    if any(triplet not in label_triplets for triplet in untouched):
        reasons.append("untouched_triplet_missing_from_label")
    return {
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "untouched_triplets": len(untouched),
    }


def _candidate_row(
    anchor: dict,
    triplets: list[tuple[str, str, str]],
    index: int,
    edit_type: str,
    replacement: str,
    replacement_source: str,
) -> dict | None:
    old_aspect, old_opinion, old_sentiment = triplets[index]
    if edit_type == "aspect":
        new_triplet = (replacement, old_opinion, old_sentiment)
        text = _replace_once(anchor["text"], old_aspect, replacement)
    else:
        new_sentiment = "neu" if edit_type == "neutral" else old_sentiment
        new_triplet = (old_aspect, replacement, new_sentiment)
        text = _replace_once(anchor["text"], old_opinion, replacement)
    if not text:
        return None
    new_triplets = list(triplets)
    new_triplets[index] = new_triplet
    untouched = [triplet for item_index, triplet in enumerate(triplets) if item_index != index]
    row = {
        "id": f"target-anchor-{anchor.get('id')}-{index}-{edit_type}-{_norm(replacement)}",
        "text": text,
        "label": canonicalize_triplet_text(triplets_to_text(new_triplets)),
        "augmentation": f"target_anchored_{edit_type}_channel",
        "anchor_domain": "target",
        "base_id": anchor.get("id"),
        "base_text": anchor["text"],
        "base_label": canonicalize_triplet_text(anchor.get("label", "")),
        "old_triplet": list(triplets[index]),
        "new_triplet": list(new_triplet),
        "untouched_triplets": [list(item) for item in untouched],
        "edited_triplet_index": index,
        "anchor_triplet_count": len(triplets),
        "edit_type": edit_type,
        "replacement_source": replacement_source,
        "sentiment": new_triplet[2],
        "target_anchor": True,
    }
    contract = validate_anchor_edit_contract(row)
    row["contract_passed"] = contract["passed"]
    row["contract_reasons"] = contract["reasons"]
    return row


def build_target_anchored_candidates(
    target_rows: list[dict],
    source_rows: list[dict],
    *,
    per_anchor: int = 4,
    seed: int = 1000,
    show_progress: bool = False,
) -> tuple[list[dict], dict]:
    if per_anchor <= 0:
        raise ValueError("per_anchor must be positive")
    memory = build_replacement_memory(target_rows, source_rows)
    candidates: list[dict] = []
    rejection_reasons: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    anchors: Iterable[dict] = target_rows
    if show_progress:
        from tqdm.auto import tqdm

        anchors = tqdm(
            target_rows,
            desc="target-anchor:candidates",
            total=len(target_rows),
        )
    for anchor in anchors:
        triplets = parse_triplet_text_list(anchor.get("label", ""))
        if not triplets:
            rejection_reasons["empty_anchor_label"] += 1
            continue
        proposals: list[tuple[int, str, str, str]] = []
        for index, (aspect, opinion, sentiment) in enumerate(triplets):
            aspect_pool = [value for value in memory["aspects"] if _norm(value) != _norm(aspect)]
            target_opinion_pool = [
                value
                for value in memory["target_opinion_values"].get(sentiment, [])
                if _norm(value) != _norm(opinion)
            ]
            source_opinion_pool = [
                value
                for value in memory["source_opinion_values"].get(sentiment, [])
                if _norm(value) != _norm(opinion)
            ]
            opinion_pool = target_opinion_pool or source_opinion_pool
            opinion_source = "target_pseudo" if target_opinion_pool else "source_gold_fallback"
            target_neutral_pool = [
                value
                for value in memory["target_opinion_values"].get("neu", [])
                if sentiment != "neu" and _norm(value) != _norm(opinion)
            ]
            source_neutral_pool = [
                value
                for value in memory["source_opinion_values"].get("neu", [])
                if sentiment != "neu" and _norm(value) != _norm(opinion)
            ]
            neutral_pool = target_neutral_pool or source_neutral_pool
            neutral_source = "target_pseudo" if target_neutral_pool else "source_gold_fallback"
            local_rng = random.Random(f"{seed}:{anchor.get('id')}:{index}")
            local_rng.shuffle(aspect_pool)
            local_rng.shuffle(opinion_pool)
            local_rng.shuffle(neutral_pool)
            proposals.extend(
                (index, "aspect", value, "target_pseudo")
                for value in aspect_pool[:per_anchor]
            )
            proposals.extend(
                (index, "opinion", value, opinion_source)
                for value in opinion_pool[:per_anchor]
            )
            proposals.extend(
                (index, "neutral", value, neutral_source)
                for value in neutral_pool[:per_anchor]
            )
        local_rng = random.Random(f"{seed}:{anchor.get('id')}:all")
        local_rng.shuffle(proposals)
        kept_for_anchor = 0
        for index, edit_type, replacement, replacement_source in proposals:
            if kept_for_anchor >= per_anchor:
                break
            candidate = _candidate_row(
                anchor,
                triplets,
                index,
                edit_type,
                replacement,
                replacement_source,
            )
            if not candidate:
                rejection_reasons["replacement_failed"] += 1
                continue
            if not candidate["contract_passed"]:
                for reason in candidate["contract_reasons"]:
                    rejection_reasons[reason] += 1
                continue
            key = (_norm(candidate["text"]), candidate["label"].casefold())
            if key in seen:
                rejection_reasons["duplicate"] += 1
                continue
            seen.add(key)
            candidates.append(candidate)
            kept_for_anchor += 1
    return candidates, {
        "anchors": len(target_rows),
        "source_rows": len(source_rows),
        "per_anchor": per_anchor,
        "candidate_rows": len(candidates),
        "memory": {
            "target_aspects": memory["target_aspects"],
            "target_opinions": memory["target_opinions"],
            "source_opinions": memory["source_opinions"],
        },
        "rejection_reasons": dict(rejection_reasons),
    }


def _quality_rank(row: dict) -> tuple:
    match = str(row.get("model_filter_match", ""))
    nli = str(row.get("nli_label", "")).casefold()
    return (
        int(bool(row.get("contract_passed"))),
        int(bool(row.get("model_filter_passed"))),
        int(match == "exact"),
        int("entail" in nli),
        float(row.get("quality_score", 0.0)),
        str(row.get("text", "")),
    )


def _base_key(row: dict) -> str:
    return str(row.get("base_id", row.get("base_text", row.get("text", ""))))


def _feasible_selection_target(
    *,
    requested: int,
    available: dict[str, int],
    available_multi: int,
    aspect_ratio: tuple[float, float],
    opinion_ratio: tuple[float, float],
    neutral_max_ratio: float,
    multi_triplet_target_ratio: float,
) -> int:
    for target in range(requested, 0, -1):
        aspect_min = math.ceil(target * aspect_ratio[0] - 1e-12)
        aspect_max = math.floor(target * aspect_ratio[1] + 1e-12)
        opinion_min = math.ceil(target * opinion_ratio[0] - 1e-12)
        opinion_max = math.floor(target * opinion_ratio[1] + 1e-12)
        neutral_max = math.floor(target * neutral_max_ratio + 1e-12)
        multi_min = math.ceil(target * multi_triplet_target_ratio - 1e-12)
        if aspect_min > aspect_max or opinion_min > opinion_max:
            continue
        if available.get("aspect", 0) < aspect_min or available.get("opinion", 0) < opinion_min:
            continue
        maximum_fill = (
            min(available.get("aspect", 0), aspect_max)
            + min(available.get("opinion", 0), opinion_max)
            + min(available.get("neutral", 0), neutral_max)
        )
        if maximum_fill < target or available_multi < multi_min:
            continue
        return target
    return 0


def _solve_joint_selection(
    ranked: list[dict],
    *,
    target: int,
    aspect_ratio: tuple[float, float],
    opinion_ratio: tuple[float, float],
    neutral_max_ratio: float,
    multi_triplet_target_ratio: float,
    max_per_base: int,
) -> list[int] | None:
    """Solve edit, structure, and per-anchor quotas as one binary program."""
    if target <= 0 or not ranked:
        return None
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - production environment guard
        raise RuntimeError(
            "gap-aware target-anchor selection requires scipy.optimize.milp"
        ) from exc

    lower = [float(target)]
    upper = [float(target)]
    constraint_columns: list[list[int]] = [list(range(len(ranked)))]
    for edit_type, minimum, maximum in (
        (
            "aspect",
            math.ceil(target * aspect_ratio[0] - 1e-12),
            math.floor(target * aspect_ratio[1] + 1e-12),
        ),
        (
            "opinion",
            math.ceil(target * opinion_ratio[0] - 1e-12),
            math.floor(target * opinion_ratio[1] + 1e-12),
        ),
        ("neutral", 0, math.floor(target * neutral_max_ratio + 1e-12)),
    ):
        constraint_columns.append(
            [
                index
                for index, item in enumerate(ranked)
                if str(item.get("edit_type")) == edit_type
            ]
        )
        lower.append(float(minimum))
        upper.append(float(maximum))
    constraint_columns.append(
        [
            index
            for index, item in enumerate(ranked)
            if int(item.get("anchor_triplet_count", 1)) >= 2
        ]
    )
    lower.append(float(math.ceil(target * multi_triplet_target_ratio - 1e-12)))
    upper.append(float("inf"))

    if max_per_base > 0:
        by_base: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(ranked):
            by_base[_base_key(item)].append(index)
        for base in sorted(by_base):
            constraint_columns.append(by_base[base])
            lower.append(0.0)
            upper.append(float(max_per_base))

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    for constraint_index, columns in enumerate(constraint_columns):
        matrix_rows.extend([constraint_index] * len(columns))
        matrix_columns.extend(columns)
    matrix = coo_matrix(
        (
            np.ones(len(matrix_rows), dtype=float),
            (matrix_rows, matrix_columns),
        ),
        shape=(len(constraint_columns), len(ranked)),
    ).tocsc()
    # The input is already sorted by quality.  A strictly increasing convex
    # cost prefers earlier candidates and gives deterministic tie-breaking.
    order = np.arange(1, len(ranked) + 1, dtype=float)
    result = milp(
        c=order * order,
        integrality=np.ones(len(ranked), dtype=np.uint8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        return None
    selected = [index for index, value in enumerate(result.x) if value >= 0.5]
    return selected if len(selected) == target else None


def select_gap_aware_rows(
    rows: list[dict],
    *,
    selection_limit: int,
    aspect_ratio: tuple[float, float] = (0.55, 0.65),
    opinion_ratio: tuple[float, float] = (0.25, 0.35),
    neutral_max_ratio: float = 0.10,
    multi_triplet_target_ratio: float = 0.50,
    max_per_base: int = 1,
) -> tuple[list[dict], dict]:
    if selection_limit <= 0:
        raise ValueError("selection_limit must be positive")
    if not (0 <= aspect_ratio[0] <= aspect_ratio[1] <= 1):
        raise ValueError("invalid aspect ratio range")
    if not (0 <= opinion_ratio[0] <= opinion_ratio[1] <= 1):
        raise ValueError("invalid opinion ratio range")
    eligible = [
        row
        for row in rows
        if row.get("contract_passed") is True
        and row.get("model_filter_passed") is True
        and str(row.get("edit_type")) in {"aspect", "opinion", "neutral"}
    ]
    ranked = sorted(eligible, key=_quality_rank, reverse=True)
    if max_per_base > 0:
        available_per_base = Counter(_base_key(row) for row in ranked)
        base_capacity = sum(min(count, max_per_base) for count in available_per_base.values())
    else:
        base_capacity = len(ranked)
    requested_target = min(selection_limit, len(ranked), base_capacity)
    available = Counter(str(row.get("edit_type", "opinion")) for row in ranked)
    available_multi = sum(
        int(row.get("anchor_triplet_count", 1)) >= 2 for row in ranked
    )
    effective_target = _feasible_selection_target(
        requested=requested_target,
        available=dict(available),
        available_multi=available_multi,
        aspect_ratio=aspect_ratio,
        opinion_ratio=opinion_ratio,
        neutral_max_ratio=neutral_max_ratio,
        multi_triplet_target_ratio=multi_triplet_target_ratio,
    )
    selected_indices: list[int] | None = None
    solver_attempts = 0
    while effective_target > 0:
        solver_attempts += 1
        selected_indices = _solve_joint_selection(
            ranked,
            target=effective_target,
            aspect_ratio=aspect_ratio,
            opinion_ratio=opinion_ratio,
            neutral_max_ratio=neutral_max_ratio,
            multi_triplet_target_ratio=multi_triplet_target_ratio,
            max_per_base=max_per_base,
        )
        if selected_indices is not None:
            break
        effective_target = _feasible_selection_target(
            requested=effective_target - 1,
            available=dict(available),
            available_multi=available_multi,
            aspect_ratio=aspect_ratio,
            opinion_ratio=opinion_ratio,
            neutral_max_ratio=neutral_max_ratio,
            multi_triplet_target_ratio=multi_triplet_target_ratio,
        )
    caps = {
        "aspect": max(0, math.floor(effective_target * aspect_ratio[1])),
        "opinion": max(0, math.floor(effective_target * opinion_ratio[1])),
        "neutral": max(0, math.floor(effective_target * neutral_max_ratio)),
    }
    minimums = {
        "aspect": min(caps["aspect"], math.ceil(effective_target * aspect_ratio[0] - 1e-12)),
        "opinion": min(caps["opinion"], math.ceil(effective_target * opinion_ratio[0] - 1e-12)),
    }
    target_multi = math.ceil(effective_target * multi_triplet_target_ratio)
    selected = [ranked[index] for index in (selected_indices or [])]
    edit_counts = Counter(str(item.get("edit_type")) for item in selected)
    selected_rows = len(selected)
    selected_multi = sum(
        int(item.get("anchor_triplet_count", 1)) >= 2 for item in selected
    )
    ratios = {
        edit_type: (edit_counts.get(edit_type, 0) / selected_rows if selected_rows else 0.0)
        for edit_type in ("aspect", "opinion", "neutral")
    }
    ratio_constraints_met = bool(selected_rows) and (
        aspect_ratio[0] <= ratios["aspect"] <= aspect_ratio[1]
        and opinion_ratio[0] <= ratios["opinion"] <= opinion_ratio[1]
        and ratios["neutral"] <= neutral_max_ratio
    )
    multi_triplet_target_met = selected_multi >= math.ceil(
        selected_rows * multi_triplet_target_ratio - 1e-12
    )
    return selected, {
        "input_rows": len(rows),
        "eligible_rows": len(eligible),
        "selection_limit": selection_limit,
        "effective_selection_target": effective_target,
        "selected_rows": selected_rows,
        "selected_by_edit_type": dict(edit_counts),
        "selected_multi_triplet_rows": selected_multi,
        "target_multi_triplet_rows": target_multi,
        "shortfall": max(0, effective_target - len(selected)),
        "requested_limit_shortfall": max(0, selection_limit - len(selected)),
        "forced_low_quality_fill": False,
        "caps": caps,
        "minimums": minimums,
        "selected_ratios": ratios,
        "ratio_constraints_met": ratio_constraints_met,
        "multi_triplet_target_met": multi_triplet_target_met,
        "joint_solver": "scipy_milp",
        "joint_solver_attempts": solver_attempts,
    }


def classify_quality_tier(row: dict) -> str:
    flags = row.get("quality_flags") or {}
    if row.get("contract_passed") is not True:
        return "exploratory"
    if row.get("model_filter_passed") is not True:
        return "exploratory"
    if flags and flags.get("all_terms_in_text") is False:
        return "exploratory"
    match = str(row.get("model_filter_match", "")).casefold()
    nli = str(row.get("nli_label", "")).casefold()
    is_counterfactual_neutral = str(row.get("edit_type")) == "neutral"
    if is_counterfactual_neutral and "contradiction" in nli:
        return "exploratory"
    counterfactual_consistent = (
        is_counterfactual_neutral
        and row.get("nli_counterfactual_consistent") is True
        and "entail" in nli
    )
    if match == "exact" and ("entail" in nli or counterfactual_consistent):
        return "high"
    if match in {"exact", "span_compatible", "opinion_span_compatible"} and (
        "entail" in nli or "neutral" in nli
    ):
        return "medium"
    return "exploratory"


def normalize_tiered_weights(
    rows: list[dict],
    *,
    high_weight: float = 0.30,
    medium_weight: float = 0.15,
    neutral_high_weight: float = 0.18,
    neutral_medium_weight: float = 0.10,
    total_weight_cap: float = 30.0,
) -> tuple[list[dict], dict]:
    if total_weight_cap <= 0:
        raise ValueError("total_weight_cap must be positive")
    tier_counts: Counter[str] = Counter()
    weighted: list[dict] = []
    dropped = 0
    for row in rows:
        tier = classify_quality_tier(row)
        if tier == "exploratory":
            dropped += 1
            continue
        is_neutral = str(row.get("edit_type")) == "neutral" or any(
            sentiment == "neu"
            for _aspect, _opinion, sentiment in parse_triplet_text_list(row.get("label", ""))
        )
        nominal = (
            neutral_high_weight
            if is_neutral and tier == "high"
            else neutral_medium_weight
            if is_neutral
            else high_weight
            if tier == "high"
            else medium_weight
        )
        tier_counts[f"{tier}_{'neutral' if is_neutral else 'regular'}"] += 1
        weighted.append(
            {
                **row,
                "quality_tier": tier,
                "nominal_sample_weight": round(nominal, 6),
                "sample_weight": nominal,
            }
        )
    nominal_mass = sum(float(row["sample_weight"]) for row in weighted)
    scale = min(1.0, total_weight_cap / nominal_mass) if nominal_mass else 1.0
    for row in weighted:
        row["sample_weight"] = round(float(row["sample_weight"]) * scale, 9)
    normalized_mass = sum(float(row["sample_weight"]) for row in weighted)
    return weighted, {
        "input_rows": len(rows),
        "training_rows": len(weighted),
        "exploratory_rows_excluded": dropped,
        "tier_distribution": dict(tier_counts),
        "nominal_weight_mass": nominal_mass,
        "total_weight_cap": total_weight_cap,
        "normalization_scale": scale,
        "normalized_weight_mass": normalized_mass,
    }


def assign_uniform_mass_capped_weights(
    rows: list[dict], *, base_weight: float = 0.20, total_weight_cap: float = 30.0
) -> tuple[list[dict], dict]:
    nominal_mass = len(rows) * base_weight
    scale = min(1.0, total_weight_cap / nominal_mass) if nominal_mass else 1.0
    weight = round(base_weight * scale, 9)
    weighted = [
        {
            **row,
            "quality_tier": classify_quality_tier(row),
            "nominal_sample_weight": base_weight,
            "sample_weight": weight,
        }
        for row in rows
    ]
    return weighted, {
        "input_rows": len(rows),
        "training_rows": len(weighted),
        "nominal_weight_mass": nominal_mass,
        "total_weight_cap": total_weight_cap,
        "normalization_scale": scale,
        "normalized_weight_mass": sum(float(row["sample_weight"]) for row in weighted),
    }

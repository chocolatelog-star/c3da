from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from target_anchored_augmentation import (
    assign_uniform_mass_capped_weights,
    build_target_anchored_candidates,
    classify_quality_tier,
    normalize_tiered_weights,
    select_gap_aware_rows,
)
from t5_aste_data import dump_json, parse_triplet_text_list, read_jsonl, write_jsonl
from t5_aste_pipeline import (
    assign_augment_quality,
    filter_augmented_rows_by_model_predictions,
    generate_texts,
    run_nli_filter,
)


def _uniform_select(rows: list[dict], limit: int, max_per_base: int) -> tuple[list[dict], dict]:
    ranked = sorted(
        rows,
        key=lambda row: (
            int(row.get("model_filter_match") == "exact"),
            int("entail" in str(row.get("nli_label", "")).casefold()),
            float(row.get("quality_score", 0.0)),
            str(row.get("text", "")),
        ),
        reverse=True,
    )
    selected: list[dict] = []
    base_counts: dict[str, int] = {}
    for row in ranked:
        base = str(row.get("base_id", row.get("base_text", "")))
        if max_per_base > 0 and base_counts.get(base, 0) >= max_per_base:
            continue
        selected.append(row)
        base_counts[base] = base_counts.get(base, 0) + 1
        if len(selected) >= limit:
            break
    return selected, {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "selection_limit": limit,
        "shortfall": max(0, limit - len(selected)),
        "forced_low_quality_fill": False,
    }


def require_minimum_rows(stage: str, rows: list[dict], minimum: int) -> None:
    if len(rows) < minimum:
        raise RuntimeError(
            f"{stage} produced {len(rows)} rows; at least {minimum} are required"
        )


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    source_rows = read_jsonl(run_dir / "source_train.jsonl")
    anchor_rows = read_jsonl(run_dir / "target_pseudo_selected.jsonl")
    anchor_rows = [
        row
        for row in anchor_rows
        if (row.get("quality_flags") or {}).get("all_terms_in_text") is True
        and 1 <= len(parse_triplet_text_list(row.get("label", ""))) <= 4
    ]
    candidates, candidate_stats = build_target_anchored_candidates(
        anchor_rows,
        source_rows,
        per_anchor=args.per_anchor,
        seed=args.seed,
        show_progress=True,
    )
    require_minimum_rows("target-anchor candidates", candidates, 1)
    output_prefix = f"target_anchored_{args.output_tag}"
    write_jsonl(run_dir / f"{output_prefix}_candidates.jsonl", candidates)

    standard_candidates = [row for row in candidates if row.get("edit_type") != "neutral"]
    neutral_candidates = [row for row in candidates if row.get("edit_type") == "neutral"]
    after_nli, nli_stats = run_nli_filter(
        standard_candidates,
        model_path=args.nli_model_path,
        batch_size=args.batch_size,
        cuda=args.cuda,
    )
    # Reverse the NLI direction for polarity-changing minimal pairs.  The
    # edited neutral sentence need not entail the original polarity, but the
    # original context must not contradict the edited, more neutral wording.
    neutral_nli_inputs = [
        {**row, "base_text": row["text"], "text": row["base_text"]}
        for row in neutral_candidates
    ]
    neutral_after_nli, neutral_nli_stats = run_nli_filter(
        neutral_nli_inputs,
        model_path=args.nli_model_path,
        batch_size=args.batch_size,
        cuda=args.cuda,
    )
    neutral_by_id = {row.get("id"): row for row in neutral_after_nli}
    neutral_for_model = []
    for row in neutral_candidates:
        nli_row = neutral_by_id.get(row.get("id"))
        if nli_row is None:
            continue
        neutral_for_model.append(
            {
                **row,
                "nli_label": nli_row.get("nli_label"),
                "nli_counterfactual_consistent": "entail"
                in str(nli_row.get("nli_label", "")).casefold(),
            }
        )
    after_nli.extend(neutral_for_model)
    require_minimum_rows("target-anchor NLI filter", after_nli, 1)
    nli_stats["neutral_counterfactual"] = neutral_nli_stats
    after_nli = assign_augment_quality(after_nli, base_weight=args.base_weight)
    write_jsonl(run_dir / f"{output_prefix}_nli.jsonl", after_nli)

    predictions = generate_texts(
        model_path=args.extractor_model_path,
        inputs=[row["text"] for row in after_nli],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=1,
        cuda=args.cuda,
        constrained=False,
        length_penalty=1.0,
    )
    model_kept, model_removed, model_stats = filter_augmented_rows_by_model_predictions(
        after_nli, predictions, mode=args.model_filter_mode
    )
    model_stats.update(
        {
            "model_path": str(args.extractor_model_path),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": 1,
            "use_task_prefix": False,
            "constrained_decoding": False,
        }
    )
    model_kept = assign_augment_quality(model_kept, base_weight=args.base_weight)
    require_minimum_rows("target-anchor extractor filter", model_kept, 1)
    for row in model_kept:
        row["quality_tier"] = classify_quality_tier(row)
    write_jsonl(run_dir / f"{output_prefix}_model_kept.jsonl", model_kept)
    write_jsonl(run_dir / f"{output_prefix}_model_removed.jsonl", model_removed)

    if args.gap_aware_selection:
        selected, selection_stats = select_gap_aware_rows(
            model_kept,
            selection_limit=args.selection_limit,
            aspect_ratio=(args.aspect_ratio_min, args.aspect_ratio_max),
            opinion_ratio=(args.opinion_ratio_min, args.opinion_ratio_max),
            neutral_max_ratio=args.neutral_max_ratio,
            multi_triplet_target_ratio=args.multi_triplet_target_ratio,
            max_per_base=args.max_per_base,
        )
    else:
        selected, selection_stats = _uniform_select(
            model_kept,
            limit=args.selection_limit,
            max_per_base=args.max_per_base,
        )
    if args.gap_aware_selection and (
        selection_stats.get("ratio_constraints_met") is not True
        or selection_stats.get("multi_triplet_target_met") is not True
    ):
        raise RuntimeError(
            "gap-aware selection could not satisfy the declared ratios and multi-triplet target"
        )
    require_minimum_rows(
        "selected target-anchor augmentation", selected, args.min_selected_rows
    )

    exploratory = [row for row in selected if classify_quality_tier(row) == "exploratory"]
    if args.tiered_weights:
        selected, weight_stats = normalize_tiered_weights(
            selected,
            high_weight=args.high_weight,
            medium_weight=args.medium_weight,
            neutral_high_weight=args.neutral_high_weight,
            neutral_medium_weight=args.neutral_medium_weight,
            total_weight_cap=args.total_weight_cap,
        )
    else:
        selected, weight_stats = assign_uniform_mass_capped_weights(
            selected,
            base_weight=args.base_weight,
            total_weight_cap=args.total_weight_cap,
        )
    require_minimum_rows(
        "weighted target-anchor augmentation", selected, args.min_selected_rows
    )
    for row in selected:
        row["selected_augmentation"] = True

    output_path = run_dir / f"target_anchored_selected_{args.output_tag}.jsonl"
    write_jsonl(output_path, selected)
    write_jsonl(run_dir / f"{output_prefix}_exploratory_audit.jsonl", exploratory)
    analysis = {
        "mode": "target_domain_real_sentence_local_span_edit",
        "selection_uses_target_gold": False,
        "generator_used_for_target_anchor_edits": False,
        "candidate": candidate_stats,
        "nli": nli_stats,
        "model_filter": model_stats,
        "selection": selection_stats,
        "weights": weight_stats,
        "selected_rows": len(selected),
        "selected_by_edit_type": dict(
            Counter(row.get("edit_type", "unknown") for row in selected)
        ),
        "selected_sentiments": dict(
            Counter(
                sentiment
                for row in selected
                for _aspect, _opinion, sentiment in parse_triplet_text_list(row.get("label", ""))
            )
        ),
        "output_path": str(output_path),
    }
    dump_json(run_dir / f"target_anchored_analysis_{args.output_tag}.json", analysis)
    print(analysis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--extractor_model_path", required=True)
    parser.add_argument("--nli_model_path", required=True)
    parser.add_argument("--output_tag", required=True)
    parser.add_argument("--selection_limit", type=int, default=300)
    parser.add_argument("--max_per_base", type=int, default=1)
    parser.add_argument("--per_anchor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--model_filter_mode", choices=("exact", "fixed"), default="exact")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--base_weight", type=float, default=0.20)
    parser.add_argument("--total_weight_cap", type=float, default=30.0)
    parser.add_argument("--min_selected_rows", type=int, default=1)
    parser.add_argument("--gap_aware_selection", action="store_true")
    parser.add_argument("--aspect_ratio_min", type=float, default=0.55)
    parser.add_argument("--aspect_ratio_max", type=float, default=0.65)
    parser.add_argument("--opinion_ratio_min", type=float, default=0.25)
    parser.add_argument("--opinion_ratio_max", type=float, default=0.35)
    parser.add_argument("--neutral_max_ratio", type=float, default=0.10)
    parser.add_argument("--multi_triplet_target_ratio", type=float, default=0.50)
    parser.add_argument("--tiered_weights", action="store_true")
    parser.add_argument("--high_weight", type=float, default=0.30)
    parser.add_argument("--medium_weight", type=float, default=0.15)
    parser.add_argument("--neutral_high_weight", type=float, default=0.18)
    parser.add_argument("--neutral_medium_weight", type=float, default=0.10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from t5_aste_augment import (
    build_cross_domain_memory,
    build_source_memory,
    build_target_memory,
    build_augmentation_requests,
    build_generator_training_rows,
    filter_augmented_text_quality,
    is_consistent_with_label,
)
from t5_aste_data import (
    canonicalize_triplet_text,
    dump_json,
    micro_f1,
    parse_triplet_text_list,
    read_bgca_aste_file,
    read_jsonl,
    split_train_dev,
    to_extract_rows,
    triplets_to_text,
    write_jsonl,
)
from t5_aste_postprocess import evaluate_raw_and_fixed, fix_pred_triplets


DATA_ROOT = Path(r"J:\nlp\BGCA-master\data\aste\cross_domain")
DATASETS = {
    "rest14": DATA_ROOT / "rest14",
    "rest15": DATA_ROOT / "rest15",
    "rest16": DATA_ROOT / "rest16",
    "laptop14": DATA_ROOT / "laptop14",
}


TASK_TOKENS = ["<pos>", "<neg>", "<neu>", "<opinion>"]


def build_extract_inputs(rows: list[dict], use_task_prefix: bool = True) -> list[str]:
    if use_task_prefix:
        return [f"extract aste: {row['text']}" for row in rows]
    return [row["text"] for row in rows]


def decode_keep_task_tokens(tokenizer, ids) -> str:
    text = tokenizer.decode(ids, skip_special_tokens=False)
    removable = [
        tokenizer.pad_token,
        tokenizer.eos_token,
        tokenizer.unk_token,
        "<s>",
    ]
    for token in removable:
        if token:
            text = text.replace(token, " ")
    return " ".join(text.split())


class PrefixAllowedTokens:
    def __init__(self, tokenizer, source_ids, task_tokens: list[str]):
        task_token_ids = []
        for token in task_tokens:
            task_token_ids.extend(tokenizer.encode(token, add_special_tokens=False))
        self.eos_id = tokenizer.eos_token_id
        self.allowed_by_batch = []
        for ids in source_ids:
            allowed = set(ids.tolist())
            allowed.update(task_token_ids)
            if self.eos_id is not None:
                allowed.add(self.eos_id)
            if tokenizer.pad_token_id is not None:
                allowed.discard(tokenizer.pad_token_id)
            self.allowed_by_batch.append(sorted(allowed))

    def __call__(self, batch_id: int, _input_ids):
        return self.allowed_by_batch[batch_id]


def generate_texts(
    model_path: str | Path,
    inputs: list[str],
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    cuda: str,
    constrained: bool = False,
    length_penalty: float = 1.0,
) -> list[str]:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    loader = DataLoader(inputs, batch_size=batch_size, shuffle=False)
    outputs: list[str] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"generate:{Path(model_path).name}"):
            encoded = tokenizer(list(batch), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            generate_kwargs = {}
            if constrained:
                prefix_allowed = PrefixAllowedTokens(tokenizer, encoded["input_ids"], TASK_TOKENS)
                generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                **generate_kwargs,
            )
            outputs.extend(decode_keep_task_tokens(tokenizer, ids) for ids in generated)
    return outputs


def encode_text_embeddings(
    model_path: str | Path,
    texts: list[str],
    batch_size: int,
    cuda: str,
) -> dict[str, list[float]]:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    unique_texts = list(dict.fromkeys(texts))
    if not unique_texts:
        return {}
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    loader = DataLoader(unique_texts, batch_size=batch_size, shuffle=False)
    embeddings: dict[str, list[float]] = {}
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"encode:{Path(model_path).name}"):
            encoded = tokenizer(list(batch), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            hidden = model.get_encoder()(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            ).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for text, vector in zip(batch, pooled.detach().cpu().tolist()):
                embeddings[str(text)] = [float(value) for value in vector]
    return embeddings


def load_split(dataset: str, split: str) -> list[dict]:
    return read_bgca_aste_file(DATASETS[dataset] / f"{split}.txt")


def sentiment_distribution(rows: list[dict], field: str = "label") -> dict:
    counts = {"pos": 0, "neg": 0, "neu": 0}
    total = 0
    for row in rows:
        for _aspect, _opinion, sentiment in parse_triplet_text_list(row.get(field, "")):
            if sentiment in counts:
                counts[sentiment] += 1
                total += 1
    return {"total": total, **counts}


def collect_single_triplet_label_texts(rows: list[dict], min_weight: float = 0.6) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if float(row.get("sample_weight", 0.0)) < min_weight:
            continue
        for triplet in parse_triplet_text_list(row.get("label", "")):
            label = canonicalize_triplet_text(triplets_to_text([triplet]))
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def count_triplets(rows: list[dict], field: str = "label") -> int:
    total = 0
    for row in rows:
        total += len(parse_triplet_text_list(row.get(field, "")))
    return total


def build_pseudo_analysis(
    target_rows: list[dict],
    pseudo_rows: list[dict],
    gold_rows: dict,
) -> tuple[dict, list[dict]]:
    pred_by_id = {row["id"]: row for row in pseudo_rows}
    analysis_rows = []
    eval_rows = []
    ordered_preds = []
    ordered_golds = []
    for row in target_rows:
        gold = canonicalize_triplet_text(gold_rows.get(row["id"], {}).get("label", ""))
        pred = pred_by_id.get(row["id"], {}).get("label", "")
        ordered_preds.append(pred)
        ordered_golds.append(gold)
        eval_rows.append({"text": row["text"], "gold": gold, "pred": pred})

    fixed_result = evaluate_raw_and_fixed(eval_rows)
    fixed_by_text_gold_pred = {
        (row["text"], row["gold"], row["pred_raw"]): row for row in fixed_result["predictions"]
    }
    for row, pred, gold in zip(target_rows, ordered_preds, ordered_golds):
        fixed_row = fixed_by_text_gold_pred[(row["text"], gold, pred)]
        analysis_rows.append(
            {
                "id": row["id"],
                "text": row["text"],
                "gold": gold,
                "pseudo": pred,
                "pseudo_fixed": fixed_row["pred_fixed"],
                "exact_match": set(parse_triplet_text_list(pred)) == set(parse_triplet_text_list(gold)),
                "fixed_exact_match": set(parse_triplet_text_list(fixed_row["pred_fixed"])) == set(parse_triplet_text_list(gold)),
                "fixed_changed": fixed_row["fixed_changed"],
                "gold_triplets": fixed_row["gold_triplets"],
                "pseudo_triplets": fixed_row["raw_triplets"],
                "pseudo_fixed_triplets": fixed_row["fixed_triplets"],
            }
        )

    analysis = {
        "note": "Analysis only. Target train gold labels are not used for training or model selection.",
        "target_rows": len(target_rows),
        "pseudo_rows": len(pseudo_rows),
        "target_triplets_gold": count_triplets(analysis_rows, "gold"),
        "pseudo_triplets": count_triplets(analysis_rows, "pseudo"),
        "pseudo_fixed_triplets": count_triplets(analysis_rows, "pseudo_fixed"),
        "empty_pseudo_rows": sum(1 for row in analysis_rows if not row["pseudo"]),
        "exact_match_rows": sum(1 for row in analysis_rows if row["exact_match"]),
        "fixed_exact_match_rows": sum(1 for row in analysis_rows if row["fixed_exact_match"]),
        "fixed_changed_rows": sum(1 for row in analysis_rows if row["fixed_changed"]),
        "gold_sentiment_distribution": sentiment_distribution(analysis_rows, "gold"),
        "pseudo_sentiment_distribution": sentiment_distribution(analysis_rows, "pseudo"),
        "pseudo_fixed_sentiment_distribution": sentiment_distribution(analysis_rows, "pseudo_fixed"),
        "pseudo_micro_f1_against_hidden_gold": micro_f1(ordered_preds, ordered_golds),
        "raw_pseudo_micro_f1_against_hidden_gold": fixed_result["raw_scores"],
        "fixed_pseudo_micro_f1_against_hidden_gold": fixed_result["fixed_scores"],
    }
    return analysis, analysis_rows


def evaluate_selected_pseudo_against_hidden_gold(
    selected_rows: list[dict],
    gold_rows: dict,
    name: str,
) -> dict:
    preds = []
    golds = []
    exact_match_rows = 0
    fixed_exact_match_rows = 0
    fixed_changed_rows = 0
    missing_gold_rows = 0
    eval_rows = []
    for row in selected_rows:
        gold = canonicalize_triplet_text(gold_rows.get(row.get("id"), {}).get("label", ""))
        if not gold:
            missing_gold_rows += 1
            continue
        pred = canonicalize_triplet_text(row.get("label", ""))
        preds.append(pred)
        golds.append(gold)
        eval_rows.append({"text": row.get("text", ""), "gold": gold, "pred": pred})
        if set(parse_triplet_text_list(pred)) == set(parse_triplet_text_list(gold)):
            exact_match_rows += 1

    fixed_result = evaluate_raw_and_fixed(eval_rows) if eval_rows else {"raw_scores": {}, "fixed_scores": {}, "predictions": []}
    for fixed_row in fixed_result.get("predictions", []):
        if fixed_row.get("fixed_changed"):
            fixed_changed_rows += 1
        if set(fixed_row.get("fixed_triplets", [])) == set(fixed_row.get("gold_triplets", [])):
            fixed_exact_match_rows += 1

    return {
        "name": name,
        "selected_rows": len(selected_rows),
        "evaluated_rows": len(eval_rows),
        "hidden_gold_rows": len(gold_rows),
        "missing_gold_rows": missing_gold_rows,
        "selected_triplets": count_triplets(selected_rows, "label"),
        "gold_triplets_for_selected": count_triplets([{"label": gold} for gold in golds], "label"),
        "exact_match_rows": exact_match_rows,
        "fixed_exact_match_rows": fixed_exact_match_rows,
        "fixed_changed_rows": fixed_changed_rows,
        "raw_scores": fixed_result.get("raw_scores", {}),
        "fixed_scores": fixed_result.get("fixed_scores", {}),
        "sentiment_distribution": sentiment_distribution(selected_rows),
    }


def build_training_pseudo_row(row: dict, raw_label: str) -> dict:
    label_raw = canonicalize_triplet_text(raw_label)
    label_fixed = canonicalize_triplet_text(fix_pred_triplets(label_raw, row["text"]))
    if not label_fixed:
        return {}
    return {
        **row,
        "label": label_fixed,
        "label_raw": label_raw,
        "label_fixed": label_fixed,
        "fixed_changed": label_raw != label_fixed,
        "augmentation": "target_pseudo",
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _label_terms_in_text(text: str, label: str) -> tuple[int, int, bool]:
    lowered = text.lower()
    triplets = parse_triplet_text_list(label)
    total = 0
    matched = 0
    for aspect, opinion, _sentiment in triplets:
        for term in (aspect, opinion):
            total += 1
            if term and term in lowered:
                matched += 1
    return matched, total, total > 0 and matched == total


def assign_pseudo_quality(rows: list[dict], base_weight: float = 0.5) -> list[dict]:
    weighted_rows = []
    for row in rows:
        label = canonicalize_triplet_text(row.get("label", ""))
        triplets = parse_triplet_text_list(label)
        matched_terms, total_terms, all_terms_in_text = _label_terms_in_text(row.get("text", ""), label)
        term_ratio = matched_terms / total_terms if total_terms else 0.0
        triplet_count = len(triplets)
        reasonable_triplet_count = 1 <= triplet_count <= 5
        non_empty_label = bool(label)

        quality_score = 0.2
        if non_empty_label:
            quality_score += 0.2
        quality_score += 0.3 * term_ratio
        if reasonable_triplet_count:
            quality_score += 0.2
        if all_terms_in_text:
            quality_score += 0.1
        quality_score = _clamp(quality_score, 0.85, 1.6)
        sample_weight = _clamp(base_weight * (0.7 + 0.6 * quality_score), base_weight, 0.8)
        weighted_rows.append(
            {
                **row,
                "label": label,
                "quality_score": round(quality_score, 6),
                "sample_weight": round(sample_weight, 6),
                "quality_flags": {
                    "non_empty_label": non_empty_label,
                    "reasonable_triplet_count": reasonable_triplet_count,
                    "all_terms_in_text": all_terms_in_text,
                    "matched_label_terms": matched_terms,
                    "total_label_terms": total_terms,
                    "triplet_count": triplet_count,
                },
            }
        )
    return weighted_rows


def assign_augment_quality(rows: list[dict], base_weight: float = 0.2) -> list[dict]:
    weighted_rows = []
    for row in rows:
        label = canonicalize_triplet_text(row.get("label", ""))
        matched_terms, total_terms, all_terms_in_text = _label_terms_in_text(row.get("text", ""), label)
        term_ratio = matched_terms / total_terms if total_terms else 0.0
        nli_label = str(row.get("nli_label", "")).lower()
        nli_entailment = "entail" in nli_label
        nli_neutral = "neutral" in nli_label
        nli_non_contradiction = "contradiction" not in nli_label if nli_label else True
        model_filter_passed = bool(row.get("model_filter_passed", False))
        model_filter_pred_raw = canonicalize_triplet_text(row.get("model_filter_pred_raw", ""))
        model_filter_pred_fixed = canonicalize_triplet_text(row.get("model_filter_pred_fixed", ""))
        label_triplets = _triplet_set(label)
        model_filter_raw_exact = bool(label_triplets) and _triplet_set(model_filter_pred_raw) == label_triplets
        model_filter_fixed_exact = bool(label_triplets) and _triplet_set(model_filter_pred_fixed) == label_triplets
        replacement_rank = row.get("replacement_rank") or {}
        replacement_rank_score = float(replacement_rank.get("score", 0.0))

        consistency_score = term_ratio
        quality_score = 0.3 + 0.4 * consistency_score
        if nli_entailment:
            quality_score += 0.3
        elif nli_neutral:
            quality_score += 0.15
        elif nli_non_contradiction:
            quality_score += 0.05
        if model_filter_raw_exact:
            quality_score += 0.25
        elif model_filter_fixed_exact:
            quality_score += 0.12
        elif model_filter_passed:
            quality_score += 0.05
        quality_score = _clamp(quality_score, 0.85, 1.75)
        sample_weight = _clamp(base_weight * (0.75 + 0.5 * quality_score), base_weight, 0.35)
        weighted_rows.append(
            {
                **row,
                "label": label,
                "quality_score": round(quality_score, 6),
                "consistency_score": round(consistency_score, 6),
                "sample_weight": round(sample_weight, 6),
                "quality_flags": {
                    "all_terms_in_text": all_terms_in_text,
                    "matched_label_terms": matched_terms,
                    "total_label_terms": total_terms,
                    "nli_entailment": nli_entailment,
                    "nli_neutral": nli_neutral,
                    "nli_non_contradiction": nli_non_contradiction,
                    "model_filter_passed": model_filter_passed,
                    "model_filter_raw_exact": model_filter_raw_exact,
                    "model_filter_fixed_exact": model_filter_fixed_exact,
                    "replacement_rank_score": replacement_rank_score,
                },
            }
        )
    return weighted_rows


def augmentation_distribution(rows: list[dict]) -> dict:
    return dict(Counter(row.get("augmentation", "unknown") for row in rows))


def augmentation_channel_analysis(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("augmentation", "unknown"), []).append(row)
    analysis = {}
    for channel, channel_rows in sorted(grouped.items()):
        analysis[channel] = {
            "rows": len(channel_rows),
            "sentiment_distribution": sentiment_distribution(channel_rows),
            "sample_weight_summary": sample_weight_summary(channel_rows),
        }
    return analysis


OPINION_AUGMENT_CHANNELS = {
    "opinion_sentiment_channel",
    "masked_opinion_sentiment_channel",
    "masked_opinion_sentiment_editor",
}
BAD_OPINION_BOUNDARY_TOKENS = {"[opi]", "[opinion]", "[asp]", "[aspect]", "<opinion>"}
BAD_SINGLE_TOKEN_OPINIONS = {"no", "none", "yes", "ok", "okay"}
BAD_OPINION_PREFIXES = {"on", "in", "at", "of", "for", "with", "by", "as", "to", "from"}


def opinion_augmented_label_boundary_valid(row: dict) -> bool:
    if row.get("augmentation") not in OPINION_AUGMENT_CHANNELS:
        return True
    triplets = parse_triplet_text_list(canonicalize_triplet_text(row.get("label", "")))
    if not triplets:
        return False
    for _aspect, opinion, _sentiment in triplets:
        normalized = " ".join(str(opinion).lower().split())
        tokens = normalized.split()
        if not normalized:
            return False
        if any(token in normalized for token in BAD_OPINION_BOUNDARY_TOKENS):
            return False
        if normalized in BAD_SINGLE_TOKEN_OPINIONS:
            return False
        if tokens and tokens[0] in BAD_OPINION_PREFIXES:
            return False
    return True


def _is_opinion_augment_channel(row: dict) -> bool:
    return row.get("augmentation") in OPINION_AUGMENT_CHANNELS


def sample_weight_summary(rows: list[dict]) -> dict:
    weights = [float(row["sample_weight"]) for row in rows if "sample_weight" in row]
    if not weights:
        return {"count": 0}
    return {
        "count": len(weights),
        "min": min(weights),
        "max": max(weights),
        "mean": sum(weights) / len(weights),
    }


NOISY_PSEUDO_TERMS = {
    "anything",
    "backlit",
    "computer",
    "item",
    "laptop",
    "laptops",
    "machine",
    "product",
    "runs",
    "thing",
    "things",
    "toshiba",
    "use",
}


def _text_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _fragment_tokens(fragment: str) -> list[str]:
    return _text_tokens(fragment)


def _find_token_span(tokens: list[str], fragment_tokens: list[str]) -> tuple[int, int] | None:
    if not tokens or not fragment_tokens:
        return None
    width = len(fragment_tokens)
    for idx in range(0, len(tokens) - width + 1):
        if tokens[idx : idx + width] == fragment_tokens:
            return idx, idx + width - 1
    return None


def aspect_opinion_token_distance(text: str, aspect: str, opinion: str) -> int | None:
    tokens = _text_tokens(text)
    aspect_span = _find_token_span(tokens, _fragment_tokens(aspect))
    opinion_span = _find_token_span(tokens, _fragment_tokens(opinion))
    if aspect_span is None or opinion_span is None:
        return None
    if aspect_span[1] < opinion_span[0]:
        return opinion_span[0] - aspect_span[1] - 1
    if opinion_span[1] < aspect_span[0]:
        return aspect_span[0] - opinion_span[1] - 1
    return 0


def pseudo_reject_reason(row: dict, min_weight: float = 0.65, max_triplets: int = 3) -> str:
    label = canonicalize_triplet_text(row.get("label", ""))
    triplets = parse_triplet_text_list(label)
    flags = row.get("quality_flags") or {}
    if not label or not triplets:
        return "empty_label"
    if float(row.get("sample_weight", 0.0)) < min_weight:
        return "low_weight"
    if row.get("fixed_changed"):
        return "fixed_changed"
    if not flags.get("all_terms_in_text", False):
        return "terms_not_in_text"
    if len(triplets) > max_triplets:
        return "too_many_triplets"
    for aspect, opinion, _sentiment in triplets:
        if aspect.strip().lower() in NOISY_PSEUDO_TERMS or opinion.strip().lower() in NOISY_PSEUDO_TERMS:
            return "noisy_term"
    return ""


def high_precision_triplet_reject_reason(
    row: dict,
    triplet: tuple[str, str, str],
    max_token_distance: int = 8,
) -> str:
    aspect, opinion, sentiment = triplet
    if sentiment not in {"pos", "neg", "neu"}:
        return "bad_sentiment"
    if aspect.strip().lower() in NOISY_PSEUDO_TERMS or opinion.strip().lower() in NOISY_PSEUDO_TERMS:
        return "noisy_term"
    matched_terms, total_terms, all_terms_in_text = _label_terms_in_text(
        row.get("text", ""),
        canonicalize_triplet_text(triplets_to_text([triplet])),
    )
    if total_terms == 0 or not all_terms_in_text or matched_terms != total_terms:
        return "terms_not_in_text"
    distance = aspect_opinion_token_distance(row.get("text", ""), aspect, opinion)
    if distance is None:
        return "distance_unknown"
    if distance > max_token_distance:
        return "distance_too_far"
    return ""


def select_high_precision_pseudo_rows(
    rows: list[dict],
    min_weight: float = 0.65,
    max_triplets: int = 3,
    max_token_distance: int = 8,
) -> tuple[list[dict], dict]:
    selected = []
    rejected_counts: Counter[str] = Counter()
    removed_triplet_counts: Counter[str] = Counter()
    changed_rows = 0
    strict_rows, strict_stats = select_high_confidence_pseudo_rows(
        rows,
        min_weight=min_weight,
        max_triplets=max_triplets,
    )
    for row in strict_rows:
        triplets = parse_triplet_text_list(canonicalize_triplet_text(row.get("label", "")))
        kept_triplets = []
        for triplet in triplets:
            reason = high_precision_triplet_reject_reason(
                row,
                triplet,
                max_token_distance=max_token_distance,
            )
            if reason:
                removed_triplet_counts[reason] += 1
                continue
            kept_triplets.append(triplet)
        if not kept_triplets:
            rejected_counts["empty_after_triplet_filter"] += 1
            continue
        new_label = canonicalize_triplet_text(triplets_to_text(kept_triplets))
        original_label = canonicalize_triplet_text(row.get("label", ""))
        changed = new_label != original_label
        if changed:
            changed_rows += 1
        selected.append(
            {
                **row,
                "label": new_label,
                "high_precision_pseudo": True,
                "high_precision_original_label": original_label if changed else row.get("high_precision_original_label", original_label),
                "high_precision_triplet_count_before": len(triplets),
                "high_precision_triplet_count_after": len(kept_triplets),
            }
        )

    stats = {
        "input_rows": len(rows),
        "strict_input_rows": strict_stats["selected_rows"],
        "selected_rows": len(selected),
        "rejected_rows": len(rows) - len(selected),
        "changed_rows": changed_rows,
        "removed_triplets_by_distance": (
            removed_triplet_counts.get("distance_too_far", 0) + removed_triplet_counts.get("distance_unknown", 0)
        ),
        "min_weight": min_weight,
        "max_triplets": max_triplets,
        "max_token_distance": max_token_distance,
        "sample_weight_summary": sample_weight_summary(selected),
        "sentiment_distribution": sentiment_distribution(selected),
    }
    for key, value in sorted(strict_stats.items()):
        if key.startswith("rejected_"):
            stats[key] = value
    for reason, count in sorted(rejected_counts.items()):
        stats[f"rejected_{reason}"] = count
    for reason, count in sorted(removed_triplet_counts.items()):
        stats[f"removed_triplets_by_{reason}"] = count
    return selected, stats


def pseudo_confidence_score(row: dict) -> float:
    label = canonicalize_triplet_text(row.get("label", ""))
    triplets = parse_triplet_text_list(label)
    flags = row.get("quality_flags") or {}
    if not label or not triplets:
        return 0.0

    score = 0.0
    if float(row.get("sample_weight", 0.0)) >= 0.65:
        score += 0.25
    if flags.get("all_terms_in_text", False):
        score += 0.3
    triplet_count = len(triplets)
    if triplet_count == 1:
        score += 0.2
    elif triplet_count <= 3:
        score += 0.12
    if row.get("fixed_changed"):
        score -= 0.08
    for aspect, opinion, _sentiment in triplets:
        if aspect.strip().lower() in NOISY_PSEUDO_TERMS or opinion.strip().lower() in NOISY_PSEUDO_TERMS:
            score -= 0.5
            break
    return round(max(0.0, min(1.0, score)), 6)


def select_high_confidence_pseudo_rows(
    rows: list[dict],
    min_weight: float = 0.65,
    max_triplets: int = 3,
) -> tuple[list[dict], dict]:
    selected = []
    rejected_counts: Counter[str] = Counter()
    for row in rows:
        reason = pseudo_reject_reason(row, min_weight=min_weight, max_triplets=max_triplets)
        if reason:
            rejected_counts[reason] += 1
            continue
        selected.append({**row, "selected_pseudo": True})

    stats = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "rejected_rows": len(rows) - len(selected),
        "min_weight": min_weight,
        "max_triplets": max_triplets,
        "sample_weight_summary": sample_weight_summary(selected),
    }
    for reason, count in sorted(rejected_counts.items()):
        stats[f"rejected_{reason}"] = count
    return selected, stats


def select_train_pseudo_rows(
    rows: list[dict],
    min_weight: float = 0.65,
    fixed_changed_min_score: float = 0.65,
    fixed_changed_weight: float = 0.35,
) -> tuple[list[dict], dict]:
    strict_selected, strict_stats = select_high_confidence_pseudo_rows(rows, min_weight=min_weight, max_triplets=3)
    selected = list(strict_selected)
    selected_ids = {row.get("id") for row in selected}
    added_counts: Counter[str] = Counter()

    for row in rows:
        if row.get("id") in selected_ids:
            continue
        reason = pseudo_reject_reason(row, min_weight=min_weight, max_triplets=3)
        score = pseudo_confidence_score(row)
        if reason == "fixed_changed" and score >= fixed_changed_min_score:
            selected.append(
                {
                    **row,
                    "sample_weight": round(fixed_changed_weight, 6),
                    "pseudo_confidence_score": score,
                    "train_selected_pseudo": True,
                    "train_selected_reason": "fixed_changed_high_confidence",
                }
            )
            selected_ids.add(row.get("id"))
            added_counts["fixed_changed_high_confidence"] += 1

    stats = {
        "input_rows": len(rows),
        "strict_selected_rows": strict_stats["selected_rows"],
        "selected_rows": len(selected),
        "added_rows": len(selected) - strict_stats["selected_rows"],
        "min_weight": min_weight,
        "fixed_changed_min_score": fixed_changed_min_score,
        "fixed_changed_weight": fixed_changed_weight,
        "sample_weight_summary": sample_weight_summary(selected),
    }
    for reason, count in sorted(added_counts.items()):
        stats[f"added_{reason}"] = count
    return selected, stats


def read_selected_pseudo_rows(run_dir: Path, fallback_to_all: bool = True) -> list[dict]:
    selected_path = run_dir / "target_pseudo_selected.jsonl"
    if selected_path.exists():
        return read_jsonl(selected_path)
    if fallback_to_all:
        return read_jsonl(run_dir / "target_pseudo.jsonl")
    return []


def read_train_selected_pseudo_rows(run_dir: Path, fallback_to_selected: bool = True) -> list[dict]:
    train_selected_path = run_dir / "target_pseudo_train_selected.jsonl"
    if train_selected_path.exists():
        return read_jsonl(train_selected_path)
    if fallback_to_selected:
        return read_selected_pseudo_rows(run_dir)
    return []


def read_pseudo_rows_for_training(
    run_dir: Path,
    source: str = "strict",
    pseudo_train_file: Path | None = None,
) -> list[dict]:
    if pseudo_train_file is not None and pseudo_train_file.exists():
        return read_jsonl(pseudo_train_file)
    if source == "strict":
        return read_selected_pseudo_rows(run_dir, fallback_to_all=True)
    if source == "high_precision":
        high_precision_path = run_dir / "target_pseudo_high_precision.jsonl"
        if high_precision_path.exists():
            return read_jsonl(high_precision_path)
        return read_selected_pseudo_rows(run_dir, fallback_to_all=True)
    if source == "train_selected":
        return read_train_selected_pseudo_rows(run_dir, fallback_to_selected=True)
    if source == "mixed_recall":
        high_precision_rows = read_pseudo_rows_for_training(run_dir, "high_precision", None)
        recall_rows = read_pseudo_rows_for_training(run_dir, "train_selected", None)
        mixed_rows, _stats = build_mixed_recall_pseudo_rows(high_precision_rows, recall_rows)
        return mixed_rows
    raise ValueError("pseudo_train_source must be 'strict', 'high_precision', 'train_selected', or 'mixed_recall'")


def build_mixed_recall_pseudo_rows(
    high_precision_rows: list[dict],
    recall_rows: list[dict],
    recall_extra_weight: float = 0.25,
    recall_extra_max_rows: int = 0,
) -> tuple[list[dict], dict]:
    mixed_rows = []
    seen: set[tuple[str, str]] = set()
    for row in high_precision_rows:
        label = canonicalize_triplet_text(row.get("label", ""))
        if not label:
            continue
        key = (str(row.get("text", "")).lower(), label.lower())
        if key in seen:
            continue
        seen.add(key)
        mixed_rows.append({**row, "label": label, "pseudo_mix_source": "high_precision"})

    extra_rows = []
    for row in recall_rows:
        label = canonicalize_triplet_text(row.get("label", ""))
        if not label:
            continue
        key = (str(row.get("text", "")).lower(), label.lower())
        if key in seen:
            continue
        seen.add(key)
        extra_rows.append(
            {
                **row,
                "label": label,
                "sample_weight": round(recall_extra_weight, 6),
                "pseudo_mix_source": "recall_extra",
            }
        )
        if recall_extra_max_rows > 0 and len(extra_rows) >= recall_extra_max_rows:
            break

    mixed_rows.extend(extra_rows)
    stats = {
        "high_precision_rows": len(high_precision_rows),
        "recall_candidate_rows": len(recall_rows),
        "high_precision_used": len(mixed_rows) - len(extra_rows),
        "recall_extra_added": len(extra_rows),
        "recall_extra_weight": recall_extra_weight,
        "recall_extra_max_rows": recall_extra_max_rows,
        "sample_weight_summary": sample_weight_summary(mixed_rows),
        "sentiment_distribution": sentiment_distribution(mixed_rows),
    }
    return mixed_rows, stats


def read_extra_augmented_rows(paths_text: str, sample_weight: float | None = None) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    path_stats = []
    if not paths_text.strip():
        return rows, {"enabled": False, "paths": [], "rows": 0}
    paths = [Path(part.strip()) for part in paths_text.split(";") if part.strip()]
    for path in paths:
        loaded = read_jsonl(path)
        converted = []
        for row in loaded:
            label = canonicalize_triplet_text(row.get("label", ""))
            if not label or not row.get("text"):
                continue
            merged = {
                **row,
                "label": label,
                "extra_augmented_source": str(path),
                "extra_augmented": True,
            }
            if sample_weight is not None and sample_weight > 0:
                merged["sample_weight"] = round(sample_weight, 6)
                merged["extra_augmented_original_sample_weight"] = row.get("sample_weight")
            converted.append(merged)
        rows.extend(converted)
        path_stats.append(
            {
                "path": str(path),
                "loaded_rows": len(loaded),
                "usable_rows": len(converted),
                "augmentation_distribution": augmentation_distribution(converted),
                "sentiment_distribution": sentiment_distribution(converted),
            }
        )
    return rows, {
        "enabled": True,
        "paths": path_stats,
        "rows": len(rows),
        "sample_weight_override": sample_weight,
        "augmentation_distribution": augmentation_distribution(rows),
        "sentiment_distribution": sentiment_distribution(rows),
        "sample_weight_summary": sample_weight_summary(rows),
    }


def tagged_output_path(run_dir: Path, filename: str, tag: str = "") -> Path:
    if not tag:
        return run_dir / filename
    path = Path(filename)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag)
    return run_dir / f"{path.stem}_{safe_tag}{path.suffix}"


def resolve_extractor_model_path(
    run_dir: Path,
    explicit_model_path: str = "",
    variant: str = "best",
) -> Path:
    if explicit_model_path:
        return Path(explicit_model_path)
    models_dir = run_dir / "models"
    if variant == "last":
        candidates = sorted(
            (path for path in models_dir.glob("extractor*last*/best") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        return models_dir / "extractor_ep25_last" / "best"
    if variant == "best":
        direct = models_dir / "extractor" / "best"
        if direct.exists():
            return direct
        candidates = sorted(
            (path for path in models_dir.glob("extractor*/best") if path.is_dir() and "last" not in path.parent.name.lower()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        return direct
    raise ValueError("extractor model variant must be 'best' or 'last'")


def select_high_value_augmented_rows(
    rows: list[dict],
    max_rows: int = 200,
    max_per_base: int = 1,
    selected_weight: float = 0.35,
    max_opinion_ratio: float = 1.0,
    require_raw_exact: bool = False,
    require_model_filter_passed: bool = False,
) -> tuple[list[dict], dict]:
    input_rows = len(rows)
    if require_raw_exact:
        rows = [row for row in rows if (row.get("quality_flags") or {}).get("model_filter_raw_exact")]
    if require_model_filter_passed:
        rows = [row for row in rows if row.get("model_filter_passed")]
    if max_rows <= 0:
        return rows, {
            "enabled": False,
            "input_rows": input_rows,
            "candidate_rows": len(rows),
            "selected_rows": len(rows),
            "max_rows": max_rows,
            "max_per_base": max_per_base,
            "selected_weight": selected_weight,
            "max_opinion_ratio": max_opinion_ratio,
            "require_raw_exact": require_raw_exact,
            "require_model_filter_passed": require_model_filter_passed,
            "skipped_by_base_limit": 0,
            "skipped_by_channel_ratio": 0,
        }

    def rank_key(row: dict) -> tuple:
        flags = row.get("quality_flags") or {}
        return (
            int(bool(flags.get("model_filter_raw_exact"))),
            int(bool(flags.get("model_filter_fixed_exact"))),
            float(row.get("quality_score", 0.0)),
            float(row.get("sample_weight", 0.0)),
            int(bool(flags.get("all_terms_in_text"))),
            row.get("text", ""),
        )

    ranked_rows = sorted(rows, key=rank_key, reverse=True)
    selected: list[dict] = []
    selected_keys: set[tuple[str, str]] = set()
    base_counts: Counter[str] = Counter()
    skipped_by_base_limit = 0
    skipped_by_channel_ratio = 0
    opinion_cap = max_rows if max_opinion_ratio >= 1.0 else max(0, int(max_rows * max_opinion_ratio))
    opinion_selected = 0

    def try_select(row: dict, enforce_channel_ratio: bool) -> bool:
        nonlocal skipped_by_base_limit, skipped_by_channel_ratio, opinion_selected
        base_key = row.get("base_id") or row.get("base_text") or row.get("text", "")
        if max_per_base > 0 and base_counts[base_key] >= max_per_base:
            skipped_by_base_limit += 1
            return False
        is_opinion_channel = _is_opinion_augment_channel(row)
        if enforce_channel_ratio and is_opinion_channel and opinion_selected >= opinion_cap:
            skipped_by_channel_ratio += 1
            return False
        selected_row = {
            **row,
            "original_sample_weight": row.get("sample_weight"),
            "sample_weight": round(selected_weight, 6),
            "selected_augmentation": True,
        }
        selected.append(
            {
                **selected_row,
            }
        )
        selected_keys.add((row.get("text", "").lower(), canonicalize_triplet_text(row.get("label", "")).lower()))
        base_counts[base_key] += 1
        if is_opinion_channel:
            opinion_selected += 1
        return True

    for row in ranked_rows:
        try_select(row, enforce_channel_ratio=True)
        if len(selected) >= max_rows:
            break
    if len(selected) < max_rows and max_opinion_ratio < 1.0:
        for row in ranked_rows:
            key = (row.get("text", "").lower(), canonicalize_triplet_text(row.get("label", "")).lower())
            if key in selected_keys:
                continue
            try_select(row, enforce_channel_ratio=False)
            if len(selected) >= max_rows:
                break

    if len(selected) >= max_rows and max_per_base > 0:
        for row in ranked_rows:
            key = (row.get("text", "").lower(), canonicalize_triplet_text(row.get("label", "")).lower())
            if key in selected_keys:
                continue
            base_key = row.get("base_id") or row.get("base_text") or row.get("text", "")
            if base_counts[base_key] >= max_per_base:
                skipped_by_base_limit += 1

    stats = {
        "enabled": True,
        "input_rows": input_rows,
        "candidate_rows": len(rows),
        "selected_rows": len(selected),
        "max_rows": max_rows,
        "max_per_base": max_per_base,
        "selected_weight": selected_weight,
        "max_opinion_ratio": max_opinion_ratio,
        "require_raw_exact": require_raw_exact,
        "require_model_filter_passed": require_model_filter_passed,
        "skipped_by_base_limit": skipped_by_base_limit,
        "skipped_by_channel_ratio": skipped_by_channel_ratio,
        "sample_weight_summary": sample_weight_summary(selected),
        "augmentation_distribution": augmentation_distribution(selected),
        "sentiment_distribution": sentiment_distribution(selected),
    }
    return selected, stats


def assign_final_training_weights(
    rows: list[dict],
    multi_triplet_gain: float = 0.0,
    neutral_gain: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[list[dict], dict]:
    weighted_rows = []
    multi_triplet_rows = 0
    neutral_rows = 0
    for row in rows:
        label = canonicalize_triplet_text(row.get("label", ""))
        triplets = parse_triplet_text_list(label)
        base_weight = float(row.get("sample_weight", 1.0))
        multiplier = 1.0
        multi_triplet = len(triplets) >= 2
        contains_neutral = any(sentiment == "neu" for _aspect, _opinion, sentiment in triplets)
        if multi_triplet:
            multiplier += multi_triplet_gain
            multi_triplet_rows += 1
        if contains_neutral:
            multiplier += neutral_gain
            neutral_rows += 1
        sample_weight = _clamp(base_weight * multiplier, 0.0, max_weight)
        weighted_rows.append(
            {
                **row,
                "label": label,
                "sample_weight": round(sample_weight, 6),
                "final_weight_flags": {
                    "triplet_count": len(triplets),
                    "multi_triplet": multi_triplet,
                    "contains_neutral": contains_neutral,
                    "base_weight": base_weight,
                    "multiplier": round(multiplier, 6),
                },
            }
        )

    stats = {
        "rows": len(rows),
        "multi_triplet_rows": multi_triplet_rows,
        "neutral_rows": neutral_rows,
        "multi_triplet_gain": multi_triplet_gain,
        "neutral_gain": neutral_gain,
        "max_weight": max_weight,
        "sample_weight_summary": sample_weight_summary(weighted_rows),
        "augmentation_distribution": augmentation_distribution(weighted_rows),
        "sentiment_distribution": sentiment_distribution(weighted_rows),
    }
    return weighted_rows, stats


def build_final_training_rows(
    source_rows: list[dict],
    pseudo_rows: list[dict],
    augmented_rows: list[dict],
    include_source: bool = True,
) -> list[dict]:
    final_rows = []
    seen = set()
    input_rows = (source_rows if include_source else []) + pseudo_rows + augmented_rows
    for row in input_rows:
        label = canonicalize_triplet_text(row.get("label", ""))
        if not label:
            continue
        key = (row["text"].lower(), label.lower())
        if key in seen:
            continue
        seen.add(key)
        final_rows.append({**row, "label": label})
    return final_rows


def run_nli_filter(
    rows: list[dict],
    model_path: str | Path,
    batch_size: int,
    cuda: str,
) -> tuple[list[dict], dict]:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    kept: list[dict] = []
    stats = {
        "enabled": True,
        "model_path": str(model_path),
        "input_rows": len(rows),
        "kept_rows": 0,
        "filtered_contradiction": 0,
        "label_counts": {},
    }
    label_counts: Counter[str] = Counter()
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=lambda batch_rows: batch_rows)
    with torch.inference_mode():
        for batch_rows in tqdm(loader, desc=f"nli-filter:{Path(model_path).name}"):
            premises = [row["base_text"] for row in batch_rows]
            hypotheses = [row["text"] for row in batch_rows]
            encoded = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            for i, pred_id in enumerate(pred_ids):
                label = id2label.get(pred_id, str(pred_id)).lower()
                label_counts[label] += 1
                row = dict(batch_rows[i])
                row["nli_label"] = label
                if "contradiction" in label:
                    stats["filtered_contradiction"] += 1
                    continue
                kept.append(row)

    stats["kept_rows"] = len(kept)
    stats["label_counts"] = dict(label_counts)
    return kept, stats


def _triplet_set(label: str) -> set[tuple[str, str, str]]:
    return set(parse_triplet_text_list(canonicalize_triplet_text(label)))


def _normalized_fragment(fragment: str) -> str:
    return " ".join(fragment.lower().strip(" .,;:!?\"'()[]{}").split())


def _normalized_triplet_for_filter(triplet: tuple[str, str, str]) -> tuple[str, str, str]:
    aspect, opinion, sentiment = triplet
    return (_normalized_fragment(aspect), _normalized_fragment(opinion), sentiment)


def _triplet_set_for_filter(label: str) -> set[tuple[str, str, str]]:
    return {
        triplet
        for triplet in (_normalized_triplet_for_filter(item) for item in parse_triplet_text_list(canonicalize_triplet_text(label)))
        if triplet[0] and triplet[1] and triplet[2] in {"pos", "neg", "neu"}
    }


def _matched_gold_triplets_for_filter(pred_label: str, gold_label: str) -> list[tuple[str, str, str]]:
    pred_triplets = _triplet_set_for_filter(pred_label)
    matched: list[tuple[str, str, str]] = []
    for gold_triplet in parse_triplet_text_list(canonicalize_triplet_text(gold_label)):
        normalized_gold = _normalized_triplet_for_filter(gold_triplet)
        if normalized_gold in pred_triplets:
            matched.append(normalized_gold)
    return matched


def _opinion_span_compatible(gold_opinion: str, pred_opinion: str) -> bool:
    gold = _normalized_fragment(gold_opinion)
    pred = _normalized_fragment(pred_opinion)
    return bool(gold and pred) and (gold == pred or gold in pred or pred in gold)


def _triplet_set_span_compatible(pred_label: str, gold_label: str) -> bool:
    pred_triplets = list(_triplet_set(pred_label))
    gold_triplets = list(_triplet_set(gold_label))
    if not pred_triplets or len(pred_triplets) != len(gold_triplets):
        return False
    used_pred: set[int] = set()
    for gold_aspect, gold_opinion, gold_sentiment in gold_triplets:
        matched_idx = None
        for idx, (pred_aspect, pred_opinion, pred_sentiment) in enumerate(pred_triplets):
            if idx in used_pred:
                continue
            if pred_sentiment != gold_sentiment:
                continue
            if _normalized_fragment(pred_aspect) != _normalized_fragment(gold_aspect):
                continue
            if not _opinion_span_compatible(gold_opinion, pred_opinion):
                continue
            matched_idx = idx
            break
        if matched_idx is None:
            return False
        used_pred.add(matched_idx)
    return True


def _triplet_set_aspect_sentiment_opinion_span_compatible(pred_label: str, gold_label: str) -> bool:
    pred_triplets = list(_triplet_set(pred_label))
    gold_triplets = list(_triplet_set(gold_label))
    if not pred_triplets or not gold_triplets or len(pred_triplets) < len(gold_triplets):
        return False
    used_pred: set[int] = set()
    for gold_aspect, gold_opinion, gold_sentiment in gold_triplets:
        matched_idx = None
        for idx, (pred_aspect, pred_opinion, pred_sentiment) in enumerate(pred_triplets):
            if idx in used_pred:
                continue
            if pred_sentiment != gold_sentiment:
                continue
            if _normalized_fragment(pred_aspect) != _normalized_fragment(gold_aspect):
                continue
            if not _opinion_span_compatible(gold_opinion, pred_opinion):
                continue
            matched_idx = idx
            break
        if matched_idx is None:
            return False
        used_pred.add(matched_idx)
    return True


def filter_augmented_rows_by_model_predictions(
    rows: list[dict],
    predictions: list[str],
    mode: str = "fixed",
) -> tuple[list[dict], list[dict], dict]:
    if mode not in {"exact", "fixed"}:
        raise ValueError("model_filter mode must be 'exact' or 'fixed'")
    if len(rows) != len(predictions):
        raise ValueError("model_filter rows and predictions must have the same length")

    kept: list[dict] = []
    removed: list[dict] = []
    changed_by_fix = 0
    for row, pred in zip(rows, predictions):
        gold_label = canonicalize_triplet_text(row.get("label", ""))
        pred_raw = canonicalize_triplet_text(pred)
        pred_fixed = canonicalize_triplet_text(fix_pred_triplets(pred_raw, row.get("text", "")))
        if pred_raw != pred_fixed:
            changed_by_fix += 1

        compare_label = pred_fixed if mode == "fixed" else pred_raw
        enriched = {
            **row,
            "model_filter_pred_raw": pred_raw,
            "model_filter_pred_fixed": pred_fixed,
            "model_filter_mode": mode,
        }
        exact_match = bool(gold_label) and _triplet_set(compare_label) == _triplet_set(gold_label)
        span_match = (
            bool(gold_label)
            and row.get("augmentation") == "label_composition_channel"
            and _triplet_set_span_compatible(compare_label, gold_label)
        )
        if exact_match or span_match:
            kept.append(
                {
                    **enriched,
                    "model_filter_passed": True,
                    "model_filter_match": "exact" if exact_match else "span_compatible",
                }
            )
        else:
            removed.append(
                {
                    **enriched,
                    "model_filter_passed": False,
                    "model_filter_reason": "label_mismatch",
                }
            )

    stats = {
        "enabled": True,
        "mode": mode,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "pred_changed_by_fixed": changed_by_fix,
        "kept_distribution": augmentation_distribution(kept),
        "removed_distribution": augmentation_distribution(removed),
        "kept_sentiment_distribution": sentiment_distribution(kept),
        "removed_sentiment_distribution": sentiment_distribution(removed),
    }
    return kept, removed, stats


def filter_augmented_rows_by_model_predictions_channel_aware(
    rows: list[dict],
    predictions: list[str],
    mode: str = "fixed",
) -> tuple[list[dict], list[dict], dict]:
    if mode not in {"exact", "fixed"}:
        raise ValueError("model_filter mode must be 'exact' or 'fixed'")
    if len(rows) != len(predictions):
        raise ValueError("model_filter rows and predictions must have the same length")

    kept: list[dict] = []
    removed: list[dict] = []
    changed_by_fix = 0
    opinion_label_replaced = 0
    opinion_kept = 0
    aspect_kept = 0
    fusion_full_kept = 0
    fusion_partial_dropped = 0
    for row, pred in zip(rows, predictions):
        gold_label = canonicalize_triplet_text(row.get("label", ""))
        pred_raw = canonicalize_triplet_text(pred)
        pred_fixed = canonicalize_triplet_text(fix_pred_triplets(pred_raw, row.get("text", "")))
        if pred_raw != pred_fixed:
            changed_by_fix += 1

        channel = str(row.get("augmentation", ""))
        is_opinion_channel = channel in {"opinion_sentiment_channel", "masked_opinion_sentiment_channel", "masked_opinion_sentiment_editor"}
        compare_label = pred_fixed if mode == "fixed" else pred_raw
        enriched = {
            **row,
            "model_filter_pred_raw": pred_raw,
            "model_filter_pred_fixed": pred_fixed,
            "model_filter_mode": mode,
        }

        if is_opinion_channel:
            new_triplet = row.get("new_triplet") or []
            target_aspect = _normalized_fragment(new_triplet[0]) if len(new_triplet) >= 1 else ""
            target_sentiment = new_triplet[2] if len(new_triplet) >= 3 else ""
            pred_triplets = list(_triplet_set(compare_label))
            has_target_triplet = any(
                _normalized_fragment(pred_aspect) == target_aspect and pred_sentiment == target_sentiment
                for pred_aspect, _pred_opinion, pred_sentiment in pred_triplets
            )
            if not compare_label:
                removed.append(
                    {
                        **enriched,
                        "model_filter_passed": False,
                        "model_filter_reason": "empty_extractor_label",
                    }
                )
                continue
            if not has_target_triplet:
                removed.append(
                    {
                        **enriched,
                        "model_filter_passed": False,
                        "model_filter_reason": "missing_target_aspect_sentiment",
                    }
                )
                continue
            updated = {
                **enriched,
                "label": compare_label,
                "model_filter_passed": True,
                "model_filter_match": "extracted",
                "model_filter_label_source": "opinion_extractor",
            }
            opinion_label_replaced += 1
            opinion_kept += 1
            kept.append(updated)
            continue

        exact_match = bool(gold_label) and _triplet_set_for_filter(compare_label) == _triplet_set_for_filter(gold_label)
        span_match = (
            bool(gold_label)
            and row.get("augmentation") in {"label_composition_channel", "sentence_fusion_composition_channel"}
            and _triplet_set_span_compatible(compare_label, gold_label)
        )
        rsda_t5_match = (
            bool(gold_label)
            and row.get("augmentation") in {"rsda_t5_label_composition_channel", "sentence_fusion_composition_channel"}
            and _triplet_set_aspect_sentiment_opinion_span_compatible(compare_label, gold_label)
        )
        is_sentence_fusion = row.get("augmentation") == "sentence_fusion_composition_channel"
        if is_sentence_fusion and not (exact_match or span_match or rsda_t5_match):
            fusion_partial_triplets = _matched_gold_triplets_for_filter(compare_label, gold_label)
            if len(fusion_partial_triplets) >= 2:
                rsda_t5_match = True
            elif len(fusion_partial_triplets) == 1:
                removed.append(
                    {
                        **enriched,
                        "model_filter_passed": False,
                        "model_filter_reason": "fusion_partial_match_dropped",
                        "model_filter_partial_triplets": len(fusion_partial_triplets),
                    }
                )
                fusion_partial_dropped += 1
                continue
            else:
                fusion_partial_dropped += 1
        compatible_aspect_sentiment = False
        if not exact_match and not span_match and not rsda_t5_match:
            new_triplet = row.get("new_triplet") or []
            target_aspect = _normalized_fragment(new_triplet[0]) if len(new_triplet) >= 1 else ""
            target_sentiment = new_triplet[2] if len(new_triplet) >= 3 else ""
            pred_triplets = list(_triplet_set(compare_label))
            compatible_aspect_sentiment = any(
                _normalized_fragment(pred_aspect) == target_aspect and pred_sentiment == target_sentiment
                for pred_aspect, _pred_opinion, pred_sentiment in pred_triplets
            )
        if exact_match or span_match or rsda_t5_match or compatible_aspect_sentiment:
            aspect_kept += 1
            if is_sentence_fusion and (exact_match or span_match or rsda_t5_match):
                fusion_full_kept += 1
            final_label = compare_label if compatible_aspect_sentiment and not (exact_match or span_match or rsda_t5_match) else gold_label
            kept.append(
                {
                    **enriched,
                    "label": final_label,
                    "model_filter_passed": True,
                    "model_filter_match": (
                        "exact"
                        if exact_match
                        else "span_compatible"
                        if span_match
                        else "aspect_sentiment_opinion_span"
                        if rsda_t5_match
                        else "aspect_sentiment_compatible"
                    ),
                    "model_filter_label_source": (
                        "candidate_label" if (exact_match or span_match or rsda_t5_match) else "aspect_channel_extractor"
                    ),
                }
            )
        else:
            removed.append(
                {
                    **enriched,
                    "model_filter_passed": False,
                    "model_filter_reason": "label_mismatch",
                }
            )

    stats = {
        "enabled": True,
        "mode": mode,
        "channel_aware": True,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "pred_changed_by_fixed": changed_by_fix,
        "opinion_label_replaced": opinion_label_replaced,
        "opinion_kept": opinion_kept,
        "aspect_kept": aspect_kept,
        "fusion_full_kept": fusion_full_kept,
        "fusion_partial_dropped": fusion_partial_dropped,
        "kept_distribution": augmentation_distribution(kept),
        "removed_distribution": augmentation_distribution(removed),
        "kept_sentiment_distribution": sentiment_distribution(kept),
        "removed_sentiment_distribution": sentiment_distribution(removed),
    }
    return kept, removed, stats


def filter_augmented_rows_with_optional_channel_awareness(
    rows: list[dict],
    predictions: list[str],
    mode: str = "fixed",
    channel_aware: bool = False,
) -> tuple[list[dict], list[dict], dict]:
    if channel_aware:
        return filter_augmented_rows_by_model_predictions_channel_aware(rows, predictions, mode=mode)
    kept, removed, stats = filter_augmented_rows_by_model_predictions(rows, predictions, mode=mode)
    stats["channel_aware"] = False
    return kept, removed, stats


def run_model_filter(
    rows: list[dict],
    model_path: str | Path,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    cuda: str,
    mode: str,
    use_task_prefix: bool,
    constrained: bool,
    channel_aware: bool = False,
) -> tuple[list[dict], list[dict], dict]:
    predictions = generate_texts(
        model_path=model_path,
        inputs=build_extract_inputs(rows, use_task_prefix=use_task_prefix),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        cuda=cuda,
        constrained=constrained,
        length_penalty=1.0,
    )
    kept, removed, stats = filter_augmented_rows_with_optional_channel_awareness(
        rows,
        predictions,
        mode=mode,
        channel_aware=channel_aware,
    )
    stats["model_path"] = str(model_path)
    stats["batch_size"] = batch_size
    stats["max_new_tokens"] = max_new_tokens
    stats["num_beams"] = num_beams
    stats["use_task_prefix"] = use_task_prefix
    stats["constrained_decoding"] = constrained
    return kept, removed, stats


def _rank_aspect_stats(
    aspect_stats: dict,
    aspects: list[str],
    sort_keys: tuple[str, ...],
    top_k: int,
) -> list[tuple[str, dict]]:
    aspect_set = set(aspects)
    rows = [(aspect, stats) for aspect, stats in aspect_stats.items() if aspect in aspect_set]

    def sort_key(item: tuple[str, dict]) -> tuple:
        aspect, stats = item
        values = []
        for key in sort_keys:
            values.append(-stats[key])
        values.append(aspect)
        return tuple(values)

    return sorted(rows, key=sort_key)[:top_k]


def build_memory_analysis(
    source_rows: list[dict],
    pseudo_rows: list[dict],
    source_memory: dict,
    target_memory: dict,
    cross_memory: dict,
    args: argparse.Namespace,
) -> dict:
    aspect_stats = target_memory["aspect_stats"]
    top_frequency_aspects = _rank_aspect_stats(
        aspect_stats,
        target_memory["aspects"],
        ("target_tf", "target_df"),
        args.top_k,
    )
    top_core_aspects = _rank_aspect_stats(
        aspect_stats,
        target_memory["core_target_aspects"],
        ("target_tf", "target_df"),
        args.top_k,
    )
    top_specific_aspects = _rank_aspect_stats(
        aspect_stats,
        target_memory["specific_target_aspects"],
        ("domain_score", "target_tf"),
        args.top_k,
    )
    return {
        "min_pseudo_weight": args.min_pseudo_weight,
        "source": {
            "rows": len(source_rows),
            "aspects": len(source_memory["aspects"]),
            "triplets": len(source_memory["triplets"]),
            "opinions_by_sentiment": {k: len(v) for k, v in source_memory["opinions_by_sentiment"].items()},
        },
        "target": {
            "rows": len(pseudo_rows),
            "aspects": len(target_memory["aspects"]),
            "core_target_aspects": len(target_memory["core_target_aspects"]),
            "specific_candidate_aspects": len(target_memory.get("specific_candidate_aspects", [])),
            "specific_target_aspects": len(target_memory["specific_target_aspects"]),
            "triplets": len(target_memory["triplets"]),
            "opinions_by_sentiment": {k: len(v) for k, v in target_memory["opinions_by_sentiment"].items()},
            "rejected_aspects_by_reason": {
                key: len(value) for key, value in target_memory["rejected_aspects_by_reason"].items()
            },
            "rejected_specific_aspects_by_reason": {
                key: len(value) for key, value in target_memory.get("rejected_specific_aspects_by_reason", {}).items()
            },
            "top_frequency_aspects": [
                {"aspect": aspect, **stats} for aspect, stats in top_frequency_aspects
            ],
            "top_core_aspects": [
                {"aspect": aspect, **stats} for aspect, stats in top_core_aspects
            ],
            "top_specific_aspects": [
                {"aspect": aspect, **stats} for aspect, stats in top_specific_aspects
            ],
        },
        "cross": {
            "target_aspects": len(cross_memory["target_aspects"]),
            "candidate_triplets": len(cross_memory["candidate_triplets"]),
            "opinions_by_sentiment": {k: len(v) for k, v in cross_memory["opinions_by_sentiment"].items()},
        },
    }


def prepare(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    source_rows = load_split(args.source_dataset, "train")
    source_dev_rows = load_split(args.source_dataset, "dev")
    if not source_dev_rows:
        source_rows, source_dev_rows = split_train_dev(source_rows, args.dev_ratio, args.seed)
    target_train_rows = load_split(args.target_dataset, "train")
    target_test_rows = load_split(args.target_dataset, "test")

    write_jsonl(run_dir / "source_train.jsonl", source_rows)
    write_jsonl(run_dir / "source_dev.jsonl", source_dev_rows)
    write_jsonl(run_dir / "target_unlabeled.jsonl", [{"id": r["id"], "text": r["text"]} for r in target_train_rows])
    write_jsonl(run_dir / "target_train_gold_analysis.jsonl", target_train_rows)
    write_jsonl(run_dir / "target_test.jsonl", target_test_rows)
    use_task_prefix = not args.no_task_prefix
    write_jsonl(run_dir / "extract_train.jsonl", to_extract_rows(source_rows, use_task_prefix=use_task_prefix))
    write_jsonl(run_dir / "extract_dev.jsonl", to_extract_rows(source_dev_rows, use_task_prefix=use_task_prefix))
    generator_train_rows = build_generator_training_rows(
        source_rows,
        args.seed,
        prompt_style=args.augment_prompt_style,
        channel_mode=args.augment_channel_mode,
        domain_name=args.source_dataset,
        domain_prefix_style=args.domain_prefix_style,
    )
    generator_dev_rows = build_generator_training_rows(
        source_dev_rows,
        args.seed + 1,
        prompt_style=args.augment_prompt_style,
        channel_mode=args.augment_channel_mode,
        domain_name=args.source_dataset,
        domain_prefix_style=args.domain_prefix_style,
    )
    generator_tag = args.generator_output_tag
    write_jsonl(tagged_output_path(run_dir, "generator_train.jsonl", generator_tag), generator_train_rows)
    write_jsonl(tagged_output_path(run_dir, "generator_dev.jsonl", generator_tag), generator_dev_rows)
    write_jsonl(tagged_output_path(run_dir, "c3da_generator_train.jsonl", generator_tag), generator_train_rows)
    write_jsonl(tagged_output_path(run_dir, "c3da_generator_dev.jsonl", generator_tag), generator_dev_rows)

    manifest = {
        "task": "cross_domain_aste_c3da_two_channel",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "source_train": len(source_rows),
        "source_dev": len(source_dev_rows),
        "generator_train": len(generator_train_rows),
        "generator_dev": len(generator_dev_rows),
        "target_unlabeled": len(target_train_rows),
        "target_test": len(target_test_rows),
        "strict_cross_domain": "target train labels are hidden; target test is used only for final evaluation",
        "analysis_only": "target_train_gold_analysis.jsonl stores target train gold labels only for pseudo-label diagnostics, never for training",
        "extractor_model": str(run_dir / "models" / "extractor" / "best"),
        "generator_model": str(run_dir / "models" / "generator" / "best"),
        "final_model": str(run_dir / "models" / "final" / "best"),
        "use_task_prefix": use_task_prefix,
        "augment_prompt_style": args.augment_prompt_style,
        "augment_channel_mode": args.augment_channel_mode,
        "domain_prefix_style": args.domain_prefix_style,
        "generator_domain_name": args.source_dataset if args.domain_prefix_style != "none" else "",
        "generator_output_tag": generator_tag,
    }
    dump_json(run_dir / "manifest.json", manifest)
    print(f"prepared {run_dir}")
    print(manifest)


def pseudo(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    target_rows = read_jsonl(run_dir / "target_unlabeled.jsonl")
    if args.max_target_unlabeled > 0:
        target_rows = target_rows[: args.max_target_unlabeled]
    model_path = resolve_extractor_model_path(
        run_dir,
        explicit_model_path=args.model_path,
        variant=args.pseudo_model_variant,
    )
    preds = generate_texts(
        model_path=model_path,
        inputs=build_extract_inputs(target_rows, use_task_prefix=not args.no_task_prefix),
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        cuda=args.cuda,
        constrained=not args.no_constrained_decoding,
        length_penalty=args.length_penalty,
    )
    pseudo_rows = []
    for row, pred in zip(target_rows, preds):
        label = canonicalize_triplet_text(pred)
        if label:
            pseudo_row = build_training_pseudo_row(row, label)
            if pseudo_row:
                pseudo_rows.append(pseudo_row)
    pseudo_rows = assign_pseudo_quality(pseudo_rows, base_weight=args.pseudo_base_weight)
    write_jsonl(run_dir / "target_pseudo.jsonl", pseudo_rows)
    selected_rows, selected_stats = select_high_confidence_pseudo_rows(
        pseudo_rows,
        min_weight=max(args.pseudo_base_weight, 0.65),
        max_triplets=3,
    )
    high_precision_rows, high_precision_stats = select_high_precision_pseudo_rows(
        pseudo_rows,
        min_weight=max(args.pseudo_base_weight, 0.65),
        max_triplets=args.high_precision_max_triplets,
        max_token_distance=args.high_precision_max_token_distance,
    )
    train_selected_rows, train_selected_stats = select_train_pseudo_rows(
        pseudo_rows,
        min_weight=max(args.pseudo_base_weight, 0.65),
        fixed_changed_min_score=args.fixed_changed_min_score,
        fixed_changed_weight=args.fixed_changed_weight,
    )
    write_jsonl(run_dir / "target_pseudo_selected.jsonl", selected_rows)
    write_jsonl(run_dir / "target_pseudo_high_precision.jsonl", high_precision_rows)
    write_jsonl(run_dir / "target_pseudo_train_selected.jsonl", train_selected_rows)
    dump_json(run_dir / "target_pseudo_selected_analysis.json", selected_stats)
    dump_json(run_dir / "target_pseudo_high_precision_analysis.json", high_precision_stats)
    dump_json(run_dir / "target_pseudo_train_selected_analysis.json", train_selected_stats)
    gold_path = run_dir / "target_train_gold_analysis.jsonl"
    if gold_path.exists():
        gold_rows = {row["id"]: row for row in read_jsonl(gold_path)}
        analysis, analysis_rows = build_pseudo_analysis(target_rows, pseudo_rows, gold_rows)
        analysis["sample_weight_summary"] = sample_weight_summary(pseudo_rows)
        selected_stats["hidden_gold_eval"] = evaluate_selected_pseudo_against_hidden_gold(
            selected_rows,
            gold_rows,
            name="strict",
        )
        high_precision_stats["hidden_gold_eval"] = evaluate_selected_pseudo_against_hidden_gold(
            high_precision_rows,
            gold_rows,
            name="high_precision",
        )
        train_selected_stats["hidden_gold_eval"] = evaluate_selected_pseudo_against_hidden_gold(
            train_selected_rows,
            gold_rows,
            name="train_selected",
        )
        dump_json(run_dir / "target_pseudo_analysis.json", analysis)
        dump_json(run_dir / "target_pseudo_selected_analysis.json", selected_stats)
        dump_json(run_dir / "target_pseudo_high_precision_analysis.json", high_precision_stats)
        dump_json(run_dir / "target_pseudo_train_selected_analysis.json", train_selected_stats)
        write_jsonl(run_dir / "target_pseudo_predictions_analysis.jsonl", analysis_rows)
        print(analysis)
    print(f"pseudo rows={len(pseudo_rows)}")


def select_pseudo(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("analysis_outputs") / run_dir.name / "pseudo_select"
    output_dir.mkdir(parents=True, exist_ok=True)
    pseudo_rows = read_jsonl(run_dir / "target_pseudo.jsonl")
    selected_rows, selected_stats = select_high_confidence_pseudo_rows(
        pseudo_rows,
        min_weight=args.min_pseudo_weight,
        max_triplets=3,
    )
    high_precision_rows, high_precision_stats = select_high_precision_pseudo_rows(
        pseudo_rows,
        min_weight=args.min_pseudo_weight,
        max_triplets=args.high_precision_max_triplets,
        max_token_distance=args.high_precision_max_token_distance,
    )
    train_selected_rows, train_selected_stats = select_train_pseudo_rows(
        pseudo_rows,
        min_weight=args.min_pseudo_weight,
        fixed_changed_min_score=args.fixed_changed_min_score,
        fixed_changed_weight=args.fixed_changed_weight,
    )
    write_jsonl(output_dir / "target_pseudo_selected.jsonl", selected_rows)
    write_jsonl(output_dir / "target_pseudo_high_precision.jsonl", high_precision_rows)
    write_jsonl(output_dir / "target_pseudo_train_selected.jsonl", train_selected_rows)
    gold_path = run_dir / "target_train_gold_analysis.jsonl"
    if gold_path.exists():
        gold_rows = {row["id"]: row for row in read_jsonl(gold_path)}
        selected_stats["hidden_gold_eval"] = evaluate_selected_pseudo_against_hidden_gold(
            selected_rows,
            gold_rows,
            name="strict",
        )
        high_precision_stats["hidden_gold_eval"] = evaluate_selected_pseudo_against_hidden_gold(
            high_precision_rows,
            gold_rows,
            name="high_precision",
        )
        train_selected_stats["hidden_gold_eval"] = evaluate_selected_pseudo_against_hidden_gold(
            train_selected_rows,
            gold_rows,
            name="train_selected",
        )
    dump_json(output_dir / "target_pseudo_selected_analysis.json", selected_stats)
    dump_json(output_dir / "target_pseudo_high_precision_analysis.json", high_precision_stats)
    dump_json(output_dir / "target_pseudo_train_selected_analysis.json", train_selected_stats)
    print(
        {
            "output_dir": str(output_dir),
            "strict_rows": len(selected_rows),
            "high_precision_rows": len(high_precision_rows),
            "train_selected_rows": len(train_selected_rows),
            "strict_hidden_gold_eval": selected_stats.get("hidden_gold_eval", {}),
            "high_precision_hidden_gold_eval": high_precision_stats.get("hidden_gold_eval", {}),
        }
    )


def memory(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    source_rows = read_jsonl(run_dir / "source_train.jsonl")
    source_dev_path = run_dir / "source_dev.jsonl"
    if source_dev_path.exists():
        source_rows = source_rows + read_jsonl(source_dev_path)
    pseudo_rows = read_selected_pseudo_rows(run_dir)

    source_memory = build_source_memory(source_rows)
    target_memory = build_target_memory(
        pseudo_rows,
        min_pseudo_weight=args.min_pseudo_weight,
        source_memory=source_memory,
        source_row_count=len(source_rows),
    )
    cross_memory = build_cross_domain_memory(source_memory, target_memory)
    analysis = build_memory_analysis(source_rows, pseudo_rows, source_memory, target_memory, cross_memory, args)
    dump_json(run_dir / "c3da_source_memory.json", source_memory)
    dump_json(run_dir / "c3da_target_memory.json", target_memory)
    dump_json(run_dir / "c3da_cross_domain_memory.json", cross_memory)
    dump_json(run_dir / "c3da_domain_memory_analysis.json", analysis)
    print(analysis)


def augment(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    source_rows = read_jsonl(run_dir / "source_train.jsonl")
    source_dev_rows = read_jsonl(run_dir / "source_dev.jsonl")
    pseudo_rows = read_selected_pseudo_rows(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    target_domain_name = manifest.get("target_dataset", "")
    domain_memory = None
    memory_path = Path(args.memory_path) if args.memory_path else run_dir / "c3da_cross_domain_memory.json"
    if memory_path.exists():
        domain_memory = json.loads(memory_path.read_text(encoding="utf-8"))
    model_path = Path(args.model_path) if args.model_path else Path(args.generator_model_path)
    label_embeddings = None
    if args.augment_prompt_style == "rsda_t5_label_composition":
        label_texts = collect_single_triplet_label_texts(pseudo_rows, min_weight=0.6)
        label_embeddings = encode_text_embeddings(
            model_path=model_path,
            texts=label_texts,
            batch_size=args.embedding_batch_size,
            cuda=args.cuda,
        )
    composition_source_rows = []
    if args.composition_source_file:
        composition_source_rows = read_jsonl(Path(args.composition_source_file))
    requests = build_augmentation_requests(
        source_rows,
        pseudo_rows,
        per_row=args.per_row,
        seed=args.seed,
        prompt_style=args.augment_prompt_style,
        domain_memory=domain_memory,
        channel_mode=args.augment_channel_mode,
        label_embeddings=label_embeddings,
        label_similarity_top_k=args.label_similarity_top_k,
        composition_source_rows=composition_source_rows,
        target_domain_name=target_domain_name,
        domain_prefix_style=args.domain_prefix_style,
        opinion_replacement_mode=args.opinion_replacement_mode,
    )
    output_tag = args.augment_output_tag
    write_jsonl(tagged_output_path(run_dir, "c3da_two_channel_requests.jsonl", output_tag), requests)

    generated_texts = generate_texts(
        model_path=model_path,
        inputs=[row["input"] for row in requests],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        cuda=args.cuda,
        constrained=False,
        length_penalty=args.length_penalty,
    )

    augmented_rows = []
    seen_aug = set()
    filtered_inconsistent = 0
    filtered_channel_inconsistent = 0
    quality_filter_counts: Counter[str] = Counter()
    quality_kept_rows = 0
    for req, generated in zip(requests, generated_texts):
        text = generated.strip()
        label = canonicalize_triplet_text(req["label"])
        channel = str(req.get("channel", ""))
        is_opinion_channel = channel in {
            "opinion_sentiment_channel",
            "masked_opinion_sentiment_channel",
            "masked_opinion_sentiment_editor",
        }
        if not text or not label:
            continue
        quality_passed, quality_reason = filter_augmented_text_quality(text)
        if not quality_passed:
            quality_filter_counts[quality_reason] += 1
            continue
        quality_kept_rows += 1
        if not args.allow_inconsistent_aug:
            if is_opinion_channel:
                if not text or len(text.split()) < 4:
                    filtered_channel_inconsistent += 1
                    continue
            elif not is_consistent_with_label(text, label):
                filtered_inconsistent += 1
                continue
        key = (text.lower(), label.lower())
        if key in seen_aug:
            continue
        seen_aug.add(key)
        augmented_rows.append(
            {
                "text": text,
                "label": label,
                "augmentation": req["channel"],
                "base_text": req["base_text"],
                "base_id": req.get("base_id"),
                "prompt": req["input"],
                "old_triplet": req.get("old_triplet"),
                "new_triplet": req.get("new_triplet"),
                "new_triplets": req.get("new_triplets"),
                "replacement_rank": req.get("replacement_rank"),
                "domain_name": req.get("domain_name"),
                "domain_prefix_style": req.get("domain_prefix_style"),
                "domain_prefix": req.get("domain_prefix"),
                "opinion_replacement_mode": req.get("opinion_replacement_mode"),
                "opinion_replacement_rank": req.get("opinion_replacement_rank"),
            }
        )
    write_jsonl(tagged_output_path(run_dir, "c3da_two_channel_augmented.jsonl", output_tag), augmented_rows)
    write_jsonl(
        tagged_output_path(run_dir, "c3da_augmented_aspect_channel.jsonl", output_tag),
        [row for row in augmented_rows if row.get("augmentation") in {"aspect_channel", "masked_aspect_channel", "rewrite_aspect_channel", "label_to_text_channel", "sentence_fusion_composition_channel"}],
    )
    write_jsonl(
        tagged_output_path(run_dir, "c3da_augmented_opinion_sentiment_channel.jsonl", output_tag),
        [row for row in augmented_rows if row.get("augmentation") in {"opinion_sentiment_channel", "masked_opinion_sentiment_channel"}],
    )
    consistency_kept_rows = len(augmented_rows)

    nli_stats = {"enabled": False}
    if args.nli_model_path:
        augmented_rows, nli_stats = run_nli_filter(
            rows=augmented_rows,
            model_path=args.nli_model_path,
            batch_size=args.nli_batch_size,
            cuda=args.cuda,
        )
        augmented_rows = assign_augment_quality(augmented_rows, base_weight=args.augment_base_weight)
        write_jsonl(tagged_output_path(run_dir, "c3da_two_channel_augmented_nli.jsonl", output_tag), augmented_rows)
    else:
        augmented_rows = assign_augment_quality(augmented_rows, base_weight=args.augment_base_weight)
    after_nli_rows = len(augmented_rows)

    model_filter_stats = {"enabled": False}
    if args.model_filter_path or args.model_filter_model_variant != "none":
        model_filter_path = resolve_extractor_model_path(
            run_dir,
            explicit_model_path=args.model_filter_path,
            variant="best" if args.model_filter_model_variant == "none" else args.model_filter_model_variant,
        )
        augmented_rows, model_filter_removed, model_filter_stats = run_model_filter(
            rows=augmented_rows,
            model_path=model_filter_path,
            batch_size=args.model_filter_batch_size,
            max_new_tokens=args.model_filter_max_new_tokens,
            num_beams=args.model_filter_num_beams,
            cuda=args.cuda,
            mode=args.model_filter_mode,
            use_task_prefix=not args.no_task_prefix,
            constrained=not args.model_filter_no_constrained_decoding,
            channel_aware=args.model_filter_channel_aware,
        )
        augmented_rows = assign_augment_quality(augmented_rows, base_weight=args.augment_base_weight)
        write_jsonl(tagged_output_path(run_dir, "c3da_two_channel_augmented_model_filter.jsonl", output_tag), augmented_rows)
        write_jsonl(
            tagged_output_path(run_dir, "c3da_augmented_opinion_extracted.jsonl", output_tag),
            [row for row in augmented_rows if row.get("model_filter_label_source") == "opinion_extractor"],
        )
        write_jsonl(tagged_output_path(run_dir, "c3da_model_filter_removed.jsonl", output_tag), model_filter_removed)
        dump_json(tagged_output_path(run_dir, "c3da_model_filter_analysis.json", output_tag), model_filter_stats)

    before_opinion_boundary_filter = len(augmented_rows)
    augmented_rows = [row for row in augmented_rows if opinion_augmented_label_boundary_valid(row)]
    filtered_opinion_boundary = before_opinion_boundary_filter - len(augmented_rows)

    selected_augmented_rows, selection_stats = select_high_value_augmented_rows(
        augmented_rows,
        max_rows=args.augment_select_max_rows,
        max_per_base=args.augment_select_max_per_base,
        selected_weight=args.augment_select_weight,
        max_opinion_ratio=args.augment_select_max_opinion_ratio,
        require_raw_exact=args.augment_select_require_raw_exact,
        require_model_filter_passed=args.augment_select_require_model_filter_passed,
    )
    write_jsonl(tagged_output_path(run_dir, "c3da_two_channel_augmented_selected.jsonl", output_tag), selected_augmented_rows)

    aug_stats = {
        "requests": len(requests),
        "generated": len(generated_texts),
        "prompt_style": args.augment_prompt_style,
        "augment_channel_mode": args.augment_channel_mode,
        "domain_prefix_style": args.domain_prefix_style,
        "opinion_replacement_mode": args.opinion_replacement_mode,
        "target_domain_name": target_domain_name if args.domain_prefix_style != "none" else "",
        "output_tag": output_tag,
        "selected_output_path": str(tagged_output_path(run_dir, "c3da_two_channel_augmented_selected.jsonl", output_tag)),
        "memory_path": str(memory_path) if domain_memory is not None else "",
        "after_quality_filter": quality_kept_rows,
        "after_consistency_filter": consistency_kept_rows,
        "after_nli_filter": after_nli_rows,
        "final_augmented_rows": len(augmented_rows),
        "filtered_inconsistent": filtered_inconsistent,
        "filtered_channel_inconsistent": filtered_channel_inconsistent,
        "filtered_opinion_boundary": filtered_opinion_boundary,
        "quality_filter": dict(quality_filter_counts),
        "filtered_prompt_leak": quality_filter_counts.get("prompt_leak", 0),
        "augmentation_distribution": augmentation_distribution(augmented_rows),
        "channel_analysis": augmentation_channel_analysis(augmented_rows),
        "sentiment_distribution": sentiment_distribution(augmented_rows),
        "sample_weight_summary": sample_weight_summary(augmented_rows),
        "selected_augmented_rows": len(selected_augmented_rows),
        "selected_sample_weight_summary": sample_weight_summary(selected_augmented_rows),
        "selection": selection_stats,
        "nli": nli_stats,
        "model_filter": model_filter_stats,
    }
    dump_json(tagged_output_path(run_dir, "c3da_augment_analysis.json", output_tag), aug_stats)
    if args.augment_prompt_style in {"label_composition", "label_to_text", "sentence_fusion_composition"}:
        dump_json(
            tagged_output_path(run_dir, "label_to_text_augment_analysis.json", output_tag),
            {
                **aug_stats,
                "generator_model_path": str(args.model_path or args.generator_model_path),
                "label_to_text_rows": sum(
                    1
                    for row in augmented_rows
                    if row.get("augmentation") in {"label_to_text_channel", "label_composition_channel", "sentence_fusion_composition_channel"}
                ),
                "selected_label_to_text_rows": sum(
                    1
                    for row in selected_augmented_rows
                    if row.get("augmentation") in {"label_to_text_channel", "label_composition_channel", "sentence_fusion_composition_channel"}
                ),
                "multi_triplet_augmented_rows": sum(
                    1 for row in augmented_rows if len(parse_triplet_text_list(row.get("label", ""))) >= 2
                ),
                "selected_multi_triplet_augmented_rows": sum(
                    1 for row in selected_augmented_rows if len(parse_triplet_text_list(row.get("label", ""))) >= 2
                ),
            },
        )

    pseudo_train_file = Path(args.pseudo_train_file) if args.pseudo_train_file else None
    train_pseudo_rows = read_pseudo_rows_for_training(run_dir, args.pseudo_train_source, pseudo_train_file)
    pseudo_mix_stats = {"enabled": False}
    if args.pseudo_train_source == "mixed_recall" and pseudo_train_file is None:
        high_precision_rows = read_pseudo_rows_for_training(run_dir, "high_precision", None)
        recall_rows = read_pseudo_rows_for_training(run_dir, args.recall_pseudo_source, None)
        train_pseudo_rows, pseudo_mix_stats = build_mixed_recall_pseudo_rows(
            high_precision_rows,
            recall_rows,
            recall_extra_weight=args.recall_extra_weight,
            recall_extra_max_rows=args.recall_extra_max_rows,
        )
        pseudo_mix_stats["enabled"] = True
        pseudo_mix_stats["recall_pseudo_source"] = args.recall_pseudo_source
        dump_json(run_dir / "target_pseudo_mixed_recall_analysis.json", pseudo_mix_stats)
    extra_aug_rows, extra_aug_stats = read_extra_augmented_rows(args.extra_augmented_files, args.extra_augmented_weight)
    pseudo_train_eval = {"enabled": False}
    gold_path = run_dir / "target_train_gold_analysis.jsonl"
    if gold_path.exists():
        gold_rows = {row["id"]: row for row in read_jsonl(gold_path)}
        pseudo_train_eval = {
            "enabled": True,
            "pseudo_train_source": args.pseudo_train_source,
            **evaluate_selected_pseudo_against_hidden_gold(
                train_pseudo_rows,
                gold_rows,
                name=args.pseudo_train_source,
            ),
        }
        dump_json(run_dir / "target_pseudo_used_for_training_analysis.json", pseudo_train_eval)
    combined_aug_rows = selected_augmented_rows + extra_aug_rows
    include_source_in_final_train = not args.no_final_train_source
    final_rows = build_final_training_rows(
        source_rows,
        train_pseudo_rows,
        combined_aug_rows,
        include_source=include_source_in_final_train,
    )
    final_rows, final_weight_stats = assign_final_training_weights(
        final_rows,
        multi_triplet_gain=args.final_multi_triplet_gain,
        neutral_gain=args.final_neutral_gain,
        max_weight=args.final_max_weight,
    )
    final_train_tag = args.final_train_output_tag
    dump_json(tagged_output_path(run_dir, "final_train_weight_analysis.json", final_train_tag), final_weight_stats)
    final_manifest = {
        "pseudo_train_source": args.pseudo_train_source,
        "pseudo_train_file": str(pseudo_train_file) if pseudo_train_file else "",
        "extra_augmented_files": args.extra_augmented_files,
        "extra_augmented_weight": args.extra_augmented_weight,
        "source_rows_available": len(source_rows),
        "source_rows_used": len(source_rows) if include_source_in_final_train else 0,
        "final_train_include_source": include_source_in_final_train,
        "pseudo_rows_used": len(train_pseudo_rows),
        "pseudo_mix_stats": pseudo_mix_stats,
        "selected_augmented_rows": len(selected_augmented_rows),
        "extra_augmented_rows": len(extra_aug_rows),
        "final_train_rows": len(final_rows),
        "domain_prefix_style": args.domain_prefix_style,
        "pseudo_train_hidden_gold_eval": pseudo_train_eval,
        "extra_augmented_analysis": extra_aug_stats,
        "final_weight_stats": final_weight_stats,
    }
    dump_json(tagged_output_path(run_dir, "final_train_composition_analysis.json", final_train_tag), final_manifest)
    write_jsonl(tagged_output_path(run_dir, "final_train.jsonl", final_train_tag), to_extract_rows(final_rows, use_task_prefix=not args.no_task_prefix))
    write_jsonl(tagged_output_path(run_dir, "final_dev.jsonl", final_train_tag), to_extract_rows(source_dev_rows, use_task_prefix=not args.no_task_prefix))
    print(
        f"requests={len(requests)}, augmented={len(augmented_rows)}, selected_augmented={len(selected_augmented_rows)}, "
        f"filtered_inconsistent={filtered_inconsistent}, "
        f"nli_filtered={nli_stats.get('filtered_contradiction', 0)}, "
        f"model_filter_removed={model_filter_stats.get('removed_rows', 0)}, "
        f"pseudo_train_source={args.pseudo_train_source}, pseudo_train_file={args.pseudo_train_file}, pseudo_rows_used={len(train_pseudo_rows)}, "
        f"source_rows_used={len(source_rows) if include_source_in_final_train else 0}, "
        f"final_train={len(final_rows)}, final_dev={len(source_dev_rows)}"
    )


def evaluate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    model_path = Path(args.model_path) if args.model_path else run_dir / "models" / "final" / "best"
    if not model_path.exists():
        model_path = run_dir / "models" / "extractor" / "best"
    rows = read_jsonl(run_dir / "target_test.jsonl")
    preds = generate_texts(
        model_path=model_path,
        inputs=build_extract_inputs(rows, use_task_prefix=not args.no_task_prefix),
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        cuda=args.cuda,
        constrained=not args.no_constrained_decoding,
        length_penalty=args.length_penalty,
    )
    preds = [canonicalize_triplet_text(pred) for pred in preds]
    golds = [canonicalize_triplet_text(row["label"]) for row in rows]
    eval_rows = [{"text": r["text"], "gold": g, "pred": p} for r, g, p in zip(rows, golds, preds)]
    result = evaluate_raw_and_fixed(eval_rows)
    raw_metrics = result["raw_scores"]
    fixed_metrics = result["fixed_scores"]
    dump_json(run_dir / "aste_metrics.json", raw_metrics)
    dump_json(run_dir / "aste_metrics_raw.json", raw_metrics)
    dump_json(run_dir / "aste_metrics_fixed.json", fixed_metrics)
    write_jsonl(run_dir / "aste_predictions.jsonl", [{"text": row["text"], "gold": row["gold"], "pred": row["pred_raw"]} for row in result["predictions"]])
    write_jsonl(run_dir / "aste_predictions_raw_fixed.jsonl", result["predictions"])
    print({"raw_scores": raw_metrics, "fixed_scores": fixed_metrics})


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--source_dataset", required=True, choices=DATASETS.keys())
    p.add_argument("--target_dataset", required=True, choices=DATASETS.keys())
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dev_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument(
        "--augment_prompt_style",
        choices=[
            "concept",
            "legacy",
            "masked_mutual",
            "rewrite_aspect",
            "label_composition",
            "label_to_text",
            "rsda_t5_label_composition",
            "sentence_fusion_composition",
        ],
        default="concept",
    )
    p.add_argument("--augment_channel_mode", choices=["all", "aspect", "opinion"], default="all")
    p.add_argument("--domain_prefix_style", choices=["none", "text", "bracket"], default="none")
    p.add_argument("--generator_output_tag", default="")
    p.add_argument("--no_task_prefix", action="store_true")
    p.set_defaults(func=prepare)

    p = sub.add_parser("pseudo")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model_path", default="")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--max_target_unlabeled", type=int, default=0)
    p.add_argument("--cuda", default="0")
    p.add_argument("--no_constrained_decoding", action="store_true")
    p.add_argument("--pseudo_base_weight", type=float, default=0.5)
    p.add_argument("--pseudo_model_variant", choices=["best", "last"], default="best")
    p.add_argument("--high_precision_max_triplets", type=int, default=1)
    p.add_argument("--high_precision_max_token_distance", type=int, default=5)
    p.add_argument("--fixed_changed_min_score", type=float, default=0.65)
    p.add_argument("--fixed_changed_weight", type=float, default=0.35)
    p.add_argument("--no_task_prefix", action="store_true")
    p.set_defaults(func=pseudo)

    p = sub.add_parser("memory")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--min_pseudo_weight", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=30)
    p.set_defaults(func=memory)

    p = sub.add_parser("augment")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model_path", default="")
    p.add_argument("--generator_model_path", default=r"J:\nlp\models\mrm8488-t5-base-finetuned-common_gen")
    p.add_argument("--nli_model_path", default=r"J:\nlp\models\nli-deberta-v3-base-mnli-fever-anli")
    p.add_argument("--nli_batch_size", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--per_row", type=int, default=1)
    p.add_argument("--dev_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument(
        "--augment_prompt_style",
        choices=[
            "concept",
            "legacy",
            "masked_mutual",
            "rewrite_aspect",
            "label_composition",
            "label_to_text",
            "rsda_t5_label_composition",
            "sentence_fusion_composition",
        ],
        default="concept",
    )
    p.add_argument("--augment_channel_mode", choices=["all", "aspect", "opinion"], default="all")
    p.add_argument("--domain_prefix_style", choices=["none", "text", "bracket"], default="none")
    p.add_argument(
        "--opinion_replacement_mode",
        choices=["coupled_random", "semantic_same_sentiment"],
        default="coupled_random",
    )
    p.add_argument("--augment_output_tag", default="")
    p.add_argument("--memory_path", default="")
    p.add_argument("--cuda", default="0")
    p.add_argument("--allow_inconsistent_aug", action="store_true")
    p.add_argument("--augment_base_weight", type=float, default=0.2)
    p.add_argument("--augment_select_max_rows", type=int, default=200)
    p.add_argument("--augment_select_max_per_base", type=int, default=1)
    p.add_argument("--augment_select_weight", type=float, default=0.35)
    p.add_argument("--augment_select_max_opinion_ratio", type=float, default=1.0)
    p.add_argument("--augment_select_require_raw_exact", action="store_true")
    p.add_argument("--augment_select_require_model_filter_passed", action="store_true")
    p.add_argument("--composition_source_file", default="")
    p.add_argument("--label_similarity_top_k", type=int, default=4)
    p.add_argument("--embedding_batch_size", type=int, default=16)
    p.add_argument(
        "--pseudo_train_source",
        choices=["strict", "high_precision", "train_selected", "mixed_recall"],
        default="high_precision",
    )
    p.add_argument("--recall_pseudo_source", choices=["strict", "train_selected"], default="train_selected")
    p.add_argument("--recall_extra_weight", type=float, default=0.25)
    p.add_argument("--recall_extra_max_rows", type=int, default=0)
    p.add_argument("--pseudo_train_file", default="")
    p.add_argument("--extra_augmented_files", default="")
    p.add_argument("--extra_augmented_weight", type=float, default=0.35)
    p.add_argument("--no_final_train_source", action="store_true")
    p.add_argument("--high_precision_max_triplets", type=int, default=1)
    p.add_argument("--high_precision_max_token_distance", type=int, default=5)
    p.add_argument("--fixed_changed_min_score", type=float, default=0.65)
    p.add_argument("--fixed_changed_weight", type=float, default=0.35)
    p.add_argument("--model_filter_path", default="")
    p.add_argument("--model_filter_batch_size", type=int, default=2)
    p.add_argument("--model_filter_mode", choices=["exact", "fixed"], default="fixed")
    p.add_argument("--model_filter_max_new_tokens", type=int, default=96)
    p.add_argument("--model_filter_num_beams", type=int, default=1)
    p.add_argument("--model_filter_model_variant", choices=["none", "best", "last"], default="none")
    p.add_argument("--model_filter_no_constrained_decoding", action="store_true")
    p.add_argument("--model_filter_channel_aware", action="store_true")
    p.add_argument("--final_multi_triplet_gain", type=float, default=0.0)
    p.add_argument("--final_neutral_gain", type=float, default=0.0)
    p.add_argument("--final_max_weight", type=float, default=1.0)
    p.add_argument("--final_train_output_tag", default="")
    p.add_argument("--no_task_prefix", action="store_true")
    p.set_defaults(func=augment)

    p = sub.add_parser("evaluate")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model_path", default="")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--cuda", default="0")
    p.add_argument("--no_constrained_decoding", action="store_true")
    p.add_argument("--no_task_prefix", action="store_true")
    p.set_defaults(func=evaluate)

    p = sub.add_parser("select_pseudo")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--output_dir", default="")
    p.add_argument("--min_pseudo_weight", type=float, default=0.65)
    p.add_argument("--high_precision_max_triplets", type=int, default=1)
    p.add_argument("--high_precision_max_token_distance", type=int, default=5)
    p.add_argument("--fixed_changed_min_score", type=float, default=0.65)
    p.add_argument("--fixed_changed_weight", type=float, default=0.35)
    p.set_defaults(func=select_pseudo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

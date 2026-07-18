from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Iterable


SENTIMENT_TO_TOKEN = {
    "POS": "pos",
    "NEG": "neg",
    "NEU": "neu",
    "positive": "pos",
    "negative": "neg",
    "neutral": "neu",
    "pos": "pos",
    "neg": "neg",
    "neu": "neu",
}

TOKEN_TO_SENTIMENT = {
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_json(path: str | Path, obj: dict | list) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _span_text(tokens: list[str], indexes: list[int]) -> str:
    if not indexes:
        return ""
    return normalize_space(" ".join(tokens[indexes[0] : indexes[-1] + 1]))


def triplets_to_text(triplets: Iterable[tuple[str, str, str]]) -> str:
    parts: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for aspect, opinion, sentiment in triplets:
        aspect = normalize_space(aspect)
        opinion = normalize_space(opinion)
        sentiment = SENTIMENT_TO_TOKEN.get(sentiment, sentiment.lower())
        if not aspect or not opinion or sentiment not in {"pos", "neg", "neu"}:
            continue
        key = (aspect.lower(), opinion.lower(), sentiment)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"<{sentiment}> {aspect} <opinion> {opinion}")
    return " ; ".join(parts)


def parse_triplet_text(label_text: str) -> set[tuple[str, str, str]]:
    return set(parse_triplet_text_list(label_text))


def parse_triplet_text_list(label_text: str) -> list[tuple[str, str, str]]:
    text = normalize_space(label_text)
    if not text:
        return []
    pattern = re.compile(
        r"<?(pos|neg|neu)>\s+(.+?)\s*<?opinion>\s*(.+?)(?=\s*;\s*<?(?:pos|neg|neu)>|\s*<?pos>|\s*<?neg>|\s*<?neu>|$)",
        re.I,
    )
    records: list[tuple[str, str, str]] = []
    for match in pattern.finditer(text):
        sentiment = match.group(1).lower()
        aspect = normalize_space(match.group(2)).lower()
        opinion = normalize_space(match.group(3)).lower()
        if aspect and opinion:
            records.append((aspect, opinion, sentiment))
    return records


def canonicalize_triplet_text(label_text: str) -> str:
    return " ; ".join(
        f"<{sentiment}> {aspect} <opinion> {opinion}"
        for aspect, opinion, sentiment in sorted(parse_triplet_text(label_text))
    )


def read_bgca_aste_file(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            sentence, raw_labels = line.split("####", 1)
            tokens = sentence.split()
            labels = ast.literal_eval(raw_labels)
            triplets = []
            for aspect_idx, opinion_idx, sentiment in labels:
                aspect = _span_text(tokens, list(aspect_idx))
                opinion = _span_text(tokens, list(opinion_idx))
                triplets.append((aspect, opinion, sentiment))
            rows.append(
                {
                    "id": idx,
                    "text": normalize_space(sentence),
                    "label": triplets_to_text(triplets),
                    "source_path": str(source_path),
                }
            )
    return rows


def split_train_dev(rows: list[dict], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_ratio)) if len(shuffled) > 10 else 0
    return shuffled[n_dev:], shuffled[:n_dev]


def to_extract_rows(rows: Iterable[dict], use_task_prefix: bool = True) -> list[dict]:
    converted = []
    for row in rows:
        label = canonicalize_triplet_text(row.get("label", ""))
        input_text = f"extract aste: {row['text']}" if use_task_prefix else row["text"]
        converted.append(
            {
                **row,
                "label": label,
                "input": input_text,
                "target": label,
            }
        )
    return converted


def micro_f1(predictions: list[str], golds: list[str]) -> dict[str, float]:
    tp = fp = fn = 0
    for pred, gold in zip(predictions, golds):
        pred_list = parse_triplet_text_list(pred)
        gold_list = parse_triplet_text_list(gold)
        matched_gold = list(gold_list)
        tp_i = 0
        for triplet in pred_list:
            if triplet in matched_gold:
                matched_gold.remove(triplet)
                tp_i += 1
        tp += tp_i
        fp += len(pred_list) - tp_i
        fn += len(gold_list) - tp_i
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def triplet_count_bucket(count: int) -> str:
    if count <= 1:
        return "count1"
    if count == 2:
        return "count2"
    if count == 3:
        return "count3"
    return "count4plus"


def micro_f1_by_triplet_count(predictions: list[str], golds: list[str]) -> dict[str, dict]:
    grouped: dict[str, tuple[list[str], list[str]]] = {
        bucket: ([], [])
        for bucket in ("count1", "count2", "count3", "count4plus")
    }
    for prediction, gold in zip(predictions, golds):
        bucket = triplet_count_bucket(len(parse_triplet_text_list(gold)))
        grouped[bucket][0].append(prediction)
        grouped[bucket][1].append(gold)

    return {
        bucket: {"rows": len(bucket_golds), **micro_f1(bucket_predictions, bucket_golds)}
        for bucket, (bucket_predictions, bucket_golds) in grouped.items()
    }


def triplet_count_diagnostics(predictions: list[str], golds: list[str]) -> dict[str, int | float]:
    exact_count_rows = under_generated_rows = over_generated_rows = 0
    rows = 0
    for prediction, gold in zip(predictions, golds):
        rows += 1
        prediction_count = len(parse_triplet_text_list(prediction))
        gold_count = len(parse_triplet_text_list(gold))
        if prediction_count == gold_count:
            exact_count_rows += 1
        elif prediction_count < gold_count:
            under_generated_rows += 1
        else:
            over_generated_rows += 1

    return {
        "rows": rows,
        "exact_count_rows": exact_count_rows,
        "under_generated_rows": under_generated_rows,
        "over_generated_rows": over_generated_rows,
        "exact_count_accuracy": exact_count_rows / rows if rows else 0.0,
    }

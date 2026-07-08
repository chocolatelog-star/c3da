from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable


SENTIMENT_MAP = {
    "positive": "pos",
    "negative": "neg",
    "neutral": "neu",
    "POS": "pos",
    "NEG": "neg",
    "NEU": "neu",
    "pos": "pos",
    "neg": "neg",
    "neu": "neu",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_json(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: str | Path, obj: dict | list) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_c3da_json(path: str | Path) -> list[dict]:
    rows = []
    for idx, item in enumerate(read_json(path)):
        text = normalize_space(" ".join(item["token"]))
        labels = []
        for aspect in item.get("aspects", []):
            aspect_text = normalize_space(" ".join(aspect.get("term", [])))
            if not aspect_text:
                continue
            sentiment = SENTIMENT_MAP.get(aspect.get("polarity", ""), "neu")
            labels.append((sentiment, aspect_text))
        rows.append(
            {
                "id": idx,
                "text": text,
                "label": labels_to_text(labels),
                "source_path": str(path),
            }
        )
    return rows


def split_train_dev(rows: list[dict], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_dev = max(1, int(len(rows) * dev_ratio)) if len(rows) > 10 else 0
    return rows[n_dev:], rows[:n_dev]


def labels_to_text(labels: Iterable[tuple[str, str]]) -> str:
    parts = []
    seen = set()
    for sentiment, aspect in labels:
        sentiment = SENTIMENT_MAP.get(sentiment, sentiment)
        aspect = normalize_space(aspect)
        if not aspect:
            continue
        key = (sentiment, aspect.lower())
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"<{sentiment}> {aspect}")
    return " ; ".join(parts)


def parse_label_text(label_text: str) -> set[tuple[str, str]]:
    label_text = normalize_space(label_text)
    if not label_text:
        return set()
    records = set()
    for chunk in [c.strip() for c in label_text.split(";") if c.strip()]:
        match = re.match(r"^<?(pos|neg|neu)>\s+(.+)$", chunk, re.I)
        if match:
            sentiment = match.group(1).lower()
            aspect = normalize_space(match.group(2)).lower()
            if aspect:
                records.add((sentiment, aspect))
    return records


def canonicalize_label_text(label_text: str) -> str:
    return " ; ".join(f"<{sentiment}> {aspect}" for sentiment, aspect in sorted(parse_label_text(label_text)))


def to_extract_rows(rows: Iterable[dict]) -> list[dict]:
    converted = []
    for row in rows:
        converted.append(
            {
                "input": f"extract aesc: {row['text']}",
                "target": canonicalize_label_text(row.get("label", "")),
                **row,
            }
        )
    return converted


def to_generate_rows(rows: Iterable[dict]) -> list[dict]:
    converted = []
    for row in rows:
        label = canonicalize_label_text(row.get("label", ""))
        if label:
            converted.append(
                {
                    "input": f"generate aesc: {label}",
                    "target": row["text"],
                    **row,
                    "label": label,
                }
            )
    return converted


def micro_f1(predictions: list[str], golds: list[str]) -> dict[str, float]:
    tp = fp = fn = 0
    for pred, gold in zip(predictions, golds):
        pred_set = parse_label_text(pred)
        gold_set = parse_label_text(gold)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
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


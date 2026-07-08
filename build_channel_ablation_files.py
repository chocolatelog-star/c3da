from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


VARIANTS = {
    "no_aug": {"target_pseudo"},
    "aspect_only": {"target_pseudo", "aspect_channel"},
    "opinion_only": {"target_pseudo", "opinion_sentiment_channel"},
    "both": {"target_pseudo", "aspect_channel", "opinion_sentiment_channel"},
}


def row_source(row: dict) -> str:
    if row.get("augmentation"):
        return row["augmentation"]
    if row.get("source_path"):
        return "source_gold"
    return "unknown"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(run_dir / "final_train.jsonl")
    print("final_train distribution:", dict(Counter(row_source(row) for row in rows)))

    for name, allowed_aug in VARIANTS.items():
        kept = []
        for row in rows:
            source = row_source(row)
            if source == "source_gold" or source in allowed_aug:
                kept.append(row)
        out_path = output_dir / f"final_train_ablation_{name}.jsonl"
        write_jsonl(out_path, kept)
        print(name, len(kept), dict(Counter(row_source(row) for row in kept)), out_path)


if __name__ == "__main__":
    main()

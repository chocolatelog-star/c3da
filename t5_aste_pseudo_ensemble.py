from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from t5_aste_data import canonicalize_triplet_text, dump_json, parse_triplet_text, read_jsonl, triplets_to_text, write_jsonl
from t5_aste_pipeline import build_pseudo_analysis


def build_ensemble_rows(runs: list[list[dict]], mode: str) -> list[dict]:
    if mode not in {"union", "vote2"}:
        raise ValueError(f"Unsupported ensemble mode: {mode}")
    if not runs:
        return []

    row_count = len(runs[0])
    for rows in runs:
        if len(rows) != row_count:
            raise ValueError("All input files must contain the same number of rows.")

    ensemble_rows = []
    for row_idx in range(row_count):
        first = runs[0][row_idx]
        triplet_counts: Counter[tuple[str, str, str]] = Counter()
        for rows in runs:
            row = rows[row_idx]
            if row["id"] != first["id"] or row["text"] != first["text"]:
                raise ValueError("Input files are not aligned by id/text.")
            triplet_counts.update(parse_triplet_text(row.get("pseudo", "")))

        if mode == "union":
            kept = sorted(triplet_counts)
        else:
            kept = sorted(triplet for triplet, count in triplet_counts.items() if count >= 2)
        ensemble_rows.append(
            {
                "id": first["id"],
                "text": first["text"],
                "gold": first.get("gold", ""),
                "pseudo": canonicalize_triplet_text(triplets_to_text(kept)),
                "ensemble_mode": mode,
                "source_votes": {str(triplet): count for triplet, count in sorted(triplet_counts.items())},
            }
        )
    return ensemble_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--mode", choices=["union", "vote2"], required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    runs = [read_jsonl(path) for path in args.inputs]
    ensemble_rows = build_ensemble_rows(runs, args.mode)
    target_rows = [{"id": row["id"], "text": row["text"]} for row in ensemble_rows]
    pseudo_rows = [
        {"id": row["id"], "text": row["text"], "label": row["pseudo"], "augmentation": f"pseudo_ensemble_{args.mode}"}
        for row in ensemble_rows
        if row["pseudo"]
    ]
    gold_rows = {row["id"]: {"label": row["gold"]} for row in ensemble_rows}
    analysis, analysis_rows = build_pseudo_analysis(target_rows, pseudo_rows, gold_rows)
    analysis["ensemble_mode"] = args.mode
    analysis["input_files"] = [str(Path(path)) for path in args.inputs]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / f"target_pseudo_ensemble_{args.mode}.jsonl", ensemble_rows)
    write_jsonl(output_dir / f"target_pseudo_ensemble_{args.mode}_analysis.jsonl", analysis_rows)
    dump_json(output_dir / f"target_pseudo_ensemble_{args.mode}_analysis.json", analysis)
    print(analysis)


if __name__ == "__main__":
    main()

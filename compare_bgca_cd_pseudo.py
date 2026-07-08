from __future__ import annotations

import argparse
from pathlib import Path

from t5_aste_data import dump_json, parse_triplet_text, read_jsonl, write_jsonl


def _triplet_key(triplet: tuple[str, str, str]) -> str:
    aspect, opinion, sentiment = triplet
    return f"{sentiment} || {aspect} || {opinion}"


def _rows_by_text(rows: list[dict]) -> dict[str, dict]:
    return {row["text"]: row for row in rows}


def compare_pseudo_rows(
    bgca_rows: list[dict],
    cd_rows: list[dict],
    bgca_field: str,
    cd_field: str,
) -> dict:
    cd_by_text = _rows_by_text(cd_rows)
    records = []
    summary = {
        "rows_compared": 0,
        "both_correct_triplets": 0,
        "bgca_only_correct_triplets": 0,
        "cd_only_correct_triplets": 0,
        "both_missed_gold_triplets": 0,
        "bgca_false_positive_triplets": 0,
        "cd_false_positive_triplets": 0,
    }

    for bgca_row in bgca_rows:
        text = bgca_row["text"]
        if text not in cd_by_text:
            continue
        cd_row = cd_by_text[text]
        gold = parse_triplet_text(bgca_row.get("gold", ""))
        bgca_pred = parse_triplet_text(bgca_row.get(bgca_field, ""))
        cd_pred = parse_triplet_text(cd_row.get(cd_field, ""))

        both_correct = sorted(gold & bgca_pred & cd_pred)
        bgca_only_correct = sorted((gold & bgca_pred) - cd_pred)
        cd_only_correct = sorted((gold & cd_pred) - bgca_pred)
        both_missed_gold = sorted(gold - bgca_pred - cd_pred)
        bgca_false_positive = sorted(bgca_pred - gold)
        cd_false_positive = sorted(cd_pred - gold)

        summary["rows_compared"] += 1
        summary["both_correct_triplets"] += len(both_correct)
        summary["bgca_only_correct_triplets"] += len(bgca_only_correct)
        summary["cd_only_correct_triplets"] += len(cd_only_correct)
        summary["both_missed_gold_triplets"] += len(both_missed_gold)
        summary["bgca_false_positive_triplets"] += len(bgca_false_positive)
        summary["cd_false_positive_triplets"] += len(cd_false_positive)

        records.append(
            {
                "text": text,
                "gold": bgca_row.get("gold", ""),
                "bgca_pseudo": bgca_row.get(bgca_field, ""),
                "cd_pseudo": cd_row.get(cd_field, ""),
                "both_correct": [_triplet_key(t) for t in both_correct],
                "bgca_only_correct": [_triplet_key(t) for t in bgca_only_correct],
                "cd_only_correct": [_triplet_key(t) for t in cd_only_correct],
                "both_missed_gold": [_triplet_key(t) for t in both_missed_gold],
                "bgca_false_positive": [_triplet_key(t) for t in bgca_false_positive],
                "cd_false_positive": [_triplet_key(t) for t in cd_false_positive],
            }
        )

    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgca", required=True)
    parser.add_argument("--cd", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bgca_field", default="pseudo_fixed")
    parser.add_argument("--cd_field", default="pseudo_fixed")
    args = parser.parse_args()

    result = compare_pseudo_rows(
        read_jsonl(args.bgca),
        read_jsonl(args.cd),
        bgca_field=args.bgca_field,
        cd_field=args.cd_field,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = result["records"]
    dump_json(output_dir / "error_type_summary.json", result["summary"])
    write_jsonl(output_dir / "pseudo_compare_all.jsonl", records)
    write_jsonl(output_dir / "bgca_only_correct.jsonl", [row for row in records if row["bgca_only_correct"]])
    write_jsonl(output_dir / "cd_only_correct.jsonl", [row for row in records if row["cd_only_correct"]])
    write_jsonl(output_dir / "both_wrong.jsonl", [row for row in records if row["both_missed_gold"]])
    write_jsonl(output_dir / "false_positive_compare.jsonl", [row for row in records if row["bgca_false_positive"] or row["cd_false_positive"]])
    print(result["summary"])


if __name__ == "__main__":
    main()

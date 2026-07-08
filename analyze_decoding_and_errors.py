from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from t5_aste_data import canonicalize_triplet_text, dump_json, parse_triplet_text_list, read_jsonl, write_jsonl
from t5_aste_pipeline import build_extract_inputs, evaluate_raw_and_fixed, generate_texts


def _triplet_set(label: str) -> set[tuple[str, str, str]]:
    return set(parse_triplet_text_list(canonicalize_triplet_text(label)))


def _sentiment_counts(triplets: set[tuple[str, str, str]]) -> Counter:
    return Counter(sentiment for _aspect, _opinion, sentiment in triplets)


def _error_summary(prediction_rows: list[dict]) -> dict:
    counters: Counter = Counter()
    by_gold_count: Counter = Counter()
    by_pred_count: Counter = Counter()
    sentiment_gold: Counter = Counter()
    sentiment_fn: Counter = Counter()
    sentiment_fp: Counter = Counter()
    examples = defaultdict(list)

    for row in prediction_rows:
        gold = _triplet_set(row.get("gold", ""))
        pred = _triplet_set(row.get("pred_fixed", row.get("pred", "")))
        tp = gold & pred
        fp = pred - gold
        fn = gold - pred

        by_gold_count[len(gold)] += 1
        by_pred_count[len(pred)] += 1
        sentiment_gold.update(_sentiment_counts(gold))
        sentiment_fn.update(_sentiment_counts(fn))
        sentiment_fp.update(_sentiment_counts(fp))

        if not pred and gold:
            counters["empty_pred_with_gold"] += 1
            if len(examples["empty_pred_with_gold"]) < 5:
                examples["empty_pred_with_gold"].append(row)
        if len(gold) > len(pred):
            counters["gold_more_than_pred"] += 1
            if len(examples["gold_more_than_pred"]) < 5:
                examples["gold_more_than_pred"].append(row)
        if len(pred) > len(gold):
            counters["pred_more_than_gold"] += 1
        if fp:
            counters["rows_with_fp"] += 1
        if fn:
            counters["rows_with_fn"] += 1

        gold_aspect_opinion = {(aspect, opinion) for aspect, opinion, _sentiment in gold}
        pred_aspect_opinion = {(aspect, opinion) for aspect, opinion, _sentiment in pred}
        if gold_aspect_opinion & pred_aspect_opinion and not tp:
            counters["possible_sentiment_error_rows"] += 1
            if len(examples["possible_sentiment_error_rows"]) < 5:
                examples["possible_sentiment_error_rows"].append(row)

    return {
        "rows": len(prediction_rows),
        "counters": dict(counters),
        "gold_triplet_count_distribution": dict(sorted(by_gold_count.items())),
        "pred_triplet_count_distribution": dict(sorted(by_pred_count.items())),
        "gold_sentiment_distribution": dict(sentiment_gold),
        "fn_sentiment_distribution": dict(sentiment_fn),
        "fp_sentiment_distribution": dict(sentiment_fp),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--settings", default="1:96,4:96,6:96,4:128,6:128")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "decoding_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    rows = read_jsonl(run_dir / "target_test.jsonl")
    settings = []
    for item in args.settings.split(","):
        beams, max_len = item.split(":")
        settings.append((int(beams), int(max_len)))

    summary = []
    for beams, max_len in settings:
        preds = generate_texts(
            model_path=model_path,
            inputs=build_extract_inputs(rows, use_task_prefix=False),
            batch_size=args.batch_size,
            max_new_tokens=max_len,
            num_beams=beams,
            cuda=args.cuda,
            constrained=True,
        )
        preds = [canonicalize_triplet_text(pred) for pred in preds]
        golds = [canonicalize_triplet_text(row["label"]) for row in rows]
        eval_rows = [{"text": r["text"], "gold": g, "pred": p} for r, g, p in zip(rows, golds, preds)]
        result = evaluate_raw_and_fixed(eval_rows)
        tag = f"beam{beams}_len{max_len}"
        dump_json(output_dir / f"decoding_metrics_raw_{tag}.json", result["raw_scores"])
        dump_json(output_dir / f"decoding_metrics_fixed_{tag}.json", result["fixed_scores"])
        write_jsonl(output_dir / f"decoding_predictions_{tag}.jsonl", result["predictions"])
        error_summary = _error_summary(result["predictions"])
        dump_json(output_dir / f"decoding_error_summary_{tag}.json", error_summary)
        row = {"tag": tag, "raw": result["raw_scores"], "fixed": result["fixed_scores"], "errors": error_summary["counters"]}
        summary.append(row)
        print(row)

    dump_json(output_dir / "decoding_summary.json", summary)


if __name__ == "__main__":
    main()

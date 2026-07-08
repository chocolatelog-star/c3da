from __future__ import annotations

import argparse
import os
from pathlib import Path

from t5_absa_data import (
    canonicalize_label_text,
    dump_json,
    load_c3da_json,
    micro_f1,
    read_jsonl,
    split_train_dev,
    to_extract_rows,
    to_generate_rows,
    write_jsonl,
)


DATASET_FILES = {
    "restaurant": {
        "train": "./dataset/Restaurants_corenlp/train.json",
        "test": "./dataset/Restaurants_corenlp/test.json",
    },
    "laptop": {
        "train": "./dataset/Laptops_corenlp/train.json",
        "test": "./dataset/Laptops_corenlp/test.json",
    },
    "twitter": {
        "train": "./dataset/Tweets_corenlp/train.json",
        "test": "./dataset/Tweets_corenlp/test.json",
    },
}


def generate_texts(
    model_path: str | Path,
    inputs: list[str],
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    cuda: str = "0",
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
    outputs = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"generate:{Path(model_path).name}"):
            encoded = tokenizer(list(batch), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, num_beams=num_beams)
            outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def prepare(args: argparse.Namespace) -> None:
    source = args.source_dataset
    target = args.target_dataset
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_c3da_json(DATASET_FILES[source]["train"])
    target_train_rows = load_c3da_json(DATASET_FILES[target]["train"])
    target_test_rows = load_c3da_json(DATASET_FILES[target]["test"])
    source_train, source_dev = split_train_dev(source_rows, args.dev_ratio, args.seed)

    write_jsonl(run_dir / "source_train.jsonl", source_train)
    write_jsonl(run_dir / "source_dev.jsonl", source_dev)
    write_jsonl(run_dir / "target_unlabeled.jsonl", [{"id": r["id"], "text": r["text"], "gold_label": r["label"]} for r in target_train_rows])
    write_jsonl(run_dir / "target_test.jsonl", target_test_rows)
    write_jsonl(run_dir / "extract_train.jsonl", to_extract_rows(source_train))
    write_jsonl(run_dir / "extract_dev.jsonl", to_extract_rows(source_dev))
    write_jsonl(run_dir / "generate_train.jsonl", to_generate_rows(source_train))
    write_jsonl(run_dir / "generate_dev.jsonl", to_generate_rows(source_dev))

    manifest = {
        "task": "aesc",
        "source_dataset": source,
        "target_dataset": target,
        "source_train": len(source_train),
        "source_dev": len(source_dev),
        "target_unlabeled": len(target_train_rows),
        "target_test": len(target_test_rows),
        "extractor_model": str(run_dir / "models" / "extractor" / "best"),
        "generator_model": str(run_dir / "models" / "generator" / "best"),
        "final_model": str(run_dir / "models" / "final" / "best"),
    }
    dump_json(run_dir / "manifest.json", manifest)
    print(f"prepared {run_dir}")
    print(manifest)


def augment(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    target_rows = read_jsonl(run_dir / "target_unlabeled.jsonl")
    if args.max_target_unlabeled > 0:
        target_rows = target_rows[: args.max_target_unlabeled]

    extractor = run_dir / "models" / "extractor" / "best"
    generator = run_dir / "models" / "generator" / "best"

    pseudo_labels = generate_texts(
        extractor,
        [f"extract aesc: {row['text']}" for row in target_rows],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        cuda=args.cuda,
    )
    pseudo_labels = [canonicalize_label_text(label) for label in pseudo_labels]
    generated_texts = generate_texts(
        generator,
        [f"generate aesc: {label}" for label in pseudo_labels],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        cuda=args.cuda,
    )

    pseudo_rows = []
    seen = set()
    for source_row, label, generated in zip(target_rows, pseudo_labels, generated_texts):
        generated = generated.strip()
        if not label or not generated:
            continue
        key = (generated.lower(), label.lower())
        if key in seen:
            continue
        seen.add(key)
        pseudo_rows.append(
            {
                "text": generated,
                "label": label,
                "premise": source_row["text"],
                "augmentation": "t5_cross_domain_pseudo",
            }
        )

    write_jsonl(run_dir / "t5_pseudo_augmented.jsonl", pseudo_rows)
    source_rows = read_jsonl(run_dir / "source_train.jsonl")
    final_rows, final_seen = [], set()
    for row in source_rows + pseudo_rows:
        label = canonicalize_label_text(row.get("label", ""))
        if not label:
            continue
        key = (row["text"].lower(), label.lower())
        if key in final_seen:
            continue
        final_seen.add(key)
        final_rows.append({**row, "label": label})
    final_train, final_dev = split_train_dev(final_rows, args.dev_ratio, args.seed)
    write_jsonl(run_dir / "final_train.jsonl", to_extract_rows(final_train))
    write_jsonl(run_dir / "final_dev.jsonl", to_extract_rows(final_dev))
    print(f"pseudo rows={len(pseudo_rows)}, final_train={len(final_train)}, final_dev={len(final_dev)}")


def evaluate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    model_path = Path(args.model_path) if args.model_path else run_dir / "models" / "final" / "best"
    if not model_path.exists():
        model_path = run_dir / "models" / "extractor" / "best"
    rows = read_jsonl(run_dir / "target_test.jsonl")
    preds = generate_texts(
        model_path,
        [f"extract aesc: {row['text']}" for row in rows],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        cuda=args.cuda,
    )
    preds = [canonicalize_label_text(pred) for pred in preds]
    golds = [canonicalize_label_text(row["label"]) for row in rows]
    metrics = micro_f1(preds, golds)
    dump_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "predictions.jsonl", [{"text": r["text"], "gold": g, "pred": p} for r, g, p in zip(rows, golds, preds)])
    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--source_dataset", required=True, choices=DATASET_FILES.keys())
    p.add_argument("--target_dataset", required=True, choices=DATASET_FILES.keys())
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dev_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1000)
    p.set_defaults(func=prepare)

    p = sub.add_parser("augment")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_target_unlabeled", type=int, default=0)
    p.add_argument("--dev_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--cuda", default="0")
    p.set_defaults(func=augment)

    p = sub.add_parser("evaluate")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model_path", default="")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--cuda", default="0")
    p.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

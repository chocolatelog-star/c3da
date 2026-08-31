from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def validate_upstream_files(upstream: Path) -> tuple[Path, Path]:
    final_data = upstream / "final_data"
    train_candidates = sorted(final_data.glob("final_train_*.jsonl"))
    dev_candidates = sorted(final_data.glob("final_dev_*.jsonl"))
    train = train_candidates[-1] if train_candidates else final_data / "final_train.jsonl"
    dev = dev_candidates[-1] if dev_candidates else final_data / "final_dev.jsonl"
    for path in (train, dev):
        if not path.is_file():
            raise FileNotFoundError(f"fixed upstream artifact is missing: {path}")
    return train, dev


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_final_only_command(*, python: Path, project_root: Path, upstream: Path,
                             output: Path, model: Path, train_batch_size: int,
                             gradient_accumulation_steps: int, epochs: int) -> list[str]:
    train, dev = validate_upstream_files(upstream)
    return [str(python), str(project_root / "t5_absa_train.py"),
            "--model_path", str(model), "--train_file", str(train), "--dev_file", str(dev),
            "--output_dir", str(output), "--num_train_epochs", str(epochs),
            "--checkpoint_selection", "best", "--resume_from_checkpoint", "auto",
            "--per_device_train_batch_size", str(train_batch_size),
            "--per_device_eval_batch_size", "2",
            "--gradient_accumulation_steps", str(gradient_accumulation_steps),
            "--learning_rate", "0.0003", "--source_weight", "1.0",
            "--pseudo_weight", "0.65", "--augment_weight", "0.2",
            "--lambda_domain_adv", "0.03", "--domain_adv_exclude_augment",
            "--lambda_sentiment_contrastive", "0.01", "--sentiment_contrastive_source_only",
            "--sentiment_contrastive_class_balanced", "--fp16", "--gradient_checkpointing",
            "--save_total_limit", "1", "--cuda", "0", "--seed", "1000"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min_free_gb", type=float, default=8.0)
    args = parser.parse_args()
    usage = shutil.disk_usage(args.output.parent)
    if usage.free / (1024**3) < args.min_free_gb:
        raise RuntimeError(f"insufficient free disk space: {usage.free / (1024**3):.2f} GiB")
    args.output.mkdir(parents=True, exist_ok=True)
    train, dev = validate_upstream_files(args.upstream)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "upstream_identity.json").write_text(
        json.dumps({"upstream": str(args.upstream.resolve()), "train": sha256_file(train), "dev": sha256_file(dev)}, indent=2),
        encoding="utf-8",
    )
    command = build_final_only_command(python=Path(sys.executable), project_root=Path(__file__).resolve().parent,
                                       upstream=args.upstream, output=args.output, model=args.model,
                                       train_batch_size=args.train_batch_size,
                                       gradient_accumulation_steps=args.gradient_accumulation_steps,
                                       epochs=args.epochs)
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()

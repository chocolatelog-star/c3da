from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm

from t5_aste_pipeline import build_final_train_from_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pseudo_file = run_dir / "target_pseudo_high_precision.jsonl"
    selected_augment_file = run_dir / "c3da_two_channel_augmented_selected_strict_aug150_w020_label_to_text_gen.jsonl"
    variants = [
        ("strict_ablation_source_pseudo", ""),
        ("strict_ablation_source_pseudo_aug", str(selected_augment_file)),
    ]
    for output_tag, augment_file in tqdm(variants, desc="build:strict-ablation-files"):
        train_path = run_dir / f"final_train_{output_tag}.jsonl"
        dev_path = run_dir / f"final_dev_{output_tag}.jsonl"
        analysis_path = run_dir / f"final_train_composition_analysis_{output_tag}.json"
        if not args.rerun and all(path.exists() for path in (train_path, dev_path, analysis_path)):
            continue
        build_final_train_from_files(
            Namespace(
                run_dir=str(run_dir),
                pseudo_train_file=str(pseudo_file),
                selected_augment_file=augment_file,
                final_train_output_tag=output_tag,
                no_final_train_source=False,
                no_task_prefix=True,
                final_multi_triplet_gain=0.0,
                final_neutral_gain=0.0,
                final_max_weight=1.0,
            )
        )


if __name__ == "__main__":
    main()

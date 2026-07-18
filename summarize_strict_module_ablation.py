from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VARIANTS = [
    ("A_source_pseudo", "strict_ablation_source_pseudo_no_dann_no_contrast", False, False, False),
    ("B_source_pseudo_aug", "strict_ablation_source_pseudo_aug_no_dann_no_contrast", True, False, False),
    ("C_source_pseudo_dann", "strict_ablation_source_pseudo_dann_l003_no_contrast", False, True, False),
    ("D_source_pseudo_aug_dann", "strict_ablation_source_pseudo_aug_dann_l003_no_contrast", True, True, False),
    (
        "E_current_best",
        "strict_aug150_w020_label_to_text_gen_sentiment_contrastive_l001_source_balanced",
        True,
        True,
        True,
    ),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary_rows(run_dir: Path) -> list[dict]:
    rows = []
    for variant, tag, augment, dann, contrastive in VARIANTS:
        raw = read_json(run_dir / f"aste_metrics_raw_{tag}.json")
        fixed = read_json(run_dir / f"aste_metrics_fixed_{tag}.json")
        structure_path = run_dir / f"aste_metrics_by_structure_{tag}.json"
        structure = read_json(structure_path) if structure_path.exists() else {}
        rows.append(
            {
                "variant": variant,
                "output_tag": tag,
                "augment": augment,
                "dann": dann,
                "sentiment_contrastive": contrastive,
                "raw_precision": raw.get("precision", ""),
                "raw_recall": raw.get("recall", ""),
                "raw_f1": raw.get("micro_f1", ""),
                "fixed_f1": fixed.get("micro_f1", ""),
                "single_raw_f1": structure.get("single_triplet_rows", {}).get("raw", {}).get("micro_f1", ""),
                "multi_raw_precision": structure.get("multi_triplet_rows", {}).get("raw", {}).get("precision", ""),
                "multi_raw_recall": structure.get("multi_triplet_rows", {}).get("raw", {}).get("recall", ""),
                "multi_raw_f1": structure.get("multi_triplet_rows", {}).get("raw", {}).get("micro_f1", ""),
            }
        )
    return rows


def _percent(value) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value) * 100:.2f}"


def write_summary(run_dir: Path, rows: list[dict]) -> tuple[Path, Path]:
    csv_path = run_dir / "results_strict_module_ablation.csv"
    md_path = run_dir / "results_strict_module_ablation_CN.md"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 严格模块消融结果",
        "",
        "主指标为 raw F1（原始 F1），fixed F1（修正 F1）只作辅助分析。五组实验使用相同源域数据、hp1 高精度伪标签、训练轮数和解码参数。",
        "",
        "| 组别 | 增强 | DANN（领域对抗） | 情感对比 | Raw P | Raw R | Raw F1 | Fixed F1 | 多三元组 Raw R | 多三元组 Raw F1 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {augment} | {dann} | {contrastive} | {raw_p} | {raw_r} | {raw_f1} | {fixed_f1} | {multi_r} | {multi_f1} |".format(
                variant=row["variant"],
                augment="是" if row["augment"] else "否",
                dann="是" if row["dann"] else "否",
                contrastive="是" if row["sentiment_contrastive"] else "否",
                raw_p=_percent(row["raw_precision"]),
                raw_r=_percent(row["raw_recall"]),
                raw_f1=_percent(row["raw_f1"]),
                fixed_f1=_percent(row["fixed_f1"]),
                multi_r=_percent(row["multi_raw_recall"]),
                multi_f1=_percent(row["multi_raw_f1"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = build_summary_rows(run_dir)
    csv_path, md_path = write_summary(run_dir, rows)
    print({"csv": str(csv_path), "md": str(md_path), "rows": len(rows)})


if __name__ == "__main__":
    main()

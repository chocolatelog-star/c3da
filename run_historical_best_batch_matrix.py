from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _find_number(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"raw_pseudo_micro_f1_against_hidden_gold", "pseudo_micro_f1_against_hidden_gold"}:
                if isinstance(item, dict) and "micro_f1" in item:
                    return item["micro_f1"]
            found = _find_number(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_number(item)
            if found is not None:
                return found
    return None


def extract_result(run_dir: Path) -> dict:
    pseudo = None
    for path in run_dir.rglob("*.json"):
        try:
            pseudo = _find_number(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
        if pseudo is not None:
            break
    final_candidates = sorted(run_dir.rglob("aste_metrics_raw_*.json"))
    final = None
    for path in final_candidates:
        if "target_test" in path.name or "4beam96" in path.name:
            try:
                final = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            if final is not None:
                break
    return {
        "pseudo_raw_f1": pseudo,
        "final_raw_f1": (final or {}).get("micro_f1"),
    }


def prune_completed_run(run_dir: Path) -> None:
    keep_names = {"manifest.json", "run_record_cn.md", "status.json", "pseudo_metrics.json", "final_metrics.json"}
    for path in sorted(run_dir.rglob("*"), reverse=True):
        if not path.is_file() or path.name in keep_names or path.suffix == ".log":
            continue
        path.unlink()
    for path in sorted(run_dir.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def run_one(args, batch: int, accumulation: int) -> dict:
    run_id = f"batch{batch}x{accumulation}_full"
    run_dir = Path(args.output_root) / args.recipe_id / run_id
    command = [
        sys.executable, str(Path(args.project_root) / "run_reproducible_pipeline.py"),
        "--recipe", str(Path(args.project_root) / "configs/recipes/rest16_to_laptop14_best_v1.json"),
        "--run_id", run_id, "--output_root", str(args.output_root), "--cuda", args.cuda,
        "--train_batch_size", str(batch), "--gradient_accumulation_steps", str(accumulation),
        "--eval_batch_size", "16", "--allow_dirty", "--skip_validation",
    ]
    log = run_dir.with_suffix(".log")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=args.project_root, stdout=handle, stderr=subprocess.STDOUT)
    result = {"batch": batch, "accumulation": accumulation, "returncode": completed.returncode}
    if completed.returncode == 0:
        metrics = extract_result(run_dir)
        (run_dir / "pseudo_metrics.json").write_text(json.dumps({"raw_f1": metrics["pseudo_raw_f1"]}, indent=2) + "\n", encoding="utf-8")
        (run_dir / "final_metrics.json").write_text(json.dumps({"raw_f1": metrics["final_raw_f1"]}, indent=2) + "\n", encoding="utf-8")
        result.update(metrics)
        prune_completed_run(run_dir)
    return result


def parse_groups(value: str) -> list[tuple[int, int]]:
    groups = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            batch, accumulation = (int(part) for part in item.split("x", 1))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"invalid group {item!r}; expected batchxaccum") from exc
        if batch < 1 or accumulation < 1:
            raise argparse.ArgumentTypeError(f"batch and accumulation must be positive: {item!r}")
        groups.append((batch, accumulation))
    if not groups:
        raise argparse.ArgumentTypeError("at least one batch group is required")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--recipe_id", default="rest16_to_laptop14_best_v1")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--groups", type=parse_groups, default=[(8, 4), (16, 2), (32, 1)], help="comma-separated batchxaccum groups")
    args = parser.parse_args()
    groups = args.groups
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda pair: run_one(args, *pair), groups))
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_matrix_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

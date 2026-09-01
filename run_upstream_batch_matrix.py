from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from experiment_runner_common import atomic_write_json, completed_file, group_units_by_gpu, matrix_unit_dir, read_json, run_command, sha256_file, write_status


DEFAULT_GROUPS = [(1, 16), (8, 2), (16, 1), (16, 2), (32, 1)]


def pipeline_run_dir(output_root: Path, recipe_id: str, run_id: str) -> Path:
    return Path(output_root) / recipe_id / run_id


def build_upstream_command(*, project_root: Path, recipe: str, output_root: Path, run_id: str, train_batch_size: int, gradient_accumulation_steps: int, cuda: str) -> list[str]:
    return [sys.executable, str(Path(project_root) / "run_reproducible_pipeline.py"), "--recipe", str(recipe), "--run_id", run_id, "--output_root", str(output_root), "--cuda", str(cuda), "--stop_after_stage", "pseudo", "--train_batch_size", str(train_batch_size), "--gradient_accumulation_steps", str(gradient_accumulation_steps), "--allow_dirty"]


def build_source_dev_command(project_root: Path, run_dir: Path, cuda: str) -> list[str]:
    extractor = run_dir / "models" / "extractor_ep25_plain_last" / "best"
    return [sys.executable, str(Path(project_root) / "t5_aste_pipeline.py"), "evaluate", "--run_dir", str(run_dir), "--model_path", str(extractor), "--eval_file", str(run_dir / "extract_dev.jsonl"), "--batch_size", "2", "--num_beams", "1", "--max_new_tokens", "128", "--no_constrained_decoding", "--no_task_prefix", "--output_tag", "source_dev", "--cuda", str(cuda)]


def parse_groups(value: str) -> list[tuple[int, int]]:
    if value == "default":
        return list(DEFAULT_GROUPS)
    return [tuple(map(int, item.split("x", 1))) for item in value.split(",")]


def run_unit(args, root: Path, recipe_id: str, batch: int, accumulation: int, gpu: str) -> dict:
    unit = matrix_unit_dir(root, batch, accumulation)
    pipeline_output = unit / "run"
    run_id = f"batch{batch}_accum{accumulation}"
    run_dir = pipeline_run_dir(pipeline_output, recipe_id, run_id)
    unit.mkdir(parents=True, exist_ok=True)
    status_path = unit / "status.json"
    if args.resume and read_json(status_path, {}).get("status") == "complete":
        return read_json(unit / "result.json", {"status": "complete", "batch": batch, "accumulation": accumulation})
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
    write_status(status_path, "running", batch=batch, accumulation=accumulation, gpu=gpu)
    command = build_upstream_command(project_root=Path(args.project_root), recipe=args.recipe, output_root=pipeline_output, run_id=run_id, train_batch_size=batch, gradient_accumulation_steps=accumulation, cuda="0")
    rc = run_command(command, unit / "pipeline.log", env)
    result = {"status": "failed" if rc else "complete", "batch": batch, "accumulation": accumulation, "effective_batch_size": batch * accumulation, "gpu": gpu, "returncode": rc, "run_dir": str(run_dir)}
    if rc == 0:
        result["source_dev_returncode"] = run_command(build_source_dev_command(Path(args.project_root), run_dir, "0"), unit / "source_dev.log", env)
        result["source_dev_metrics"] = read_json(run_dir / "aste_metrics_source_dev.json", {})
        result["pseudo_analysis"] = read_json(run_dir / "target_pseudo_analysis.json", {})
        result["high_precision_pseudo_analysis"] = read_json(run_dir / "target_pseudo_high_precision_analysis.json", {})
        model = run_dir / "models" / "extractor_ep25_plain_last" / "best" / "model.safetensors"
        result["extractor_sha256"] = sha256_file(model) if completed_file(model) else ""
        if result["source_dev_returncode"] != 0:
            result["status"] = "failed"
    atomic_write_json(unit / "result.json", result)
    status_fields = {key: value for key, value in result.items() if key != "status"}
    write_status(status_path, result["status"], **status_fields)
    return result


def run_gpu_queue(args, root: Path, recipe_id: str, gpu: str, groups: list[tuple[int, int]]) -> list[dict]:
    return [run_unit(args, root, recipe_id, batch, accumulation, gpu) for batch, accumulation in groups]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run upstream-only batch matrix")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--groups", default="default")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    recipe_data = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    recipe_id = recipe_data["recipe_id"]
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    groups = parse_groups(args.groups)
    queues = group_units_by_gpu(groups, gpus)
    rows = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_gpu_queue, args, root, recipe_id, gpu, queue) for gpu, queue in queues.items() if queue]
        for future in tqdm(as_completed(futures), total=len(futures), desc="upstream-gpu-queues"):
            rows.extend(future.result())
    rows.sort(key=lambda row: (row["effective_batch_size"], row["batch"]))
    atomic_write_json(root / "upstream_batch_matrix.json", rows)
    (root / "upstream_batch_matrix.md").write_text("# 上游 Batch 矩阵\n\n" + "\n".join(f"- batch={r['batch']}, accumulation={r['accumulation']}, status={r['status']}, source-dev={r.get('source_dev_metrics', {})}" for r in rows) + "\n", encoding="utf-8")
    return 0 if all(r["status"] == "complete" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

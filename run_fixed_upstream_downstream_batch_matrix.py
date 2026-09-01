from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from experiment_runner_common import atomic_write_json, group_units_by_gpu, matrix_unit_dir, read_json, run_command, sha256_file, write_status


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in Path(root).rglob(pattern) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"missing shared upstream artifacts: {pattern}")
    return matches[-1]


def discover_shared_artifacts(shared_run: Path) -> dict[str, Path]:
    root = Path(shared_run)
    final_data_root = root / "final_data"
    if not final_data_root.is_dir():
        raise FileNotFoundError(f"missing shared upstream artifacts: {final_data_root}")
    return {
        "extractor_config": _one(root, "models/extractor_ep25_plain_last/best/config.json"),
        "pseudo": _one(root, "target_pseudo_high_precision.jsonl"),
        "generator_config": _one(root, "models/generator*/best/config.json"),
        "selected_augment": _one(root, "c3da_two_channel_augmented_selected*.jsonl"),
        "final_train": _one(final_data_root, "final_train_*.jsonl"),
        "final_dev": _one(final_data_root, "final_dev_*.jsonl"),
        "target_test": _one(final_data_root, "target_test.jsonl"),
    }


def build_shared_manifest(shared_upstream: Path) -> dict:
    artifacts = discover_shared_artifacts(shared_upstream)
    return {"root": str(Path(shared_upstream).resolve()), "artifacts": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in artifacts.items()}}


def build_downstream_command(*, project_root: Path, shared_final_train: Path, shared_final_dev: Path, output_dir: Path, train_batch_size: int, accumulation: int, cuda: str) -> list[str]:
    return [sys.executable, str(Path(project_root) / "t5_absa_train.py"), "--model_path", str(Path(project_root) / "models" / "t5-base-py"), "--train_file", str(shared_final_train), "--dev_file", str(shared_final_dev), "--output_dir", str(output_dir), "--num_train_epochs", "5", "--source_weight", "1.0", "--pseudo_weight", "0.65", "--augment_weight", "0.2", "--checkpoint_selection", "best", "--resume_from_checkpoint", "auto", "--save_total_limit", "1", "--lambda_domain_adv", "0.03", "--domain_adv_grl_lambda", "1.0", "--domain_adv_hidden_size", "256", "--domain_adv_exclude_augment", "--lambda_sentiment_contrastive", "0.01", "--lambda_pairing_loss", "0.0", "--pairing_temperature", "0.1", "--sentiment_contrastive_temperature", "0.1", "--sentiment_contrastive_min_weight", "0.65", "--neutral_generation_loss_gain", "0.0", "--neutral_generation_max_effective_weight", "0.0", "--sentiment_contrastive_source_only", "--sentiment_contrastive_class_balanced", "--per_device_train_batch_size", str(train_batch_size), "--per_device_eval_batch_size", "2", "--gradient_accumulation_steps", str(accumulation), "--learning_rate", "0.0003", "--fp16", "--gradient_checkpointing", "--cuda", str(cuda), "--seed", "1000"]


def build_evaluate_command(project_root: Path, run_dir: Path, model_dir: Path, target_test: Path, cuda: str) -> list[str]:
    return [sys.executable, str(Path(project_root) / "t5_aste_pipeline.py"), "evaluate", "--run_dir", str(run_dir), "--model_path", str(model_dir / "best"), "--eval_file", str(target_test), "--batch_size", "2", "--num_beams", "1", "--max_new_tokens", "128", "--no_constrained_decoding", "--no_task_prefix", "--output_tag", "fixed_upstream", "--cuda", str(cuda)]


def run_unit(args, root: Path, paths: dict[str, Path], manifest: dict, batch: int, accumulation: int, gpu: str) -> dict:
    unit = matrix_unit_dir(root, batch, accumulation); status_path = unit / "status.json"
    if args.resume and read_json(status_path, {}).get("status") == "complete":
        return read_json(unit / "result.json", {})
    unit.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
    model_dir = unit / "model"
    write_status(status_path, "running", batch=batch, accumulation=accumulation, gpu=gpu)
    train = build_downstream_command(project_root=Path(args.project_root), shared_final_train=paths["final_train"], shared_final_dev=paths["final_dev"], output_dir=model_dir, train_batch_size=batch, accumulation=accumulation, cuda="0")
    rc = run_command(train, unit / "train.log", env)
    eval_rc = run_command(build_evaluate_command(Path(args.project_root), unit, model_dir, paths["target_test"], "0"), unit / "evaluate.log", env) if rc == 0 else None
    result = {"status": "complete" if rc == 0 and eval_rc == 0 else "failed", "batch": batch, "accumulation": accumulation, "effective_batch_size": batch * accumulation, "gpu": gpu, "train_returncode": rc, "evaluate_returncode": eval_rc, "shared_manifest_sha256": sha256_file(root / "shared_upstream_manifest.json"), "fixed_metrics": read_json(unit / "aste_metrics_fixed_fixed_upstream.json", {})}
    atomic_write_json(unit / "result.json", result)
    status_fields = {key: value for key, value in result.items() if key != "status"}
    write_status(status_path, result["status"], **status_fields)
    return result


def run_gpu_queue(args, root: Path, paths: dict[str, Path], manifest: dict, gpu: str, groups: list[tuple[int, int]]) -> list[dict]:
    return [run_unit(args, root, paths, manifest, batch, accumulation, gpu) for batch, accumulation in groups]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final-train batch matrix with identical frozen inputs")
    parser.add_argument("--shared_run", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--groups", default="8x2,16x1,16x2,32x1")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    manifest = build_shared_manifest(Path(args.shared_run))
    atomic_write_json(root / "shared_upstream_manifest.json", manifest)
    paths = {name: Path(value["path"]) for name, value in manifest["artifacts"].items()}
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    groups = [tuple(map(int, item.split("x", 1))) for item in args.groups.split(",")]
    queues = group_units_by_gpu(groups, gpus); rows = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_gpu_queue, args, root, paths, manifest, gpu, queue) for gpu, queue in queues.items() if queue]
        for future in tqdm(as_completed(futures), total=len(futures), desc="downstream-gpu-queues"):
            rows.extend(future.result())
    rows.sort(key=lambda row: (row["effective_batch_size"], row["batch"]))
    atomic_write_json(root / "fixed_upstream_downstream_batch_matrix.json", rows)
    (root / "fixed_upstream_downstream_batch_matrix.md").write_text("# 固定上游下游 Batch 矩阵\n\n" + "\n".join(f"- batch={r['batch']}, accumulation={r['accumulation']}, status={r['status']}, fixed={r.get('fixed_metrics', {})}" for r in rows) + "\n", encoding="utf-8")
    return 0 if all(r["status"] == "complete" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

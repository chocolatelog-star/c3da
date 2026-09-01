from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiment_runner_common import atomic_write_json, read_json
from run_plan_a_graph_best import build_adapter_manifest, build_full_command


GRAPH_ONLY_FIELDS = {"name", "graph_enabled", "output_dir"}


def build_ab_specs(root: Path, *, train_batch_size: int, accumulation: int) -> list[dict]:
    common = {"seed": 1000, "train_batch_size": int(train_batch_size), "gradient_accumulation_steps": int(accumulation), "lambda_domain_adv": 0.03, "lambda_sentiment_contrastive": 0.01}
    root = Path(root)
    return [
        {"name": "control", "graph_enabled": False, "output_dir": str(root / "control"), **common},
        {"name": "graph", "graph_enabled": True, "output_dir": str(root / "graph"), **common},
    ]


def compare_ab_identity(control: dict, graph: dict) -> dict:
    keys = (set(control) | set(graph)) - GRAPH_ONLY_FIELDS
    mismatches = {key: {"control": control.get(key), "graph": graph.get(key)} for key in sorted(keys) if control.get(key) != graph.get(key)}
    return {"matched": not mismatches, "mismatches": mismatches, "allowed_graph_fields": sorted(GRAPH_ONLY_FIELDS)}


def phase_a_artifacts_complete(phase_root: Path) -> bool:
    root = Path(phase_root)
    return all(
        (root / variant / "models" / "extractor" / "best" / "config.json").is_file()
        and (root / variant / "target_pseudo_selected.jsonl").is_file()
        for variant in ("control", "treatment")
    )


def _metrics(output_root: Path) -> dict:
    candidates = sorted(Path(output_root).rglob("aste_metrics_fixed*.json"))
    fixed = read_json(candidates[-1], {}) if candidates else {}
    raw_candidates = sorted(Path(output_root).rglob("aste_metrics_raw*.json"))
    raw = read_json(raw_candidates[-1], {}) if raw_candidates else {}
    structure_candidates = sorted(Path(output_root).rglob("aste_metrics_by_structure*.json"))
    structure = read_json(structure_candidates[-1], {}) if structure_candidates else {}
    return {"raw": raw, "fixed": fixed, "structure": structure}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fair Graph OFF/ON complete A/B")
    parser.add_argument("--phase_a_output", required=True)
    parser.add_argument("--full_output_root", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--graph_cache_dir", required=True)
    parser.add_argument("--parser_dir", default="models/stanza_resources")
    parser.add_argument("--model_path", default="models/t5-base-py")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    phase_root = Path(args.phase_a_output).resolve()
    full_root = Path(args.full_output_root).resolve()
    phase_cmd = [sys.executable, "m1_syntactic_rgat_pseudo_quick_ablation.py", "--recipe", args.recipe, "--output_dir", str(phase_root), "--model_path", args.model_path, "--graph_cache_dir", args.graph_cache_dir, "--parser_dir", args.parser_dir, "--extractor_train_batch_size", str(args.train_batch_size), "--gradient_accumulation_steps", str(args.gradient_accumulation_steps), "--cuda", args.cuda]
    if args.resume:
        phase_cmd.append("--resume")
    if args.dry_run:
        print(" ".join(phase_cmd))
        return 0
    phase_result = subprocess.run(phase_cmd, cwd=project_root, check=False)
    if phase_result.returncode != 0 and not phase_a_artifacts_complete(phase_root):
        raise subprocess.CalledProcessError(phase_result.returncode, phase_cmd)
    specs = build_ab_specs(full_root, train_batch_size=args.train_batch_size, accumulation=args.gradient_accumulation_steps)
    identity = compare_ab_identity(specs[0], specs[1])
    if not identity["matched"]:
        raise RuntimeError(f"Graph A/B common identity mismatch: {identity['mismatches']}")
    for spec in specs:
        variant = "treatment" if spec["graph_enabled"] else "control"
        adapter = full_root / f"{spec['name']}_adapter"
        build_adapter_manifest(phase_root / variant, adapter, source="laptop14", target="rest15", seed=args.seed, variant=variant)
        command = build_full_command(project_root=project_root, adapter=adapter, output_root=Path(spec["output_dir"]), model_path=args.model_path, train_batch_size=args.train_batch_size, accumulation=args.gradient_accumulation_steps, cuda=args.cuda, seed=args.seed)
        subprocess.run(command, cwd=project_root, check=True)
        spec["metrics"] = _metrics(Path(spec["output_dir"]))
    control_f1 = specs[0].get("metrics", {}).get("fixed", {}).get("micro_f1")
    graph_f1 = specs[1].get("metrics", {}).get("fixed", {}).get("micro_f1")
    result = {"schema_version": 1, "identity": identity, "variants": specs, "fixed_f1_delta": graph_f1 - control_f1 if isinstance(graph_f1, (int, float)) and isinstance(control_f1, (int, float)) else None}
    atomic_write_json(full_root / "graph_control_ab.json", result)
    (full_root / "graph_control_ab.md").write_text(f"# Graph OFF/ON A/B\n\n- Control Fixed F1: {control_f1}\n- Graph Fixed F1: {graph_f1}\n- Delta: {result['fixed_f1_delta']}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

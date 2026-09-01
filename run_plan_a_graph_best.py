"""Plan A: reuse M1 graph treatment pseudo labels in the historical best tail."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def required_adapter_paths(adapter: Path) -> tuple[Path, ...]:
    return (
        adapter / "models" / "extractor_ep25_plain_last" / "best" / "config.json",
        adapter / "target_pseudo.jsonl",
        adapter / "target_pseudo_high_precision.jsonl",
        adapter / "target_pseudo_high_precision_analysis.json",
        adapter / "target_pseudo_analysis.json",
        adapter / "target_pseudo_generation_state.json",
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        os.symlink(source, destination, target_is_directory=source.is_dir())
    except (OSError, NotImplementedError):
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def build_adapter_manifest(treatment_dir: Path, adapter_dir: Path, *, source: str, target: str, seed: int, variant: str = "treatment") -> dict:
    treatment_dir = Path(treatment_dir).resolve()
    adapter_dir = Path(adapter_dir).resolve()
    model = treatment_dir / "models" / "extractor" / "best"
    pseudo = treatment_dir / "target_pseudo_selected.jsonl"
    if not (model / "config.json").exists() or not pseudo.exists():
        raise FileNotFoundError("M1 treatment must contain extractor/best/config.json and target_pseudo_selected.jsonl")
    _link_or_copy(model, adapter_dir / "models" / "extractor_ep25_plain_last" / "best")
    _link_or_copy(pseudo, adapter_dir / "target_pseudo.jsonl")
    _link_or_copy(pseudo, adapter_dir / "target_pseudo_high_precision.jsonl")
    analysis = {
        "schema_version": 1,
        "status": "complete",
        "model_path": str((adapter_dir / "models" / "extractor_ep25_plain_last" / "best").resolve()),
        "pseudo_source_tag": "extractor_ep25_plain_last",
        "target_test_access": False,
        "source": str(treatment_dir),
    }
    (adapter_dir / "target_pseudo_high_precision_analysis.json").parent.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "target_pseudo_high_precision_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    (adapter_dir / "target_pseudo_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    state = dict(analysis)
    state["resolved_model_path"] = analysis["model_path"]
    (adapter_dir / "target_pseudo_generation_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"schema_version": 1, "source": str(treatment_dir), "adapter": str(adapter_dir), "source_dataset": source, "target_dataset": target, "seed": seed, "target_test_access": False, "pseudo_file": str(pseudo)}
    manifest["variant"] = variant
    manifest["graph_enabled"] = variant == "treatment"
    (adapter_dir / f"graph_{variant}_adapter.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_full_command(*, project_root: Path, adapter: Path, output_root: Path, model_path: str, train_batch_size: int, accumulation: int, cuda: str, seed: int) -> list[str]:
    return [sys.executable, str(Path(project_root) / "run_bgca_aste_stage1_pairs.py"), "--pairs", "laptop14:rest15", "--output_root", str(output_root), "--reuse_upstream_run_dir", str(adapter), "--extractor_model_path", model_path, "--generator_model_path", model_path, "--generator_prompt_style", "label_to_text", "--augment_prompt_style", "masked_mutual", "--complete_multi_extra_weight", "0.25", "--final_pseudo_weight", "0.75", "--final_augment_weight", "0.2", "--lambda_sentiment_contrastive", "0.01", "--sentiment_contrastive_source_only", "--sentiment_contrastive_class_balanced", "--lambda_domain_adv", "0.03", "--train_batch_size", str(train_batch_size), "--gradient_accumulation_steps", str(accumulation), "--eval_batch_size", "2", "--learning_rate", "0.0003", "--cuda", str(cuda), "--seed", str(seed)]


def main() -> int:
    p = argparse.ArgumentParser(description="Run Plan A graph treatment followed by historical-best tail")
    p.add_argument("--phase_a_output", required=True)
    p.add_argument("--full_output_root", required=True)
    p.add_argument("--recipe", default="configs/recipes/laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_v1.json")
    p.add_argument("--graph_cache_dir", required=True)
    p.add_argument("--parser_dir", default="models/stanza_resources")
    p.add_argument("--model_path", default="models/t5-base-py")
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--cuda", default="0")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    phase_a = Path(args.phase_a_output).resolve()
    adapter = phase_a.parent / (phase_a.name + "_upstream_adapter")
    phase_cmd = [sys.executable, "m1_syntactic_rgat_pseudo_quick_ablation.py", "--recipe", args.recipe, "--output_dir", str(phase_a), "--model_path", args.model_path, "--graph_cache_dir", args.graph_cache_dir, "--parser_dir", args.parser_dir, "--cuda", args.cuda]
    if args.dry_run:
        print(" ".join(phase_cmd))
        return 0
    subprocess.run(phase_cmd, cwd=root, check=True)
    manifest = build_adapter_manifest(phase_a / "treatment", adapter, source="laptop14", target="rest15", seed=args.seed)
    full_cmd = build_full_command(project_root=root, adapter=adapter, output_root=Path(args.full_output_root), model_path=args.model_path, train_batch_size=args.train_batch_size, accumulation=args.gradient_accumulation_steps, cuda=args.cuda, seed=args.seed)
    (phase_a / "plan_a_full_command.json").write_text(json.dumps({"adapter": manifest, "command": full_cmd, "target_test_access": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(full_cmd, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

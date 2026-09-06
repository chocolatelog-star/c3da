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
    # Reused upstream runs must expose the same shared inputs expected by the
    # downstream pipeline.  The previous adapter only linked model/pseudo
    # artifacts, causing augmentation to fail on source_train.jsonl.
    for name in (
        "source_train.jsonl",
        "source_dev.jsonl",
        "target_unlabeled.jsonl",
        "target_train_gold_analysis.jsonl",
        "target_test.jsonl",
        "extract_train.jsonl",
        "extract_dev.jsonl",
    ):
        source_path = treatment_dir / name
        if source_path.is_file():
            _link_or_copy(source_path, adapter_dir / name)
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


def build_full_command(*, project_root: Path, adapter: Path, output_root: Path, model_path: str, train_batch_size: int, accumulation: int, cuda: str, seed: int, source_dataset: str = "laptop14", target_dataset: str = "rest15", final_lambda_domain_adv: float = 0.0, structure_preserving_augmentation: bool = False, minimal_outputs: bool = True) -> list[str]:
    model_root = Path(model_path).resolve().parent
    nli_model_path = model_root / "nli-deberta-v3-base-mnli-fever-anli"
    glove_path = model_root / "glove" / "glove.6B.300d.txt"
    final_pseudo_weight = "0.75" if (source_dataset, target_dataset) == ("laptop14", "rest15") else "0.65"
    command = [sys.executable, str(Path(project_root) / "run_bgca_aste_stage1_pairs.py"), "--pairs", f"{source_dataset}:{target_dataset}", "--output_root", str(output_root), "--reuse_upstream_run_dir", str(adapter), "--extractor_model_path", model_path, "--generator_model_path", model_path, "--generator_prompt_style", "label_to_text", "--augment_prompt_style", "masked_mutual", "--nli_model_path", str(nli_model_path), "--sentiment_vector_model_path", model_path, "--glove_path", str(glove_path), "--complete_multi_extra_weight", "0.25", "--final_pseudo_weight", final_pseudo_weight, "--final_augment_weight", "0.2", "--lambda_sentiment_contrastive", "0.01", "--sentiment_contrastive_source_only", "--sentiment_contrastive_class_balanced", "--train_batch_size", str(train_batch_size), "--gradient_accumulation_steps", str(accumulation), "--eval_batch_size", "16", "--learning_rate", "0.0003", "--final_lambda_domain_adv", str(final_lambda_domain_adv), "--cuda", str(cuda), "--seed", str(seed)]
    if structure_preserving_augmentation:
        command.append("--structure_preserving_augmentation")
    if minimal_outputs:
        command.append("--minimal_outputs")
    return command


def resolve_final_domain_adv(*, no_dann: bool, explicit: float | None, recipe_value: float) -> float:
    """Resolve final ASTE DANN independently from the upstream switch.

    ``no_dann`` controls Phase A only. An explicit final value is therefore
    authoritative, which is required for the controlled 0.01/0.02/0.03 sweep.
    """
    if explicit is not None:
        value = float(explicit)
    elif no_dann:
        value = 0.0
    else:
        value = float(recipe_value)
    if value < 0.0:
        raise ValueError("final domain adversarial weight must be non-negative")
    return value


def main() -> int:
    p = argparse.ArgumentParser(description="Run Plan A graph treatment followed by historical-best tail")
    p.add_argument("--phase_a_output", required=True)
    p.add_argument("--full_output_root", required=True)
    p.add_argument("--recipe", default="configs/recipes/laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_v1.json")
    p.add_argument("--graph_cache_dir", required=True)
    p.add_argument("--parser_dir", default="models/stanza_resources")
    p.add_argument("--model_path", default="models/t5-base-py")
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=16,
                   help="下游评估批次；当前历史最佳尾部固定为16，保留此参数用于命令兼容性")
    p.add_argument("--cuda", default="0")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--source_dataset", default="laptop14")
    p.add_argument("--target_dataset", default="rest15")
    p.add_argument("--no_dann", action="store_true")
    p.add_argument("--variant", choices=("G0", "G1", "G2", "G3", "G4"), default="G1")
    p.add_argument("--final_lambda_domain_adv", type=float, default=None,
                   help="下游最终训练的DANN权重；省略时读取recipe中的lambda_domain_adv")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--structure_preserving_augmentation", action="store_true")
    p.add_argument("--keep_intermediates", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    phase_a = Path(args.phase_a_output).resolve()
    adapter = phase_a.parent / (phase_a.name + "_upstream_adapter")
    phase_cmd = [sys.executable, "m1_syntactic_rgat_pseudo_quick_ablation.py", "--recipe", args.recipe, "--output_dir", str(phase_a), "--model_path", args.model_path, "--graph_cache_dir", args.graph_cache_dir, "--parser_dir", args.parser_dir, "--extractor_train_batch_size", str(args.train_batch_size), "--gradient_accumulation_steps", str(args.gradient_accumulation_steps), "--cuda", args.cuda, "--variant", args.variant, "--treatment_only"]
    if args.no_dann:
        phase_cmd.append("--no_dann")
    if args.dry_run:
        print(" ".join(phase_cmd))
        return 0
    subprocess.run(phase_cmd, cwd=root, check=True)
    manifest = build_adapter_manifest(phase_a / "treatment", adapter, source=args.source_dataset, target=args.target_dataset, seed=args.seed)
    recipe_data = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    recipe_lambda = float(recipe_data.get("training", {}).get("lambda_domain_adv", 0.03))
    final_lambda = resolve_final_domain_adv(no_dann=args.no_dann, explicit=args.final_lambda_domain_adv, recipe_value=recipe_lambda)
    full_cmd = build_full_command(project_root=root, adapter=adapter, output_root=Path(args.full_output_root), model_path=args.model_path, train_batch_size=args.train_batch_size, accumulation=args.gradient_accumulation_steps, cuda=args.cuda, seed=args.seed, source_dataset=args.source_dataset, target_dataset=args.target_dataset, final_lambda_domain_adv=final_lambda, structure_preserving_augmentation=args.structure_preserving_augmentation, minimal_outputs=not args.keep_intermediates)
    (phase_a / "plan_a_full_command.json").write_text(json.dumps({"adapter": manifest, "command": full_cmd, "target_test_access": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(full_cmd, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

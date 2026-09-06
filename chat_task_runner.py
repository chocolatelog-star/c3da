#!/usr/bin/env python3
"""C3DA G0-G4 task runner: reuse upstream checkpoints for audit, train G4 only."""
from __future__ import annotations
import argparse, gzip, hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def triplets(row):
    # Labels use <pos> aspect <opinion> opinion [<neg>/<neu>].
    label = str(row.get("label_fixed", row.get("label", "")))
    return max(0, label.count("<pos>") + label.count("<neg>") + label.count("<neu>"))

def stats(path: Path):
    rs = rows(path) if path.is_file() else []
    ts = [triplets(r) for r in rs]
    return {"rows": len(rs), "nonempty_rows": sum(x > 0 for x in ts),
            "triplets": sum(ts), "mean_triplets_per_row": (sum(ts) / len(rs) if rs else 0.0),
            "single_rows": sum(x == 1 for x in ts), "multi_rows": sum(x >= 2 for x in ts),
            "3plus_rows": sum(x >= 3 for x in ts)}

def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)

def audit_one(variant: str, source_run: Path, out_root: Path, model_path: str, parser_dir: str, cuda: str):
    treatment = source_run / "treatment"
    model = treatment / "models" / "extractor" / "best"
    if not (model / "config.json").is_file():
        candidates = sorted(source_run.rglob("models/extractor/best/config.json"))
        if candidates:
            model = candidates[0].parent
    if not (model / "config.json").is_file():
        candidates = sorted(source_run.rglob("models/extractor_ep25_plain_last/best/config.json"))
        if candidates:
            model = candidates[0].parent
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"{variant}: missing checkpoint {model}")
    out = out_root / variant
    out.mkdir(parents=True, exist_ok=True)
    for name in ("source_train.jsonl", "source_dev.jsonl", "target_unlabeled.jsonl"):
        src = treatment / name
        if not src.is_file(): src = source_run / "inputs" / name
        if not src.is_file(): raise FileNotFoundError(f"{variant}: missing {name}")
        dst = out / name
        if not dst.exists(): dst.symlink_to(src)
    cache = None
    ident = source_run / "graph_cache_identity.json"
    if ident.is_file():
        cache = json.loads(ident.read_text(encoding="utf-8")).get("cache_dir")
    if not cache: raise FileNotFoundError(f"{variant}: graph_cache_identity.json/cache_dir missing")
    run([PY, "t5_aste_pipeline.py", "evaluate", "--run_dir", out, "--model_path", model,
         "--eval_file", out / "source_dev.jsonl", "--batch_size", "2", "--num_beams", "1",
         "--max_new_tokens", "128", "--length_penalty", "1.0", "--cuda", cuda,
         "--no_task_prefix", "--no_constrained_decoding", "--use_syntactic_graph_adapter",
         "--syntactic_graph_cache_dir", cache, "--syntactic_graph_parser_dir", parser_dir,
         "--syntactic_graph_cache_tokenizer_path", model_path, "--syntactic_graph_split", "source_dev",
         "--output_tag", "source_dev"])
    run([PY, "t5_aste_pipeline.py", "pseudo", "--run_dir", out, "--model_path", model,
         "--batch_size", "1", "--num_beams", "1", "--max_new_tokens", "128", "--length_penalty", "1.0",
         "--max_target_unlabeled", "0", "--pseudo_model_variant", "best", "--pseudo_base_weight", "0.75",
         "--high_precision_max_triplets", "1", "--high_precision_max_token_distance", "5",
         "--fixed_changed_min_score", "0.65", "--fixed_changed_weight", "0.35", "--cuda", cuda,
         "--no_task_prefix", "--no_constrained_decoding", "--use_syntactic_graph_adapter",
         "--syntactic_graph_cache_dir", cache, "--syntactic_graph_parser_dir", parser_dir,
         "--syntactic_graph_cache_tokenizer_path", model_path, "--syntactic_graph_split", "target_unlabeled"])
    files = {"raw": out / "target_pseudo.jsonl", "qualified": out / "target_pseudo_selected.jsonl",
             "hp": out / "target_pseudo_high_precision.jsonl", "selected": out / "target_pseudo_selected.jsonl"}
    result = {"variant": variant, "checkpoint": str(model), "checkpoint_sha256": sha256(model / "model.safetensors") if (model / "model.safetensors").is_file() else None,
              "source_dev": {}, "pseudo": {}, "hidden_gold_audit": "UNAVAILABLE", "target_test_access": False}
    pred = out / "aste_predictions_raw_fixed_source_dev.jsonl"
    if pred.is_file(): result["source_dev"]["prediction_file"] = str(pred)
    for key, path in files.items():
        result["pseudo"][key] = stats(path)
    (out / "upstream_audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("audit", "g4"), required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--parser_dir", required=True)
    p.add_argument("--cuda", default="0")
    p.add_argument("--g0", default="")
    p.add_argument("--g1", default="")
    p.add_argument("--g2", default="")
    p.add_argument("--g3", default="")
    p.add_argument("--g4_phase_a", default="")
    p.add_argument("--g4_full", default="")
    args = p.parse_args()
    out = Path(args.output_root).resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        result = [audit_one(v, Path(getattr(args, v)), out, args.model_path, args.parser_dir, args.cuda) for v in ("g0", "g1", "g2", "g3")]
        (out / "G0_G1_G2_G3_upstream_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if not args.g4_phase_a or not args.g4_full: raise ValueError("g4 requires --g4_phase_a and --g4_full")
        run([PY, "run_plan_a_graph_best.py", "--phase_a_output", args.g4_phase_a, "--full_output_root", args.g4_full,
             "--recipe", "configs/recipes/laptop14_to_rest15_m1_syntactic_rgat_pseudo_quick_ablation_v1.json",
             "--graph_cache_dir", str(Path(args.g4_phase_a).parent / "graph_cache"), "--parser_dir", args.parser_dir,
             "--model_path", args.model_path, "--train_batch_size", "16", "--gradient_accumulation_steps", "2",
             "--eval_batch_size", "16", "--cuda", args.cuda, "--seed", "1000", "--source_dataset", "laptop14",
             "--target_dataset", "rest15", "--no_dann", "--variant", "G4"])

if __name__ == "__main__": main()

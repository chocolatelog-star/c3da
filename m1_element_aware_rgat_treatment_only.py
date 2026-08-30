from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from m1_syntactic_rgat_pseudo_quick_ablation import _build_input_rows, _serialize_rows
from t5_aste_data import parse_triplet_text_list, micro_f1, read_jsonl
import t5_absa_train as train_mod

TASK_ID = "M1_ELEMENT_AWARE_COMPONENT_ATTRIBUTION_V1"
FROZEN_TRAIN_BATCH_SIZE = 1
FROZEN_GRADIENT_ACCUMULATION_STEPS = 16
FROZEN_EVAL_BATCH_SIZE = 2


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_serialize_rows(rows))


def resolve_variant(args: argparse.Namespace) -> dict:
    if args.focus_only:
        return {"name": "focus_only", "focus_enabled": True, "coverage_enabled": False}
    if args.coverage_only:
        return {"name": "coverage_only", "focus_enabled": False, "coverage_enabled": True}
    return {
        "name": "focus_plus_coverage",
        "focus_enabled": True,
        "coverage_enabled": True,
    }


def validate_frozen_training_recipe(args: argparse.Namespace) -> None:
    if args.train_batch_size != FROZEN_TRAIN_BATCH_SIZE:
        raise ValueError("formal component ablations require V9e train_batch_size=1")
    if args.gradient_accumulation_steps != FROZEN_GRADIENT_ACCUMULATION_STEPS:
        raise ValueError(
            "formal component ablations require V9e gradient_accumulation_steps=16"
        )


def build_train_args(
    args: argparse.Namespace,
    root: Path,
    variant: dict,
) -> tuple[list[str], dict]:
    validate_frozen_training_recipe(args)
    focus_weight = "0.05" if variant["focus_enabled"] else "0"
    coverage_weight = "0.05" if variant["coverage_enabled"] else "0"
    train_args = [
        "--model_path",
        args.model_path,
        "--train_file",
        str(root / "source_train.jsonl"),
        "--dev_file",
        str(root / "source_dev.jsonl"),
        "--output_dir",
        str(root / "models" / "extractor"),
        "--num_train_epochs",
        "25",
        "--source_weight",
        "1.0",
        "--pseudo_weight",
        "0.75",
        "--augment_weight",
        "0.2",
        "--lambda_structure_loss",
        "0",
        "--lambda_consistency_loss",
        "0",
        "--lambda_pairing_loss",
        "0",
        "--multi_triplet_loss_gain",
        "0",
        "--neutral_loss_gain",
        "0",
        "--checkpoint_selection",
        "last",
        "--resume_from_checkpoint",
        "auto",
        "--per_device_train_batch_size",
        str(FROZEN_TRAIN_BATCH_SIZE),
        "--per_device_eval_batch_size",
        str(FROZEN_EVAL_BATCH_SIZE),
        "--max_source_length",
        "128",
        "--max_target_length",
        "96",
        "--gradient_accumulation_steps",
        str(FROZEN_GRADIENT_ACCUMULATION_STEPS),
        "--learning_rate",
        "0.0003",
        "--lambda_domain_adv",
        "0",
        "--domain_adv_grl_lambda",
        "1.0",
        "--domain_adv_hidden_size",
        "256",
        "--pairing_temperature",
        "0.1",
        "--max_effective_weight",
        "1.0",
        "--neutral_generation_loss_gain",
        "0",
        "--neutral_generation_max_effective_weight",
        "0",
        "--max_pairing_triplets",
        "4",
        "--min_pairing_triplets",
        "2",
        "--min_pairing_sample_weight",
        "0.65",
        "--fp16",
        "--gradient_checkpointing",
        "--cuda",
        args.cuda,
        "--seed",
        "1000",
        "--legacy_stochastic",
        "--use_syntactic_graph_adapter",
        "--syntactic_graph_cache_dir",
        args.graph_cache_dir,
        "--syntactic_graph_parser_dir",
        args.parser_dir,
        "--element_aware_attention",
        "--element_focus_weight",
        focus_weight,
        "--element_coverage_weight",
        coverage_weight,
        "--target_unlabeled_file",
        str(root / "target_unlabeled.jsonl"),
        "--initialization_audit_path",
        str(root / "phase_a_initialization_audit.json"),
    ]
    if variant["focus_enabled"]:
        train_args.append("--element_focus_loss")
    if variant["coverage_enabled"]:
        train_args.append("--multi_element_coverage_loss")
    config = {
        "variant": variant["name"],
        "focus_enabled": variant["focus_enabled"],
        "coverage_enabled": variant["coverage_enabled"],
        "focus_weight": float(focus_weight),
        "coverage_weight": float(coverage_weight),
        "train_batch_size": FROZEN_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": FROZEN_GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": (
            FROZEN_TRAIN_BATCH_SIZE * FROZEN_GRADIENT_ACCUMULATION_STEPS
        ),
        "eval_batch_size": FROZEN_EVAL_BATCH_SIZE,
        "dann": 0.0,
    }
    return train_args, config


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--graph_cache_dir", required=True)
    p.add_argument("--parser_dir", required=True)
    p.add_argument("--cuda", default="0")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    g=p.add_mutually_exclusive_group()
    g.add_argument("--focus_only", action="store_true")
    g.add_argument("--coverage_only", action="store_true")
    a=p.parse_args()
    variant = resolve_variant(a)
    validate_frozen_training_recipe(a)
    root=Path(a.output_dir); root.mkdir(parents=True, exist_ok=True)
    project_root=Path(__file__).resolve().parent
    data_root=project_root / "data" / "aste" / "cross_domain"
    external={"source_train":{"path":str(data_root / "laptop14" / "train.txt")},"source_dev":{"path":str(data_root / "laptop14" / "dev.txt")},"target_unlabeled":{"path":str(data_root / "rest15" / "train.txt")}}
    rows=_build_input_rows("laptop14","rest15", external_inputs=external)
    write_jsonl(root/"source_train.jsonl", rows["source_train"])
    write_jsonl(root/"source_dev.jsonl", rows["source_dev"])
    write_jsonl(root/"target_unlabeled.jsonl", rows["target_unlabeled"])
    train_args, frozen_config = build_train_args(a, root, variant)
    train_mod.run_phase_a_training(train_args)
    model=root/"models"/"extractor"/"best"
    common=[sys.executable,"t5_aste_pipeline.py"]
    subprocess.run(common+["evaluate","--run_dir",str(root),"--model_path",str(model),
      "--eval_file",str(root/"source_dev.jsonl"),"--batch_size","2","--num_beams","1",
      "--max_new_tokens","128","--cuda",a.cuda,"--no_task_prefix","--no_constrained_decoding",
      "--output_tag","element_aware_source_dev","--use_syntactic_graph_adapter",
      "--syntactic_graph_cache_dir",a.graph_cache_dir,"--syntactic_graph_parser_dir",a.parser_dir,
      "--syntactic_graph_cache_tokenizer_path",a.model_path,"--syntactic_graph_split","source_dev"],check=True)
    subprocess.run(common+["pseudo","--run_dir",str(root),"--model_path",str(model),
      "--batch_size","1","--num_beams","1","--max_new_tokens","128","--cuda",a.cuda,
      "--no_task_prefix","--no_constrained_decoding","--use_syntactic_graph_adapter",
      "--syntactic_graph_cache_dir",a.graph_cache_dir,"--syntactic_graph_parser_dir",a.parser_dir,
      "--syntactic_graph_cache_tokenizer_path",a.model_path],check=True)
    result={"task":TASK_ID,"phase":"A",
      "treatment_only":True,"dann":0,"target_test_accessed":False,"phase_b_started":False,
      "variant":variant["name"],"training":frozen_config,
      "output_dir":str(root),"model_path":str(model)}
    (root/"treatment_only_entry.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))
if __name__=="__main__":
    main()

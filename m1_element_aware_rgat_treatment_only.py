from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from m1_syntactic_rgat_pseudo_quick_ablation import _build_input_rows, _serialize_rows
from t5_aste_data import parse_triplet_text_list, micro_f1, read_jsonl
import t5_absa_train as train_mod

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_serialize_rows(rows))

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
    focus_enabled = not a.coverage_only
    coverage_enabled = not a.focus_only
    root=Path(a.output_dir); root.mkdir(parents=True, exist_ok=True)
    project_root=Path(__file__).resolve().parent
    data_root=project_root / "data" / "aste" / "cross_domain"
    external={"source_train":{"path":str(data_root / "laptop14" / "train.txt")},"source_dev":{"path":str(data_root / "laptop14" / "dev.txt")},"target_unlabeled":{"path":str(data_root / "rest15" / "train.txt")}}
    rows=_build_input_rows("laptop14","rest15", external_inputs=external)
    write_jsonl(root/"source_train.jsonl", rows["source_train"])
    write_jsonl(root/"source_dev.jsonl", rows["source_dev"])
    write_jsonl(root/"target_unlabeled.jsonl", rows["target_unlabeled"])
    train_args=[
      "--model_path",a.model_path,"--train_file",str(root/"source_train.jsonl"),
      "--dev_file",str(root/"source_dev.jsonl"),"--output_dir",str(root/"models"/"extractor"),
      "--num_train_epochs","25","--source_weight","1.0","--pseudo_weight","0.75",
      "--augment_weight","0.2","--lambda_structure_loss","0","--lambda_consistency_loss","0",
      "--lambda_pairing_loss","0","--multi_triplet_loss_gain","0","--neutral_loss_gain","0",
      "--checkpoint_selection","last","--resume_from_checkpoint","none",
      "--per_device_train_batch_size",str(a.train_batch_size),"--per_device_eval_batch_size","2",
      "--max_source_length","128","--max_target_length","96","--gradient_accumulation_steps",str(a.gradient_accumulation_steps),
      "--learning_rate","0.0003","--lambda_domain_adv","0","--domain_adv_grl_lambda","1.0",
      "--domain_adv_hidden_size","256","--pairing_temperature","0.1","--max_effective_weight","1.0",
      "--neutral_generation_loss_gain","0","--neutral_generation_max_effective_weight","0",
      "--max_pairing_triplets","4","--min_pairing_triplets","2","--min_pairing_sample_weight","0.65",
      "--fp16","--gradient_checkpointing","--cuda",a.cuda,"--seed","1000","--legacy_stochastic",
      "--use_syntactic_graph_adapter","--syntactic_graph_cache_dir",a.graph_cache_dir,
      "--syntactic_graph_parser_dir",a.parser_dir,"--element_aware_attention",
      "--element_focus_loss" if focus_enabled else  "--multi_element_coverage_loss" if coverage_enabled else "", "--element_focus_weight","0.05" if focus_enabled else "0",
      "--element_coverage_weight","0.05" if coverage_enabled else "0","--target_unlabeled_file",str(root/"target_unlabeled.jsonl"),"--initialization_audit_path",str(root/"phase_a_initialization_audit.json")
    ]
    train_mod._PHASE_A_GRAPH_TRAINING_AUTHORIZED=True
    train_mod._PHASE_A_LIFECYCLE_CLEANUP_REQUESTED=True
    old=sys.argv; sys.argv=["t5_absa_train.py",*train_args]
    try: train_mod.main()
    finally: sys.argv=old
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
    result={"task":"M1_ELEMENT_AWARE_MULTI_TRIPLET_RGAT_QUICK_ABLATION_V1","phase":"A",
      "treatment_only":True,"dann":0,"target_test_accessed":False,"phase_b_started":False,
      "output_dir":str(root),"model_path":str(model)}
    (root/"treatment_only_entry.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))
if __name__=="__main__":
    main()

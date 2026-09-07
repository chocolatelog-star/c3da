from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from t5_absa_data import read_jsonl
from t5_absa_train import (
    JsonlSeq2SeqDataset,
    WeightedSeq2SeqTrainer,
    build_checkpoint_selection_config,
    configure_reproducibility,
    cleanup_training_checkpoints,
    reproducibility_training_args,
)


def combine_source_target_loss(source_loss: float, target_loss: float, target_weight: float) -> float:
    return float(source_loss) + float(target_weight) * float(target_loss)


class DualStreamDataset(Dataset):
    def __init__(self, source_data, target_data):
        if not source_data or not target_data:
            raise ValueError("source and target generator datasets must both be non-empty")
        self.source_data = source_data
        self.target_data = target_data

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, index):
        return {"source_index": index, "target_index": index % len(self.target_data)}


class DualStreamCollator:
    def __init__(self, source_data, target_data, base_collator):
        self.source_data = source_data
        self.target_data = target_data
        self.base_collator = base_collator

    def __call__(self, features):
        if not features or "source_index" not in features[0]:
            return self.base_collator(features)
        source_features = [self.source_data[item["source_index"]] for item in features]
        target_features = [self.target_data[item["target_index"]] for item in features]
        return {
            "source_batch": self.base_collator(source_features),
            "target_batch": self.base_collator(target_features),
        }


class TargetWeightedTrainer(WeightedSeq2SeqTrainer):
    def __init__(self, *args, target_generator_loss_weight: float, **kwargs):
        super().__init__(*args, **kwargs)
        if target_generator_loss_weight < 0:
            raise ValueError("target_generator_loss_weight must be non-negative")
        self.target_generator_loss_weight = float(target_generator_loss_weight)
        self._dual_audit = {"source_exposures": 0, "target_exposures": 0, "batches": 0}

    @staticmethod
    def _generation_loss(model, batch):
        clean = dict(batch)
        for key in JsonlSeq2SeqDataset._generation_only_input_keys if hasattr(JsonlSeq2SeqDataset, "_generation_only_input_keys") else ():
            clean.pop(key, None)
        for key in (
            "sample_weight", "domain_weight", "domain_label", "structure_weight", "consistency_group",
            "pairing_aspect_spans", "pairing_opinion_spans", "pairing_mask", "sentiment_contrastive_spans",
            "sentiment_contrastive_labels", "sentiment_contrastive_mask", "sentiment_contrastive_weights",
        ):
            clean.pop(key, None)
        labels = clean["labels"]
        outputs = model(**clean, return_dict=True, output_hidden_states=False)
        token_loss = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), ignore_index=-100, reduction="none"
        ).view(labels.size())
        mask = labels.ne(-100)
        return (token_loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1)).mean(), outputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "source_batch" not in inputs:
            loss, outputs = self._generation_loss(model, inputs)
            return (loss, outputs) if return_outputs else loss
        source_loss, source_outputs = self._generation_loss(model, inputs["source_batch"])
        target_loss, _target_outputs = self._generation_loss(model, inputs["target_batch"])
        total = source_loss + self.target_generator_loss_weight * target_loss
        if model.training:
            self._track_component("source_train_loss", source_loss)
            self._track_component("target_train_loss", target_loss)
            self._track_component("weighted_target_loss", self.target_generator_loss_weight * target_loss)
            self._track_component("joint_total_loss", total)
            self._dual_audit["batches"] += 1
            self._dual_audit["source_exposures"] += int(inputs["source_batch"]["input_ids"].size(0))
            self._dual_audit["target_exposures"] += int(inputs["target_batch"]["input_ids"].size(0))
        return (total, source_outputs) if return_outputs else total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--target_train_file", required=True)
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_generator_loss_weight", type=float, required=True)
    parser.add_argument("--source_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_weight", type=float, default=1.0)
    parser.add_argument("--augment_weight", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=float, default=25)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--checkpoint_selection", choices=["best", "last"], default="best")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_reproducibility(args.seed, "deterministic" if args.deterministic else "seeded")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    source_rows = read_jsonl(args.train_file)
    target_rows = read_jsonl(args.target_train_file)
    dev_rows = read_jsonl(args.dev_file)
    source_data = JsonlSeq2SeqDataset(source_rows, tokenizer, args.max_source_length, args.max_target_length, 1.0, 1.0, 1.0)
    target_data = JsonlSeq2SeqDataset(target_rows, tokenizer, args.max_source_length, args.max_target_length, 1.0, 1.0, 1.0)
    dev_data = JsonlSeq2SeqDataset(dev_rows, tokenizer, args.max_source_length, args.max_target_length, 1.0, 1.0, 1.0)
    dual = DualStreamDataset(source_data, target_data)
    collator = DualStreamCollator(source_data, target_data, DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model))
    selection = build_checkpoint_selection_config(args.checkpoint_selection)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        remove_unused_columns=False,
        report_to=[],
        **reproducibility_training_args(args.seed, "deterministic" if args.deterministic else "seeded"),
        **selection,
    )
    trainer = TargetWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dual,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        data_collator=collator,
        target_generator_loss_weight=args.target_generator_loss_weight,
    )
    trainer.train(resume_from_checkpoint=False)
    best_dir = Path(args.output_dir) / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    cleanup_training_checkpoints(Path(args.output_dir))
    audit = {
        "target_generator_loss_weight": args.target_generator_loss_weight,
        "source_rows": len(source_rows),
        "target_rows": len(target_rows),
        "source_batches": len(trainer.get_train_dataloader()),
        "optimizer_steps": trainer.state.global_step,
        "best_epoch": trainer.state.best_model_checkpoint,
        "log_history": [
            item for item in trainer.state.log_history
            if any(key in item for key in ("loss", "eval_loss", "source_train_loss", "target_train_loss"))
        ],
        **trainer._dual_audit,
    }
    (Path(args.output_dir) / "target_weighted_training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(audit)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from t5_absa_data import read_jsonl
from t5_aste_data import (
    micro_f1,
    micro_f1_by_triplet_count,
    parse_triplet_text_list,
    triplet_count_diagnostics,
)
from syntactic_graph import (
    CompositeGraphCache,
    GraphCacheError,
    build_parser_identity,
    build_tokenizer_identity,
    load_graph_cache_directory,
)
from syntactic_graph_adapter import load_seq2seq_model


def reproducibility_training_args(seed: int, mode: str) -> dict:
    if mode == "legacy":
        return {"seed": seed}
    if mode not in {"seeded", "deterministic"}:
        raise ValueError(f"unsupported reproducibility mode: {mode}")
    return {
        "seed": seed,
        "data_seed": seed,
        "full_determinism": mode == "deterministic",
        "dataloader_num_workers": 0,
    }


def configure_reproducibility(seed: int, mode: str) -> dict:
    deterministic = mode == "deterministic"
    if mode == "legacy":
        os.environ.pop("PYTHONHASHSEED", None)
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "mode": mode,
        "deterministic": deterministic,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


TASK_SPECIAL_TOKENS = ["<pos>", "<neg>", "<neu>", "<opinion>", "<aspect>"]
CSA_AUGMENT_CHANNELS = {
    "aspect_channel",
    "opinion_sentiment_channel",
    "masked_aspect_channel",
    "masked_opinion_sentiment_channel",
    "label_composition_channel",
    "label_to_text_channel",
    "sentence_fusion_composition_channel",
}
TAG_INIT_WORDS = {
    "<pos>": "positive",
    "<neg>": "negative",
    "<neu>": "neutral",
    "<opinion>": "opinion",
    "<aspect>": "aspect",
}
SENTIMENT_LABEL_IDS = {"pos": 0, "neg": 1, "neu": 2}


def build_target_unlabeled_domain_rows(
    rows: list[dict],
    use_task_prefix: bool = True,
) -> list[dict]:
    """Build target rows which contribute only to the existing DANN loss."""
    domain_rows = []
    for row in rows:
        text = str(row.get("text", ""))
        if not text:
            raise ValueError(f"target-unlabeled row has empty text: {row.get('id')}")
        input_text = f"extract aste: {text}" if use_task_prefix else text
        domain_rows.append(
            {
                "id": row["id"],
                "text": text,
                "input": input_text,
                "target": "",
                "augmentation": "target_unlabeled",
                "sample_weight": 0.0,
                "domain_weight": 0.0,
                "structure_weight": 0.0,
                "domain_label": 1,
            }
        )
    return domain_rows


def decode_keep_aste_task_tokens(tokenizer, token_ids) -> str:
    token_ids = [tokenizer.pad_token_id if int(token) < 0 else int(token) for token in token_ids]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    for token in (
        tokenizer.pad_token,
        tokenizer.eos_token,
        tokenizer.unk_token,
        "<s>",
    ):
        if token:
            text = text.replace(token, " ")
    return " ".join(text.split())


def _metric_input_to_numpy(value, name: str) -> np.ndarray:
    if isinstance(value, tuple):
        if not value:
            raise ValueError(f"{name} tuple must not be empty")
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a numpy array") from exc


def build_aste_compute_metrics(tokenizer):
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be defined for ASTE metrics")

    def compute_metrics(eval_prediction):
        predictions = _metric_input_to_numpy(eval_prediction.predictions, "predictions")
        labels = _metric_input_to_numpy(eval_prediction.label_ids, "labels")
        if predictions.ndim == 3:
            if predictions.shape[-1] == 0:
                raise ValueError("predictions logits dimension must not be empty")
            predictions = predictions.argmax(axis=-1)
        if predictions.ndim != 2:
            raise ValueError(
                f"predictions dimension must be 2 for token ids or 3 for logits; got {predictions.ndim}"
            )
        if labels.ndim != 2:
            raise ValueError(f"labels dimension must be 2; got {labels.ndim}")
        if predictions.shape[0] != labels.shape[0]:
            raise ValueError(
                "predictions and labels batch lengths must match; "
                f"got {predictions.shape[0]} and {labels.shape[0]}"
            )
        labels = labels.copy()
        labels[labels == -100] = tokenizer.pad_token_id

        prediction_texts = [decode_keep_aste_task_tokens(tokenizer, row) for row in predictions]
        gold_texts = [decode_keep_aste_task_tokens(tokenizer, row) for row in labels]
        overall = micro_f1(prediction_texts, gold_texts)
        grouped = micro_f1_by_triplet_count(prediction_texts, gold_texts)
        diagnostics = triplet_count_diagnostics(prediction_texts, gold_texts)

        multi_predictions = []
        multi_golds = []
        for prediction, gold in zip(prediction_texts, gold_texts):
            if len(parse_triplet_text_list(gold)) >= 2:
                multi_predictions.append(prediction)
                multi_golds.append(gold)
        multi = micro_f1(multi_predictions, multi_golds)

        metrics = {
            "micro_f1": overall["micro_f1"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "multi_micro_f1": multi["micro_f1"],
            "exact_count_accuracy": diagnostics["exact_count_accuracy"],
            "under_generated_rows": diagnostics["under_generated_rows"],
            "over_generated_rows": diagnostics["over_generated_rows"],
        }
        for bucket in ("count1", "count2", "count3", "count4plus"):
            metrics[f"{bucket}_micro_f1"] = grouped[bucket]["micro_f1"]
        metrics["selection_score"] = metrics["micro_f1"] + 0.001 * metrics["multi_micro_f1"]
        return metrics

    return compute_metrics


def build_checkpoint_selection_config(checkpoint_selection: str) -> dict:
    if checkpoint_selection not in {"last", "best", "aste_f1"}:
        raise ValueError(f"unsupported checkpoint selection: {checkpoint_selection}")
    if checkpoint_selection == "aste_f1":
        return {
            "predict_with_generate": True,
            "generation_num_beams": 1,
            "generation_max_length": 128,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_selection_score",
            "greater_is_better": True,
        }
    return {
        "predict_with_generate": True,
        "load_best_model_at_end": checkpoint_selection == "best",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }


class JsonlSeq2SeqDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        max_source_length: int,
        max_target_length: int,
        source_weight: float,
        pseudo_weight: float,
        augment_weight: float,
        multi_triplet_loss_gain: float = 0.0,
        neutral_loss_gain: float = 0.0,
        max_effective_weight: float = 1.0,
        neutral_generation_loss_gain: float = 0.0,
        neutral_generation_max_effective_weight: float | None = None,
        force_domain_weights: bool = False,
        max_pairing_triplets: int = 4,
        min_pairing_triplets: int = 2,
        min_pairing_sample_weight: float = 0.65,
        pairing_source_only: bool = False,
        domain_adv_exclude_augment: bool = False,
        sentiment_contrastive_min_weight: float = 0.65,
        sentiment_contrastive_exclude_augment: bool = False,
        sentiment_contrastive_source_only: bool = False,
        graph_cache=None,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_weight = source_weight
        self.pseudo_weight = pseudo_weight
        self.augment_weight = augment_weight
        self.multi_triplet_loss_gain = multi_triplet_loss_gain
        self.neutral_loss_gain = neutral_loss_gain
        self.max_effective_weight = max_effective_weight
        self.neutral_generation_loss_gain = neutral_generation_loss_gain
        self.neutral_generation_max_effective_weight = (
            1.0
            if neutral_generation_max_effective_weight is None or neutral_generation_max_effective_weight <= 0
            else neutral_generation_max_effective_weight
        )
        self.force_domain_weights = force_domain_weights
        self.max_pairing_triplets = max_pairing_triplets
        self.min_pairing_triplets = min_pairing_triplets
        self.min_pairing_sample_weight = min_pairing_sample_weight
        self.pairing_source_only = pairing_source_only
        self.domain_adv_exclude_augment = domain_adv_exclude_augment
        self.sentiment_contrastive_min_weight = sentiment_contrastive_min_weight
        self.sentiment_contrastive_exclude_augment = sentiment_contrastive_exclude_augment
        self.sentiment_contrastive_source_only = sentiment_contrastive_source_only
        self.graph_cache = graph_cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        model_inputs = self.tokenizer(
            row["input"],
            max_length=self.max_source_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=row["target"],
            max_length=self.max_target_length,
            truncation=True,
        )
        if row.get("augmentation") == "target_unlabeled":
            labels["input_ids"] = [-100] * len(labels["input_ids"])
        model_inputs["labels"] = labels["input_ids"]
        sample_weight = self.sample_weight(row)
        model_inputs["sample_weight"] = sample_weight
        model_inputs["domain_weight"] = self.generation_weight(row, sample_weight)
        model_inputs["domain_label"] = self.domain_label(row)
        model_inputs["structure_weight"] = self.structure_weight(row, sample_weight)
        model_inputs["consistency_group"] = self.consistency_group(row, idx)
        model_inputs.update(self.pairing_features(row, model_inputs["input_ids"], sample_weight))
        model_inputs.update(self.sentiment_contrastive_features(row, model_inputs["input_ids"], sample_weight))
        if self.graph_cache is not None:
            model_inputs.update(self.graph_cache.get(row))
        return model_inputs

    def sample_weight(self, row: dict) -> float:
        if row.get("augmentation") == "target_unlabeled":
            return 0.0
        if "sample_weight" in row and not self.force_domain_weights:
            return float(row["sample_weight"])
        augmentation = row.get("augmentation")
        if augmentation == "target_pseudo":
            return self.pseudo_weight
        if augmentation in CSA_AUGMENT_CHANNELS:
            return self.augment_weight
        return self.source_weight

    def domain_label(self, row: dict) -> int:
        augmentation = row.get("augmentation")
        if augmentation == "target_unlabeled":
            return 1
        if self.domain_adv_exclude_augment and augmentation in CSA_AUGMENT_CHANNELS:
            return -100
        if augmentation == "target_pseudo" or augmentation in CSA_AUGMENT_CHANNELS:
            return 1
        return 0

    def structure_weight(self, row: dict, domain_weight: float) -> float:
        triplets = parse_triplet_text_list(row.get("target", ""))
        multiplier = 1.0
        if len(triplets) >= 2:
            multiplier += self.multi_triplet_loss_gain * min(len(triplets) - 1, 2)
        if any(sentiment == "neu" for _aspect, _opinion, sentiment in triplets):
            multiplier += self.neutral_loss_gain
        return min(domain_weight * multiplier, self.max_effective_weight)

    def generation_weight(self, row: dict, sample_weight: float) -> float:
        triplets = parse_triplet_text_list(row.get("target", ""))
        if not any(sentiment == "neu" for _aspect, _opinion, sentiment in triplets):
            return sample_weight
        return min(
            sample_weight * (1.0 + self.neutral_generation_loss_gain),
            self.neutral_generation_max_effective_weight,
        )

    def consistency_group(self, row: dict, idx: int) -> int:
        if row.get("base_id") is not None:
            return stable_group_id(row["base_id"])
        if row.get("id") is not None:
            return stable_group_id(row["id"])
        return int(idx)

    def pairing_features(self, row: dict, input_ids: list[int], sample_weight: float) -> dict:
        target = row.get("target", "")
        triplets = parse_triplet_text_list(target)
        augmentation = row.get("augmentation")
        if self.pairing_source_only and (
            augmentation == "target_pseudo" or augmentation in CSA_AUGMENT_CHANNELS
        ):
            return {
                "pairing_aspect_spans": [],
                "pairing_opinion_spans": [],
                "pairing_mask": [],
            }
        if len(triplets) < self.min_pairing_triplets:
            return {
                "pairing_aspect_spans": [],
                "pairing_opinion_spans": [],
                "pairing_mask": [],
            }
        if sample_weight < self.min_pairing_sample_weight and row.get("augmentation") != "target_pseudo":
            return {
                "pairing_aspect_spans": [],
                "pairing_opinion_spans": [],
                "pairing_mask": [],
            }
        aspect_spans: list[list[int]] = []
        opinion_spans: list[list[int]] = []
        mask: list[int] = []
        for aspect, opinion, _sentiment in triplets[: self.max_pairing_triplets]:
            aspect_span = find_fragment_span_in_input(
                self.tokenizer, row.get("input", ""), input_ids, aspect
            )
            opinion_span = find_fragment_span_in_input(
                self.tokenizer, row.get("input", ""), input_ids, opinion
            )
            if aspect_span is None or opinion_span is None:
                continue
            aspect_spans.append(list(aspect_span))
            opinion_spans.append(list(opinion_span))
            mask.append(1)
        if len(mask) < self.min_pairing_triplets:
            return {
                "pairing_aspect_spans": [],
                "pairing_opinion_spans": [],
                "pairing_mask": [],
            }
        return {
            "pairing_aspect_spans": aspect_spans,
            "pairing_opinion_spans": opinion_spans,
            "pairing_mask": mask,
        }

    def sentiment_contrastive_features(self, row: dict, input_ids: list[int], domain_weight: float) -> dict:
        augmentation = row.get("augmentation")
        if domain_weight < self.sentiment_contrastive_min_weight:
            return self.empty_sentiment_contrastive_features()
        if self.sentiment_contrastive_exclude_augment and augmentation in CSA_AUGMENT_CHANNELS:
            return self.empty_sentiment_contrastive_features()
        if self.sentiment_contrastive_source_only and (
            augmentation == "target_pseudo" or augmentation in CSA_AUGMENT_CHANNELS
        ):
            return self.empty_sentiment_contrastive_features()
        spans = []
        labels = []
        for _aspect, opinion, sentiment in parse_triplet_text_list(row.get("target", "")):
            sentiment_id = SENTIMENT_LABEL_IDS.get(sentiment)
            span = find_opinion_span_in_input(self.tokenizer, row.get("input", ""), input_ids, opinion)
            if sentiment_id is None or span is None:
                continue
            spans.append(list(span))
            labels.append(sentiment_id)
        return {
            "sentiment_contrastive_spans": spans,
            "sentiment_contrastive_labels": labels,
            "sentiment_contrastive_mask": [1] * len(labels),
            "sentiment_contrastive_weights": [float(domain_weight)] * len(labels),
        }

    @staticmethod
    def empty_sentiment_contrastive_features() -> dict:
        return {
            "sentiment_contrastive_spans": [],
            "sentiment_contrastive_labels": [],
            "sentiment_contrastive_mask": [],
            "sentiment_contrastive_weights": [],
        }


def stable_group_id(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
        return int(digest[:12], 16)


def find_token_subsequence_span(sequence: list[int], subsequence: list[int]) -> tuple[int, int] | None:
    if not sequence or not subsequence or len(subsequence) > len(sequence):
        return None
    width = len(subsequence)
    for start in range(0, len(sequence) - width + 1):
        if sequence[start : start + width] == subsequence:
            return start, start + width
    return None


def find_fragment_span_in_input(
    tokenizer,
    text: str,
    input_ids: list[int],
    fragment: str,
) -> tuple[int, int] | None:
    candidates = [fragment]
    lower_text = text.lower()
    lower_fragment = fragment.lower()
    start = 0
    while lower_fragment and (match_start := lower_text.find(lower_fragment, start)) >= 0:
        candidates.append(text[match_start : match_start + len(fragment)])
        start = match_start + max(1, len(fragment))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        span = find_token_subsequence_span(input_ids, tokenizer.encode(candidate, add_special_tokens=False))
        if span is not None:
            return span
    return None


def find_opinion_span_in_input(tokenizer, text: str, input_ids: list[int], opinion: str) -> tuple[int, int] | None:
    return find_fragment_span_in_input(tokenizer, text, input_ids, opinion)


class DataCollatorForSeq2SeqWithPairing:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features: list[dict]) -> dict:
        graph_keys = (
            "word_to_subword",
            "word_mask",
            "edge_src",
            "edge_dst",
            "relation_id",
            "dependency_relation_id",
            "pos_pair_id",
            "edge_mask",
        )
        graph_present = [key in feature for feature in features for key in graph_keys]
        if any(graph_present) and not all(graph_present):
            raise ValueError("graph fields must be present for every feature or for none of them")
        graph_values = {key: [feature.pop(key, None) for feature in features] for key in graph_keys}
        pairing_aspect_spans = [feature.pop("pairing_aspect_spans", []) for feature in features]
        pairing_opinion_spans = [feature.pop("pairing_opinion_spans", []) for feature in features]
        pairing_masks = [feature.pop("pairing_mask", []) for feature in features]
        sentiment_spans = [feature.pop("sentiment_contrastive_spans", []) for feature in features]
        sentiment_labels = [feature.pop("sentiment_contrastive_labels", []) for feature in features]
        sentiment_masks = [feature.pop("sentiment_contrastive_mask", []) for feature in features]
        sentiment_weights = [feature.pop("sentiment_contrastive_weights", []) for feature in features]
        batch = self.base_collator(features)
        max_pairs = max([len(mask) for mask in pairing_masks] + [0])
        if max_pairs == 0:
            batch["pairing_aspect_spans"] = torch.zeros((len(features), 0, 2), dtype=torch.long)
            batch["pairing_opinion_spans"] = torch.zeros((len(features), 0, 2), dtype=torch.long)
            batch["pairing_mask"] = torch.zeros((len(features), 0), dtype=torch.long)
        else:
            aspect_tensor = torch.zeros((len(features), max_pairs, 2), dtype=torch.long)
            opinion_tensor = torch.zeros((len(features), max_pairs, 2), dtype=torch.long)
            mask_tensor = torch.zeros((len(features), max_pairs), dtype=torch.long)
            for row_idx, (aspect_spans, opinion_spans, mask) in enumerate(
                zip(pairing_aspect_spans, pairing_opinion_spans, pairing_masks)
            ):
                for pair_idx, (aspect_span, opinion_span, active) in enumerate(zip(aspect_spans, opinion_spans, mask)):
                    if pair_idx >= max_pairs:
                        break
                    aspect_tensor[row_idx, pair_idx] = torch.tensor(aspect_span, dtype=torch.long)
                    opinion_tensor[row_idx, pair_idx] = torch.tensor(opinion_span, dtype=torch.long)
                    mask_tensor[row_idx, pair_idx] = int(active)
            batch["pairing_aspect_spans"] = aspect_tensor
            batch["pairing_opinion_spans"] = opinion_tensor
            batch["pairing_mask"] = mask_tensor

        max_sentiments = max([len(mask) for mask in sentiment_masks] + [0])
        sentiment_span_tensor = torch.zeros((len(features), max_sentiments, 2), dtype=torch.long)
        sentiment_label_tensor = torch.full((len(features), max_sentiments), -100, dtype=torch.long)
        sentiment_mask_tensor = torch.zeros((len(features), max_sentiments), dtype=torch.long)
        sentiment_weight_tensor = torch.zeros((len(features), max_sentiments), dtype=torch.float)
        for row_idx, (spans, labels, mask, weights) in enumerate(
            zip(sentiment_spans, sentiment_labels, sentiment_masks, sentiment_weights)
        ):
            for item_idx, (span, label, active, weight) in enumerate(zip(spans, labels, mask, weights)):
                sentiment_span_tensor[row_idx, item_idx] = torch.tensor(span, dtype=torch.long)
                sentiment_label_tensor[row_idx, item_idx] = int(label)
                sentiment_mask_tensor[row_idx, item_idx] = int(active)
                sentiment_weight_tensor[row_idx, item_idx] = float(weight)
        batch["sentiment_contrastive_spans"] = sentiment_span_tensor
        batch["sentiment_contrastive_labels"] = sentiment_label_tensor
        batch["sentiment_contrastive_mask"] = sentiment_mask_tensor
        batch["sentiment_contrastive_weights"] = sentiment_weight_tensor
        if any(graph_present):
            max_words = max(len(value) for value in graph_values["word_to_subword"])
            max_subwords = max(
                max((len(indices) for indices in value), default=0)
                for value in graph_values["word_to_subword"]
            )
            max_edges = max(len(value) for value in graph_values["edge_src"])
            word_to_subword = torch.full(
                (len(features), max_words, max(1, max_subwords)),
                -1,
                dtype=torch.long,
            )
            word_mask = torch.zeros((len(features), max_words), dtype=torch.bool)
            edge_tensors = {
                key: torch.zeros((len(features), max_edges), dtype=torch.long)
                for key in (
                    "edge_src",
                    "edge_dst",
                    "relation_id",
                    "dependency_relation_id",
                    "pos_pair_id",
                )
            }
            edge_mask = torch.zeros((len(features), max_edges), dtype=torch.bool)
            for row_index, values in enumerate(zip(*graph_values.values())):
                row_word_to_subword, row_word_mask, row_src, row_dst, row_relation, row_dependency, row_pos, row_edge_mask = values
                word_mask[row_index, : len(row_word_mask)] = torch.tensor(row_word_mask, dtype=torch.bool)
                for word_index, indices in enumerate(row_word_to_subword):
                    word_to_subword[row_index, word_index, : len(indices)] = torch.tensor(indices, dtype=torch.long)
                edge_count = len(row_src)
                edge_mask[row_index, :edge_count] = torch.tensor(row_edge_mask, dtype=torch.bool)
                edge_tensors["edge_src"][row_index, :edge_count] = torch.tensor(row_src, dtype=torch.long)
                edge_tensors["edge_dst"][row_index, :edge_count] = torch.tensor(row_dst, dtype=torch.long)
                edge_tensors["relation_id"][row_index, :edge_count] = torch.tensor(row_relation, dtype=torch.long)
                edge_tensors["dependency_relation_id"][row_index, :edge_count] = torch.tensor(row_dependency, dtype=torch.long)
                edge_tensors["pos_pair_id"][row_index, :edge_count] = torch.tensor(row_pos, dtype=torch.long)
            batch["graph_word_to_subword"] = word_to_subword
            batch["graph_word_mask"] = word_mask
            batch["graph_edge_src"] = edge_tensors["edge_src"]
            batch["graph_edge_dst"] = edge_tensors["edge_dst"]
            batch["graph_relation_id"] = edge_tensors["relation_id"]
            batch["graph_dependency_relation_id"] = edge_tensors["dependency_relation_id"]
            batch["graph_pos_pair_id"] = edge_tensors["pos_pair_id"]
            batch["graph_edge_mask"] = edge_mask
        return batch


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        ctx.grl_lambda = grl_lambda
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.grl_lambda * grad_output, None


def gradient_reverse(inputs: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(inputs, grl_lambda)


class DomainAdversarialHead(nn.Module):
    def __init__(self, hidden_size: int, classifier_hidden_size: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden_size, 2),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled_hidden)


class SentimentPrototypeHead(nn.Module):
    def __init__(self, hidden_size: int, num_sentiments: int = 3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_sentiments, hidden_size))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, p=2, dim=-1)


def build_sentiment_prototype_centroids(
    vectors: torch.Tensor,
    labels: torch.Tensor,
    num_sentiments: int = 3,
) -> tuple[torch.Tensor, list[int]]:
    if vectors.ndim != 2 or labels.ndim != 1 or vectors.size(0) != labels.size(0):
        raise ValueError("vectors and labels must have aligned [N, H] and [N] shapes")
    centroids = []
    counts = []
    for sentiment_id in range(num_sentiments):
        class_vectors = vectors[labels == sentiment_id]
        counts.append(int(class_vectors.size(0)))
        if class_vectors.size(0) == 0:
            raise ValueError(f"cannot initialize sentiment prototype {sentiment_id}: no examples")
        centroids.append(F.normalize(class_vectors.mean(dim=0), p=2, dim=0))
    return torch.stack(centroids), counts


def initialize_sentiment_prototypes_from_context(
    model,
    tokenizer,
    rows: list[dict],
    batch_size: int,
    max_source_length: int,
) -> dict:
    source_rows = [
        row for row in rows
        if row.get("augmentation") != "target_pseudo"
        and row.get("augmentation") not in CSA_AUGMENT_CHANNELS
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    was_training = model.training
    model.eval()
    collected_vectors = []
    collected_labels = []
    for start in tqdm(range(0, len(source_rows), batch_size), desc="init-sentiment-prototypes"):
        batch_rows = source_rows[start : start + batch_size]
        encoded = tokenizer(
            [row["input"] for row in batch_rows],
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            encoder_hidden = model.get_encoder()(**encoded, return_dict=True).last_hidden_state
        for row_idx, row in enumerate(batch_rows):
            row_input_ids = encoded["input_ids"][row_idx].tolist()
            for _aspect, opinion, sentiment in parse_triplet_text_list(row.get("target", "")):
                sentiment_id = SENTIMENT_LABEL_IDS.get(sentiment)
                span = find_opinion_span_in_input(
                    tokenizer,
                    row.get("input", ""),
                    row_input_ids,
                    opinion,
                )
                if sentiment_id is None or span is None:
                    continue
                collected_vectors.append(encoder_hidden[row_idx, span[0] : span[1]].mean(dim=0).float().cpu())
                collected_labels.append(sentiment_id)
    if was_training:
        model.train()
    if not collected_vectors:
        raise ValueError("no opinion context vectors were collected for sentiment prototype initialization")
    vectors = torch.stack(collected_vectors)
    labels = torch.tensor(collected_labels, dtype=torch.long)
    centroids, counts = build_sentiment_prototype_centroids(vectors, labels)
    with torch.no_grad():
        model.sentiment_prototype_head.prototypes.copy_(
            centroids.to(
                model.sentiment_prototype_head.prototypes.device,
                dtype=model.sentiment_prototype_head.prototypes.dtype,
            )
        )
    return {
        "source_rows": len(source_rows),
        "embedded_triplets": len(collected_vectors),
        "sentiment_counts": dict(zip(("pos", "neg", "neu"), counts)),
        "prototype_norms": [round(float(value), 6) for value in centroids.norm(dim=-1)],
        "device": str(device),
    }


def mean_pool_encoder_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def compute_domain_adversarial_loss(
    encoder_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    domain_labels: torch.Tensor,
    domain_adversarial_head: nn.Module,
    grl_lambda: float = 1.0,
) -> torch.Tensor | None:
    """Compute the existing DANN loss on post-graph encoder states."""
    pooled_hidden = mean_pool_encoder_hidden(encoder_hidden, attention_mask)
    reversed_hidden = gradient_reverse(pooled_hidden, grl_lambda)
    domain_logits = domain_adversarial_head(reversed_hidden)
    domain_targets = domain_labels.to(domain_logits.device, dtype=torch.long).view(-1)
    domain_valid_mask = domain_targets.ne(-100)
    if not domain_valid_mask.any():
        return None
    return F.cross_entropy(domain_logits[domain_valid_mask], domain_targets[domain_valid_mask])


class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    _generation_only_input_keys = {
        "sample_weight",
        "domain_weight",
        "domain_label",
        "structure_weight",
        "consistency_group",
        "pairing_aspect_spans",
        "pairing_opinion_spans",
        "pairing_mask",
        "sentiment_contrastive_spans",
        "sentiment_contrastive_labels",
        "sentiment_contrastive_mask",
        "sentiment_contrastive_weights",
    }
    _graph_input_keys = {
        "graph_word_to_subword",
        "graph_word_mask",
        "graph_edge_src",
        "graph_edge_dst",
        "graph_relation_id",
        "graph_dependency_relation_id",
        "graph_pos_pair_id",
        "graph_edge_mask",
    }

    def __init__(
        self,
        *args,
        lambda_structure_loss: float = 0.0,
        lambda_consistency_loss: float = 0.0,
        lambda_pairing_loss: float = 0.0,
        pairing_temperature: float = 0.1,
        lambda_domain_adv: float = 0.0,
        domain_adv_grl_lambda: float = 1.0,
        lambda_sentiment_contrastive: float = 0.0,
        sentiment_contrastive_temperature: float = 0.1,
        sentiment_contrastive_class_weights: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_structure_loss = lambda_structure_loss
        self.lambda_consistency_loss = lambda_consistency_loss
        self.lambda_pairing_loss = lambda_pairing_loss
        self.pairing_temperature = pairing_temperature
        self.lambda_domain_adv = lambda_domain_adv
        self.domain_adv_grl_lambda = domain_adv_grl_lambda
        self.lambda_sentiment_contrastive = lambda_sentiment_contrastive
        self.sentiment_contrastive_temperature = sentiment_contrastive_temperature
        self.sentiment_contrastive_class_weights = sentiment_contrastive_class_weights
        self._component_sums: dict[str, float] = {}
        self._component_counts: dict[str, int] = {}
        self._component_reductions: dict[str, str] = {}

    @classmethod
    def _strip_generation_only_inputs(cls, inputs: dict, keep_graph: bool = False) -> dict:
        cleaned = dict(inputs)
        for key in cls._generation_only_input_keys:
            cleaned.pop(key, None)
        if not keep_graph:
            for key in cls._graph_input_keys:
                cleaned.pop(key, None)
        return cleaned

    def _track_component(self, name: str, value: torch.Tensor | float, reduction: str = "mean") -> None:
        numeric = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        self._component_sums[name] = self._component_sums.get(name, 0.0) + numeric
        self._component_counts[name] = self._component_counts.get(name, 0) + 1
        self._component_reductions[name] = reduction

    def log(self, logs: dict, *args, **kwargs) -> None:
        if "loss" in logs and self._component_sums:
            for name, total in self._component_sums.items():
                if self._component_reductions.get(name) == "sum":
                    logs[name] = round(total, 6)
                else:
                    logs[name] = round(total / max(1, self._component_counts.get(name, 1)), 6)
            self._component_sums.clear()
            self._component_counts.clear()
            self._component_reductions.clear()
        super().log(logs, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        cleaned_inputs = self._strip_generation_only_inputs(
            inputs,
            keep_graph=bool(getattr(model, "use_syntactic_graph_adapter", False)),
        )
        return super().prediction_step(
            model,
            cleaned_inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
            **kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weight = inputs.pop("sample_weight", None)
        domain_weight = inputs.pop("domain_weight", sample_weight)
        domain_label = inputs.pop("domain_label", None)
        structure_weight = inputs.pop("structure_weight", None)
        consistency_group = inputs.pop("consistency_group", None)
        pairing_aspect_spans = inputs.pop("pairing_aspect_spans", None)
        pairing_opinion_spans = inputs.pop("pairing_opinion_spans", None)
        pairing_mask = inputs.pop("pairing_mask", None)
        sentiment_contrastive_spans = inputs.pop("sentiment_contrastive_spans", None)
        sentiment_contrastive_labels = inputs.pop("sentiment_contrastive_labels", None)
        sentiment_contrastive_mask = inputs.pop("sentiment_contrastive_mask", None)
        sentiment_contrastive_weights = inputs.pop("sentiment_contrastive_weights", None)
        graph_inputs = {
            key: inputs.pop(key, None)
            for key in self._graph_input_keys
        }
        if getattr(model, "use_syntactic_graph_adapter", False):
            if any(value is None for value in graph_inputs.values()):
                missing = [key for key, value in graph_inputs.items() if value is None]
                raise ValueError(f"syntactic graph model received incomplete graph batch: {missing}")
            inputs.update(graph_inputs)
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        outputs = model(**inputs, return_dict=True, output_hidden_states=False)
        logits = outputs.logits
        token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.size())
        token_mask = labels.ne(-100)
        per_sample_loss = token_loss.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        if domain_weight is not None:
            domain_weights = domain_weight.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
            if structure_weight is not None:
                structure_weights = structure_weight.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
                loss = joint_weighted_loss(
                    per_sample_loss,
                    domain_weights,
                    structure_weights,
                    self.lambda_structure_loss,
                )
            else:
                loss = weighted_loss_mean(per_sample_loss, domain_weights)
        else:
            loss = per_sample_loss.mean()
        generation_loss = loss
        if consistency_group is not None and self.lambda_consistency_loss > 0:
            consistency_loss = grouped_representation_consistency_loss(
                outputs.encoder_last_hidden_state,
                attention_mask,
                consistency_group,
            )
            loss = loss + self.lambda_consistency_loss * consistency_loss
        if self.lambda_pairing_loss > 0 and pairing_aspect_spans is not None and pairing_opinion_spans is not None:
            encoder_hidden = outputs.encoder_last_hidden_state
            if encoder_hidden is not None:
                pair_loss, pairing_stats = encoder_pairing_contrastive_loss(
                    encoder_hidden,
                    pairing_aspect_spans,
                    pairing_opinion_spans,
                    pairing_mask,
                    temperature=self.pairing_temperature,
                    return_stats=True,
                )
                loss = loss + self.lambda_pairing_loss * pair_loss
                if model.training:
                    self._track_component("pairing_loss", pair_loss)
                    for name, value in pairing_stats.items():
                        reduction = "sum" if name in {"pairing_active_rows", "pairing_active_pairs"} else "mean"
                        self._track_component(name, value, reduction=reduction)
        if (
            self.lambda_sentiment_contrastive > 0
            and sentiment_contrastive_spans is not None
            and hasattr(model, "sentiment_prototype_head")
        ):
            encoder_hidden = outputs.encoder_last_hidden_state
            if encoder_hidden is not None:
                sentiment_loss, sentiment_stats = sentiment_prototype_contrastive_loss(
                    encoder_hidden,
                    sentiment_contrastive_spans,
                    sentiment_contrastive_labels,
                    sentiment_contrastive_mask,
                    model.sentiment_prototype_head,
                    temperature=self.sentiment_contrastive_temperature,
                    sample_weights=sentiment_contrastive_weights,
                    class_weights=(
                        torch.tensor(self.sentiment_contrastive_class_weights, device=encoder_hidden.device)
                        if self.sentiment_contrastive_class_weights else None
                    ),
                    return_stats=True,
                )
                loss = loss + self.lambda_sentiment_contrastive * sentiment_loss
                if model.training:
                    self._track_component("sentiment_contrastive_loss", sentiment_loss)
                    for name, value in sentiment_stats.items():
                        self._track_component(name, value)
        if (
            model.training
            and self.lambda_domain_adv > 0
            and domain_label is not None
            and hasattr(model, "domain_adversarial_head")
            and outputs.encoder_last_hidden_state is not None
        ):
            domain_adv_loss = compute_domain_adversarial_loss(
                outputs.encoder_last_hidden_state,
                attention_mask,
                domain_label,
                model.domain_adversarial_head,
                grl_lambda=self.domain_adv_grl_lambda,
            )
            if domain_adv_loss is not None:
                loss = loss + self.lambda_domain_adv * domain_adv_loss
                self._track_component("domain_adv_loss", domain_adv_loss)
        if model.training:
            self._track_component("generation_loss", generation_loss)
            self._track_component("joint_total_loss", loss)
        return (loss, outputs) if return_outputs else loss


def weighted_loss_mean(per_sample_loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (per_sample_loss * weights).mean()


def joint_weighted_loss(
    per_sample_loss: torch.Tensor,
    domain_weights: torch.Tensor,
    structure_weights: torch.Tensor,
    lambda_structure: float,
) -> torch.Tensor:
    domain_loss = weighted_loss_mean(per_sample_loss, domain_weights)
    if lambda_structure <= 0:
        return domain_loss
    structure_loss = weighted_loss_mean(per_sample_loss, structure_weights)
    return domain_loss + lambda_structure * structure_loss


def grouped_representation_consistency_loss(
    representations: torch.Tensor,
    attention_mask: torch.Tensor | None,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    if representations is None or group_ids is None:
        return torch.tensor(0.0, device=representations.device if representations is not None else None)
    if representations.size(0) <= 1:
        return representations.new_tensor(0.0)
    pooled = representations
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(pooled.dtype)
        pooled = (pooled * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    group_ids = group_ids.to(pooled.device).view(-1)
    unique_group_ids = torch.unique(group_ids)
    losses = []
    for group_id in unique_group_ids:
        member_idx = torch.nonzero(group_ids == group_id, as_tuple=False).view(-1)
        if member_idx.numel() < 2:
            continue
        group_repr = F.normalize(pooled.index_select(0, member_idx), p=2, dim=-1)
        center = F.normalize(group_repr.mean(dim=0, keepdim=True), p=2, dim=-1)
        losses.append(1.0 - F.cosine_similarity(group_repr, center.expand_as(group_repr), dim=-1).mean())
    if not losses:
        return representations.new_tensor(0.0)
    return torch.stack(losses).mean()


def span_mean(hidden: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
    vectors = []
    seq_len = hidden.size(1)
    for batch_idx, batch_spans in enumerate(spans):
        row_vectors = []
        for start, end in batch_spans.tolist():
            start = max(0, min(int(start), seq_len - 1))
            end = max(start + 1, min(int(end), seq_len))
            row_vectors.append(hidden[batch_idx, start:end].mean(dim=0))
        vectors.append(torch.stack(row_vectors, dim=0) if row_vectors else hidden.new_zeros((0, hidden.size(-1))))
    if not vectors:
        return hidden.new_zeros((0, 0, hidden.size(-1)))
    return torch.stack(vectors, dim=0)


def pairing_contrastive_loss(
    decoder_hidden: torch.Tensor,
    aspect_spans: torch.Tensor,
    opinion_spans: torch.Tensor,
    pairing_mask: torch.Tensor | None,
    temperature: float = 0.1,
) -> torch.Tensor:
    if decoder_hidden is None or aspect_spans is None or opinion_spans is None:
        return decoder_hidden.new_tensor(0.0) if decoder_hidden is not None else torch.tensor(0.0)
    if aspect_spans.numel() == 0 or opinion_spans.numel() == 0:
        return decoder_hidden.new_tensor(0.0)
    aspect_spans = aspect_spans.to(decoder_hidden.device)
    opinion_spans = opinion_spans.to(decoder_hidden.device)
    if pairing_mask is None:
        pairing_mask = torch.ones(aspect_spans.shape[:2], device=decoder_hidden.device, dtype=torch.bool)
    else:
        pairing_mask = pairing_mask.to(decoder_hidden.device).bool()
    aspect_repr = F.normalize(span_mean(decoder_hidden, aspect_spans), p=2, dim=-1)
    opinion_repr = F.normalize(span_mean(decoder_hidden, opinion_spans), p=2, dim=-1)
    losses = []
    for batch_idx in range(aspect_repr.size(0)):
        active_idx = torch.nonzero(pairing_mask[batch_idx], as_tuple=False).view(-1)
        if active_idx.numel() < 2:
            continue
        aspects = aspect_repr[batch_idx].index_select(0, active_idx)
        opinions = opinion_repr[batch_idx].index_select(0, active_idx)
        logits = aspects @ opinions.transpose(0, 1) / temperature
        targets = torch.arange(active_idx.numel(), device=decoder_hidden.device)
        losses.append(F.cross_entropy(logits, targets))
    if not losses:
        return decoder_hidden.new_tensor(0.0)
    return torch.stack(losses).mean()


def _multi_positive_direction_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, float, int]:
    losses = []
    correct = 0
    active_anchors = 0
    for anchor_idx in range(logits.size(0)):
        positives = positive_mask[anchor_idx]
        if not positives.any() or positives.all():
            continue
        anchor_logits = logits[anchor_idx]
        losses.append(torch.logsumexp(anchor_logits, dim=0) - torch.logsumexp(anchor_logits[positives], dim=0))
        predicted_idx = int(anchor_logits.argmax().item())
        correct += int(bool(positives[predicted_idx]))
        active_anchors += 1
    if not losses:
        return logits.new_tensor(0.0), 0.0, 0
    return torch.stack(losses).mean(), correct / active_anchors, active_anchors


def encoder_pairing_contrastive_loss(
    encoder_hidden: torch.Tensor,
    aspect_spans: torch.Tensor,
    opinion_spans: torch.Tensor,
    pairing_mask: torch.Tensor | None,
    temperature: float = 0.1,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    zero = encoder_hidden.new_tensor(0.0)
    empty_stats = {
        "pairing_aspect_accuracy": 0.0,
        "pairing_opinion_accuracy": 0.0,
        "pairing_active_rows": 0.0,
        "pairing_active_pairs": 0.0,
    }
    if aspect_spans is None or opinion_spans is None or aspect_spans.numel() == 0 or opinion_spans.numel() == 0:
        return (zero, empty_stats) if return_stats else zero
    aspect_spans = aspect_spans.to(encoder_hidden.device)
    opinion_spans = opinion_spans.to(encoder_hidden.device)
    if pairing_mask is None:
        pairing_mask = torch.ones(aspect_spans.shape[:2], device=encoder_hidden.device, dtype=torch.bool)
    else:
        pairing_mask = pairing_mask.to(encoder_hidden.device).bool()
    aspect_repr = F.normalize(span_mean(encoder_hidden, aspect_spans), p=2, dim=-1)
    opinion_repr = F.normalize(span_mean(encoder_hidden, opinion_spans), p=2, dim=-1)
    losses = []
    aspect_correct_weighted = 0.0
    opinion_correct_weighted = 0.0
    aspect_anchor_count = 0
    opinion_anchor_count = 0
    active_rows = 0
    active_pairs = 0
    for batch_idx in range(aspect_repr.size(0)):
        active_idx = torch.nonzero(pairing_mask[batch_idx], as_tuple=False).view(-1)
        if active_idx.numel() < 2:
            continue
        aspects = aspect_repr[batch_idx].index_select(0, active_idx)
        opinions = opinion_repr[batch_idx].index_select(0, active_idx)
        active_aspect_spans = aspect_spans[batch_idx].index_select(0, active_idx)
        active_opinion_spans = opinion_spans[batch_idx].index_select(0, active_idx)
        aspect_same = (active_aspect_spans[:, None, :] == active_aspect_spans[None, :, :]).all(dim=-1)
        opinion_same = (active_opinion_spans[:, None, :] == active_opinion_spans[None, :, :]).all(dim=-1)
        positive_mask = (aspect_same[:, None, :] & opinion_same[None, :, :]).any(dim=-1)
        logits = aspects @ opinions.transpose(0, 1) / max(float(temperature), 1e-6)
        aspect_loss, aspect_accuracy, aspect_anchors = _multi_positive_direction_loss(logits, positive_mask)
        opinion_loss, opinion_accuracy, opinion_anchors = _multi_positive_direction_loss(
            logits.transpose(0, 1), positive_mask.transpose(0, 1)
        )
        row_losses = []
        if aspect_anchors:
            row_losses.append(aspect_loss)
            aspect_correct_weighted += aspect_accuracy * aspect_anchors
            aspect_anchor_count += aspect_anchors
        if opinion_anchors:
            row_losses.append(opinion_loss)
            opinion_correct_weighted += opinion_accuracy * opinion_anchors
            opinion_anchor_count += opinion_anchors
        if row_losses:
            losses.append(torch.stack(row_losses).mean())
            active_rows += 1
            active_pairs += int(active_idx.numel())
    loss = torch.stack(losses).mean() if losses else zero
    stats = {
        "pairing_aspect_accuracy": aspect_correct_weighted / max(1, aspect_anchor_count),
        "pairing_opinion_accuracy": opinion_correct_weighted / max(1, opinion_anchor_count),
        "pairing_active_rows": float(active_rows),
        "pairing_active_pairs": float(active_pairs),
    }
    return (loss, stats) if return_stats else loss


def sentiment_prototype_contrastive_loss(
    contextual_hidden: torch.Tensor,
    opinion_spans: torch.Tensor,
    sentiment_labels: torch.Tensor,
    sentiment_mask: torch.Tensor,
    prototype_head: SentimentPrototypeHead,
    temperature: float = 0.1,
    sample_weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    if opinion_spans is None or opinion_spans.numel() == 0:
        zero = contextual_hidden.new_tensor(0.0)
        return (zero, {}) if return_stats else zero
    opinion_spans = opinion_spans.to(contextual_hidden.device)
    sentiment_labels = sentiment_labels.to(contextual_hidden.device, dtype=torch.long)
    valid_mask = sentiment_mask.to(contextual_hidden.device).bool() & sentiment_labels.ne(-100)
    if not valid_mask.any():
        zero = contextual_hidden.new_tensor(0.0)
        return (zero, {}) if return_stats else zero
    opinion_repr = F.normalize(span_mean(contextual_hidden, opinion_spans), p=2, dim=-1)
    logits = opinion_repr[valid_mask] @ prototype_head.normalized_prototypes().transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    targets = sentiment_labels[valid_mask]
    per_item_loss = F.cross_entropy(
        logits,
        targets,
        weight=class_weights.to(logits.device, dtype=logits.dtype) if class_weights is not None else None,
        reduction="none",
    )
    valid_weights = (
        sample_weights.to(logits.device, dtype=logits.dtype)[valid_mask]
        if sample_weights is not None else torch.ones_like(per_item_loss)
    )
    loss = (per_item_loss * valid_weights).sum() / valid_weights.sum().clamp_min(1e-6)
    if not return_stats:
        return loss
    predictions = logits.argmax(dim=-1)
    stats = {}
    for sentiment_id, sentiment_name in enumerate(("pos", "neg", "neu")):
        class_mask = targets.eq(sentiment_id)
        if class_mask.any():
            stats[f"sentiment_{sentiment_name}_accuracy"] = float(predictions[class_mask].eq(targets[class_mask]).float().mean())
    stats["sentiment_prototype_accuracy"] = float(predictions.eq(targets).float().mean())
    return loss, stats


def summarize_sample_weights(
    rows: list[dict],
    source_weight: float,
    pseudo_weight: float,
    augment_weight: float,
    force_domain_weights: bool = False,
) -> dict:
    counts = {"source_gold": 0, "target_pseudo": 0, "c3da_augment": 0, "target_unlabeled": 0}
    weights = []
    for row in rows:
        augmentation = row.get("augmentation")
        if augmentation == "target_unlabeled":
            counts["target_unlabeled"] += 1
            fallback_weight = 0.0
        elif augmentation == "target_pseudo":
            counts["target_pseudo"] += 1
            fallback_weight = pseudo_weight
        elif augmentation in CSA_AUGMENT_CHANNELS:
            counts["c3da_augment"] += 1
            fallback_weight = augment_weight
        else:
            counts["source_gold"] += 1
            fallback_weight = source_weight
        weights.append(float(fallback_weight if force_domain_weights else row.get("sample_weight", fallback_weight)))
    by_source = {}
    for name, predicate, fallback_weight in [
        ("source_gold", lambda row: row.get("augmentation") not in {"target_pseudo", "target_unlabeled", *CSA_AUGMENT_CHANNELS}, source_weight),
        ("target_pseudo", lambda row: row.get("augmentation") == "target_pseudo", pseudo_weight),
        ("c3da_augment", lambda row: row.get("augmentation") in CSA_AUGMENT_CHANNELS, augment_weight),
    ]:
        source_weights = [
            float(fallback_weight if force_domain_weights else row.get("sample_weight", fallback_weight))
            for row in rows
            if predicate(row)
        ]
        if source_weights:
            by_source[f"{name}_weight_mean"] = sum(source_weights) / len(source_weights)
            by_source[f"{name}_weight_min"] = min(source_weights)
            by_source[f"{name}_weight_max"] = max(source_weights)
    return {
        **counts,
        "source_weight": source_weight,
        "pseudo_weight": pseudo_weight,
        "augment_weight": augment_weight,
        "force_domain_weights": force_domain_weights,
        "sample_weight_min": min(weights) if weights else None,
        "sample_weight_max": max(weights) if weights else None,
        "sample_weight_mean": sum(weights) / len(weights) if weights else None,
        **by_source,
    }


def summarize_generation_weights(dataset: JsonlSeq2SeqDataset) -> dict:
    neutral_weights = []
    non_neutral_weights = []
    for row in dataset.rows:
        if row.get("augmentation") == "target_unlabeled":
            continue
        domain_weight = dataset.sample_weight(row)
        effective_weight = dataset.generation_weight(row, domain_weight)
        triplets = parse_triplet_text_list(row.get("target", ""))
        target = (
            neutral_weights
            if any(sentiment == "neu" for _aspect, _opinion, sentiment in triplets)
            else non_neutral_weights
        )
        target.append(effective_weight)

    def weight_stats(name: str, weights: list[float]) -> dict:
        if not weights:
            return {
                f"{name}_rows": 0,
                f"{name}_weight_mean": None,
                f"{name}_weight_min": None,
                f"{name}_weight_max": None,
            }
        return {
            f"{name}_rows": len(weights),
            f"{name}_weight_mean": sum(weights) / len(weights),
            f"{name}_weight_min": min(weights),
            f"{name}_weight_max": max(weights),
        }

    return {
        **weight_stats("neutral", neutral_weights),
        **weight_stats("non_neutral", non_neutral_weights),
    }


def summarize_sentiment_contrastive_rows(
    rows: list[dict],
    min_weight: float,
    exclude_augment: bool,
    source_only: bool = False,
) -> dict:
    counts = {"pos": 0, "neg": 0, "neu": 0}
    eligible_rows = 0
    for row in rows:
        augmentation = row.get("augmentation")
        fallback_weight = 0.65 if augmentation == "target_pseudo" else (0.2 if augmentation in CSA_AUGMENT_CHANNELS else 1.0)
        weight = float(row.get("sample_weight", fallback_weight) or fallback_weight)
        if weight < min_weight or (exclude_augment and augmentation in CSA_AUGMENT_CHANNELS):
            continue
        if source_only and (augmentation == "target_pseudo" or augmentation in CSA_AUGMENT_CHANNELS):
            continue
        eligible_rows += 1
        for _aspect, _opinion, sentiment in parse_triplet_text_list(row.get("target", "")):
            if sentiment in counts:
                counts[sentiment] += 1
    return {"eligible_rows": eligible_rows, "triplets": sum(counts.values()), **counts}


def build_sentiment_class_weights(counts: dict[str, int]) -> list[float]:
    raw = [1.0 / math.sqrt(max(1, int(counts.get(name, 0)))) for name in ("pos", "neg", "neu")]
    mean_weight = sum(raw) / len(raw)
    return [value / mean_weight for value in raw]


def add_task_special_tokens(tokenizer, model, rows: list[dict]) -> None:
    text = "\n".join(f"{row.get('input', '')}\n{row.get('target', '')}" for row in rows[:2000])
    needed = [tok for tok in TASK_SPECIAL_TOKENS if tok in text]
    if not needed:
        return
    added = tokenizer.add_special_tokens({"additional_special_tokens": needed})
    if added:
        model.resize_token_embeddings(len(tokenizer))
        print(f"added special tokens: {needed}")
    for token in needed:
        init_word = TAG_INIT_WORDS.get(token)
        if not init_word:
            continue
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        init_ids = tokenizer.encode(init_word, add_special_tokens=False)
        if len(token_ids) != 1 or not init_ids:
            continue
        with torch.no_grad():
            model.shared.weight[token_ids[0]] = model.shared.weight[init_ids[0]].clone()
        print(f"initialized {token} from {init_word}")


_PHASE_A_GRAPH_TRAINING_AUTHORIZED = False


def enforce_graph_training_boundary(use_syntactic_graph_adapter: bool) -> None:
    """Keep direct graph training closed; only the Phase A API may authorize it."""
    if use_syntactic_graph_adapter and not _PHASE_A_GRAPH_TRAINING_AUTHORIZED:
        raise RuntimeError(
            "syntactic graph training is not approved; run "
            "m1_syntactic_graph_entry_audit.py for zero-update audit only, or use the approved "
            "m1_syntactic_rgat_pseudo_quick_ablation.py Phase A entry"
        )


def run_phase_a_training(argv: list[str]) -> None:
    """Run the existing trainer through a narrow in-process Phase A entry.

    The direct ``t5_absa_train.py`` command never sets the private authorization
    flag, so its graph-training hard stop remains active. The dedicated Phase A
    runner calls this API only after validating its frozen recipe and identities.
    """
    global _PHASE_A_GRAPH_TRAINING_AUTHORIZED
    previous_argv = sys.argv
    previous_authorization = _PHASE_A_GRAPH_TRAINING_AUTHORIZED
    sys.argv = ["t5_absa_train.py", *argv]
    _PHASE_A_GRAPH_TRAINING_AUTHORIZED = True
    try:
        main()
    finally:
        sys.argv = previous_argv
        _PHASE_A_GRAPH_TRAINING_AUTHORIZED = previous_authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=r"J:\nlp\models\t5-base-py")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=float, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=96)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", choices=["none", "auto"], default="none")
    parser.add_argument("--seed", type=int, default=1000)
    reproducibility_group = parser.add_mutually_exclusive_group()
    reproducibility_group.add_argument("--deterministic", action="store_true")
    reproducibility_group.add_argument("--legacy_stochastic", action="store_true")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_syntactic_graph_adapter", action="store_true")
    parser.add_argument("--syntactic_graph_cache_dir", default="")
    parser.add_argument("--syntactic_graph_parser_dir", default=r"J:\nlp\models\stanza_resources")
    parser.add_argument("--target_unlabeled_file", default="")
    parser.add_argument("--source_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_weight", type=float, default=0.5)
    parser.add_argument("--augment_weight", type=float, default=0.2)
    parser.add_argument("--force_domain_weights", action="store_true")
    parser.add_argument("--lambda_structure_loss", type=float, default=0.15)
    parser.add_argument("--lambda_consistency_loss", type=float, default=0.0)
    parser.add_argument("--lambda_pairing_loss", type=float, default=0.0)
    parser.add_argument("--pairing_temperature", type=float, default=0.1)
    parser.add_argument("--pairing_source_only", action="store_true")
    parser.add_argument("--lambda_domain_adv", type=float, default=0.0)
    parser.add_argument("--domain_adv_hidden_size", type=int, default=256)
    parser.add_argument("--domain_adv_grl_lambda", type=float, default=1.0)
    parser.add_argument("--domain_adv_exclude_augment", action="store_true")
    parser.add_argument("--lambda_sentiment_contrastive", type=float, default=0.0)
    parser.add_argument("--sentiment_contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--sentiment_contrastive_min_weight", type=float, default=0.65)
    parser.add_argument("--sentiment_contrastive_exclude_augment", action="store_true")
    parser.add_argument("--sentiment_contrastive_source_only", action="store_true")
    parser.add_argument("--sentiment_contrastive_class_balanced", action="store_true")
    parser.add_argument("--sentiment_prototype_initialize_from_context", action="store_true")
    parser.add_argument("--sentiment_prototype_init_batch_size", type=int, default=2)
    parser.add_argument("--max_pairing_triplets", type=int, default=4)
    parser.add_argument("--min_pairing_triplets", type=int, default=2)
    parser.add_argument("--min_pairing_sample_weight", type=float, default=0.65)
    parser.add_argument("--multi_triplet_loss_gain", type=float, default=0.1)
    parser.add_argument("--neutral_loss_gain", type=float, default=0.15)
    parser.add_argument("--max_effective_weight", type=float, default=1.0)
    parser.add_argument("--neutral_generation_loss_gain", type=float, default=0.0)
    parser.add_argument("--neutral_generation_max_effective_weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint_selection",
        choices=["last", "best", "aste_f1"],
        default="last",
        help=(
            "last saves the final training step; best selects the lowest dev eval_loss; "
            "aste_f1 selects the highest dev ASTE micro-F1 with multi-triplet F1 as a near-tie breaker."
        ),
    )
    args = parser.parse_args()

    enforce_graph_training_boundary(args.use_syntactic_graph_adapter)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    reproducibility_mode = "deterministic" if args.deterministic else "legacy"
    reproducibility_config = configure_reproducibility(args.seed, reproducibility_mode)
    print("reproducibility:", reproducibility_config)
    output_dir = Path(args.output_dir)
    checkpoint_dirs = list(output_dir.glob("checkpoint-*")) if output_dir.exists() else []
    resume_from_checkpoint = args.resume_from_checkpoint == "auto" and bool(checkpoint_dirs)

    if args.use_syntactic_graph_adapter:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file)
    train_graph_cache = None
    dev_graph_cache = None
    target_domain_rows = []
    graph_relation_vocab_size = 1
    if args.use_syntactic_graph_adapter:
        if not args.syntactic_graph_cache_dir:
            raise GraphCacheError("--syntactic_graph_cache_dir is required when graph adapter is enabled")
        if not args.target_unlabeled_file:
            raise GraphCacheError("--target_unlabeled_file is required when graph adapter is enabled")
        tokenizer_identity = build_tokenizer_identity(args.model_path, tokenizer)
        parser_identity = build_parser_identity(args.syntactic_graph_parser_dir)
        train_graph_cache = load_graph_cache_directory(
            args.syntactic_graph_cache_dir,
            "source_train",
            train_rows,
            tokenizer_identity=tokenizer_identity,
            parser_identity=parser_identity,
        )
        dev_graph_cache = load_graph_cache_directory(
            args.syntactic_graph_cache_dir,
            "source_dev",
            dev_rows,
            tokenizer_identity=tokenizer_identity,
            parser_identity=parser_identity,
        )
        target_rows = read_jsonl(args.target_unlabeled_file)
        target_graph_cache = load_graph_cache_directory(
            args.syntactic_graph_cache_dir,
            "target_unlabeled",
            target_rows,
            tokenizer_identity=tokenizer_identity,
            parser_identity=parser_identity,
        )
        if train_graph_cache.relation_vocab != dev_graph_cache.relation_vocab or train_graph_cache.relation_vocab != target_graph_cache.relation_vocab:
            raise GraphCacheError("graph relation vocabulary mismatch across required cache splits")
        graph_relation_vocab_size = train_graph_cache.relation_vocab_size
        if args.lambda_domain_adv > 0:
            target_domain_rows = build_target_unlabeled_domain_rows(
                target_rows,
                use_task_prefix=train_graph_cache.use_task_prefix,
            )
            train_rows = train_rows + target_domain_rows
            train_graph_cache = CompositeGraphCache(
                {
                    "source_train": train_graph_cache,
                    "target_unlabeled": target_graph_cache,
                }
            )
    elif args.target_unlabeled_file and args.lambda_domain_adv > 0:
        target_rows = read_jsonl(args.target_unlabeled_file)
        target_domain_rows = build_target_unlabeled_domain_rows(target_rows, use_task_prefix=False)
        train_rows = train_rows + target_domain_rows
    model = load_seq2seq_model(
        args.model_path,
        use_syntactic_graph_adapter=args.use_syntactic_graph_adapter,
        relation_vocab_size=graph_relation_vocab_size,
    )
    add_task_special_tokens(tokenizer, model, train_rows + dev_rows)
    if args.lambda_domain_adv > 0:
        hidden_size = int(getattr(model.config, "d_model", model.get_input_embeddings().embedding_dim))
        model.domain_adversarial_head = DomainAdversarialHead(
            hidden_size=hidden_size,
            classifier_hidden_size=args.domain_adv_hidden_size,
        )
    if args.lambda_sentiment_contrastive > 0:
        hidden_size = int(getattr(model.config, "d_model", model.get_input_embeddings().embedding_dim))
        model.sentiment_prototype_head = SentimentPrototypeHead(hidden_size=hidden_size)
        if args.sentiment_prototype_initialize_from_context and not resume_from_checkpoint:
            prototype_init_stats = initialize_sentiment_prototypes_from_context(
                model,
                tokenizer,
                train_rows,
                batch_size=args.sentiment_prototype_init_batch_size,
                max_source_length=args.max_source_length,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            init_path = output_dir / "sentiment_prototype_init.json"
            init_path.write_text(json.dumps(prototype_init_stats, ensure_ascii=False, indent=2), encoding="utf-8")
            print("sentiment prototype initialization:", {"path": str(init_path), **prototype_init_stats})
        elif args.sentiment_prototype_initialize_from_context:
            print("sentiment prototype initialization: skipped because training will resume from checkpoint")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    print(
        "sample weights:",
        summarize_sample_weights(
            train_rows,
            args.source_weight,
            args.pseudo_weight,
            args.augment_weight,
            force_domain_weights=args.force_domain_weights,
        ),
    )
    print(
        "joint loss:",
        {
            "lambda_structure_loss": args.lambda_structure_loss,
            "lambda_consistency_loss": args.lambda_consistency_loss,
            "lambda_pairing_loss": args.lambda_pairing_loss,
            "pairing_temperature": args.pairing_temperature,
            "pairing_source_only": args.pairing_source_only,
            "lambda_domain_adv": args.lambda_domain_adv,
            "domain_adv_hidden_size": args.domain_adv_hidden_size,
            "domain_adv_grl_lambda": args.domain_adv_grl_lambda,
            "domain_adv_exclude_augment": args.domain_adv_exclude_augment,
            "lambda_sentiment_contrastive": args.lambda_sentiment_contrastive,
            "sentiment_contrastive_temperature": args.sentiment_contrastive_temperature,
            "sentiment_contrastive_min_weight": args.sentiment_contrastive_min_weight,
            "sentiment_contrastive_exclude_augment": args.sentiment_contrastive_exclude_augment,
            "sentiment_contrastive_source_only": args.sentiment_contrastive_source_only,
            "sentiment_contrastive_class_balanced": args.sentiment_contrastive_class_balanced,
            "sentiment_prototype_initialize_from_context": args.sentiment_prototype_initialize_from_context,
            "sentiment_prototype_init_batch_size": args.sentiment_prototype_init_batch_size,
            "max_pairing_triplets": args.max_pairing_triplets,
            "min_pairing_triplets": args.min_pairing_triplets,
            "min_pairing_sample_weight": args.min_pairing_sample_weight,
            "multi_triplet_loss_gain": args.multi_triplet_loss_gain,
            "neutral_loss_gain": args.neutral_loss_gain,
            "max_effective_weight": args.max_effective_weight,
            "neutral_generation_loss_gain": args.neutral_generation_loss_gain,
            "neutral_generation_max_effective_weight": args.neutral_generation_max_effective_weight,
            "use_syntactic_graph_adapter": args.use_syntactic_graph_adapter,
            "syntactic_graph_cache_dir": args.syntactic_graph_cache_dir,
            "syntactic_graph_parser_dir": args.syntactic_graph_parser_dir,
            "graph_layers": 1 if args.use_syntactic_graph_adapter else 0,
            "graph_hidden_size": 256 if args.use_syntactic_graph_adapter else 0,
            "graph_attention_heads": 4 if args.use_syntactic_graph_adapter else 0,
            "graph_head_size": 64 if args.use_syntactic_graph_adapter else 0,
        },
    )
    if args.lambda_sentiment_contrastive > 0:
        sentiment_summary = summarize_sentiment_contrastive_rows(
            train_rows,
            args.sentiment_contrastive_min_weight,
            args.sentiment_contrastive_exclude_augment,
            args.sentiment_contrastive_source_only,
        )
        sentiment_class_weights = (
            build_sentiment_class_weights(sentiment_summary)
            if args.sentiment_contrastive_class_balanced else None
        )
        print("sentiment contrastive samples:", sentiment_summary)
        print("sentiment contrastive class weights:", sentiment_class_weights)
    else:
        sentiment_class_weights = None
    train_data = JsonlSeq2SeqDataset(
        train_rows,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        args.source_weight,
        args.pseudo_weight,
        args.augment_weight,
        multi_triplet_loss_gain=args.multi_triplet_loss_gain,
        neutral_loss_gain=args.neutral_loss_gain,
        max_effective_weight=args.max_effective_weight,
        neutral_generation_loss_gain=args.neutral_generation_loss_gain,
        neutral_generation_max_effective_weight=args.neutral_generation_max_effective_weight,
        force_domain_weights=args.force_domain_weights,
        max_pairing_triplets=args.max_pairing_triplets,
        min_pairing_triplets=args.min_pairing_triplets,
        min_pairing_sample_weight=args.min_pairing_sample_weight,
        pairing_source_only=args.pairing_source_only,
        domain_adv_exclude_augment=args.domain_adv_exclude_augment,
        sentiment_contrastive_min_weight=args.sentiment_contrastive_min_weight,
        sentiment_contrastive_exclude_augment=args.sentiment_contrastive_exclude_augment,
        sentiment_contrastive_source_only=args.sentiment_contrastive_source_only,
        graph_cache=train_graph_cache,
    )
    print("effective generation weights:", summarize_generation_weights(train_data))
    dev_data = JsonlSeq2SeqDataset(
        dev_rows,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        1.0,
        1.0,
        1.0,
        max_pairing_triplets=args.max_pairing_triplets,
        min_pairing_triplets=args.min_pairing_triplets,
        min_pairing_sample_weight=args.min_pairing_sample_weight,
        pairing_source_only=args.pairing_source_only,
        graph_cache=dev_graph_cache,
    )

    checkpoint_selection_config = build_checkpoint_selection_config(args.checkpoint_selection)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
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
        report_to=[],
        **reproducibility_training_args(args.seed, reproducibility_mode),
        **checkpoint_selection_config,
    )
    collator = DataCollatorForSeq2SeqWithPairing(DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model))
    trainer = WeightedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        data_collator=collator,
        lambda_structure_loss=args.lambda_structure_loss,
        lambda_consistency_loss=args.lambda_consistency_loss,
        lambda_pairing_loss=args.lambda_pairing_loss,
        pairing_temperature=args.pairing_temperature,
        lambda_domain_adv=args.lambda_domain_adv,
        domain_adv_grl_lambda=args.domain_adv_grl_lambda,
        lambda_sentiment_contrastive=args.lambda_sentiment_contrastive,
        sentiment_contrastive_temperature=args.sentiment_contrastive_temperature,
        sentiment_contrastive_class_weights=sentiment_class_weights,
        compute_metrics=(
            build_aste_compute_metrics(tokenizer)
            if args.checkpoint_selection == "aste_f1"
            else None
        ),
    )
    if resume_from_checkpoint:
        print(f"resuming from latest checkpoint in {output_dir}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    best_dir = output_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"saved {args.checkpoint_selection} model to {best_dir}")


if __name__ == "__main__":
    main()

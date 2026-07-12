from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from t5_absa_data import read_jsonl
from t5_aste_data import parse_triplet_text_list


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
        force_domain_weights: bool = False,
        max_pairing_triplets: int = 4,
        min_pairing_triplets: int = 2,
        min_pairing_sample_weight: float = 0.65,
        domain_adv_exclude_augment: bool = False,
        sentiment_contrastive_min_weight: float = 0.65,
        sentiment_contrastive_exclude_augment: bool = False,
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
        self.force_domain_weights = force_domain_weights
        self.max_pairing_triplets = max_pairing_triplets
        self.min_pairing_triplets = min_pairing_triplets
        self.min_pairing_sample_weight = min_pairing_sample_weight
        self.domain_adv_exclude_augment = domain_adv_exclude_augment
        self.sentiment_contrastive_min_weight = sentiment_contrastive_min_weight
        self.sentiment_contrastive_exclude_augment = sentiment_contrastive_exclude_augment

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
        model_inputs["labels"] = labels["input_ids"]
        domain_weight = self.sample_weight(row)
        model_inputs["sample_weight"] = domain_weight
        model_inputs["domain_weight"] = domain_weight
        model_inputs["domain_label"] = self.domain_label(row)
        model_inputs["structure_weight"] = self.structure_weight(row, domain_weight)
        model_inputs["consistency_group"] = self.consistency_group(row, idx)
        model_inputs.update(self.pairing_features(row))
        model_inputs.update(self.sentiment_contrastive_features(row, labels["input_ids"], domain_weight))
        return model_inputs

    def sample_weight(self, row: dict) -> float:
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

    def consistency_group(self, row: dict, idx: int) -> int:
        if row.get("base_id") is not None:
            return stable_group_id(row["base_id"])
        if row.get("id") is not None:
            return stable_group_id(row["id"])
        return int(idx)

    def pairing_features(self, row: dict) -> dict:
        target = row.get("target", "")
        triplets = parse_triplet_text_list(target)
        if len(triplets) < self.min_pairing_triplets:
            return {
                "pairing_aspect_spans": [],
                "pairing_opinion_spans": [],
                "pairing_mask": [],
            }
        if float(row.get("sample_weight", 0.0) or 0.0) < self.min_pairing_sample_weight and row.get("augmentation") != "target_pseudo":
            return {
                "pairing_aspect_spans": [],
                "pairing_opinion_spans": [],
                "pairing_mask": [],
            }
        aspect_spans: list[list[int]] = []
        opinion_spans: list[list[int]] = []
        mask: list[int] = []
        for aspect, opinion, _sentiment in triplets[: self.max_pairing_triplets]:
            aspect_span = find_token_subsequence_span(
                self.tokenizer.encode(target, add_special_tokens=False),
                self.tokenizer.encode(aspect, add_special_tokens=False),
            )
            opinion_span = find_token_subsequence_span(
                self.tokenizer.encode(target, add_special_tokens=False),
                self.tokenizer.encode(opinion, add_special_tokens=False),
            )
            if aspect_span is None or opinion_span is None:
                continue
            aspect_spans.append(list(aspect_span))
            opinion_spans.append(list(opinion_span))
            mask.append(1)
        return {
            "pairing_aspect_spans": aspect_spans,
            "pairing_opinion_spans": opinion_spans,
            "pairing_mask": mask,
        }

    def sentiment_contrastive_features(self, row: dict, target_ids: list[int], domain_weight: float) -> dict:
        augmentation = row.get("augmentation")
        if domain_weight < self.sentiment_contrastive_min_weight:
            return self.empty_sentiment_contrastive_features()
        if self.sentiment_contrastive_exclude_augment and augmentation in CSA_AUGMENT_CHANNELS:
            return self.empty_sentiment_contrastive_features()
        spans = []
        labels = []
        for _aspect, opinion, sentiment in parse_triplet_text_list(row.get("target", "")):
            sentiment_id = SENTIMENT_LABEL_IDS.get(sentiment)
            opinion_ids = self.tokenizer.encode(opinion, add_special_tokens=False)
            span = find_token_subsequence_span(target_ids, opinion_ids)
            if sentiment_id is None or span is None:
                continue
            spans.append(list(span))
            labels.append(sentiment_id)
        return {
            "sentiment_contrastive_spans": spans,
            "sentiment_contrastive_labels": labels,
            "sentiment_contrastive_mask": [1] * len(labels),
        }

    @staticmethod
    def empty_sentiment_contrastive_features() -> dict:
        return {
            "sentiment_contrastive_spans": [],
            "sentiment_contrastive_labels": [],
            "sentiment_contrastive_mask": [],
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


class DataCollatorForSeq2SeqWithPairing:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features: list[dict]) -> dict:
        pairing_aspect_spans = [feature.pop("pairing_aspect_spans", []) for feature in features]
        pairing_opinion_spans = [feature.pop("pairing_opinion_spans", []) for feature in features]
        pairing_masks = [feature.pop("pairing_mask", []) for feature in features]
        sentiment_spans = [feature.pop("sentiment_contrastive_spans", []) for feature in features]
        sentiment_labels = [feature.pop("sentiment_contrastive_labels", []) for feature in features]
        sentiment_masks = [feature.pop("sentiment_contrastive_mask", []) for feature in features]
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
        for row_idx, (spans, labels, mask) in enumerate(zip(sentiment_spans, sentiment_labels, sentiment_masks)):
            for item_idx, (span, label, active) in enumerate(zip(spans, labels, mask)):
                sentiment_span_tensor[row_idx, item_idx] = torch.tensor(span, dtype=torch.long)
                sentiment_label_tensor[row_idx, item_idx] = int(label)
                sentiment_mask_tensor[row_idx, item_idx] = int(active)
        batch["sentiment_contrastive_spans"] = sentiment_span_tensor
        batch["sentiment_contrastive_labels"] = sentiment_label_tensor
        batch["sentiment_contrastive_mask"] = sentiment_mask_tensor
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


def mean_pool_encoder_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        lambda_structure_loss: float = 0.0,
        lambda_consistency_loss: float = 0.0,
        lambda_pairing_loss: float = 0.0,
        lambda_domain_adv: float = 0.0,
        domain_adv_grl_lambda: float = 1.0,
        lambda_sentiment_contrastive: float = 0.0,
        sentiment_contrastive_temperature: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_structure_loss = lambda_structure_loss
        self.lambda_consistency_loss = lambda_consistency_loss
        self.lambda_pairing_loss = lambda_pairing_loss
        self.lambda_domain_adv = lambda_domain_adv
        self.domain_adv_grl_lambda = domain_adv_grl_lambda
        self.lambda_sentiment_contrastive = lambda_sentiment_contrastive
        self.sentiment_contrastive_temperature = sentiment_contrastive_temperature

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
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        needs_decoder_hidden = self.lambda_pairing_loss > 0 or self.lambda_sentiment_contrastive > 0
        outputs = model(**inputs, return_dict=True, output_hidden_states=needs_decoder_hidden)
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
        if consistency_group is not None and self.lambda_consistency_loss > 0:
            consistency_loss = grouped_representation_consistency_loss(
                outputs.encoder_last_hidden_state,
                attention_mask,
                consistency_group,
            )
            loss = loss + self.lambda_consistency_loss * consistency_loss
        if self.lambda_pairing_loss > 0 and pairing_aspect_spans is not None and pairing_opinion_spans is not None:
            decoder_hidden = outputs.decoder_hidden_states[-1] if outputs.decoder_hidden_states else None
            if decoder_hidden is not None:
                pair_loss = pairing_contrastive_loss(
                    decoder_hidden,
                    pairing_aspect_spans,
                    pairing_opinion_spans,
                    pairing_mask,
                )
                loss = loss + self.lambda_pairing_loss * pair_loss
        if (
            self.lambda_sentiment_contrastive > 0
            and sentiment_contrastive_spans is not None
            and hasattr(model, "sentiment_prototype_head")
        ):
            decoder_hidden = outputs.decoder_hidden_states[-1] if outputs.decoder_hidden_states else None
            if decoder_hidden is not None:
                sentiment_loss = sentiment_prototype_contrastive_loss(
                    decoder_hidden,
                    sentiment_contrastive_spans,
                    sentiment_contrastive_labels,
                    sentiment_contrastive_mask,
                    model.sentiment_prototype_head,
                    temperature=self.sentiment_contrastive_temperature,
                )
                loss = loss + self.lambda_sentiment_contrastive * sentiment_loss
        if (
            model.training
            and self.lambda_domain_adv > 0
            and domain_label is not None
            and hasattr(model, "domain_adversarial_head")
            and outputs.encoder_last_hidden_state is not None
        ):
            pooled_hidden = mean_pool_encoder_hidden(outputs.encoder_last_hidden_state, attention_mask)
            reversed_hidden = gradient_reverse(pooled_hidden, self.domain_adv_grl_lambda)
            domain_logits = model.domain_adversarial_head(reversed_hidden)
            domain_targets = domain_label.to(domain_logits.device, dtype=torch.long).view(-1)
            domain_valid_mask = domain_targets.ne(-100)
            if domain_valid_mask.any():
                domain_adv_loss = F.cross_entropy(domain_logits[domain_valid_mask], domain_targets[domain_valid_mask])
                loss = loss + self.lambda_domain_adv * domain_adv_loss
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


def sentiment_prototype_contrastive_loss(
    decoder_hidden: torch.Tensor,
    opinion_spans: torch.Tensor,
    sentiment_labels: torch.Tensor,
    sentiment_mask: torch.Tensor,
    prototype_head: SentimentPrototypeHead,
    temperature: float = 0.1,
) -> torch.Tensor:
    if opinion_spans is None or opinion_spans.numel() == 0:
        return decoder_hidden.new_tensor(0.0)
    opinion_spans = opinion_spans.to(decoder_hidden.device)
    sentiment_labels = sentiment_labels.to(decoder_hidden.device, dtype=torch.long)
    valid_mask = sentiment_mask.to(decoder_hidden.device).bool() & sentiment_labels.ne(-100)
    if not valid_mask.any():
        return decoder_hidden.new_tensor(0.0)
    opinion_repr = F.normalize(span_mean(decoder_hidden, opinion_spans), p=2, dim=-1)
    logits = opinion_repr[valid_mask] @ prototype_head.normalized_prototypes().transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    return F.cross_entropy(logits, sentiment_labels[valid_mask])


def summarize_sample_weights(
    rows: list[dict],
    source_weight: float,
    pseudo_weight: float,
    augment_weight: float,
    force_domain_weights: bool = False,
) -> dict:
    counts = {"source_gold": 0, "target_pseudo": 0, "c3da_augment": 0}
    weights = []
    for row in rows:
        augmentation = row.get("augmentation")
        if augmentation == "target_pseudo":
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
        ("source_gold", lambda row: row.get("augmentation") not in {"target_pseudo", *CSA_AUGMENT_CHANNELS}, source_weight),
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


def summarize_sentiment_contrastive_rows(
    rows: list[dict],
    min_weight: float,
    exclude_augment: bool,
) -> dict:
    counts = {"pos": 0, "neg": 0, "neu": 0}
    eligible_rows = 0
    for row in rows:
        augmentation = row.get("augmentation")
        fallback_weight = 0.65 if augmentation == "target_pseudo" else (0.2 if augmentation in CSA_AUGMENT_CHANNELS else 1.0)
        weight = float(row.get("sample_weight", fallback_weight) or fallback_weight)
        if weight < min_weight or (exclude_augment and augmentation in CSA_AUGMENT_CHANNELS):
            continue
        eligible_rows += 1
        for _aspect, _opinion, sentiment in parse_triplet_text_list(row.get("target", "")):
            if sentiment in counts:
                counts[sentiment] += 1
    return {"eligible_rows": eligible_rows, "triplets": sum(counts.values()), **counts}


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
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--source_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_weight", type=float, default=0.5)
    parser.add_argument("--augment_weight", type=float, default=0.2)
    parser.add_argument("--force_domain_weights", action="store_true")
    parser.add_argument("--lambda_structure_loss", type=float, default=0.15)
    parser.add_argument("--lambda_consistency_loss", type=float, default=0.0)
    parser.add_argument("--lambda_pairing_loss", type=float, default=0.0)
    parser.add_argument("--lambda_domain_adv", type=float, default=0.0)
    parser.add_argument("--domain_adv_hidden_size", type=int, default=256)
    parser.add_argument("--domain_adv_grl_lambda", type=float, default=1.0)
    parser.add_argument("--domain_adv_exclude_augment", action="store_true")
    parser.add_argument("--lambda_sentiment_contrastive", type=float, default=0.0)
    parser.add_argument("--sentiment_contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--sentiment_contrastive_min_weight", type=float, default=0.65)
    parser.add_argument("--sentiment_contrastive_exclude_augment", action="store_true")
    parser.add_argument("--max_pairing_triplets", type=int, default=4)
    parser.add_argument("--min_pairing_triplets", type=int, default=2)
    parser.add_argument("--min_pairing_sample_weight", type=float, default=0.65)
    parser.add_argument("--multi_triplet_loss_gain", type=float, default=0.1)
    parser.add_argument("--neutral_loss_gain", type=float, default=0.15)
    parser.add_argument("--max_effective_weight", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint_selection",
        choices=["last", "best"],
        default="last",
        help="last saves the model after the final training step; best saves the checkpoint with the lowest dev eval_loss.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file)
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
            "lambda_domain_adv": args.lambda_domain_adv,
            "domain_adv_hidden_size": args.domain_adv_hidden_size,
            "domain_adv_grl_lambda": args.domain_adv_grl_lambda,
            "domain_adv_exclude_augment": args.domain_adv_exclude_augment,
            "lambda_sentiment_contrastive": args.lambda_sentiment_contrastive,
            "sentiment_contrastive_temperature": args.sentiment_contrastive_temperature,
            "sentiment_contrastive_min_weight": args.sentiment_contrastive_min_weight,
            "sentiment_contrastive_exclude_augment": args.sentiment_contrastive_exclude_augment,
            "max_pairing_triplets": args.max_pairing_triplets,
            "min_pairing_triplets": args.min_pairing_triplets,
            "min_pairing_sample_weight": args.min_pairing_sample_weight,
            "multi_triplet_loss_gain": args.multi_triplet_loss_gain,
            "neutral_loss_gain": args.neutral_loss_gain,
            "max_effective_weight": args.max_effective_weight,
        },
    )
    if args.lambda_sentiment_contrastive > 0:
        print(
            "sentiment contrastive samples:",
            summarize_sentiment_contrastive_rows(
                train_rows,
                args.sentiment_contrastive_min_weight,
                args.sentiment_contrastive_exclude_augment,
            ),
        )
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
        force_domain_weights=args.force_domain_weights,
        max_pairing_triplets=args.max_pairing_triplets,
        min_pairing_triplets=args.min_pairing_triplets,
        min_pairing_sample_weight=args.min_pairing_sample_weight,
        domain_adv_exclude_augment=args.domain_adv_exclude_augment,
        sentiment_contrastive_min_weight=args.sentiment_contrastive_min_weight,
        sentiment_contrastive_exclude_augment=args.sentiment_contrastive_exclude_augment,
    )
    dev_data = JsonlSeq2SeqDataset(
        dev_rows,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        1.0,
        1.0,
        1.0,
    )

    output_dir = Path(args.output_dir)
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
        load_best_model_at_end=args.checkpoint_selection == "best",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        report_to=[],
        seed=args.seed,
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
        lambda_domain_adv=args.lambda_domain_adv,
        domain_adv_grl_lambda=args.domain_adv_grl_lambda,
        lambda_sentiment_contrastive=args.lambda_sentiment_contrastive,
        sentiment_contrastive_temperature=args.sentiment_contrastive_temperature,
    )
    checkpoint_dirs = list(output_dir.glob("checkpoint-*")) if output_dir.exists() else []
    resume_from_checkpoint = args.resume_from_checkpoint == "auto" and bool(checkpoint_dirs)
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

import torch
from transformers import T5Config

from syntactic_graph import CompositeGraphCache
from syntactic_graph_adapter import SyntacticGraphT5ForConditionalGeneration, graph_model_config
from t5_absa_train import (
    DataCollatorForSeq2SeqWithPairing,
    DomainAdversarialHead,
    JsonlSeq2SeqDataset,
    build_target_unlabeled_domain_rows,
    compute_domain_adversarial_loss,
)
from test_syntactic_graph_adapter import graph_inputs
from test_syntactic_graph_training import BaseCollator, TinyTokenizer


def test_target_unlabeled_rows_are_domain_only_and_have_no_aste_weight():
    rows = build_target_unlabeled_domain_rows(
        [{"id": "target-1", "text": "The staff is friendly."}],
        use_task_prefix=False,
    )

    assert rows == [
        {
            "id": "target-1",
            "text": "The staff is friendly.",
            "input": "The staff is friendly.",
            "target": "",
            "augmentation": "target_unlabeled",
            "sample_weight": 0.0,
            "domain_weight": 0.0,
            "structure_weight": 0.0,
            "domain_label": 1,
        }
    ]

    dataset = JsonlSeq2SeqDataset(
        rows,
        TinyTokenizer(),
        max_source_length=32,
        max_target_length=8,
        source_weight=1.0,
        pseudo_weight=0.5,
        augment_weight=0.2,
    )
    item = dataset[0]
    assert item["domain_label"] == 1
    assert item["domain_weight"] == 0.0
    assert item["structure_weight"] == 0.0
    assert all(label == -100 for label in item["labels"])


def test_composite_graph_cache_routes_source_and_target_domain_rows():
    class NamedCache:
        relation_vocab = ["self"]
        relation_vocab_size = 1

        def __init__(self, marker):
            self.marker = marker

        def get(self, row):
            return {"marker": self.marker, "row_id": row["id"]}

    cache = CompositeGraphCache(
        {
            "source_train": NamedCache("source"),
            "target_unlabeled": NamedCache("target"),
        }
    )
    assert cache.get({"id": "s1", "augmentation": "source_gold"})["marker"] == "source"
    assert cache.get({"id": "t1", "augmentation": "target_unlabeled"})["marker"] == "target"


def test_target_unlabeled_domain_rows_use_real_graph_collator_fields():
    rows = build_target_unlabeled_domain_rows(
        [{"id": "target-1", "text": "The staff is friendly."}],
        use_task_prefix=False,
    )
    dataset = JsonlSeq2SeqDataset(
        rows,
        TinyTokenizer(),
        max_source_length=32,
        max_target_length=8,
        source_weight=1.0,
        pseudo_weight=0.5,
        augment_weight=0.2,
        graph_cache=type(
            "TargetCache",
            (),
            {"get": lambda self, row: {"word_to_subword": [[0]], "word_mask": [1], "edge_src": [0], "edge_dst": [0], "relation_id": [0], "dependency_relation_id": [0], "pos_pair_id": [0], "edge_mask": [1]}},
        )(),
    )
    batch = DataCollatorForSeq2SeqWithPairing(BaseCollator())([dataset[0]])
    assert batch["graph_word_to_subword"].shape == (1, 1, 1)
    assert batch["graph_edge_mask"].tolist() == [[True]]


def test_dann_loss_from_post_graph_encoder_reaches_zero_initialized_output_projection():
    torch.manual_seed(1000)
    config = T5Config(
        vocab_size=32,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    graph_model_config(config, 8)
    model = SyntacticGraphT5ForConditionalGeneration(config).eval()
    model.domain_adversarial_head = DomainAdversarialHead(hidden_size=8, classifier_hidden_size=8)
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]])
    attention_mask = torch.ones_like(input_ids)
    fields = {"graph_" + key: value for key, value in graph_inputs().items()}
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.tensor([[1, 2], [1, 2]]),
        **fields,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    loss = compute_domain_adversarial_loss(
        outputs.encoder_last_hidden_state,
        attention_mask,
        labels,
        model.domain_adversarial_head,
        grl_lambda=1.0,
    )
    loss.backward()

    projection_grad = model.syntactic_graph_adapter.output_projection.weight.grad
    assert torch.isfinite(loss)
    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert projection_grad.abs().sum() > 0

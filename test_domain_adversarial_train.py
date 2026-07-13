import torch

from t5_absa_train import (
    DomainAdversarialHead,
    JsonlSeq2SeqDataset,
    SentimentPrototypeHead,
    build_sentiment_class_weights,
    gradient_reverse,
    mean_pool_encoder_hidden,
    sentiment_prototype_contrastive_loss,
)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text=None, text_target=None, max_length=None, truncation=True):
        value = text_target if text_target is not None else text
        return {"input_ids": [ord(ch) % 97 + 1 for ch in str(value)]}

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 97 + 1 for ch in str(text)]


def test_dataset_assigns_domain_labels():
    rows = [
        {"input": "source", "target": "<pos> food <opinion> good"},
        {"input": "pseudo", "target": "<pos> keyboard <opinion> good", "augmentation": "target_pseudo"},
        {"input": "augment", "target": "<pos> screen <opinion> bright", "augmentation": "masked_aspect_channel"},
    ]
    dataset = JsonlSeq2SeqDataset(rows, TinyTokenizer(), 16, 16, 1.0, 0.5, 0.2)

    assert [dataset[i]["domain_label"] for i in range(3)] == [0, 1, 1]


def test_gradient_reverse_flips_gradient_sign():
    value = torch.tensor([2.0], requires_grad=True)
    gradient_reverse(value, 0.5).sum().backward()

    assert torch.allclose(value.grad, torch.tensor([-0.5]))


def test_domain_head_pooling_produces_domain_logits():
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[5.0, 7.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = mean_pool_encoder_hidden(hidden, mask)
    head = DomainAdversarialHead(hidden_size=2, classifier_hidden_size=4)

    logits = head(pooled)

    assert pooled.tolist() == [[2.0, 3.0], [5.0, 7.0]]
    assert list(logits.shape) == [2, 2]


def test_dataset_builds_clean_sentiment_contrastive_features():
    rows = [
        {"input": "source", "target": "<neu> food <opinion> average"},
        {"input": "pseudo", "target": "<pos> screen <opinion> bright", "augmentation": "target_pseudo", "sample_weight": 0.65},
        {"input": "augment", "target": "<neg> fan <opinion> loud", "augmentation": "masked_opinion_sentiment_channel", "sample_weight": 0.2},
    ]
    dataset = JsonlSeq2SeqDataset(
        rows, TinyTokenizer(), 64, 64, 1.0, 0.5, 0.2,
        sentiment_contrastive_min_weight=0.65,
        sentiment_contrastive_exclude_augment=True,
        sentiment_contrastive_source_only=True,
    )

    assert dataset[0]["sentiment_contrastive_labels"] == [2]
    assert dataset[1]["sentiment_contrastive_labels"] == []
    assert dataset[2]["sentiment_contrastive_labels"] == []
    assert dataset[0]["sentiment_contrastive_weights"] == [1.0]


def test_sentiment_prototype_loss_works_with_one_triplet():
    hidden = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    spans = torch.tensor([[[0, 2]]])
    labels = torch.tensor([[0]])
    mask = torch.tensor([[1]])
    sample_weights = torch.tensor([[0.7]])
    head = SentimentPrototypeHead(hidden_size=2)
    loss = sentiment_prototype_contrastive_loss(
        hidden, spans, labels, mask, head, temperature=0.1,
        sample_weights=sample_weights,
        class_weights=torch.tensor([1.0, 2.0, 3.0]),
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()
    assert hidden.grad is not None


def test_sentiment_class_weights_upweight_rare_neutral_class():
    weights = build_sentiment_class_weights({"pos": 1200, "neg": 500, "neu": 50})

    assert len(weights) == 3
    assert weights[2] > weights[1] > weights[0]


if __name__ == "__main__":
    test_dataset_assigns_domain_labels()
    test_gradient_reverse_flips_gradient_sign()
    test_domain_head_pooling_produces_domain_logits()
    test_dataset_builds_clean_sentiment_contrastive_features()
    test_sentiment_prototype_loss_works_with_one_triplet()
    test_sentiment_class_weights_upweight_rare_neutral_class()
    print("domain adversarial training tests passed")

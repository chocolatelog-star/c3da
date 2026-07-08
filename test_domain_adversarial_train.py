import torch

from t5_absa_train import (
    DomainAdversarialHead,
    JsonlSeq2SeqDataset,
    gradient_reverse,
    mean_pool_encoder_hidden,
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


if __name__ == "__main__":
    test_dataset_assigns_domain_labels()
    test_gradient_reverse_flips_gradient_sign()
    test_domain_head_pooling_produces_domain_logits()
    print("domain adversarial training tests passed")

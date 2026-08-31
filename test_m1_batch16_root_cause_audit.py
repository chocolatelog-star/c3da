import torch
from pathlib import Path

from m1_batch16_root_cause_audit import audit_loss_reductions, generation_per_example_loss, load_source_audit_rows


def test_generation_loss_is_per_example_token_normalized():
    logits = torch.zeros((2, 4, 3))
    labels = torch.tensor([[0, 1, -100, -100], [0, 1, 2, 2]])
    losses = generation_per_example_loss(logits, labels)
    assert losses.shape == (2,)
    assert torch.allclose(losses, torch.full((2,), torch.log(torch.tensor(3.0))))


def test_audit_covers_all_effective_batch_16_decompositions_without_target_data():
    result = audit_loss_reductions()
    assert {(item["micro_batch"], item["accumulation"]) for item in result["splits"]} == {(1, 16), (4, 4), (8, 2), (16, 1)}
    assert all(item["effective_batch_size"] == 16 for item in result["splits"])
    assert result["conclusion"]["target_test_gold"] is False


def test_source_audit_sample_is_exactly_16_rows_and_preserves_ids(tmp_path: Path):
    source = tmp_path / "train.txt"
    source.write_text("\n".join(f"row {i}####[]" for i in range(20)), encoding="utf-8")
    rows = load_source_audit_rows(source, start=2)
    assert len(rows) == 16
    assert [row["id"] for row in rows] == list(range(2, 18))

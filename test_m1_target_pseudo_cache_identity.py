import hashlib
import sys
from types import ModuleType
from unittest.mock import patch

import torch

import t5_aste_pipeline
from syntactic_graph import GraphCache, GraphCacheError, normalized_text_sha256


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    unk_token = "<unk>"

    def __call__(self, texts, **_kwargs):
        return _FakeBatch(
            input_ids=torch.ones((len(texts), 1), dtype=torch.long),
            attention_mask=torch.ones((len(texts), 1), dtype=torch.long),
        )

    def decode(self, _ids, skip_special_tokens=False):
        del skip_special_tokens
        return "decoded"


class _FakeModel:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, **_kwargs):
        return torch.ones((1, 1), dtype=torch.long)


def _make_cache(rows):
    records = []
    for row in rows:
        text = str(row["text"])
        records.append(
            {
                "row_id": str(row["id"]),
                "normalized_text_sha256": normalized_text_sha256(text),
                "input_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "word_to_subword": [[0]],
                "edges": [],
            }
        )
    return GraphCache(records, ["self_loop"], "target_unlabeled", use_task_prefix=False)


def _run_fake_generation(tmp_path, full_rows, graph_rows, inputs, *, pass_identity_rows=True):
    tokenizer = _FakeTokenizer()
    fake_transformers = ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return tokenizer

    class _AutoModelForSeq2SeqLM:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _FakeModel()

    fake_transformers.AutoTokenizer = _AutoTokenizer
    fake_transformers.AutoModelForSeq2SeqLM = _AutoModelForSeq2SeqLM
    loaded_expected_rows = []
    accessed_rows = []
    cache = _make_cache(full_rows)

    def load_cache(_cache_dir, _split, expected_rows, **_kwargs):
        loaded_expected_rows.append(expected_rows)
        return cache

    graph_kwargs = {"graph_word_mask": torch.ones((len(graph_rows), 1), dtype=torch.bool)}
    with patch.dict(sys.modules, {"transformers": fake_transformers}), patch.object(
        t5_aste_pipeline,
        "build_tokenizer_identity",
        return_value={"name": "fake"},
    ), patch.object(
        t5_aste_pipeline,
        "build_parser_identity",
        return_value={"name": "fake-parser"},
    ), patch.object(
        t5_aste_pipeline,
        "load_graph_cache_directory",
        side_effect=load_cache,
    ), patch.object(
        t5_aste_pipeline,
        "load_seq2seq_model",
        return_value=_FakeModel(),
    ), patch.object(
        t5_aste_pipeline,
        "_collate_graph_cache_rows",
        side_effect=lambda rows: (accessed_rows.extend(graph_rows[: len(rows)]) or graph_kwargs),
    ), patch("torch.cuda.is_available", return_value=False):
        identity_rows = full_rows if pass_identity_rows else None
        outputs = t5_aste_pipeline.generate_texts(
            model_path="fake-model",
            inputs=inputs,
            batch_size=1,
            max_new_tokens=1,
            num_beams=1,
            cuda="0",
            use_syntactic_graph_adapter=True,
            graph_cache_dir=tmp_path,
            graph_rows=graph_rows,
            graph_cache_identity_rows=identity_rows,
            graph_parser_dir="fake-parser",
            graph_split="target_unlabeled",
        )
    return outputs, loaded_expected_rows, accessed_rows


def test_complete_identity_rows_allow_a_single_target_inference_subset(tmp_path):
    full_rows = [
        {"id": "t1", "text": "first"},
        {"id": "t2", "text": "second"},
    ]
    subset = full_rows[:1]

    outputs, loaded_rows, accessed_rows = _run_fake_generation(
        tmp_path, full_rows, subset, ["first"]
    )

    assert outputs == ["decoded"]
    assert loaded_rows == [full_rows]
    assert accessed_rows == subset


def test_target_subset_outside_complete_identity_rows_is_rejected(tmp_path):
    full_rows = [{"id": "t1", "text": "first"}]
    outside = [{"id": "t9", "text": "outside"}]

    try:
        _run_fake_generation(tmp_path, full_rows, outside, ["outside"])
    except GraphCacheError as exc:
        assert "missing graph cache row" in str(exc)
    else:
        raise AssertionError("a subset outside the full cache identity must be rejected")


def test_target_subset_with_modified_text_is_rejected(tmp_path):
    full_rows = [{"id": "t1", "text": "first"}]
    modified = [{"id": "t1", "text": "changed"}]

    try:
        _run_fake_generation(tmp_path, full_rows, modified, ["changed"])
    except GraphCacheError as exc:
        assert "text hash mismatch" in str(exc)
    else:
        raise AssertionError("a modified subset row must be rejected")


def test_target_subset_with_modified_id_is_rejected(tmp_path):
    full_rows = [{"id": "t1", "text": "first"}]
    modified = [{"id": "changed-id", "text": "first"}]

    try:
        _run_fake_generation(tmp_path, full_rows, modified, ["first"])
    except GraphCacheError as exc:
        assert "missing graph cache row" in str(exc)
    else:
        raise AssertionError("a modified subset id must be rejected")


def test_graph_rows_and_inputs_length_mismatch_is_rejected(tmp_path):
    full_rows = [{"id": "t1", "text": "first"}]

    try:
        _run_fake_generation(tmp_path, full_rows, full_rows, ["first", "extra"])
    except GraphCacheError as exc:
        assert "identical lengths" in str(exc)
    else:
        raise AssertionError("graph rows and inputs must have identical lengths")


def test_omitting_identity_rows_preserves_the_original_full_generation_contract(tmp_path):
    full_rows = [{"id": "t1", "text": "first"}]
    outputs, loaded_rows, accessed_rows = _run_fake_generation(
        tmp_path,
        full_rows,
        full_rows,
        ["first"],
        pass_identity_rows=False,
    )

    assert outputs == ["decoded"]
    assert loaded_rows == [full_rows]
    assert accessed_rows == full_rows


def test_audit_source_wires_full_target_identity_rows_to_generate_texts():
    import inspect

    from m1_syntactic_graph_entry_audit import _run_model_audit

    source = inspect.getsource(_run_model_audit)
    assert "graph_cache_identity_rows=target_rows" in source
    assert "graph_rows=target_sample" in source
    assert "inputs=[row[\"text\"] for row in target_sample]" in source

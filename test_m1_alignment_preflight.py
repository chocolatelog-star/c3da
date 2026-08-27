from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from m1_alignment_preflight import (
    ALIGNMENT_POLICY_VERSION,
    AlignmentPreflightError,
    AlignmentPreflightInterrupted,
    build_preflight_summary,
    load_preflight_rows,
    run_preflight,
    scan_alignment_row,
    scan_split_rows,
)


class FakeWord:
    def __init__(self, text, start, end, sentence_index=0):
        self.text = text
        self.start_char = start
        self.end_char = end
        self.upos = "X"
        self.head = 0
        self.deprel = "root"
        self.sentence_index = sentence_index


class FakeSentence:
    def __init__(self, words):
        self.words = words


class FakeDoc:
    def __init__(self, sentences):
        self.sentences = sentences


class FakeParser:
    def __init__(self, doc=None, error=None):
        self.doc = doc
        self.error = error

    def __call__(self, _text):
        if self.error:
            raise self.error
        return self.doc


class FakeTokenizer:
    is_fast = True
    vocab_size = 100

    def __init__(self, offsets, tokens=None, full_length=None, full_offsets=None):
        self.offsets = offsets
        self.tokens = tokens or [f"tok-{i}" for i in range(len(offsets))]
        self.full_length = full_length or len(offsets)
        self.full_offsets = full_offsets

    def __call__(self, _text, truncation=True, **_kwargs):
        offsets = self.offsets if truncation else self.full_offsets or [(0, 1)] * self.full_length
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

    def convert_ids_to_tokens(self, input_ids):
        return [self.tokens[index] for index in input_ids]


def _row(text="iPhone"):
    return {"id": "r1", "text": text, "input": text}


def _words(*items):
    return [FakeWord(*item) for item in items]


def test_preflight_records_legal_iphone_contiguous_sharing():
    parser = FakeParser(FakeDoc([FakeSentence(_words(("i", 0, 1), ("Phone", 1, 6)))]))
    tokenizer = FakeTokenizer([(0, 6)], tokens=["▁iPhone"])

    result = scan_alignment_row("source_train", _row(), tokenizer, parser, max_source_length=128)

    assert result["failed"] is False
    assert result["stats"]["legal_contiguous_shared_subword_count"] == 1
    assert result["stats"]["character_coverage_incomplete_count"] == 0
    assert "legal_contiguous_shared_subword" in result["issue_types"]
    assert "subword_maps_to_multiple_parser_words" in result["issue_types"]


def test_preflight_rejects_cross_space_shared_subword():
    parser = FakeParser(FakeDoc([FakeSentence(_words(("a", 0, 1), ("b", 2, 3)))]))
    tokenizer = FakeTokenizer([(0, 3)], tokens=["▁a▁b"])

    result = scan_alignment_row("source_train", _row("a b"), tokenizer, parser, max_source_length=128)

    assert "cross_space_shared_subword" in result["issue_types"]
    assert "legal_contiguous_shared_subword" not in result["issue_types"]


def test_preflight_rejects_cross_sentence_shared_subword():
    parser = FakeParser(
        FakeDoc(
            [
                FakeSentence(_words(("a", 0, 1, 0))),
                FakeSentence(_words(("b", 1, 2, 1))),
            ]
        )
    )
    tokenizer = FakeTokenizer([(0, 2)], tokens=["▁a▁b"])

    result = scan_alignment_row("source_train", _row("a b"), tokenizer, parser, max_source_length=128)

    assert "cross_sentence_shared_subword" in result["issue_types"]


def test_preflight_records_truncation_uncovered_word():
    parser = FakeParser(FakeDoc([FakeSentence(_words(("a", 0, 1), ("b", 2, 3)))]))
    tokenizer = FakeTokenizer(
        [(0, 1)],
        tokens=["a"],
        full_length=4,
        full_offsets=[(0, 1), (2, 3), (0, 0), (0, 0)],
    )

    result = scan_alignment_row("source_train", _row("a b"), tokenizer, parser, max_source_length=2)

    assert result["stats"]["truncated_rows"] == 1
    assert result["stats"]["truncation_uncovered_word_count"] == 1
    assert "truncation_uncovered_word" in result["issue_types"]


def test_preflight_continues_after_a_failed_row():
    rows = [_row("broken"), {"id": "r2", "text": "ok", "input": "ok"}]
    parsers = [
        FakeParser(error=RuntimeError("synthetic parser failure")),
        FakeParser(FakeDoc([FakeSentence(_words(("ok", 0, 2)))])),
    ]
    tokenizers = [FakeTokenizer([(0, 1)]), FakeTokenizer([(0, 2)])]

    result = scan_split_rows("source_train", rows, tokenizers, parsers, max_source_length=128)

    assert result["completed_rows"] == 2
    assert result["failed_rows"] == 1
    assert result["successful_rows"] == 1
    assert result["row_results"][1]["failed"] is False


def test_summary_is_machine_readable_and_has_target_test_isolation():
    split_reports = {
        name: {
            "sentence_count": 1,
            "word_count": 2,
            "subword_count": 2,
            "successful_rows": 1,
            "failed_rows": 0,
            "stats": {
                "unaligned_word_count": 0,
                "character_coverage_incomplete_count": 0,
                "non_contiguous_shared_count": 0,
                "cross_sentence_shared_count": 0,
                "out_of_bounds_count": 0,
                "legal_contiguous_shared_subword_count": 1,
                "legal_contiguous_shared_row_count": 1,
                "truncated_rows": 0,
                "truncation_uncovered_word_count": 0,
            },
            "distributions": {
                "word_subword_count": {"1": 2},
                "subword_parser_word_count": {"1": 2},
            },
        }
        for name in ("source_train", "source_dev", "target_unlabeled")
    }
    identity = {
        "git_commit": "abc",
        "input_files": {},
        "parser": {"sha256": {}},
        "tokenizer": {"files_sha256": {}},
        "code": {"files_sha256": {}},
        "alignment_policy_version": ALIGNMENT_POLICY_VERSION,
        "max_source_length": 128,
    }

    summary = build_preflight_summary(split_reports, identity, max_source_length=128)

    assert summary["status"] in {"PASS", "BLOCKED"}
    assert summary["target_test_access"] is False
    assert summary["identity"]["alignment_policy_version"] == ALIGNMENT_POLICY_VERSION
    assert "word_subword_count" in summary["splits"]["source_train"]["distributions"]


def test_preflight_loads_only_source_train_dev_and_target_train():
    calls = []

    def fake_load_split(dataset, split):
        calls.append((dataset, split))
        return [{"id": f"{dataset}-{split}", "text": "x", "input": "x"}]

    with patch("m1_alignment_preflight.load_split", side_effect=fake_load_split):
        rows = load_preflight_rows("laptop14", "rest15")

    assert set(rows) == {"source_train", "source_dev", "target_unlabeled"}
    assert calls == [("laptop14", "train"), ("laptop14", "dev"), ("rest15", "train")]
    assert all(split != "test" for _, split in calls)


def test_run_preflight_resumes_rows_and_rejects_identity_mismatch(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"id":"r1","text":"iPhone"}\n', encoding="utf-8")
    rows = {
        split: [{"id": f"{split}-r1", "text": "iPhone", "input": "iPhone", "source_path": str(input_path)}]
        for split in ("source_train", "source_dev", "target_unlabeled")
    }
    parser = FakeParser(FakeDoc([FakeSentence(_words(("i", 0, 1), ("Phone", 1, 6)))]))
    tokenizer = FakeTokenizer([(0, 6)], tokens=["▁iPhone"])
    args = SimpleNamespace(
        output_dir=str(tmp_path / "preflight"),
        source_dataset="laptop14",
        target_dataset="rest15",
        max_source_length=128,
        max_subwords_per_word=8,
    )
    identities = {"class": "fake", "files_sha256": {}}

    try:
        run_preflight(
            args,
            rows_by_split=rows,
            tokenizer=tokenizer,
            parser=parser,
            tokenizer_identity=identities,
            parser_identity={"name": "fake"},
            repo_dir=Path(__file__).resolve().parent,
            stop_after_rows=1,
        )
    except AlignmentPreflightInterrupted:
        pass
    else:
        raise AssertionError("stop_after_rows must interrupt after a durable row")

    resumed = run_preflight(
        args,
        rows_by_split=rows,
        tokenizer=tokenizer,
        parser=parser,
        tokenizer_identity=identities,
        parser_identity={"name": "fake"},
        repo_dir=Path(__file__).resolve().parent,
    )
    assert resumed["status"] == "PASS"
    assert all(report["completed_rows"] == 1 for report in resumed["splits"].values())
    assert sorted(path.name for path in (tmp_path / "preflight").iterdir()) == [
        "alignment_preflight_CN.md",
        "alignment_preflight_summary.json",
        "alignment_suspicious_rows.jsonl",
    ]

    mismatch_args = SimpleNamespace(**{**vars(args), "max_source_length": 64})
    try:
        run_preflight(
            mismatch_args,
            rows_by_split=rows,
            tokenizer=tokenizer,
            parser=parser,
            tokenizer_identity=identities,
            parser_identity={"name": "fake"},
            repo_dir=Path(__file__).resolve().parent,
        )
    except AlignmentPreflightError as exc:
        assert "identity mismatch" in str(exc)
    else:
        raise AssertionError("resume with a changed max_source_length must be blocked")

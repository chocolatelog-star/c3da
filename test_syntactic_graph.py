import json
from pathlib import Path

from syntactic_graph import (
    ALIGNMENT_POLICY_VERSION,
    GRAPH_SCHEMA_VERSION,
    GraphCacheError,
    align_parser_words_to_subwords,
    build_graph_cache_records,
    build_graph_record,
    build_typed_edges,
    canonical_row_json,
    load_graph_cache_directory,
    load_graph_cache_rows,
    sha256_file,
    verify_tokenizer_input_equivalence,
    validate_alignment_policy,
)


class FakeWord:
    def __init__(self, text, start, end, upos, head, deprel):
        self.text = text
        self.start_char = start
        self.end_char = end
        self.upos = upos
        self.head = head
        self.deprel = deprel


class FakeSentence:
    def __init__(self, words):
        self.words = words


class FakeDoc:
    def __init__(self, sentences):
        self.sentences = sentences


class FakeParser:
    def __init__(self, doc):
        self.doc = doc

    def __call__(self, _text):
        return self.doc


class FakeTokenizer:
    is_fast = True
    name_or_path = "fake-t5"

    def __call__(self, text, **_kwargs):
        offsets = []
        input_ids = []
        for index, char in enumerate(text):
            if char.isspace():
                continue
            input_ids.append(index + 1)
            offsets.append((index, index + 1))
        input_ids.append(0)
        offsets.append((0, 0))
        return {"input_ids": input_ids, "offset_mapping": offsets}


class FakeTaskTokenTokenizer(FakeTokenizer):
    def __init__(self, *, shift_inputs=False):
        self.shift_inputs = shift_inputs

    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        if self.shift_inputs:
            encoded["input_ids"] = [value + 1 for value in encoded["input_ids"]]
        return encoded


def test_graph_cache_identity_accepts_output_only_task_tokens_when_inputs_are_equivalent():
    rows = [{"id": "row-1", "text": "The service was excellent."}]
    report = verify_tokenizer_input_equivalence(
        FakeTokenizer(),
        FakeTaskTokenTokenizer(),
        rows,
        use_task_prefix=False,
        max_length=128,
    )
    assert report == {"rows_checked": 1, "differences": 0}


def test_graph_cache_identity_rejects_any_input_tokenization_change():
    rows = [{"id": "row-1", "text": "The service was excellent."}]
    try:
        verify_tokenizer_input_equivalence(
            FakeTokenizer(),
            FakeTaskTokenTokenizer(shift_inputs=True),
            rows,
            use_task_prefix=False,
            max_length=128,
        )
    except GraphCacheError as exc:
        assert "row-1" in str(exc)
    else:
        raise AssertionError("input tokenization change was not rejected")


def test_typed_edges_include_dependency_directions_neighbors_and_self_loops():
    words = [
        {"index": 0, "text": "staff", "upos": "NOUN", "head": 1, "deprel": "nsubj", "start": 0, "end": 5},
        {"index": 1, "text": "is", "upos": "AUX", "head": -1, "deprel": "root", "start": 6, "end": 8},
        {"index": 2, "text": ".", "upos": "PUNCT", "head": 1, "deprel": "punct", "start": 8, "end": 9},
    ]

    edges = build_typed_edges(words)
    keys = {(edge["src"], edge["dst"], edge["kind"]) for edge in edges}

    assert (1, 0, "dependency_forward") in keys
    assert (0, 1, "dependency_reverse") in keys
    assert (0, 1, "pos_neighbor_forward") in keys
    assert (1, 0, "pos_neighbor_reverse") in keys
    assert all((index, index, "self_loop") in keys for index in range(3))
    assert all(0 <= edge["src"] < 3 and 0 <= edge["dst"] < 3 for edge in edges)


def test_alignment_uses_offsets_and_supports_repeated_words_and_subwords():
    text = "staff staff"
    words = [
        {"index": 0, "text": "staff", "upos": "NOUN", "head": -1, "deprel": "root", "start": 0, "end": 5},
        {"index": 1, "text": "staff", "upos": "NOUN", "head": 0, "deprel": "dep", "start": 6, "end": 11},
    ]
    parser = FakeParser(FakeDoc([FakeSentence([])]))
    parser.doc = FakeDoc([FakeSentence([FakeWord("staff", 0, 5, "NOUN", 0, "root"), FakeWord("staff", 6, 11, "NOUN", 1, "dep")])])

    record = build_graph_record(
        row_id="row-1",
        text=text,
        input_text="extract aste: " + text,
        parser=parser,
        tokenizer=FakeTokenizer(),
        tokenizer_identity={"name_or_path": "fake-t5"},
        parser_identity={"name": "fake"},
    )

    assert record["word_to_subword"] == [[12, 13, 14, 15, 16], [17, 18, 19, 20, 21]]
    assert record["alignment"]["valid_word_count"] == 2
    assert record["alignment"]["unaligned_word_count"] == 0


def test_alignment_allows_one_subword_to_cover_adjacent_parser_fragments():
    words = [
        {"index": 0, "sentence_index": 0, "text": "i", "start": 19, "end": 20},
        {"index": 1, "sentence_index": 0, "text": "Phone", "start": 20, "end": 25},
    ]

    aligned = align_parser_words_to_subwords(
        words,
        input_text="Pairing it with an iPhone",
        graph_text="Pairing it with an iPhone",
        offset_mapping=[(19, 25), (0, 0)],
    )

    assert aligned == [[0], [0]]


def test_alignment_policy_shared_validator_accepts_iphone_and_formal_alignment_matches():
    words = [
        {"index": 0, "sentence_index": 0, "text": "i", "start": 0, "end": 1},
        {"index": 1, "sentence_index": 0, "text": "Phone", "start": 1, "end": 6},
    ]

    validation = validate_alignment_policy(words, "iPhone", "iPhone", [(0, 6)])

    assert validation["valid"] is True
    assert validation["aligned"] == [[0], [0]]
    assert align_parser_words_to_subwords(words, "iPhone", "iPhone", [(0, 6)]) == validation["aligned"]


def test_alignment_policy_accepts_real_partial_contiguous_shared_subwords():
    cases = [
        (
            "didnt",
            [("did", 0, 3), ("nt", 3, 5)],
            [(0, 4), (4, 5)],
        ),
        (
            "doesnt",
            [("does", 0, 4), ("nt", 4, 6)],
            [(0, 5), (5, 6)],
        ),
        (
            "cant",
            [("ca", 0, 2), ("nt", 2, 4)],
            [(0, 3), (3, 4)],
        ),
        (
            "wont",
            [("wo", 0, 2), ("nt", 2, 4)],
            [(0, 3), (3, 4)],
        ),
        (
            "wasnt",
            [("was", 0, 3), ("nt", 3, 5)],
            [(0, 4), (4, 5)],
        ),
        (
            "Registration/1st",
            [("Registration", 0, 12), ("/", 12, 13), ("1st", 13, 16)],
            [(0, 12), (12, 14), (14, 15), (15, 16)],
        ),
        (
            "WIth",
            [("WIt", 0, 3), ("h", 3, 4)],
            [(0, 2), (2, 4)],
        ),
    ]

    for text, raw_words, offsets in cases:
        words = [
            {"index": index, "sentence_index": 0, "text": word, "start": start, "end": end}
            for index, (word, start, end) in enumerate(raw_words)
        ]
        validation = validate_alignment_policy(words, text, text, offsets)

        assert validation["valid"] is True, text
        partial = [item for item in validation["shared_subwords"] if item["partial_contiguous_shared_subword"]]
        assert len(partial) == 1, text
        assert partial[0]["exact_union"] is False, text
        assert partial[0]["contained_in_parser_union"] is True, text
        assert align_parser_words_to_subwords(words, text, text, offsets) == validation["aligned"]


def test_alignment_policy_rejects_abx_incomplete_shared_union_in_preflight_and_formal_code():
    words = [
        {"index": 0, "sentence_index": 0, "text": "a", "start": 0, "end": 1},
        {"index": 1, "sentence_index": 0, "text": "b", "start": 1, "end": 2},
    ]
    validation = validate_alignment_policy(words, "abx", "abx", [(0, 3)])

    assert validation["valid"] is False
    assert any(item["issue_type"] == "shared_subword_outside_parser_union" for item in validation["violations"])
    try:
        align_parser_words_to_subwords(words, "abx", "abx", [(0, 3)])
    except GraphCacheError as exc:
        assert "shared_subword_outside_parser_union" in str(exc)
    else:
        raise AssertionError("formal alignment must reject an incomplete shared union")


def test_alignment_policy_rejects_cross_space_non_contiguous_cross_sentence_and_out_of_bounds():
    cases = [
        (
            "cross_space",
            [
                {"index": 0, "sentence_index": 0, "text": "a", "start": 0, "end": 1},
                {"index": 1, "sentence_index": 0, "text": "b", "start": 2, "end": 3},
            ],
            "a b",
            [(0, 3)],
            "cross_space_shared_subword",
        ),
        (
            "non_contiguous",
            [
                {"index": 0, "sentence_index": 0, "text": "a", "start": 0, "end": 1},
                {"index": 1, "sentence_index": 0, "text": "x", "start": 4, "end": 5},
                {"index": 2, "sentence_index": 0, "text": "b", "start": 1, "end": 2},
            ],
            "abxxx",
            [(0, 2)],
            "non_contiguous_shared_subword",
        ),
        (
            "cross_sentence",
            [
                {"index": 0, "sentence_index": 0, "text": "a", "start": 0, "end": 1},
                {"index": 1, "sentence_index": 1, "text": "b", "start": 1, "end": 2},
            ],
            "ab",
            [(0, 2)],
            "cross_sentence_shared_subword",
        ),
        (
            "out_of_bounds",
            [{"index": 0, "sentence_index": 0, "text": "a", "start": -1, "end": 1}],
            "a",
            [(0, 1)],
            "out_of_bounds_mapping",
        ),
        (
            "incomplete_character_coverage",
            [{"index": 0, "sentence_index": 0, "text": "staff", "start": 0, "end": 5}],
            "staff",
            [(0, 2), (3, 5)],
            "incomplete_character_coverage",
        ),
    ]

    for _name, words, text, offsets, issue_type in cases:
        validation = validate_alignment_policy(words, text, text, offsets)
        assert validation["valid"] is False
        assert any(item["issue_type"] == issue_type for item in validation["violations"])
        try:
            align_parser_words_to_subwords(words, text, text, offsets)
        except GraphCacheError:
            pass
        else:
            raise AssertionError(f"formal alignment must reject {_name}")


def test_alignment_policy_accepts_one_to_one_and_one_to_many_subword_mapping():
    word = [{"index": 0, "sentence_index": 0, "text": "staff", "start": 0, "end": 5}]

    assert validate_alignment_policy(word, "staff", "staff", [(0, 5)])["valid"] is True
    assert validate_alignment_policy(word, "staff", "staff", [(0, 2), (2, 5)])["valid"] is True


def test_alignment_rejects_shared_subword_across_a_character_gap():
    words = [
        {"index": 0, "sentence_index": 0, "text": "a", "start": 0, "end": 1},
        {"index": 1, "sentence_index": 0, "text": "b", "start": 2, "end": 3},
    ]

    try:
        align_parser_words_to_subwords(
            words,
            input_text="a b",
            graph_text="a b",
            offset_mapping=[(0, 3), (0, 0)],
        )
    except GraphCacheError as exc:
        assert "cross_space_shared_subword" in str(exc)
    else:
        raise AssertionError("a shared subword must not bridge a character gap")


def test_cache_row_serialization_is_canonical_and_rejects_identity_mismatch(tmp_path):
    row = {"row_id": "1", "text": "A sentence", "normalized_text_sha256": "abc", "edges": []}
    first = canonical_row_json(row)
    second = canonical_row_json(json.loads(first))
    assert first == second

    cache = Path(tmp_path) / "source_train.jsonl"
    cache.write_text(first + "\n", encoding="utf-8")
    try:
        load_graph_cache_rows(cache, [{"id": "1", "text": "Different"}], expected_split="source_train")
    except GraphCacheError as exc:
        assert "text hash mismatch" in str(exc)
    else:
        raise AssertionError("cache identity mismatch must fail hard")


def test_graph_cache_resume_is_rowwise_and_byte_identical(tmp_path):
    rows = {
        "source_train": [{"id": "s1", "text": "staff"}],
        "source_dev": [{"id": "d1", "text": "staff"}],
        "target_unlabeled": [{"id": "t1", "text": "staff"}],
    }
    parser = FakeParser(FakeDoc([FakeSentence([FakeWord("staff", 0, 5, "NOUN", 0, "root")])]))
    tokenizer = FakeTokenizer()
    tokenizer_identity = {"class": "fake", "files_sha256": {}}
    parser_identity = {"name": "fake"}
    interrupted_dir = Path(tmp_path) / "interrupted"
    try:
        build_graph_cache_records(
            rows,
            interrupted_dir,
            tokenizer,
            parser,
            tokenizer_identity,
            parser_identity,
            use_task_prefix=False,
            stop_after_rows=1,
        )
    except GraphCacheError as exc:
        assert "interrupted" in str(exc)
    else:
        raise AssertionError("test interruption must stop after one row")

    build_graph_cache_records(
        rows,
        interrupted_dir,
        tokenizer,
        parser,
        tokenizer_identity,
        parser_identity,
        use_task_prefix=False,
    )
    clean_dir = Path(tmp_path) / "clean"
    build_graph_cache_records(
        rows,
        clean_dir,
        tokenizer,
        parser,
        tokenizer_identity,
        parser_identity,
        use_task_prefix=False,
    )

    for name in ("source_train.jsonl", "source_dev.jsonl", "target_unlabeled.jsonl", "relation_vocab.json", "manifest.json"):
        assert (interrupted_dir / name).read_bytes() == (clean_dir / name).read_bytes()

    loaded = load_graph_cache_directory(
        interrupted_dir,
        "source_train",
        rows["source_train"],
        tokenizer_identity=tokenizer_identity,
        parser_identity=parser_identity,
    )
    assert loaded.relation_vocab_size > 0
    manifest = json.loads((interrupted_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == GRAPH_SCHEMA_VERSION
    assert manifest["alignment_policy_version"] == ALIGNMENT_POLICY_VERSION


def test_graph_cache_resume_rejects_an_older_alignment_policy(tmp_path):
    rows = {
        "source_train": [{"id": "s1", "text": "staff"}],
        "source_dev": [{"id": "d1", "text": "staff"}],
        "target_unlabeled": [{"id": "t1", "text": "staff"}],
    }
    parser = FakeParser(FakeDoc([FakeSentence([FakeWord("staff", 0, 5, "NOUN", 0, "root")])]))
    output_dir = Path(tmp_path) / "resume-policy"
    try:
        build_graph_cache_records(
            rows,
            output_dir,
            FakeTokenizer(),
            parser,
            {"class": "fake", "files_sha256": {}},
            {"name": "fake"},
            use_task_prefix=False,
            stop_after_rows=1,
        )
    except GraphCacheError:
        pass
    progress_path = output_dir / "source_train.progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["alignment_policy_version"] == ALIGNMENT_POLICY_VERSION
    progress["alignment_policy_version"] = "legacy-containment-v1"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    try:
        build_graph_cache_records(
            rows,
            output_dir,
            FakeTokenizer(),
            parser,
            {"class": "fake", "files_sha256": {}},
            {"name": "fake"},
            use_task_prefix=False,
        )
    except GraphCacheError as exc:
        assert "alignment_policy_version" in str(exc)
    else:
        raise AssertionError("a cache from another alignment policy must not resume")


def test_graph_cache_target_test_split_is_forbidden(tmp_path):
    vocab_path = Path(tmp_path) / "relation_vocab.json"
    vocab_path.write_text("[]\n", encoding="utf-8")
    (Path(tmp_path) / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "alignment_policy_version": ALIGNMENT_POLICY_VERSION,
                "target_test_access": False,
                "relation_vocab_sha256": sha256_file(vocab_path),
            }
        ),
        encoding="utf-8",
    )

    try:
        load_graph_cache_directory(Path(tmp_path), "target_test", [])
    except GraphCacheError as exc:
        assert "target_test graph cache is forbidden" in str(exc)
    else:
        raise AssertionError("target_test graph caches must remain forbidden")

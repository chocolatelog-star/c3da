from t5_aste_augment import build_syntax_candidate_banks, rank_syntax_candidates, syntax_compatibility_score


def test_syntax_rank_prefers_compatible_candidate():
    source = {"upos": "NOUN", "dependency_relation": "nsubj", "head_pos": "VERB"}
    candidates = [
        {"text": "bad", "upos": "ADJ", "dependency_relation": "amod", "head_pos": "NOUN"},
        {"text": "good", "upos": "NOUN", "dependency_relation": "nsubj", "head_pos": "VERB"},
    ]
    ranked, audit = rank_syntax_candidates(source, candidates, min_acceptable_score=1)
    assert ranked[0]["text"] == "good"
    assert audit["syntax_fallback"] is False
    assert audit["high_compatibility"] == 1


def test_syntax_rank_falls_back_when_no_candidate_is_acceptable():
    source = {"upos": "NOUN", "dependency_relation": "nsubj", "head_pos": "VERB"}
    candidates = [{"text": "bad", "upos": "ADV", "dependency_relation": "advmod", "head_pos": "ADJ"}]
    ranked, audit = rank_syntax_candidates(source, candidates, min_acceptable_score=3)
    assert ranked[0]["text"] == "bad"
    assert audit["syntax_fallback"] is True
    assert audit["no_compatible_candidate"] is True


def test_missing_syntax_is_not_invented_as_compatible():
    result = syntax_compatibility_score({"upos": "NOUN"}, {"text": "screen"})
    assert result["available_features"] == 0
    assert result["score"] == 0
    assert result["high_compatibility"] is False


def test_graph_cache_metadata_is_attached_to_triplet_occurrence():
    rows = [{"id": "1", "text": "good battery", "label": "<pos> battery <opinion> good"}]
    cache = [{
        "row_id": "1",
        "text": "good battery",
        "parser_tokens": [
            {"index": 0, "text": "good", "upos": "ADJ", "deprel": "amod", "head": 1},
            {"index": 1, "text": "battery", "upos": "NOUN", "deprel": "root", "head": -1},
        ],
    }]
    result = build_syntax_candidate_banks(rows, cache)
    assert result["syntax_cache_rows"] == 1
    assert rows[0]["triplet_syntax"]["battery|good|pos"]["aspect"]["upos"] == "NOUN"
    assert result["syntax_candidate_opinions"][0]["head_pos"] == "NOUN"

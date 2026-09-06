from t5_aste_augment import (
    build_augmentation_requests,
    build_opinion_provenance_banks,
    build_syntax_candidate_banks,
    rank_syntax_candidates,
    syntax_compatibility_score,
)


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


def test_target_priority_bank_contains_only_pseudo_opinions_with_provenance():
    banks = build_opinion_provenance_banks(
        [{"id": "s1", "label": "<pos> screen <opinion> bright"}],
        [{"id": "t1", "label": "<pos> display <opinion> vivid"}],
    )
    assert banks["source_gold"]["pos"][0]["source_domain"] == "source_gold"
    assert banks["target_pseudo"]["pos"][0]["source_domain"] == "target_pseudo"
    assert banks["target_pseudo"]["pos"][0]["source_row_ids"] == ["t1"]


def test_target_priority_keeps_coupled_random_sentiment_transition():
    source = [{"id": "s1", "text": "screen is bright", "label": "<pos> screen <opinion> bright"}]
    pseudo = [{"id": "t1", "text": "display is vivid", "label": "<pos> display <opinion> vivid", "sample_weight": 0.9}]
    memory = {
        "target_aspects": ["display"],
        "opinions_by_sentiment": {"pos": ["vivid"], "neg": ["bad"]},
        "candidate_opinions_by_sentiment": {"pos": ["vivid"], "neg": ["bad"]},
    }
    requests = build_augmentation_requests(
        source, pseudo, 1, 1000, prompt_style="masked_mutual", channel_mode="opinion",
        domain_memory=memory, target_domain_opinion_priority=True,
    )
    assert requests
    request = requests[0]
    assert request["new_triplet"][2] != request["old_triplet"][2]
    assert request["target_bank_hit"] is False or request["opinion_source_domain"] == "target_pseudo"

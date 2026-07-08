from t5_aste_augment import build_cross_domain_memory, build_source_memory, build_target_memory


def test_source_memory_keeps_gold_opinion_sentiment_pairs():
    source_rows = [
        {"text": "The food is good.", "label": "<pos> food <opinion> good"},
        {"text": "The service is rude.", "label": "<neg> service <opinion> rude"},
    ]

    memory = build_source_memory(source_rows)

    assert "food" in memory["aspects"]
    assert "good" in memory["opinions_by_sentiment"]["pos"]
    assert "rude" in memory["opinions_by_sentiment"]["neg"]
    assert ["food", "good", "pos"] in memory["triplets"]


def test_target_memory_filters_low_confidence_and_bad_aspects():
    target_rows = [
        {
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
            "sample_weight": 0.65,
        },
        {
            "text": "The thing is bad.",
            "label": "<neg> thing <opinion> bad",
            "sample_weight": 0.65,
        },
        {
            "text": "The screen is bright.",
            "label": "<pos> screen <opinion> bright",
            "sample_weight": 0.4,
        },
    ]

    memory = build_target_memory(target_rows, min_pseudo_weight=0.6)

    assert "battery life" in memory["aspects"]
    assert "thing" not in memory["aspects"]
    assert "screen" not in memory["aspects"]
    assert "battery life" in memory["rejected_aspects_by_reason"]["kept"]


def test_cross_domain_memory_uses_target_aspects_and_shared_opinions():
    source_memory = build_source_memory(
        [{"text": "The food is good.", "label": "<pos> food <opinion> good"}]
    )
    target_memory = build_target_memory(
        [
            {
                "text": "The keyboard is good.",
                "label": "<pos> keyboard <opinion> good",
                "sample_weight": 0.65,
            }
        ],
        min_pseudo_weight=0.6,
    )

    cross = build_cross_domain_memory(source_memory, target_memory)

    assert "keyboard" in cross["target_aspects"]
    assert "good" in cross["opinions_by_sentiment"]["pos"]
    assert ["keyboard", "good", "pos"] in cross["candidate_triplets"]


def test_target_memory_scores_frequency_and_domain_specificity():
    source_rows = [
        {"text": "The service is good.", "label": "<pos> service <opinion> good"},
        {"text": "The food is good.", "label": "<pos> food <opinion> good"},
    ]
    target_rows = [
        {"text": "The battery life is good.", "label": "<pos> battery life <opinion> good", "sample_weight": 0.65},
        {"text": "Battery life is long.", "label": "<pos> battery life <opinion> long", "sample_weight": 0.65},
        {"text": "The trackpad is responsive.", "label": "<pos> trackpad <opinion> responsive", "sample_weight": 0.65},
    ]
    source_memory = build_source_memory(source_rows)

    target_memory = build_target_memory(
        target_rows,
        min_pseudo_weight=0.6,
        source_memory=source_memory,
        source_row_count=len(source_rows),
    )

    battery_stats = target_memory["aspect_stats"]["battery life"]
    trackpad_stats = target_memory["aspect_stats"]["trackpad"]

    assert battery_stats["target_tf"] == 2
    assert battery_stats["target_df"] == 2
    assert battery_stats["source_tf"] == 0
    assert battery_stats["domain_score"] > 0
    assert "battery life" in target_memory["core_target_aspects"]
    assert "trackpad" in target_memory["specific_candidate_aspects"]
    assert "trackpad" not in target_memory["specific_target_aspects"]
    assert "trackpad" in target_memory["rejected_specific_aspects_by_reason"]["low_frequency"]
    assert trackpad_stats["mean_pseudo_weight"] == 0.65


def test_target_memory_rejects_event_and_opinion_like_aspects():
    target_rows = [
        {"text": "The laptop has good syncing.", "label": "<pos> syncing <opinion> good", "sample_weight": 0.65},
        {"text": "The keyboard is responsive.", "label": "<pos> keyboard <opinion> responsive", "sample_weight": 0.65},
        {"text": "The display is bright.", "label": "<pos> bright <opinion> bright", "sample_weight": 0.65},
    ]

    source_rows = [
        {"text": f"The food {i} is good.", "label": "<pos> food <opinion> good"}
        for i in range(30)
    ]
    source_memory = build_source_memory(source_rows)
    target_memory = build_target_memory(
        target_rows,
        min_pseudo_weight=0.6,
        source_memory=source_memory,
        source_row_count=len(source_rows),
    )

    assert "keyboard" in target_memory["aspects"]
    assert "syncing" not in target_memory["aspects"]
    assert "bright" not in target_memory["aspects"]
    assert "syncing" in target_memory["rejected_aspects_by_reason"]["event_or_action_like"]
    assert "bright" in target_memory["rejected_aspects_by_reason"]["opinion_like"]


def test_cross_domain_memory_prunes_candidate_triplets_with_top_opinions():
    source_memory = build_source_memory(
        [
            {"text": f"The food is good{i}.", "label": f"<pos> food <opinion> good{i}"}
            for i in range(10)
        ]
    )
    target_memory = build_target_memory(
        [
            {"text": "The keyboard is responsive.", "label": "<pos> keyboard <opinion> responsive", "sample_weight": 0.65},
            {"text": "The battery is poor.", "label": "<neg> battery <opinion> poor", "sample_weight": 0.65},
        ],
        min_pseudo_weight=0.6,
    )

    cross = build_cross_domain_memory(source_memory, target_memory, max_candidate_opinions_per_sentiment=2)

    assert ["keyboard", "responsive", "pos"] in cross["candidate_triplets"]
    assert ["battery", "poor", "neg"] in cross["candidate_triplets"]
    assert len(cross["candidate_triplets"]) < 20
    assert cross["candidate_generation"]["max_candidate_opinions_per_sentiment"] == 2


def test_target_memory_keeps_only_high_confidence_specific_aspects():
    target_rows = [
        {
            "text": "Mac osx is stable and fast.",
            "label": "<pos> mac osx <opinion> stable ; <pos> mac osx <opinion> fast",
            "sample_weight": 0.65,
        },
        {"text": "The abilitiy is good.", "label": "<pos> abilitiy <opinion> good", "sample_weight": 0.65},
        {"text": "The navigate is easy.", "label": "<pos> navigate <opinion> easy", "sample_weight": 0.65},
        {"text": "The visual is good.", "label": "<pos> visual <opinion> good", "sample_weight": 0.65},
        {"text": "It works right out.", "label": "<pos> right out <opinion> works", "sample_weight": 0.65},
        {"text": "The accessories are useful.", "label": "<pos> accessories <opinion> useful", "sample_weight": 0.65},
        {
            "text": "Extended warranties are useful and helpful.",
            "label": "<pos> extended warranties <opinion> useful ; <pos> extended warranties <opinion> helpful",
            "sample_weight": 0.605,
        },
        {"text": "The laptop is good.", "label": "<pos> laptop <opinion> good", "sample_weight": 0.65},
        {"text": "The laptop is fast.", "label": "<pos> laptop <opinion> fast", "sample_weight": 0.65},
    ]

    source_rows = [
        {"text": f"The food {i} is good.", "label": "<pos> food <opinion> good"}
        for i in range(30)
    ]
    target_memory = build_target_memory(
        target_rows,
        min_pseudo_weight=0.6,
        source_memory=build_source_memory(source_rows),
        source_row_count=len(source_rows),
    )

    assert "laptop" in target_memory["core_target_aspects"]
    assert "mac osx" in target_memory["specific_target_aspects"]
    assert "accessories" not in target_memory["specific_target_aspects"]
    assert "extended warranties" not in target_memory["specific_target_aspects"]
    assert "abilitiy" not in target_memory["specific_target_aspects"]
    assert "navigate" not in target_memory["specific_target_aspects"]
    assert "visual" not in target_memory["specific_target_aspects"]
    assert "right out" not in target_memory["specific_target_aspects"]
    assert "accessories" in target_memory["rejected_specific_aspects_by_reason"]["low_frequency"]
    assert "extended warranties" in target_memory["rejected_specific_aspects_by_reason"]["low_mean_weight"]
    assert "abilitiy" in target_memory["rejected_specific_aspects_by_reason"]["bad_specific_shape"]

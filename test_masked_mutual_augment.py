from t5_aste_augment import (
    aspect_replacement_compatible,
    build_aspect_rewrite_prompt,
    build_augmentation_requests,
    build_domain_memory,
    build_generator_training_rows,
    rank_replacement_aspects,
)
from t5_aste_data import parse_triplet_text_list


def test_domain_memory_keeps_high_confidence_target_aspects():
    rows = [
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

    memory = build_domain_memory(rows, min_pseudo_weight=0.6)

    assert "battery life" in memory["aspects"]
    assert "thing" not in memory["aspects"]
    assert "screen" not in memory["aspects"]


def test_masked_mutual_generator_training_uses_source_masked_prompts():
    rows = [
        {
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
        }
    ]

    train_rows = build_generator_training_rows(rows, seed=13, prompt_style="masked_mutual")

    assert any("[ASP]" in row["input"] for row in train_rows)
    assert any("[OPI]" in row["input"] for row in train_rows)
    assert {row["channel"] for row in train_rows} == {"masked_aspect_editor", "masked_opinion_sentiment_editor"}


def test_generator_training_can_add_text_domain_prefix():
    rows = [
        {
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
        }
    ]

    train_rows = build_generator_training_rows(
        rows,
        seed=13,
        prompt_style="masked_mutual",
        domain_name="laptop14",
        domain_prefix_style="text",
    )

    assert train_rows
    assert all(row["input"].startswith("target domain: [laptop14] ; ") for row in train_rows)
    assert all(row["domain_name"] == "laptop14" for row in train_rows)
    assert all(row["domain_prefix_style"] == "text" for row in train_rows)


def test_label_to_text_generator_training_can_add_bracket_domain_prefix():
    rows = [
        {"text": "The battery life is long.", "label": "<pos> battery life <opinion> long"},
    ]

    train_rows = build_generator_training_rows(
        rows,
        seed=13,
        prompt_style="label_to_text",
        domain_name="rest14",
        domain_prefix_style="bracket",
    )

    assert train_rows
    assert all(row["input"].startswith("[rest14] ") for row in train_rows)
    assert all("generate aste sentence:" in row["input"] for row in train_rows)


def test_masked_mutual_aspect_training_uses_supervised_replacement():
    rows = [
        {
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
        },
        {
            "text": "The keyboard is responsive.",
            "label": "<pos> keyboard <opinion> responsive",
        },
    ]

    train_rows = build_generator_training_rows(
        rows,
        seed=13,
        prompt_style="masked_mutual",
        channel_mode="aspect",
    )

    assert train_rows
    assert {row["channel"] for row in train_rows} == {"masked_aspect_editor"}
    assert all("[ASP]" in row["input"] for row in train_rows)
    assert all("[OPI]" not in row["input"] for row in train_rows)
    assert any(row["target"] != "The battery life is long." for row in train_rows)
    assert any("new_triplet" in row and row["old_triplet"] != row["new_triplet"] for row in train_rows)


def test_rewrite_aspect_training_uses_full_sentence_rewrite_prompt():
    rows = [
        {"text": "The service is great.", "label": "<pos> service <opinion> great"},
        {"text": "The food is great.", "label": "<pos> food <opinion> great"},
    ]

    train_rows = build_generator_training_rows(
        rows,
        seed=13,
        prompt_style="rewrite_aspect",
        channel_mode="aspect",
    )

    assert train_rows
    assert {row["channel"] for row in train_rows} == {"rewrite_aspect_editor"}
    assert all(row["input"].startswith("rewrite aspect:") for row in train_rows)
    assert all("[ASP]" not in row["input"] for row in train_rows)
    assert any(row["target"] == "The food is great." for row in train_rows)
    assert any(row["old_triplet"] != row["new_triplet"] for row in train_rows)


def test_rewrite_aspect_augmentation_requests_use_rewrite_prompt():
    pseudo_rows = [
        {
            "id": "t1",
            "text": "The keyboard is responsive.",
            "label": "<pos> keyboard <opinion> responsive",
            "sample_weight": 0.65,
        },
        {
            "id": "t2",
            "text": "The trackpad is responsive.",
            "label": "<pos> trackpad <opinion> responsive",
            "sample_weight": 0.65,
        },
    ]

    requests = build_augmentation_requests(
        [],
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="rewrite_aspect",
        channel_mode="aspect",
    )

    assert requests
    assert {req["channel"] for req in requests} == {"rewrite_aspect_channel"}
    assert all(req["input"].startswith("rewrite aspect:") for req in requests)
    assert all("[ASP]" not in req["input"] for req in requests)


def test_label_composition_training_uses_label_to_text_generation():
    rows = [
        {"text": "The battery life is long.", "label": "<pos> battery life <opinion> long"},
        {"text": "The keyboard is stiff.", "label": "<neg> keyboard <opinion> stiff"},
    ]

    train_rows = build_generator_training_rows(
        rows,
        seed=13,
        prompt_style="label_composition",
        channel_mode="aspect",
    )

    assert train_rows
    assert {row["channel"] for row in train_rows} == {"label_composition_generator"}
    assert all(row["input"].startswith("generate aste sentence:") for row in train_rows)
    assert any(row["target"] == "The battery life is long." for row in train_rows)


def test_label_to_text_training_uses_source_label_to_text_generation():
    rows = [
        {"text": "The battery life is long.", "label": "<pos> battery life <opinion> long"},
        {"text": "The keyboard is stiff.", "label": "<neg> keyboard <opinion> stiff"},
    ]

    train_rows = build_generator_training_rows(
        rows,
        seed=13,
        prompt_style="label_to_text",
        channel_mode="aspect",
    )

    assert train_rows
    assert {row["channel"] for row in train_rows} == {"label_to_text_generator"}
    assert all(row["input"].startswith("generate aste sentence:") for row in train_rows)
    assert {row["target"] for row in train_rows} == {"The battery life is long.", "The keyboard is stiff."}


def test_label_to_text_augmentation_requests_generate_from_single_target_labels():
    pseudo_rows = [
        {
            "id": "t1",
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
            "sample_weight": 0.65,
        },
        {
            "id": "t2",
            "text": "The keyboard is stiff.",
            "label": "<neg> keyboard <opinion> stiff",
            "sample_weight": 0.65,
        },
    ]

    requests = build_augmentation_requests(
        [],
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="label_to_text",
        channel_mode="aspect",
    )

    assert requests
    assert {req["channel"] for req in requests} == {"label_to_text_channel"}
    assert all(req["input"].startswith("generate aste sentence:") for req in requests)
    assert all(len(req["new_triplets"]) == 1 for req in requests)


def test_label_composition_augmentation_requests_combine_target_triplets():
    pseudo_rows = [
        {
            "id": "t1",
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
            "sample_weight": 0.65,
        },
        {
            "id": "t2",
            "text": "The keyboard is stiff.",
            "label": "<neg> keyboard <opinion> stiff",
            "sample_weight": 0.65,
        },
        {
            "id": "t3",
            "text": "The screen is bright.",
            "label": "<pos> screen <opinion> bright",
            "sample_weight": 0.65,
        },
    ]

    requests = build_augmentation_requests(
        [],
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="label_composition",
        channel_mode="aspect",
    )

    assert requests
    assert {req["channel"] for req in requests} == {"label_composition_channel"}
    assert all(req["input"].startswith("generate aste sentence:") for req in requests)
    assert all(len(req["new_triplets"]) >= 2 for req in requests)
    assert any(" ; " in req["label"] for req in requests)


def test_sentence_fusion_composition_requests_use_filtered_single_augments():
    composition_rows = [
        {
            "id": "single1",
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
            "sample_weight": 0.2,
            "augmentation": "label_to_text_channel",
        },
        {
            "id": "single2",
            "text": "The keyboard is stiff.",
            "label": "<neg> keyboard <opinion> stiff",
            "sample_weight": 0.2,
            "augmentation": "label_to_text_channel",
        },
    ]

    requests = build_augmentation_requests(
        [],
        [],
        per_row=1,
        seed=7,
        prompt_style="sentence_fusion_composition",
        channel_mode="aspect",
        composition_source_rows=composition_rows,
    )

    assert len(requests) == 1
    assert requests[0]["channel"] == "sentence_fusion_composition_channel"
    assert requests[0]["input"].startswith("fuse aste sentences:")
    assert "The battery life is long." in requests[0]["input"]
    assert "The keyboard is stiff." in requests[0]["input"]
    assert set(parse_triplet_text_list(requests[0]["label"])) == {
        ("battery life", "long", "pos"),
        ("keyboard", "stiff", "neg"),
    }
    assert len(requests[0]["new_triplets"]) == 2


def test_rsda_t5_label_composition_prefers_similar_label_embeddings():
    pseudo_rows = [
        {
            "id": "input1",
            "text": "The keyboard is responsive.",
            "label": "<pos> keyboard <opinion> responsive",
            "sample_weight": 0.65,
        },
        {
            "id": "input2",
            "text": "The touchpad is smooth.",
            "label": "<pos> touchpad <opinion> smooth",
            "sample_weight": 0.65,
        },
        {
            "id": "battery1",
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
            "sample_weight": 0.65,
        },
        {
            "id": "battery2",
            "text": "The charger is reliable.",
            "label": "<pos> charger <opinion> reliable",
            "sample_weight": 0.65,
        },
    ]
    label_embeddings = {
        "<pos> keyboard <opinion> responsive": [1.0, 0.0],
        "<pos> touchpad <opinion> smooth": [0.7, 0.7],
        "<pos> battery life <opinion> long": [0.0, 1.0],
        "<pos> charger <opinion> reliable": [0.7, 0.7],
    }

    requests = build_augmentation_requests(
        [],
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="rsda_t5_label_composition",
        channel_mode="aspect",
        label_embeddings=label_embeddings,
    )

    labels = {req["label"] for req in requests}
    assert requests
    assert {req["channel"] for req in requests} == {"rsda_t5_label_composition_channel"}
    assert "<pos> keyboard <opinion> responsive ; <pos> touchpad <opinion> smooth" in labels
    assert "<pos> battery life <opinion> long ; <pos> charger <opinion> reliable" in labels
    assert all(
        not ("keyboard" in label and "battery life" in label)
        for label in labels
    )


def test_rsda_t5_label_composition_rejects_near_duplicate_aspects():
    pseudo_rows = [
        {
            "id": "battery1",
            "text": "The battery is long.",
            "label": "<pos> battery <opinion> long",
            "sample_weight": 0.65,
        },
        {
            "id": "battery2",
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
            "sample_weight": 0.65,
        },
        {
            "id": "screen1",
            "text": "The screen is bright.",
            "label": "<pos> screen <opinion> bright",
            "sample_weight": 0.65,
        },
    ]
    label_embeddings = {
        "<pos> battery <opinion> long": [1.0, 0.0],
        "<pos> battery life <opinion> long": [0.99, 0.01],
        "<pos> screen <opinion> bright": [0.6, 0.8],
    }

    requests = build_augmentation_requests(
        [],
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="rsda_t5_label_composition",
        channel_mode="aspect",
        label_embeddings=label_embeddings,
    )

    assert requests
    assert all(
        not ("battery <opinion> long" in req["label"] and "battery life <opinion> long" in req["label"])
        for req in requests
    )


def test_label_composition_requests_filter_noisy_candidate_triplets():
    pseudo_rows = [
        {
            "id": "bad1",
            "text": "The use is horrible.",
            "label": "<neg> use <opinion> horrible",
            "sample_weight": 0.65,
        },
        {
            "id": "bad2",
            "text": "The system runs quick.",
            "label": "<pos> runs <opinion> quick",
            "sample_weight": 0.65,
        },
        {
            "id": "bad3",
            "text": "Toshiba did not do anything.",
            "label": "<neg> toshiba <opinion> anything",
            "sample_weight": 0.65,
        },
        {
            "id": "good1",
            "text": "The keyboard is responsive.",
            "label": "<pos> keyboard <opinion> responsive",
            "sample_weight": 0.65,
        },
        {
            "id": "good2",
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
            "sample_weight": 0.65,
        },
    ]

    requests = build_augmentation_requests(
        [],
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="label_composition",
        channel_mode="aspect",
    )

    labels = " ".join(req["label"] for req in requests)
    assert requests
    assert "use" not in labels
    assert "runs" not in labels
    assert "anything" not in labels
    assert "keyboard" in labels
    assert "battery life" in labels


def test_masked_mutual_augmentation_requests_carry_channel_metadata():
    source_rows = [
        {
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
        }
    ]
    pseudo_rows = [
        {
            "id": "t1",
            "text": "The keyboard is responsive.",
            "label": "<pos> keyboard <opinion> responsive",
            "sample_weight": 0.65,
        },
        {
            "id": "t2",
            "text": "The trackpad is responsive.",
            "label": "<pos> trackpad <opinion> responsive",
            "sample_weight": 0.65,
        }
    ]

    requests = build_augmentation_requests(source_rows, pseudo_rows, per_row=1, seed=7, prompt_style="masked_mutual")

    assert any(req["channel"] == "masked_aspect_channel" for req in requests)
    assert any(req["channel"] == "masked_opinion_sentiment_channel" for req in requests)
    assert any("[ASP]" in req["input"] for req in requests)
    assert any("[OPI]" in req["input"] for req in requests)
    assert all("old_triplet" in req and "new_triplet" in req for req in requests)


def test_masked_mutual_augmentation_requests_can_add_target_domain_prefix():
    source_rows = [
        {
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
        }
    ]
    pseudo_rows = [
        {
            "id": "t1",
            "text": "The keyboard is responsive.",
            "label": "<pos> keyboard <opinion> responsive",
            "sample_weight": 0.65,
        },
        {
            "id": "t2",
            "text": "The trackpad is responsive.",
            "label": "<pos> trackpad <opinion> responsive",
            "sample_weight": 0.65,
        },
    ]

    requests = build_augmentation_requests(
        source_rows,
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="masked_mutual",
        target_domain_name="laptop14",
        domain_prefix_style="text",
    )

    assert requests
    assert all(req["input"].startswith("target domain: [laptop14] ; ") for req in requests)
    assert all(req["domain_name"] == "laptop14" for req in requests)
    assert all(req["domain_prefix_style"] == "text" for req in requests)
    assert all(req["domain_prefix"] == "target domain: [laptop14] ; " for req in requests)


def test_masked_mutual_opinion_channel_changes_sentiment_when_possible():
    source_rows = [
        {
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
        }
    ]
    pseudo_rows = [
        {
            "id": "t1",
            "text": "The keyboard is responsive.",
            "label": "<pos> keyboard <opinion> responsive",
            "sample_weight": 0.65,
        },
        {
            "id": "t2",
            "text": "The trackpad is terrible.",
            "label": "<neg> trackpad <opinion> terrible",
            "sample_weight": 0.65,
        },
    ]

    requests = build_augmentation_requests(
        source_rows,
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="masked_mutual",
        channel_mode="opinion",
    )

    opinion_requests = [req for req in requests if req["channel"] == "masked_opinion_sentiment_channel"]
    assert opinion_requests
    assert any(req["old_triplet"][2] != req["new_triplet"][2] for req in opinion_requests)


def test_semantic_same_sentiment_opinion_channel_keeps_sentiment_and_prefers_domain_match():
    pseudo_rows = [
        {
            "id": "t1",
            "text": "The keyboard is responsive.",
            "label": "<pos> keyboard <opinion> responsive",
            "sample_weight": 0.65,
        },
        {
            "id": "t2",
            "text": "The trackpad is smooth.",
            "label": "<pos> trackpad <opinion> smooth",
            "sample_weight": 0.65,
        },
        {
            "id": "t3",
            "text": "The service is terrible.",
            "label": "<neg> service <opinion> terrible",
            "sample_weight": 0.65,
        },
    ]
    memory = {
        "candidate_opinions_by_sentiment": {
            "pos": ["smooth", "bright"],
            "neg": ["terrible"],
        },
        "opinion_aspect_counts": {
            "smooth|pos": {"keyboard": 4},
            "bright|pos": {"keyboard": 0},
        },
        "target_triplet_counts": {
            "keyboard|smooth|pos": 2,
        },
    }

    requests = build_augmentation_requests(
        [],
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="masked_mutual",
        channel_mode="opinion",
        domain_memory=memory,
        opinion_replacement_mode="semantic_same_sentiment",
    )

    opinion_requests = [req for req in requests if req["channel"] == "masked_opinion_sentiment_channel"]
    assert opinion_requests
    assert all(req["old_triplet"][2] == req["new_triplet"][2] for req in opinion_requests)
    assert any(req["old_triplet"][1] == "responsive" and req["new_triplet"][1] == "smooth" for req in opinion_requests)
    assert all(req["opinion_replacement_mode"] == "semantic_same_sentiment" for req in opinion_requests)
    assert all("opinion_replacement_rank" in req for req in opinion_requests)


def test_augmentation_requests_can_use_explicit_cross_domain_memory():
    source_rows = [
        {
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
        }
    ]
    pseudo_rows = [
        {
            "id": "t1",
            "text": "The keyboard is responsive.",
            "label": "<pos> keyboard <opinion> responsive",
            "sample_weight": 0.65,
        },
        {
            "id": "t2",
            "text": "The trackpad is responsive.",
            "label": "<pos> trackpad <opinion> responsive",
            "sample_weight": 0.65,
        }
    ]
    memory = {
        "target_aspects": ["trackpad"],
        "candidate_opinions_by_sentiment": {"pos": ["smooth"]},
    }

    requests = build_augmentation_requests(
        source_rows,
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="masked_mutual",
        domain_memory=memory,
    )

    assert any(req["new_triplet"][0] == "trackpad" for req in requests)
    assert any(req["new_triplet"][1] == "smooth" for req in requests)


def test_augmentation_requests_can_run_aspect_channel_only():
    source_rows = [
        {
            "text": "The battery life is long.",
            "label": "<pos> battery life <opinion> long",
        }
    ]
    pseudo_rows = [
        {
            "id": "t1",
            "text": "The keyboard is responsive.",
            "label": "<pos> keyboard <opinion> responsive",
            "sample_weight": 0.65,
        },
        {
            "id": "t2",
            "text": "The trackpad is responsive.",
            "label": "<pos> trackpad <opinion> responsive",
            "sample_weight": 0.65,
        }
    ]

    requests = build_augmentation_requests(
        source_rows,
        pseudo_rows,
        per_row=1,
        seed=7,
        prompt_style="masked_mutual",
        channel_mode="aspect",
    )

    assert requests
    assert {req["channel"] for req in requests} == {"masked_aspect_channel"}
    assert all("[ASP]" in req["input"] for req in requests)


def test_rank_replacement_aspects_prefers_opinion_compatible_target_aspects():
    memory = {
        "target_aspects": ["keyboard", "warranty"],
        "aspect_types": {"keyboard": "hardware", "warranty": "service"},
        "opinion_aspect_counts": {
            "responsive|pos": {"keyboard": 3, "warranty": 0},
        },
        "target_triplet_counts": {
            "keyboard|responsive|pos": 3,
        },
    }

    ranked = rank_replacement_aspects(
        text="The trackpad is responsive.",
        old_aspect="trackpad",
        opinion="responsive",
        sentiment="pos",
        aspect_bank=["warranty", "keyboard"],
        domain_memory=memory,
    )

    assert [item["aspect"] for item in ranked] == ["keyboard", "warranty"]
    assert ranked[0]["score"] > ranked[1]["score"]
    assert ranked[0]["features"]["opinion_aspect_count"] == 3
    assert ranked[0]["features"]["target_triplet_count"] == 3


def test_aspect_replacement_compatibility_filters_bad_slots():
    assert not aspect_replacement_compatible(
        "I charge it at night because of the good battery life.",
        "battery life",
        "desktop",
    )
    assert not aspect_replacement_compatible(
        "I can barely use any usb devices because they will not stay connected properly.",
        "usb devices",
        "unibody design",
    )
    assert not aspect_replacement_compatible(
        "talk about painless syncing - used to take me forever.",
        "syncing",
        "games",
    )

    assert aspect_replacement_compatible("Great laptop that offers features.", "laptop", "keyboard")
    assert aspect_replacement_compatible("The speed is incredible.", "speed", "processor")
    assert aspect_replacement_compatible("The machine is slow to boot up.", "machine", "bios")


def test_aspect_replacement_compatibility_filters_observed_bad_augments():
    assert not aspect_replacement_compatible(
        "I can barely use any usb devices because they will not stay connected properly.",
        "usb devices",
        "fit",
    )
    assert not aspect_replacement_compatible("Customer support crashed twice.", "customer support", "hard disc")
    assert not aspect_replacement_compatible("Customer support just died.", "customer support", "unibody design")
    assert not aspect_replacement_compatible("The sales associate was knowledgeable.", "sales associate", "internet speed")
    assert not aspect_replacement_compatible("I paid for this warranty.", "warranty", "bios")
    assert not aspect_replacement_compatible("I upgraded the memory to four gigabytes.", "memory", "keyboard")
    assert not aspect_replacement_compatible("It has 10 hour battery life.", "battery life", "desktop")
    assert not aspect_replacement_compatible("Talk about painless syncing.", "syncing", "power supply")
    assert not aspect_replacement_compatible("I bought a laptop.", "laptop", "aero")
    assert not aspect_replacement_compatible("The good battery life is useful.", "battery life", "software")

    assert aspect_replacement_compatible("I upgraded the memory to four gigabytes.", "memory", "ram")
    assert aspect_replacement_compatible("The sales associate was knowledgeable.", "sales associate", "customer support")
    assert aspect_replacement_compatible("The good battery life is useful.", "battery life", "screen")


def test_augmentation_requests_skip_incompatible_aspect_replacements():
    source_rows = [
        {"text": "The battery life is long.", "label": "<pos> battery life <opinion> long"}
    ]
    pseudo_rows = [
        {
            "id": "bad",
            "text": "I charge it at night because of the good battery life.",
            "label": "<pos> battery life <opinion> good",
            "sample_weight": 0.65,
        },
        {
            "id": "good",
            "text": "Great laptop that offers features.",
            "label": "<pos> laptop <opinion> great",
            "sample_weight": 0.65,
        },
    ]
    memory = {
        "target_aspects": ["desktop", "keyboard"],
        "candidate_opinions_by_sentiment": {"pos": ["great"]},
    }

    requests = build_augmentation_requests(
        source_rows,
        pseudo_rows,
        per_row=1,
        seed=3,
        prompt_style="masked_mutual",
        domain_memory=memory,
        channel_mode="aspect",
    )

    assert requests
    assert all(req["new_triplet"][0] != "desktop" for req in requests)
    assert any(req["new_triplet"][0] == "keyboard" for req in requests)

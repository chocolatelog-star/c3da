from argparse import Namespace

from t5_aste_pipeline import build_memory_analysis


def test_memory_analysis_top_specific_only_uses_specific_aspects():
    source_memory = {
        "aspects": [],
        "triplets": [],
        "opinions_by_sentiment": {},
    }
    target_memory = {
        "aspects": ["battery life", "trackpad"],
        "core_target_aspects": ["battery life"],
        "specific_target_aspects": ["trackpad"],
        "triplets": [],
        "opinions_by_sentiment": {},
        "rejected_aspects_by_reason": {},
        "aspect_stats": {
            "battery life": {
                "target_tf": 10,
                "target_df": 10,
                "source_tf": 0,
                "source_df": 0,
                "mean_pseudo_weight": 0.65,
                "in_text_ratio": 1.0,
                "domain_score": 5.0,
            },
            "trackpad": {
                "target_tf": 1,
                "target_df": 1,
                "source_tf": 0,
                "source_df": 0,
                "mean_pseudo_weight": 0.65,
                "in_text_ratio": 1.0,
                "domain_score": 2.0,
            },
        },
    }
    cross_memory = {
        "target_aspects": ["battery life", "trackpad"],
        "candidate_triplets": [],
        "opinions_by_sentiment": {},
    }

    analysis = build_memory_analysis(
        source_rows=[],
        pseudo_rows=[],
        source_memory=source_memory,
        target_memory=target_memory,
        cross_memory=cross_memory,
        args=Namespace(min_pseudo_weight=0.6, top_k=10),
    )

    specific_names = [row["aspect"] for row in analysis["target"]["top_specific_aspects"]]
    core_names = [row["aspect"] for row in analysis["target"]["top_core_aspects"]]

    assert specific_names == ["trackpad"]
    assert core_names == ["battery life"]

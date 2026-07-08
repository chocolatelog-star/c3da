from t5_aste_pseudo_ensemble import build_ensemble_rows


def test_build_ensemble_rows_supports_union_and_vote2():
    runs = [
        [{"id": 0, "text": "x", "gold": "<pos> a <opinion> good", "pseudo": "<pos> a <opinion> good"}],
        [{"id": 0, "text": "x", "gold": "<pos> a <opinion> good", "pseudo": "<pos> a <opinion> good ; <neg> b <opinion> bad"}],
        [{"id": 0, "text": "x", "gold": "<pos> a <opinion> good", "pseudo": "<neg> b <opinion> bad"}],
    ]

    union_rows = build_ensemble_rows(runs, mode="union")
    vote_rows = build_ensemble_rows(runs, mode="vote2")

    assert union_rows[0]["pseudo"] == "<pos> a <opinion> good ; <neg> b <opinion> bad"
    assert vote_rows[0]["pseudo"] == "<pos> a <opinion> good ; <neg> b <opinion> bad"

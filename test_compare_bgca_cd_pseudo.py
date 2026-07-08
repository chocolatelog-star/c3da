from compare_bgca_cd_pseudo import compare_pseudo_rows


def test_compare_pseudo_rows_splits_correct_sets():
    bgca_rows = [
        {
            "text": "x",
            "gold": "<pos> a <opinion> good ; <neg> b <opinion> bad",
            "pseudo_fixed": "<pos> a <opinion> good",
        }
    ]
    cd_rows = [
        {
            "text": "x",
            "gold": "<pos> a <opinion> good ; <neg> b <opinion> bad",
            "pseudo_fixed": "<neg> b <opinion> bad ; <neu> c <opinion> ok",
        }
    ]

    result = compare_pseudo_rows(bgca_rows, cd_rows, bgca_field="pseudo_fixed", cd_field="pseudo_fixed")

    assert result["summary"]["bgca_only_correct_triplets"] == 1
    assert result["summary"]["cd_only_correct_triplets"] == 1
    assert result["summary"]["both_correct_triplets"] == 0
    assert result["summary"]["bgca_false_positive_triplets"] == 0
    assert result["summary"]["cd_false_positive_triplets"] == 1

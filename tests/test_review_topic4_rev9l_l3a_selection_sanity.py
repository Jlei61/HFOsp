from scripts.review_topic4_rev9l_l3a_selection_sanity import _corrected_score


def test_corrected_score_uses_mode_specific_readable_and_ood(monkeypatch):
    captured = {}

    def fake_score(descriptors, floor, readable, ood, **kwargs):
        captured.update(readable=readable, ood=ood, kwargs=kwargs)
        return {"objective": 1.0}

    monkeypatch.setattr(
        "scripts.review_topic4_rev9l_l3a_selection_sanity.score_candidate",
        fake_score,
    )
    row = {
        "mode_descriptors": {"modes": {}},
        "geometry": {
            "component_2": {"curve_usable_fraction": 0.7, "ood_fraction": 0.2},
            "component_1": {"curve_usable_fraction": 0.8, "ood_fraction": 0.1},
        },
    }
    base = {
        "primary_mapping": {
            "mode_A_source": "component_2", "mode_B_source": "component_1"},
        "objective": {
            "readable_fraction_penalty_weight": 2.0,
            "weakest_mode_lse_tau": 0.25,
            "ood_weight": 0.1,
        },
    }
    assert _corrected_score(row, {"floor": {}}, base) == {"objective": 1.0}
    assert captured["readable"] == {"A": 0.7, "B": 0.8}
    assert captured["ood"] == {"A": 0.2, "B": 0.1}

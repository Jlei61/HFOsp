from scripts import accept_topic5_event_innovation_v3_0 as accept


def route(kind, medians, primary_p=0.01):
    names = (
        ("propagation_gain", "true_minus_matched", "future_minus_past")
        if kind == "local"
        else ("cumulative_gain", "true_minus_matched", "alignment")
    )
    return {
        "route": kind,
        "cohort_inference": {
            name: {
                "n": 10,
                "median": value,
                "wilcoxon_two_sided_p": primary_p if index == 0 else 1.0,
            }
            for index, (name, value) in enumerate(zip(names, medians))
        },
    }


def test_level2_requires_one_complete_positive_route():
    v27 = {
        "scientific_adjudication": {
            "status": "ACCEPTED_REPAIR_ONLY_STATE_TRACKING_FINAL"
        }
    }
    level, routes = accept.assign_evidence_level(
        route("local", [1.0, 2.0, 3.0]),
        route("cumulative", [1.0, -1.0, 1.0]),
        v27,
    )
    assert level == 2
    assert routes == ["goal2_local"]


def test_incomplete_positive_pattern_falls_back_to_v27_level1():
    v27 = {
        "scientific_adjudication": {
            "status": "ACCEPTED_REPAIR_ONLY_STATE_TRACKING_FINAL"
        }
    }
    level, routes = accept.assign_evidence_level(
        route("local", [1.0, -1.0, 1.0]),
        route("cumulative", [-1.0, 1.0, 1.0]),
        v27,
    )
    assert level == 1
    assert routes == []


def test_positive_but_unsupported_primary_gain_falls_back_to_level1():
    v27 = {
        "scientific_adjudication": {
            "status": "ACCEPTED_REPAIR_ONLY_STATE_TRACKING_FINAL"
        }
    }
    level, routes = accept.assign_evidence_level(
        route("local", [1.0, 2.0, 3.0], primary_p=0.2),
        route("cumulative", [-1.0, -1.0, -1.0]),
        v27,
    )
    assert level == 1
    assert routes == []


def test_no_v27_support_falls_back_to_level0():
    level, routes = accept.assign_evidence_level(
        route("local", [-1.0, -1.0, -1.0]),
        route("cumulative", [-1.0, -1.0, -1.0]),
        {},
    )
    assert level == 0
    assert routes == []

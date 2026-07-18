import pandas as pd
import pytest

from scripts.run_topic5_gradient_multiband_original_f2_fixed_sigma_shared_only import (
    filter_shared_inputs,
    shared_subject_ids,
)
from scripts.run_topic5_gradient_multiband_original_f2_fixed_sigma_cohort_matched import (
    cohort_matched_subject_ids,
    filter_cohort_matched_inputs,
    selector_exclusion_reasons,
    validate_exact_cohort_difference,
    validate_source_summary,
)


def _routing():
    return pd.DataFrame([
        {"subject": "s1", "field_plane": "shared", "smoothing_policy": "subject_fixed", "sigma_common": 0.2},
        {"subject": "s2", "field_plane": "shared", "smoothing_policy": "subject_fixed", "sigma_common": 0.3},
        {"subject": "s3", "field_plane": "own_fallback", "smoothing_policy": "subject_fixed", "sigma_common": 0.4},
    ])


def test_shared_subject_selection_is_route_based():
    assert shared_subject_ids(_routing(), expected_n=2) == ["s1", "s2"]


def test_shared_selector_rejects_per_model_source():
    routing = _routing()
    routing.loc[0, "smoothing_policy"] = "frozen_per_model"
    with pytest.raises(ValueError, match="fixed-sigma"):
        shared_subject_ids(routing, expected_n=2)


def test_filter_shared_inputs_removes_own_rows_and_keeps_full_draws(monkeypatch):
    import scripts.run_topic5_gradient_multiband_original_f2_fixed_sigma_shared_only as module
    monkeypatch.setattr(module, "MIN_PERM", 2)
    subjects = pd.DataFrame([
        {"subject": sid, "band": band, "field_plane": plane}
        for sid, plane in (("s1", "shared"), ("s2", "shared"), ("s3", "own_fallback"))
        for band in ("a", "b")
    ])
    perm = pd.DataFrame([
        {"subject": sid, "band": band, "perm_id": draw, "perm_subject_median": 0.1}
        for sid in ("s1", "s2", "s3") for band in ("a", "b") for draw in (-1, 0, 1)
    ])
    sub, prm, route = filter_shared_inputs(subjects, perm, _routing(), expected_n=2)
    assert set(sub.subject) == set(prm.subject) == set(route.subject) == {"s1", "s2"}
    assert len(prm) == 2 * 2 * 3


def test_cohort_matched_selector_uses_group_and_keeps_e139_e1146():
    selector = pd.DataFrame([
        {"subject": "epilepsiae_139", "group_id": "all_phenotype_matched"},
        {"subject": "epilepsiae_1146", "group_id": "all_phenotype_matched"},
        {"subject": "epilepsiae_1077", "group_id": "all_phenotype_matched"},
        {"subject": "epilepsiae_583", "group_id": "strict_broadband"},
    ])
    selected = cohort_matched_subject_ids(
        selector,
        expected_n=3,
        required_included=("epilepsiae_139", "epilepsiae_1146"),
    )
    assert selected == ["epilepsiae_1077", "epilepsiae_1146", "epilepsiae_139"]
    assert "epilepsiae_583" not in selected


def test_cohort_matched_filter_keeps_complete_subject_band_draw_contract():
    bands = ("a", "b")
    canonical = ("s1", "s2", "s3")
    matched = ("s1", "s2")
    subjects = pd.DataFrame([
        {
            "subject": sid,
            "band": band,
            "field_plane": "shared" if sid == "s1" else "own_fallback",
            "delta": 0.1,
        }
        for sid in canonical for band in bands
    ])
    perm = pd.DataFrame([
        {
            "subject": sid,
            "band": band,
            "perm_id": draw,
            "perm_subject_median": 0.1,
        }
        for sid in canonical for band in bands for draw in (-1, 0, 1)
    ])
    routing = pd.DataFrame([
        {
            "subject": sid,
            "smoothing_policy": "subject_fixed",
            "sigma_common": 0.2,
        }
        for sid in canonical
    ])
    sub, prm, route = filter_cohort_matched_inputs(
        subjects,
        perm,
        routing,
        matched,
        bands=bands,
        expected_n=2,
        n_perm=2,
    )
    assert set(sub.subject) == set(prm.subject) == set(route.subject) == set(matched)
    assert len(sub) == 2 * 2
    assert len(prm) == 2 * 2 * 3


def test_cohort_matched_contract_requires_fixed_sigma_and_exact_difference():
    summary = {
        "contract": "topic5_gradient_shared_else_own_original_f2_subject_fixed_sigma_v1",
        "counts": {"analysis_subjects": 3},
        "smoothing": {"policy": "subject_fixed"},
    }
    validate_source_summary(summary, expected_n=3)
    assert validate_exact_cohort_difference(
        ["s1", "s2", "e583"],
        ["s1", "s2"],
        expected_source_n=3,
        expected_matched_n=2,
        expected_exclusions=("e583",),
    ) == ["e583"]

    bad = {**summary, "smoothing": {"policy": "frozen_per_model"}}
    with pytest.raises(ValueError, match="subject-fixed-sigma"):
        validate_source_summary(bad, expected_n=3)
    with pytest.raises(ValueError, match="exclusions drifted"):
        validate_exact_cohort_difference(
            ["s1", "s2", "e583"],
            ["s1", "s2"],
            expected_source_n=3,
            expected_matched_n=2,
            expected_exclusions=("wrong_subject",),
        )


def test_cohort_matched_selector_exclusion_reason_is_frozen():
    drops = pd.DataFrame([
        {
            "subject": "epilepsiae_583",
            "group_id": "all_phenotype_matched",
            "drop_reason": "no_strict_broadband_or_gamma_nonbroadband_event",
        },
        {
            "subject": "yuquan_zhangkexuan",
            "group_id": "all_phenotype_matched",
            "drop_reason": "no_strict_broadband_or_gamma_nonbroadband_event",
        },
    ])
    reasons = selector_exclusion_reasons(
        drops, ("epilepsiae_583", "yuquan_zhangkexuan")
    )
    assert set(reasons) == {"epilepsiae_583", "yuquan_zhangkexuan"}
    drops.loc[0, "drop_reason"] = "geometry_failure"
    with pytest.raises(ValueError, match="unexpected selector exclusion reasons"):
        selector_exclusion_reasons(
            drops, ("epilepsiae_583", "yuquan_zhangkexuan")
        )

import numpy as np
import pandas as pd

from scripts.run_topic5_gradient_multiband_original_f2 import (
    EXPECTED_WINDOWS,
    build_original_null_groups,
    fold_windows_to_subject,
    permute_by_groups,
    rebuild_subject_fixed_scorers,
    select_original_f2_rows,
)
from src.topic5_v2_band_scan import spatial_constrained_permute


def test_original_f2_selector_keeps_all_five_windows():
    rows = []
    for start, end in EXPECTED_WINDOWS:
        rows.append({
            "subject": "139", "seizure": 1, "band": "gamma_LVFA",
            "feature": "raw", "used_fixed_mask": True,
            "win_start_rel": start, "win_end_rel": end,
            "ictal_fraction": 0.5,
        })
    rows.append({**rows[0], "feature": "common_resid"})
    out = select_original_f2_rows(pd.DataFrame(rows), ["gamma_LVFA"])
    assert len(out) == 5
    assert tuple(map(tuple, out[["win_start_rel", "win_end_rel"]].to_numpy())) == EXPECTED_WINDOWS


def test_original_fold_is_window_then_seizure_then_subject():
    scores = np.array([1.0, 3.0, 10.0, 20.0, 30.0])
    seizures = [0, 0, 1, 1, 1]
    # seizure medians are 2 and 20; subject median is 11.
    assert fold_windows_to_subject(scores, seizures) == 11.0
    draws = np.vstack([scores, scores + 2.0])
    np.testing.assert_allclose(fold_windows_to_subject(draws, seizures), [11.0, 13.0])


def test_group_builder_matches_original_spatial_permutation():
    names = ["A1", "A2", "A3", "A4", "B1", "B2", "C1", "C2"]
    values = np.arange(len(names), dtype=float)
    coords = {name: np.array([i, (i % 2) * 0.2]) for i, name in enumerate(names)}
    groups, strength = build_original_null_groups(
        names, np.ones(len(names), bool), names, coords, min_group=4,
    )
    assert strength == "distance_bin_fallback"
    seed = 91
    got = permute_by_groups(values, groups, np.random.default_rng(seed))
    expected, meta = spatial_constrained_permute(
        names,
        dict(zip(names, values)),
        {name: parse for name, parse in zip(names, ["A"] * 4 + ["B"] * 2 + ["C"] * 2)},
        coords,
        np.random.default_rng(seed),
        "within_shaft",
        4,
    )
    np.testing.assert_allclose(got, [expected[name] for name in names])
    assert meta["spatial_null_strength"] == strength


def test_singleton_remainder_is_weak_and_not_permuted():
    names = ["A1", "A2", "A3", "A4", "B1"]
    coords = {name: np.array([i, 0.0]) for i, name in enumerate(names)}
    groups, strength = build_original_null_groups(
        names, np.ones(5, bool), names, coords, min_group=4,
    )
    assert strength == "subject_wide_weak"
    got = permute_by_groups(np.arange(5.0), groups, np.random.default_rng(2))
    assert got[-1] == 4.0


def _field_record_with_models(include_shared=False):
    from src.topic5_template_axis_field import make_field_scorer
    names = [f"A{i}" for i in range(1, 7)]
    points_a = np.column_stack((np.linspace(0, 1, 6), np.zeros(6)))
    points_b = np.column_stack((np.linspace(0, 1, 6), np.linspace(-0.2, 0.2, 6)))
    values_a = np.linspace(1, -1, 6)
    values_b = values_a[::-1]
    support_a = np.linspace(0.5, 1.0, 6)
    support_b = support_a[::-1]
    planes = {
        "own_a": {"points": points_a.tolist(), "sigma": 0.4},
        "own_b": {"points": points_b.tolist(), "sigma": 0.1},
    }
    models = {
        "own_a": make_field_scorer(values_a, points_a, support_a, 0.4),
        "own_b": make_field_scorer(values_b, points_b, support_b, 0.1),
    }
    if include_shared:
        planes["shared"] = {"points": points_a.tolist(), "sigma": 0.25}
        models["shared_a"] = make_field_scorer(values_a, points_a, support_a, 0.25)
        models["shared_b"] = make_field_scorer(values_b, points_a, support_b, 0.25)
    field = {
        "status": "ok", "contact_order": names,
        "coords": np.column_stack((points_a, np.zeros(6))).tolist(),
        "shafts": ["A"] * 6,
        "rank_a": list(range(6)), "rank_b": list(range(5, -1, -1)),
        "earliness_a": values_a.tolist(), "earliness_b": values_b.tolist(),
        "support_a": support_a.tolist(), "support_b": support_b.tolist(),
        "n_contacts": 6, "planes": planes,
    }
    return field, models


def test_subject_fixed_own_scorers_reuse_own_a_sigma_for_b():
    field, frozen = _field_record_with_models()
    scorers, plane, key, meta = rebuild_subject_fixed_scorers(field, frozen)
    assert (plane, key) == ("own_fallback", "own_maxab")
    assert scorers["own_a"]["sigma"] == scorers["own_b"]["sigma"] == 0.4
    assert meta["sigma_original"] == {"own_a": 0.4, "own_b": 0.1}


def test_subject_fixed_shared_scorers_keep_shared_sigma():
    field, frozen = _field_record_with_models(include_shared=True)
    scorers, plane, key, meta = rebuild_subject_fixed_scorers(field, frozen)
    assert (plane, key) == ("shared", "shared_maxab")
    assert scorers["shared_a"]["sigma"] == scorers["shared_b"]["sigma"] == 0.25
    assert meta["sigma_common"] == 0.25

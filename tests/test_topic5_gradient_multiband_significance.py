import numpy as np
import pandas as pd

from src.propagation_skeleton_geometry import parse_shaft
from src.topic5_v2_band_scan import spatial_constrained_permute
from scripts.run_topic5_gradient_multiband_significance import (
    band_window_activation,
    build_cohort_table,
    load_primary_band_contract,
    make_original_spatial_permutations,
    plot_gradient_multiband_figure,
    select_reference_window_rows,
)


def test_primary_band_contract_matches_original_f2_family():
    rows = load_primary_band_contract()
    assert [(r["band"], r["low_hz"], r["high_hz"], r["interval"]) for r in rows] == [
        ("delta_HYP_slow", 1.0, 4.0, "half_open"),
        ("theta_preictal_PAC", 4.0, 8.0, "half_open"),
        ("alpha_sharp_leq13", 8.0, 13.0, "half_open"),
        ("beta_LVFA_low", 13.0, 30.0, "half_open"),
        ("gamma_LVFA", 30.0, 80.0, "half_open"),
        ("hg_low_ripple", 80.0, 150.0, "half_open"),
        ("ripple_high", 150.0, 250.0, "half_open"),
    ]


def test_reference_window_filter_keeps_only_original_exact_zero_to_ten_rows():
    frame = pd.DataFrame([
        {"subject": "1", "seizure": 0, "band": "b1", "feature": "raw",
         "used_fixed_mask": True, "win_start_rel": 0, "win_end_rel": 10,
         "ictal_fraction": 1.0},
        {"subject": "1", "seizure": 0, "band": "b1", "feature": "raw",
         "used_fixed_mask": True, "win_start_rel": 5, "win_end_rel": 15,
         "ictal_fraction": 1.0},
        {"subject": "1", "seizure": 0, "band": "b2", "feature": "raw",
         "used_fixed_mask": False, "win_start_rel": 0, "win_end_rel": 10,
         "ictal_fraction": 1.0},
    ])
    out = select_reference_window_rows(frame, ["b1", "b2"])
    assert out[["subject", "seizure", "band"]].to_dict("records") == [
        {"subject": "1", "seizure": 0, "band": "b1"}
    ]


def test_band_activation_uses_only_onset_zero_to_ten_seconds():
    cache = {
        "gamma_LVFA__zt__2": np.asarray([
            [100.0, 1.0, 3.0, 5.0, 200.0],
            [100.0, 2.0, 4.0, 6.0, 200.0],
        ]),
        "gamma_LVFA__relt__2": np.asarray([-0.1, 0.0, 5.0, 10.0, 10.1]),
    }
    np.testing.assert_allclose(
        band_window_activation(cache, "gamma_LVFA", 2), [3.0, 4.0]
    )


def test_original_spatial_permutations_preserve_within_shaft_groups():
    names = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    coords = np.column_stack([
        np.arange(8, dtype=float), np.zeros(8), np.zeros(8)
    ])
    permutations, strength = make_original_spatial_permutations(
        names, coords, np.ones(8, bool), 20, 3, min_group=4
    )
    assert strength == "within_shaft_strong"
    assert permutations.shape == (20, 8)
    assert all(set(row[:4]) == set(range(4)) for row in permutations)
    assert all(set(row[4:]) == set(range(4, 8)) for row in permutations)


def test_original_spatial_permutations_report_subject_wide_remainder():
    names = ["A1", "A2", "B1", "B2", "C1", "C2"]
    coords = np.column_stack([
        np.arange(6, dtype=float), np.zeros(6), np.zeros(6)
    ])
    permutations, strength = make_original_spatial_permutations(
        names, coords, np.ones(6, bool), 10, 5, min_group=4
    )
    assert strength == "subject_wide_weak"
    assert permutations.shape == (10, 6)


def test_vectorized_spatial_permutations_match_original_producer_draws():
    names = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2", "C3"]
    coords = np.column_stack([
        np.arange(10, dtype=float), np.zeros(10), np.zeros(10)
    ])
    seed = 17
    observed, strength = make_original_spatial_permutations(
        names, coords, np.ones(10, bool), 12, seed, min_group=4
    )
    rng = np.random.default_rng(seed)
    values = {name: float(i) for i, name in enumerate(names)}
    shafts = {name: parse_shaft(name)[0] for name in names}
    xyz = {name: coords[i] for i, name in enumerate(names)}
    expected = []
    strengths = []
    for _ in range(12):
        permuted, info = spatial_constrained_permute(
            names, values, shafts, xyz, rng, "within_shaft", 4
        )
        expected.append([int(permuted[name]) for name in names])
        strengths.append(info["spatial_null_strength"])
    np.testing.assert_array_equal(observed, np.asarray(expected, int))
    assert strength == "subject_wide_weak"
    assert set(strengths) == {strength}


def test_cohort_table_uses_seven_band_family_max_statistic():
    contract = [
        {"band": "a", "low_hz": 1.0, "high_hz": 2.0,
         "interval": "half_open", "label": "A"},
        {"band": "b", "low_hz": 2.0, "high_hz": 3.0,
         "interval": "half_open", "label": "B"},
    ]
    subject_rows = []
    perm_rows = []
    for subject, shift in (("s1", 0.4), ("s2", 0.3)):
        for band, offset in (("a", 0.0), ("b", 0.1)):
            observed = 0.8 + offset
            null = np.asarray([0.2, 0.3, 0.4]) + offset
            subject_rows.append({
                "dataset": "epilepsiae", "subject": subject, "band": band,
                "field_plane": "shared" if subject == "s1" else "own_fallback",
                "spatial_null_strength": "within_shaft_strong",
                "delta": shift, "n_seizures": 1,
            })
            base = {"subject": subject, "feature": "raw_gradient",
                    "null_type": "spatial", "band": band}
            perm_rows.append({**base, "perm_id": -1,
                              "perm_subject_median": observed})
            perm_rows.extend({**base, "perm_id": i,
                              "perm_subject_median": value}
                             for i, value in enumerate(null))
    out = build_cohort_table(
        pd.DataFrame(subject_rows), pd.DataFrame(perm_rows), contract
    )
    assert out["band"].tolist() == ["a", "b"]
    assert out["n_subjects"].tolist() == [2, 2]
    assert out["n_shared_subjects"].tolist() == [1, 1]
    assert np.isfinite(out["max_over_bands_p"]).all()


def test_gradient_multiband_plot_writes_png_and_pdf(tmp_path):
    contract = [
        {"band": "a", "low_hz": 1.0, "high_hz": 2.0,
         "interval": "half_open", "label": "A\n1–2"},
        {"band": "b", "low_hz": 2.0, "high_hz": 3.0,
         "interval": "half_open", "label": "B\n2–3"},
    ]
    subjects = pd.DataFrame([
        {"subject": subject, "band": band, "delta": value}
        for subject, value in (("s1", 0.1), ("s2", -0.02), ("s3", 0.04))
        for band in ("a", "b")
    ])
    cohort = pd.DataFrame([
        {"band": "a", "passes_fwer_0p05": True,
         "cohort_perm_delta_spatial": 0.04,
         "max_over_bands_p": 0.02, "n_subjects": 3},
        {"band": "b", "passes_fwer_0p05": False,
         "cohort_perm_delta_spatial": 0.02,
         "max_over_bands_p": 0.40, "n_subjects": 3},
    ])
    png = tmp_path / "figure.png"
    plot_gradient_multiband_figure(subjects, cohort, contract, png, seed=1)
    assert png.exists()
    assert png.with_suffix(".pdf").exists()

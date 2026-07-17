import copy
import hashlib
import json

import numpy as np
import pytest

from src.topic5_template_axis_field import (
    build_interictal_template_field_record,
    make_field_scorer,
    score_scorer_bundle,
    scorers_from_interictal_record,
)
from src.topic5_tspectral_field_concordance import (
    aggregate_complete_windows,
    annotation_provenance,
    apply_fixed_permutations,
    distal_baseline_robust_z,
    eligibility_drop_reason,
    exact_name_align_matrix,
    fold_seizure_null_draws,
    fixed_window_sign_flip_maxt,
    independent_label_permutation_maxt,
    make_complete_window_grid,
    make_contact_permutations,
    phenotype_selector_sets,
    score_permutation_matrix,
    sign_flip_cluster_maxt,
    subject_first_fold,
    tspectral_reference_for_raw_eeg,
    tspectral_zeroed_times,
)
from scripts.run_topic5_tspectral_field_concordance import (
    _cohort_fixed_field_statistics,
    _fixed_field_subject_rows,
    _phenotype_matched_cohort_statistics,
    _phenotype_matched_fixed_subject_rows,
    _phenotype_matched_relation_statistics,
)
from scripts.run_topic5_unstratified_channel_scaffold_diagnostic import (
    _checkpoint_valid,
    _high_quality_collinear_mask,
    _paired_summary,
    _strict_reversed_mask,
)
from scripts.run_topic5_eeg_onset_shared_field_concordance import (
    available_bands_for_fs,
    is_strict_reversed,
    select_shared_scorers,
    true_eeg_relative_times,
    _eeg_offset_from_inventory,
)
from scripts.run_topic5_clinical_onset_shared_field_concordance import (
    clinical_relative_times,
    clinical_stratum,
    highest_valid_broadband_upper,
)
from scripts.run_topic5_old_cache_hybrid_field_comparison import (
    select_shared_else_own_scorers,
)


FIXED_ORDER = ("distal", "pre20", "pre10", "post10", "post20", "late20_30")


def _fixed_band_result(observed, null_level):
    boundaries = {
        "distal": (-120.0, -90.0), "pre20": (-20.0, 0.0),
        "pre10": (-10.0, 0.0), "post10": (0.0, 10.0),
        "post20": (0.0, 20.0), "late20_30": (20.0, 30.0),
    }
    rows = []
    for j, name in enumerate(FIXED_ORDER):
        start, end = boundaries[name]
        rows.append({
            "window_scale": "fixed", "window_region": "fixed", "fixed_window": name,
            "window_start_sec": start, "window_end_sec": end,
            "window_center_sec": (start + end) / 2,
            "n_target_contacts": 8, "n_matched": 8, "n_finite": 8,
            "spatial_median_delta_energy": float(j),
            "own_a_abs": observed + 0.01 * j, "own_b_abs": observed - 0.05,
            "own_a_signed": observed + 0.01 * j, "own_b_signed": -(observed - 0.05),
            "own_maxab": observed + 0.01 * j,
            "shared_a_abs": np.nan, "shared_b_abs": np.nan,
            "shared_a_signed": np.nan, "shared_b_signed": np.nan,
            "shared_maxab": np.nan,
        })
    null = np.full((9, len(FIXED_ORDER)), float(null_level))
    return {
        "rows": rows,
        "fixed_null": {
            "window_names": list(FIXED_ORDER),
            "within_shaft": {"own_maxab": null},
            "all_contact": {"own_maxab": null + 0.02},
        },
    }


def _record(seed=0):
    rng = np.random.default_rng(seed)
    n = 18
    coords = rng.normal(size=(n, 3))
    names = [f"S{i // 6 + 1}{i % 6 + 1}" for i in range(n)]
    shafts = [f"S{i // 6 + 1}" for i in range(n)]
    a = coords @ np.array([0.8, -0.3, 0.2])
    b = coords @ np.array([-0.7, 0.4, 0.1])
    return build_interictal_template_field_record(
        subject_id="test", dataset="test", subject="test", stable_k=2,
        names=names, coords=coords, rank_ta=a, rank_tb=b, shafts=shafts,
        support_ta=np.ones(n), support_tb=np.linspace(0.4, 1.0, n),
        support_source="test", n_axis_boot=10, n_pair_boot=10, seed=seed,
    )


def test_tspectral_rezero_is_exact():
    np.testing.assert_allclose(tspectral_zeroed_times([-3, 1, 7], 1), [-4, 0, 6])


def test_raw_eeg_rezero_uses_eeg_not_clinical_cache_coordinate():
    event = {
        "t_spectral_rel_eeg_sec": 1.75,
        "t_spectral_rel_cache_zero_sec": -5.25,
    }
    reference = tspectral_reference_for_raw_eeg(event)
    assert reference == 1.75
    assert tspectral_zeroed_times([1.75], reference)[0] == 0.0


def test_yuquan_provenance_never_fabricates_clinical_onset():
    out = annotation_provenance("yuquan", {
        "clinical_onset_rel_tspectral_sec": 9.0,
        "eeg_onset_rel_tspectral_sec": -4.0,
    })
    assert out["annotation_mode"] == "eeg_only"
    assert out["clinical_onset_available"] is False
    assert out["clinical_onset_rel_tspectral_sec"] is None
    assert out["eeg_onset_rel_tspectral_sec"] == -4.0


def test_broadband_and_gamma_nonbroadband_selectors_are_disjoint():
    meta = {
        "seizure_idxs": [1, 2, 3],
        "early_spectral_phenotype_selectors": {
            "accepted_tspectral_labeled_idxs": [1, 2],
            "accepted_tspectral_strict_broadband_idxs": [1],
            "accepted_tspectral_simple_phenotype_idxs": {"gamma_nonbroadband": [2]},
        },
    }
    out = phenotype_selector_sets(meta)
    assert out["broadband_1_150"] == {1}
    assert out["gamma_nonbroadband"] == {2}
    bad = copy.deepcopy(meta)
    bad["early_spectral_phenotype_selectors"]["accepted_tspectral_simple_phenotype_idxs"]["gamma_nonbroadband"] = [1]
    with pytest.raises(ValueError, match="overlap"):
        phenotype_selector_sets(bad)


def test_fingerprint_drift_fails_closed():
    record = _record(1)
    scorers_from_interictal_record(record)
    tampered = copy.deepcopy(record)
    tampered["interictal_field"]["support_a"][0] += 0.01
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        scorers_from_interictal_record(tampered)


def test_activation_matrix_joins_by_contact_name_not_array_position():
    record = _record(2)
    target = list(record["interictal_field"]["contact_order"])
    names = target[::-1]
    values = np.vstack((np.arange(len(names))[::-1], np.arange(len(names))[::-1] + 100)).T
    aligned = exact_name_align_matrix(record, names, values)
    np.testing.assert_allclose(aligned["values"][:, 0], np.arange(len(target)))
    np.testing.assert_allclose(aligned["values"][:, 1], np.arange(len(target)) + 100)


def test_batch_null_scorer_matches_rowwise_and_reselects_mirror_and_maxab():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(12, 2))
    scorers = {
        "own_a": make_field_scorer(pts[:, 0] + 0.8 * pts[:, 1], pts, np.ones(12), 0.5),
        "own_b": make_field_scorer(-pts[:, 0] + 0.4 * pts[:, 1], pts, np.ones(12), 0.5),
    }
    values = rng.normal(size=(4, 12))
    perms = make_contact_permutations([f"S{i // 4}{i}" for i in range(12)],
                                      np.ones(12, bool), 7, 42, mode="all_contact")
    got = score_permutation_matrix(scorers, values, perms, chunk_draws=3)
    permuted = apply_fixed_permutations(values, perms)
    for draw in range(len(perms)):
        for window in range(len(values)):
            expected = score_scorer_bundle(scorers, permuted[draw, window])
            assert np.isclose(got["own_a_abs"][draw, window], expected["own_a_abs"])
            assert np.isclose(got["own_b_abs"][draw, window], expected["own_b_abs"])
            assert np.isclose(got["own_maxab"][draw, window], expected["own_maxab"])


def test_one_null_draw_uses_one_contact_permutation_across_all_timepoints():
    values = np.arange(24).reshape(4, 6)
    perms = make_contact_permutations(["A1", "A2", "A3", "B1", "B2", "B3"],
                                      np.ones(6, bool), 5, 9, mode="all_contact")
    shuffled = apply_fixed_permutations(values, perms)
    for draw in range(len(perms)):
        np.testing.assert_array_equal(shuffled[draw] - shuffled[draw, :1],
                                      values[:, perms[draw]] - values[:1, perms[draw]])


def test_subject_first_fold_does_not_pool_seizures_as_cohort_units():
    rows = [
        {"subject": "s1", "seizure_idx": 0, "time": 0, "D": 0.0},
        {"subject": "s1", "seizure_idx": 1, "time": 0, "D": 10.0},
        {"subject": "s2", "seizure_idx": 0, "time": 0, "D": 1.0},
    ]
    folded = subject_first_fold(rows, "D", ["time"])
    assert len(folded) == 2
    assert {r["subject"]: r["D"] for r in folded} == {"s1": 5.0, "s2": 1.0}
    null = fold_seizure_null_draws([np.zeros((3, 2)), np.full((3, 2), 10.0)])
    np.testing.assert_allclose(null, 5.0)


def test_fixed_window_null_is_folded_seizure_to_subject_before_q95():
    names = ["pre10", "post10"]
    within = [
        np.array([[0.10, 0.20], [0.20, 0.30], [0.30, 0.40]]),
        np.array([[0.20, 0.40], [0.40, 0.50], [0.60, 0.60]]),
    ]
    events = []
    for seizure_idx, offset in enumerate((0.0, 0.2)):
        fixed_rows = []
        for j, name in enumerate(names):
            fixed_rows.append({
                "window_scale": "fixed", "window_region": "fixed", "fixed_window": name,
                "window_start_sec": -10.0 if name == "pre10" else 0.0,
                "window_end_sec": 0.0 if name == "pre10" else 10.0,
                "window_center_sec": -5.0 if name == "pre10" else 5.0,
                "n_target_contacts": 6, "n_matched": 6, "n_finite": 6,
                "spatial_median_delta_energy": float(j + offset),
                "own_a_abs": 0.50 + 0.10 * j + offset,
                "own_b_abs": 0.30 + 0.05 * j,
                "own_a_signed": 0.50 + 0.10 * j + offset,
                "own_b_signed": -0.30 - 0.05 * j,
                "own_maxab": 0.50 + 0.10 * j + offset,
                "shared_a_abs": np.nan, "shared_b_abs": np.nan,
                "shared_a_signed": np.nan, "shared_b_signed": np.nan,
                "shared_maxab": np.nan,
            })
        result = {
            "rows": fixed_rows,
            "fixed_null": {
                "window_names": names,
                "within_shaft": {"own_maxab": within[seizure_idx]},
                "all_contact": {"own_maxab": within[seizure_idx] + 0.01},
            },
            "fixed_activation": {
                "pre10": np.arange(6, dtype=float) + offset,
                "post10": np.arange(6, dtype=float) + 1.0 + offset,
            },
        }
        events.append({"phenotype": "broadband_1_150", "seizure_idx": seizure_idx,
                       "bands": {"broadband_1_150": result}})
    quality = {"field_ready": True, "geometry_2d_supported": True,
               "strict_stability_pass": True, "axis_relation": "same",
               "shared_field_available": False}
    rows, display = _fixed_field_subject_rows(
        "epilepsiae_test", "epilepsiae", quality, events
    )
    pre = next(row for row in rows if row["fixed_window"] == "pre10")
    folded = fold_seizure_null_draws(within)[:, 0]
    assert np.isclose(pre["own_maxab"], 0.60)
    assert np.isclose(pre["own_within_shaft_null_p95_folded"], np.percentile(folded, 95))
    assert np.isclose(pre["own_within_shaft_null_median_folded"], np.median(folded))
    assert pre["own_within_shaft_exceeds_p95"] is True
    np.testing.assert_allclose(
        display["broadband_1_150"]["windows"]["pre10"], np.arange(6) + 0.1
    )


def test_distal_tspectral_coordinates_do_not_split_the_cohort_window():
    import pandas as pd

    rows = []
    for subject, start, delta in (("s1", -121.0, 0.1), ("s2", -116.0, 0.3)):
        rows.append({
            "dataset": "epilepsiae", "subject": subject,
            "phenotype": "broadband_1_150", "band": "broadband_1_150",
            "band_label": "Broadband 1–150 Hz", "band_role": "primary",
            "fixed_window": "distal", "window_start_sec": start,
            "window_end_sec": start + 30.0, "window_center_sec": start + 15.0,
            "n_seizures": 1, "own_within_shaft_delta_null_median": delta,
            "own_all_contact_delta_null_median": delta + 0.01,
            "shared_within_shaft_delta_null_median": np.nan,
            "shared_all_contact_delta_null_median": np.nan,
        })
    out = _cohort_fixed_field_statistics(
        pd.DataFrame(rows), n_boot=100, n_perm=100, seed=3
    )
    own = out[(out.dataset_stratum == "epilepsiae") &
              (out.field_plane == "own") &
              (out.null_type == "within_shaft")]
    assert len(own) == 1
    assert int(own.iloc[0].n_subjects) == 2
    assert np.isclose(float(own.iloc[0].window_start_sec), np.median([-121.0, -116.0]))


def test_phenotype_matched_pool_uses_one_selected_readout_per_seizure():
    events = [
        {"seizure_idx": 1, "phenotype": "broadband_1_150",
         "bands": {"broadband_1_150": _fixed_band_result(0.20, 0.10)}},
        {"seizure_idx": 2, "phenotype": "gamma_nonbroadband",
         "bands": {
             "hfa_60_100": _fixed_band_result(0.80, 0.30),
             "gamma_30_80_sensitivity": _fixed_band_result(0.40, 0.20),
         }},
    ]
    quality = {
        "field_ready": True, "geometry_2d_supported": True,
        "strict_stability_pass": True, "axis_quality_tier": "strict_2d",
        "axis_relation": "reversed", "shared_field_available": False,
    }
    primary = _phenotype_matched_fixed_subject_rows(
        "epilepsiae_test", "epilepsiae", quality, events, readout_family="primary"
    )
    sensitivity = _phenotype_matched_fixed_subject_rows(
        "epilepsiae_test", "epilepsiae", quality, events,
        readout_family="gamma_30_80_substitution_sensitivity",
    )
    assert len(primary) == len(sensitivity) == 6
    assert {row["n_seizures"] for row in primary} == {2}
    assert {row["n_gamma_seizures"] for row in primary} == {1}
    assert set(primary[0]["selected_bands"].split(";")) == {
        "broadband_1_150", "hfa_60_100"
    }
    assert set(sensitivity[0]["selected_bands"].split(";")) == {
        "broadband_1_150", "gamma_30_80_sensitivity"
    }
    assert np.isclose(primary[0]["own_maxab"], 0.50)
    assert np.isclose(sensitivity[0]["own_maxab"], 0.30)
    assert np.isclose(primary[0]["own_within_shaft_null_median_folded"], 0.20)
    assert np.isclose(sensitivity[0]["own_within_shaft_null_median_folded"], 0.15)


def test_fixed_window_signflip_reuses_subject_sign_and_is_reproducible():
    values = np.array([
        [0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.2, 0.5],
        [0.4, 0.3, 0.6], [0.5, 0.4, 0.7], [0.6, 0.5, 0.8],
    ])
    a = fixed_window_sign_flip_maxt(values, n_perm=100, seed=17)
    b = fixed_window_sign_flip_maxt(values, n_perm=100, seed=17)
    np.testing.assert_allclose(a["raw_p"], b["raw_p"])
    np.testing.assert_allclose(a["maxt_p"], b["maxt_p"])
    assert np.all(a["maxt_p"] >= a["raw_p"])
    assert a["n_permutations"] == 2 ** len(values)


def test_relation_permutation_keeps_same_descriptive_and_geometry_fail_closed():
    import pandas as pd

    rows = []
    subjects = [
        ("e_r1", "epilepsiae", "reversed", True, 0.35),
        ("e_r2", "epilepsiae", "reversed", True, 0.30),
        ("e_d1", "epilepsiae", "different", True, 0.05),
        ("y_d2", "yuquan", "different", True, 0.00),
        ("e_s1", "epilepsiae", "same", True, 0.15),
        ("y_s2", "yuquan", "same", True, 0.10),
        ("e_bad", "epilepsiae", "reversed", False, 0.90),
    ]
    for subject, dataset, relation, geometry, base in subjects:
        for j, window in enumerate(FIXED_ORDER):
            rows.append({
                "subject": subject, "dataset": dataset, "readout_family": "primary",
                "fixed_window": window, "window_start_sec": float(j),
                "window_end_sec": float(j + 1), "window_center_sec": float(j + 0.5),
                "n_seizures": 1, "n_broadband_seizures": 1, "n_gamma_seizures": 0,
                "own_within_shaft_delta_null_median": base + 0.01 * j,
                "geometry_2d_supported": geometry,
                "axis_quality_tier": "non_strict_2d" if geometry else "geometry_unsupported",
                "axis_relation": relation,
            })
    frame = pd.DataFrame(rows)
    relation = _phenotype_matched_relation_statistics(frame, n_perm=200, seed=4)
    combined = relation[(relation.quality_scope == "geometry_2d_supported") &
                        (relation.dataset_stratum == "combined")]
    assert len(combined) == 6
    assert set(combined.n_reversed) == {2}
    assert set(combined.n_different) == {2}
    assert set(combined.n_same) == {2}
    assert np.isfinite(combined.same_median).all()
    direct = independent_label_permutation_maxt(
        frame[frame.fixed_window == "post10"][[
            "own_within_shaft_delta_null_median"
        ]].to_numpy(float),
        frame[frame.fixed_window == "post10"].axis_relation,
        "reversed", "different", n_perm=100, seed=4,
    )
    assert direct["n_group_a"] == 3  # direct call includes the unsupported row


def test_combined_phenotype_matched_cohort_includes_yuquan_as_subject_unit():
    import pandas as pd

    rows = []
    for subject, dataset, delta in (
        ("e1", "epilepsiae", 0.2), ("e2", "epilepsiae", 0.3),
        ("y1", "yuquan", 0.1), ("y2", "yuquan", 0.4),
    ):
        for j, window in enumerate(FIXED_ORDER):
            rows.append({
                "subject": subject, "dataset": dataset, "readout_family": "primary",
                "fixed_window": window, "window_start_sec": float(j),
                "window_end_sec": float(j + 1), "window_center_sec": float(j + 0.5),
                "n_seizures": 2, "n_broadband_seizures": 1, "n_gamma_seizures": 1,
                "own_within_shaft_delta_null_median": delta + 0.01 * j,
            })
    stats = _phenotype_matched_cohort_statistics(
        pd.DataFrame(rows), n_boot=50, n_perm=100, seed=9
    )
    combined = stats[(stats.dataset_stratum == "combined") &
                     (stats.field_plane == "own") &
                     (stats.null_type == "within_shaft")]
    assert len(combined) == 6
    assert set(combined.n_subjects) == {4}
    assert set(combined.n_seizures) == {8}


@pytest.mark.parametrize("width,step,n", [(2, 0.5, 117), (5, 1, 56), (10, 2, 26), (20, 2, 21)])
def test_multiscale_window_boundaries_are_complete(width, step, n):
    grid = make_complete_window_grid(-30, 30, width, step)
    assert len(grid) == n
    assert np.isclose(grid[0, 0], -30)
    assert np.isclose(grid[-1, 1], 30)
    assert np.allclose(grid[:, 2], (grid[:, 0] + grid[:, 1]) / 2)


def test_distal_baseline_delta_is_explicitly_zero_centered():
    times = np.arange(-130, 31, 0.5)
    power = np.vstack((times * 0.01 + 3, np.sin(times / 7) + 2))
    out = distal_baseline_robust_z(power, times, (-120, -90), min_frames=50)
    center = np.nanmedian(out["delta"][:, out["baseline_mask"]], axis=1)
    np.testing.assert_allclose(center, 0.0, atol=1e-12)
    assert np.nanmax(np.abs(out["baseline_z_center"])) < 1e-12


def test_fixed_seed_permutations_are_reproducible_and_within_shaft():
    names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    a = make_contact_permutations(names, np.ones(6, bool), 10, 123, mode="within_shaft")
    b = make_contact_permutations(names, np.ones(6, bool), 10, 123, mode="within_shaft")
    np.testing.assert_array_equal(a, b)
    assert np.all(a[:, :3] < 3)
    assert np.all(a[:, 3:] >= 3)


def test_missing_band_axis_and_fewer_than_six_contacts_are_explicit_drops():
    assert eligibility_drop_reason(band_available=False, field_status="ok", fingerprint_ok=True,
                                   n_finite_contacts=8) == "missing_band"
    assert eligibility_drop_reason(band_available=True, field_status="axis_not_available", fingerprint_ok=True,
                                   n_finite_contacts=8) == "missing_axis_or_field"
    assert eligibility_drop_reason(band_available=True, field_status="ok", fingerprint_ok=True,
                                   n_finite_contacts=5) == "fewer_than_6_finite_contacts"
    assert eligibility_drop_reason(band_available=True, field_status="ok", fingerprint_ok=False,
                                   n_finite_contacts=8) == "fingerprint_drift"


def test_tspectral_cache_npz_and_selector_are_read_only(tmp_path):
    path = tmp_path / "cache.npz"
    np.savez(path, seizure_idxs=np.array([1, 2]), values=np.arange(5))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    phenotype_selector_sets({
        "seizure_idxs": [1, 2],
        "early_spectral_phenotype_selectors": {
            "accepted_tspectral_labeled_idxs": [1],
            "accepted_tspectral_strict_broadband_idxs": [1],
            "accepted_tspectral_simple_phenotype_idxs": {},
        },
    })
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after


def test_cluster_sign_flips_preserve_subject_time_shape_and_are_reproducible():
    rng = np.random.default_rng(8)
    values = rng.normal(0, 0.1, size=(8, 20))
    values[:, 8:13] += 0.8
    time = np.arange(20) * 0.5
    a = sign_flip_cluster_maxt(values, time, n_perm=200, seed=11)
    b = sign_flip_cluster_maxt(values, time, n_perm=200, seed=11)
    np.testing.assert_allclose(a["maxt_p"], b["maxt_p"])
    assert len(a["t_observed"]) == values.shape[1]
    assert any(c["sign"] == "positive" for c in a["clusters"])


def test_aggregate_windows_rejects_source_boundary_extrapolation():
    times = np.arange(-29.5, 30.0, 0.5)
    values = np.ones((6, len(times)))
    windows = np.array([[-30, -28, -29], [28, 30, 29], [-31, -29, -30]])
    _, complete = aggregate_complete_windows(values, times, windows, spectral_window_sec=1.0)
    np.testing.assert_array_equal(complete, [True, True, False])


def test_unstratified_channel_scaffold_summary_is_subject_paired():
    rows = [
        {"subject_id": "s1", "data": 0.8, "null": 0.4, "n_seizures": 2},
        {"subject_id": "s2", "data": 0.7, "null": 0.5, "n_seizures": 3},
        {"subject_id": "s3", "data": 0.6, "null": 0.3, "n_seizures": 9},
    ]
    summary = _paired_summary(rows, seed=17)
    assert summary["n_subjects"] == 3
    assert summary["n_seizures"] == 14
    assert summary["n_data_gt_null"] == 3
    assert np.isclose(summary["margin_median"], 0.3)


def test_unstratified_checkpoint_fails_closed_on_contract_drift(tmp_path):
    path = tmp_path / "subject.json"
    path.write_text(json.dumps({
        "contract": "topic5_unstratified_channel_scaffold_diagnostic_v1",
        "n_perm": 1000, "seed": 11,
        "cache_sha256": "cache", "field_sha256": "field",
    }))
    assert _checkpoint_valid(
        path, n_perm=1000, seed=11, cache_hash="cache", field_hash="field"
    )
    assert not _checkpoint_valid(
        path, n_perm=999, seed=11, cache_hash="cache", field_hash="field"
    )
    assert not _checkpoint_valid(
        path, n_perm=1000, seed=11, cache_hash="changed", field_hash="field"
    )


def test_high_quality_collinear_subset_uses_preexisting_axis_fields_only():
    import pandas as pd

    frame = pd.DataFrame({
        "axis_quality_tier": ["strict_2d", "strict_2d", "strict_2d",
                              "non_strict_2d", "geometry_unsupported"],
        "axis_relation": ["same", "reversed", "different", "reversed", "same"],
        "own_maxab": [0.1, 0.9, 0.99, 0.8, 0.7],
    })
    np.testing.assert_array_equal(
        _high_quality_collinear_mask(frame).to_numpy(bool),
        [True, True, False, False, False],
    )
    np.testing.assert_array_equal(
        _strict_reversed_mask(frame).to_numpy(bool),
        [False, True, False, False, False],
    )


def test_true_eeg_onset_zero_uses_annotation_not_clinical_zero():
    # Spectral center at clinical -7.5 s is EEG +2.5 s when EEG leads by 10 s.
    times_crop = np.array([112.5, 120.0, 122.5])
    rel = true_eeg_relative_times(times_crop, pre_sec=120.0,
                                  eeg_onset_minus_clinical_sec=-10.0)
    np.testing.assert_allclose(rel, [2.5, 10.0, 12.5])


def test_missing_epilepsiae_eeg_onset_does_not_fall_back_to_clinical():
    with pytest.raises(ValueError, match="missing_eeg_onset"):
        _eeg_offset_from_inventory(
            "epilepsiae", {"clin_onset_epoch": "100", "eeg_onset_epoch": ""}
        )
    assert _eeg_offset_from_inventory("yuquan", {"eeg_onset_epoch": "100"}) == 0.0


def test_shared_only_scorer_selection_excludes_own_fields():
    scorers = {
        "own_a": {"marker": 1}, "own_b": {"marker": 2},
        "shared_a": {"marker": 3}, "shared_b": {"marker": 4},
    }
    selected = select_shared_scorers(scorers)
    assert set(selected) == {"shared_a", "shared_b"}
    assert all(not key.startswith("own_") for key in selected)
    with pytest.raises(ValueError, match="missing_shared"):
        select_shared_scorers({"shared_a": {}})


def test_strict_reversed_filter_is_preexisting_quality_not_result_selected():
    record = {
        "interictal_field": {
            "status": "ok", "field_models": {"shared_a": {}, "shared_b": {}}
        },
        "axis_pair": {
            "geometry_2d_supported": True, "strict_stability_pass": True,
            "relation": {"relation": "reversed"},
        },
    }
    assert is_strict_reversed(record)
    record["axis_pair"]["relation"]["relation"] = "same"
    assert not is_strict_reversed(record)


def test_low_nyquist_drops_exact_1_150_but_retains_1_45_sensitivity():
    low = available_bands_for_fs(256.0)
    high = available_bands_for_fs(400.0)
    assert "broadband_1_150" not in low
    assert "broadband_1_45_sensitivity" in low
    assert "broadband_1_150" in high
    assert "broadband_1_45_sensitivity" in high


def test_clinical_onset_zero_does_not_apply_eeg_shift():
    np.testing.assert_allclose(
        clinical_relative_times([120.0, 125.0, 130.0], pre_sec=120.0),
        [0.0, 5.0, 10.0],
    )


def test_adaptive_broadband_uses_highest_non_nyquist_fft_bin():
    assert highest_valid_broadband_upper(256.0) == 127.0
    assert highest_valid_broadband_upper(300.0) == 149.0
    assert highest_valid_broadband_upper(400.0) == 150.0
    assert highest_valid_broadband_upper(1024.0) == 150.0


@pytest.mark.parametrize(
    "tier,relation,expected",
    [
        ("strict_2d", "reversed", "strict_reversed"),
        ("non_strict_2d", "reversed", "non_strict_reversed"),
        ("geometry_unsupported", "reversed", None),
        ("strict_2d", "same", None),
    ],
)
def test_clinical_reversed_strata_are_preexisting_quality_levels(tier, relation, expected):
    record = {
        "interictal_field": {
            "status": "ok", "field_models": {"shared_a": {}, "shared_b": {}}
        },
        "axis_pair": {
            "geometry_2d_supported": tier != "geometry_unsupported",
            "strict_stability_pass": tier == "strict_2d",
            "relation": {"relation": relation},
        },
    }
    assert clinical_stratum(record) == expected


def test_old_cache_hybrid_routing_prefers_shared_then_falls_back_to_own():
    all_fields = {
        "own_a": {"id": "oa"}, "own_b": {"id": "ob"},
        "shared_a": {"id": "sa"}, "shared_b": {"id": "sb"},
    }
    selected, plane, key = select_shared_else_own_scorers(all_fields)
    assert set(selected) == {"shared_a", "shared_b"}
    assert (plane, key) == ("shared", "shared_maxab")

    selected, plane, key = select_shared_else_own_scorers({
        "own_a": {"id": "oa"}, "own_b": {"id": "ob"},
    })
    assert set(selected) == {"own_a", "own_b"}
    assert (plane, key) == ("own_fallback", "own_maxab")

    with pytest.raises(ValueError, match="neither_complete"):
        select_shared_else_own_scorers({"own_a": {}})

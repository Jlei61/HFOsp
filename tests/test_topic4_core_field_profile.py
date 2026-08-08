"""Stage 3 rev5 -- the observable is the per-event profile shape.

The old objective crushed each event into a direction sign and averaged within
sign, which threw away the one thing that separates a mid-sheet source from an
end source. These tests lock the replacement: the same code runs on both sides,
the patient's phantom ranks never enter, and -- the fix that mattered most --
"can it tell positions apart" and "can it recover a position" are kept as two
separate gates, because the earliest-contact statistic passes the first and
fails the second.
"""
import numpy as np
import pytest

from src.topic4_core_field_profile import (NOT_A_POSITION, OBJECTIVE_FEATURES,
                                           REPORT_ONLY, argmin_axial_position,
                                           assert_not_interpreted_as_position,
                                           binned_distance, event_shape,
                                           objective_features,
                                           passes_sensitivity, recovery_score,
                                           shape_table, split_by_block)

AX = {f"C{i}": float(s) for i, s in enumerate(np.linspace(-8, 8, 11))}


def _monotone(sign=1, n=11, noise=0.0, seed=0):
    """Ranks running cleanly from one end of the axis to the other."""
    rng = np.random.default_rng(seed)
    names = list(AX)[:n]
    order = sorted(names, key=lambda k: sign * AX[k])
    return {k: float(i) + noise * rng.normal() for i, k in enumerate(order)}


def _vee(n=11, seed=0):
    """Ranks growing outward from the middle -- what a mid-sheet source gives."""
    names = list(AX)[:n]
    return {k: abs(AX[k]) for k in names}


# ---------------------------------------------------- shape discriminates
def test_a_clean_one_way_profile_is_monotone():
    s = event_shape(_monotone(+1), AX)
    assert s["r2"] > 0.99 and s["slope"] > 0


def test_the_other_direction_flips_the_slope_not_the_fit():
    a, b = event_shape(_monotone(+1), AX), event_shape(_monotone(-1), AX)
    assert a["r2"] == pytest.approx(b["r2"], abs=1e-9)
    assert np.sign(a["slope"]) == -np.sign(b["slope"])


def test_a_mid_sheet_source_is_not_monotone_and_curves_up():
    # THE contract: this is what separates a middle source from an end source,
    # measured at 0% vs 94% of events on the real sweep
    s = event_shape(_vee(), AX)
    assert s["r2"] < 0.2
    assert s["curvature"] > 0


# ------------------------------------------------- one code path, two sides
def test_phantom_ranks_never_enter():
    # the patient's rank matrix gives every channel a finite value whether or
    # not it took part; only the participation mask says who did
    real = _monotone(+1, n=8)
    polluted = dict(real)
    for k in list(AX)[8:]:
        polluted[k] = 99.0                      # phantom: finite but not a participant
    assert event_shape(real, AX) == event_shape(
        polluted, AX, participating=set(real))


def test_none_ranks_are_treated_as_absent():
    r = _monotone(+1, n=8)
    r[list(AX)[8]] = None
    assert event_shape(r, AX)["n_part"] == 8


def test_contacts_outside_the_frozen_support_are_ignored():
    r = dict(_monotone(+1, n=8)); r["NOT_IN_SUPPORT"] = 0.0
    assert event_shape(r, AX)["n_part"] == 8


def test_too_few_participants_yields_nothing_rather_than_a_guess():
    assert event_shape(_monotone(+1, n=5), AX) is None
    assert event_shape(_monotone(+1, n=6), AX) is not None


def test_degenerate_input_yields_nothing():
    flat = {k: 1.0 for k in list(AX)[:8]}
    assert event_shape(flat, AX) is None            # no rank variation


# ------------------------------------- two gates, not one (2026-08-08 fix)
def test_discrimination_and_recovery_are_separate_questions():
    # conflating them is how a statistic gets banned on grounds it passes:
    # the earliest-contact statistic DOES differ across source positions
    # (ratio 4.38 on the real sweep) but does NOT track them (slope 0.25)
    assert "argmin_axial" in NOT_A_POSITION
    assert "argmin_axial" not in OBJECTIVE_FEATURES
    assert "argmin_axial" in REPORT_ONLY          # kept, for reporting


def test_a_locational_claim_is_refused_for_a_failed_estimator():
    with pytest.raises(ValueError, match="recovery gate"):
        assert_not_interpreted_as_position("argmin_axial")
    assert_not_interpreted_as_position("slope")    # not a position claim at all


def test_recovery_passes_when_the_estimate_tracks_the_truth():
    x = np.linspace(-8, 8, 20)
    assert recovery_score(x + 0.3 * np.sin(x), x)["passed"] is True


def test_recovery_fails_at_the_measured_quarter_slope():
    # 0.25 is what the earliest-contact statistic actually scored
    x = np.linspace(-8, 8, 20)
    r = recovery_score(0.25 * x, x)
    assert r["passed"] is False and r["slope"] == pytest.approx(0.25, abs=1e-9)


def test_objective_features_assembles_and_reports_unknown_names():
    tab = shape_table([_monotone(+1), _vee()], AX)
    assert objective_features(tab).shape[1] == len(OBJECTIVE_FEATURES)
    with pytest.raises(ValueError, match="no such shape statistic"):
        objective_features(tab, features=("slope", "not_a_statistic"))


def test_the_disqualified_estimator_still_exists_for_reporting():
    assert argmin_axial_position(_vee(), AX) == pytest.approx(0.0, abs=1.7)


# ------------------------------------------------- sensitivity gate
def test_a_statistic_that_separates_positions_passes():
    rng = np.random.default_rng(0)
    groups = [rng.normal(mu, 0.2, size=40) for mu in (-3.0, 0.0, 3.0)]
    assert passes_sensitivity(groups)["passed"] is True


def test_a_statistic_swamped_by_seed_noise_fails():
    rng = np.random.default_rng(0)
    groups = [rng.normal(mu, 5.0, size=40) for mu in (-0.2, 0.0, 0.2)]
    assert passes_sensitivity(groups)["passed"] is False


def test_the_gate_reports_its_numbers_not_just_a_verdict():
    rng = np.random.default_rng(0)
    g = passes_sensitivity([rng.normal(m, 0.2, size=40) for m in (-3.0, 0.0, 3.0)])
    assert {"between_over_within", "n_groups", "threshold"} <= set(g)


# ------------------------------------------------------ distribution distance
def test_identical_samples_are_at_zero_distance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2))
    assert binned_distance(x, x) == pytest.approx(0.0, abs=1e-12)


def test_distance_is_symmetric():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 2)), rng.normal(1.0, size=(300, 2))
    assert binned_distance(a, b) == pytest.approx(binned_distance(b, a))


def test_distance_grows_with_separation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 2))
    near = rng.normal(0.3, size=(400, 2))
    far = rng.normal(3.0, size=(400, 2))
    assert binned_distance(a, near) < binned_distance(a, far)


def test_distance_uses_frozen_edges_so_two_calls_are_comparable():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 2)), rng.normal(size=(200, 2))
    # a wider second sample must not silently rescale the bins
    wide = rng.normal(0.0, 10.0, size=(200, 2))
    d1 = binned_distance(a, b)
    d2 = binned_distance(a, b)
    assert d1 == pytest.approx(d2)
    assert binned_distance(a, wide) > d1


# --------------------------------------------------- held-out by block
def test_held_out_splits_whole_recordings_not_events():
    blocks = np.repeat(np.arange(10), 50)
    tr, te = split_by_block(blocks, frac=0.3, seed=1)
    assert not (set(blocks[tr]) & set(blocks[te]))
    assert len(tr) + len(te) == len(blocks)


def test_the_split_is_reproducible_from_its_seed():
    blocks = np.repeat(np.arange(10), 50)
    assert np.array_equal(split_by_block(blocks, 0.3, 1)[0],
                          split_by_block(blocks, 0.3, 1)[0])


def test_a_split_that_leaks_a_block_is_rejected():
    # events inside one recording are not independent, so an event-wise split
    # would badly overstate how well the fit generalises
    from src.topic4_core_field_profile import assert_block_disjoint
    blocks = np.repeat(np.arange(4), 10)
    with pytest.raises(ValueError, match="shares"):
        assert_block_disjoint(blocks, np.arange(0, 20), np.arange(15, 40))


# --------------------------------- regression lock on the real calibration
import glob
import json
import os

_CELLS = "results/topic4_sef_hfo/data_driven_core_field_stage3/cells/sigma1.2"
_CFG = "results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json"


def _real_axial():
    from src.topic4_core_field_runner import _placement
    cfg = json.load(open(_CFG))
    reg = _placement(cfg)
    u, c = reg["axis_unit_vec"], reg["center"]
    return {str(n): float((p - c) @ u) for n, p in
            zip(reg["montage_sheet"].names,
                np.asarray(reg["montage_sheet"].contacts, float))}


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists(_CFG) or not glob.glob(f"{_CELLS}/*.json"),
                    reason="position sweep not on disk")
def test_shape_separates_a_middle_source_from_an_end_source():
    """The measurement the whole rev5 objective rests on.

    Statistical unit is the network: each seed contributes the median of its own
    events, and each ground-truth position is one group.
    """
    axial = _real_axial()
    sweep = json.load(open("results/topic4_sef_hfo/data_driven_core_field_stage3"
                           "/config/sweep_config.json"))
    groups = {f: [] for f in ("slope", "r2", "argmin_axial")}
    for i in range(len(sweep["grid"]["centers"])):
        per_seed = {f: [] for f in groups}
        for path in glob.glob(f"{_CELLS}/c{i:03d}_s*.json"):
            rec = json.load(open(path))
            vals = {f: [] for f in groups}
            for ev in rec.get("events", []):
                s = event_shape(ev.get("ranks"), axial)
                if s is not None:
                    vals["slope"].append(s["slope"])
                    vals["r2"].append(s["r2"])
                a = argmin_axial_position(ev.get("ranks"), axial)
                if a is not None:
                    vals["argmin_axial"].append(a)
            for f in groups:
                if len(vals[f]) >= 3:
                    per_seed[f].append(float(np.median(vals[f])))
        for f in groups:
            if len(per_seed[f]) >= 2:
                groups[f].append(per_seed[f])

    # shape statistics carry the objective
    assert passes_sensitivity(groups["slope"])["passed"] is True
    assert passes_sensitivity(groups["r2"])["passed"] is True
    # and so does the earliest-contact statistic -- which is exactly why the
    # discrimination gate alone must not be used to disqualify it
    assert passes_sensitivity(groups["argmin_axial"])["passed"] is True


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists(_CFG) or not glob.glob(f"{_CELLS}/*.json"),
                    reason="position sweep not on disk")
def test_the_earliest_contact_statistic_cannot_recover_position():
    """Locks the measurement that disqualifies it as a location.

    Only near-axis cells are used: the contacts lie close to the axis, so a cell
    far off it is not a fair test of axial recovery.
    """
    axial = _real_axial()
    from src.topic4_core_field_runner import _placement
    cfg = json.load(open(_CFG))
    reg = _placement(cfg)
    u, c = reg["axis_unit_vec"], reg["center"]
    perp = np.array([-u[1], u[0]])
    sweep = json.load(open("results/topic4_sef_hfo/data_driven_core_field_stage3"
                           "/config/sweep_config.json"))
    truth, est = [], []
    for i, xy in enumerate(np.asarray(sweep["grid"]["centers"], float)):
        if abs(float((xy - c) @ perp)) >= 4.0:
            continue
        vals = []
        for path in glob.glob(f"{_CELLS}/c{i:03d}_s*.json"):
            for ev in json.load(open(path)).get("events", []):
                a = argmin_axial_position(ev.get("ranks"), axial)
                if a is not None:
                    vals.append(a)
        if len(vals) >= 10:
            truth.append(float((xy - c) @ u))
            est.append(float(np.median(vals)))

    r = recovery_score(est, truth)
    assert r["n"] >= 10
    assert r["passed"] is False
    assert r["slope"] < 0.5          # measured 0.25

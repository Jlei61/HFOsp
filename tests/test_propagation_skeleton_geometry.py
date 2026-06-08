import numpy as np
import pytest
from src.propagation_skeleton_geometry import parse_shaft


def test_parse_shaft_depth_and_grid_and_junk():
    assert parse_shaft("D13") == ("D", 13)
    assert parse_shaft("FLA2") == ("FLA", 2)
    assert parse_shaft("GA1") == ("GA", 1)
    assert parse_shaft("A1'") == ("A'", 1)   # prime-marked shaft
    assert parse_shaft("EKG") == (None, None)
    assert parse_shaft("") == (None, None)


# ---------------------------------------------------------------------------
# Task 2: Endpoint cores + numeric eligibility gate
# ---------------------------------------------------------------------------
from src.propagation_skeleton_geometry import build_endpoint_cores


def _axis(n, participating):
    """template axis: ascending ranks on participating idx, NaN elsewhere."""
    ax = np.full(n, np.nan)
    ax[participating] = np.arange(len(participating), dtype=float)
    return ax


def test_cores_k3_primary_tier():
    # 8 participating + coord-mapped channels -> n_eff=8 -> k=3
    ax = _axis(8, np.arange(8))
    eligible = np.ones(8, bool)
    r = build_endpoint_cores(ax, eligible, k_primary=3)
    assert r["tier"] == "primary"
    assert r["k_used"] == 3
    assert r["n_eff"] == 8
    assert sorted(r["source_idx"]) == [0, 1, 2]      # lowest ranks
    assert sorted(r["sink_idx"]) == [5, 6, 7]        # highest ranks
    assert len(r["interior_idx"]) == 2               # 3,4
    assert set(r["source_idx"]).isdisjoint(r["sink_idx"])


def test_cores_phantom_never_in_core():
    # channel 0 is NON-participating (NaN) and would be index-0; must NOT be source
    n = 9
    ax = np.full(n, np.nan)
    ax[1:8] = np.arange(7, dtype=float)   # participating = idx 1..7 (n_eff=7)
    eligible = ~np.isnan(ax)
    r = build_endpoint_cores(ax, eligible, k_primary=3)
    assert 0 not in r["source_idx"]
    assert 8 not in r["sink_idx"]         # idx 8 is NaN too
    assert all(eligible[i] for i in r["source_idx"] + r["sink_idx"])


def test_cores_fallback_k2_when_neff_5_or_6():
    ax = _axis(6, np.arange(6))
    r = build_endpoint_cores(ax, np.ones(6, bool), k_primary=3)
    assert r["tier"] == "fallback"
    assert r["k_used"] == 2
    assert len(r["interior_idx"]) == 2


def test_cores_descriptive_only_when_neff_below_5():
    ax = _axis(4, np.arange(4))
    r = build_endpoint_cores(ax, np.ones(4, bool), k_primary=3)
    assert r["tier"] == "descriptive_only"
    assert r["source_idx"] == [] and r["sink_idx"] == []


# ---------------------------------------------------------------------------
# Task 3: Axis-coordinate frame (centroids, axis, along/off per channel)
# ---------------------------------------------------------------------------
from src.propagation_skeleton_geometry import compute_axis_frame


def test_axis_frame_along_and_off_are_orthogonal_decomposition():
    # source core at x=0, sink core at x=10 (along +x). A channel at (5, 3, 0)
    # must have along=5 (projection) and off=3 (perp distance).
    coords = np.array([
        [0., 0., 0.], [0., 0., 0.],   # source core (centroid x=0)
        [10., 0., 0.], [10., 0., 0.],  # sink core (centroid x=10)
        [5., 3., 0.],                  # test channel
    ])
    fr = compute_axis_frame(coords, source_idx=[0, 1], sink_idx=[2, 3])
    assert np.allclose(fr["axis_length"], 10.0)
    assert np.allclose(fr["along_axis"][4], 5.0)
    assert np.allclose(fr["off_axis"][4], 3.0)
    # on-axis channel -> off == 0
    assert np.allclose(fr["off_axis"][0], 0.0)


def test_axis_frame_degenerate_axis_flagged():
    coords = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    fr = compute_axis_frame(coords, source_idx=[0, 1], sink_idx=[2, 3])
    assert fr["degenerate_axis"] is True


def test_axis_frame_nan_source_core_raises():
    # all-NaN source-core coords -> non-finite centroid -> ValueError (a NaN L
    # would silently pass `L < 1e-9` and propagate NaN everywhere).
    coords = np.array([
        [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],   # source core, all NaN
        [10., 0., 0.], [10., 0., 0.],                          # sink core
        [5., 3., 0.],
    ])
    with pytest.raises(ValueError):
        compute_axis_frame(coords, source_idx=[0, 1], sink_idx=[2, 3])


# ---------------------------------------------------------------------------
# Task 4: Core radii (RMS / MEB / max-pairwise) — bimodal-core guard
# ---------------------------------------------------------------------------
from src.propagation_skeleton_geometry import core_radii


def test_core_radii_compact_vs_split():
    centroid = np.array([0., 0., 0.])
    compact = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
    r = core_radii(compact, centroid)
    assert r["rms_mm"] == pytest.approx(1.0, abs=1e-9)
    assert r["max_pairwise_mm"] == pytest.approx(2.0, abs=1e-6)   # [1,0,0]-[-1,0,0]
    assert r["meb_mm"] == pytest.approx(1.0, abs=1e-6)            # circumradius branch (right triangle: circumradius = hyp/2 = 1.0)

    # split core: two contacts 20mm apart -> centroid in the gap.
    split = np.array([[10., 0., 0.], [-10., 0., 0.]])
    split_centroid = split.mean(axis=0)
    rs = core_radii(split, split_centroid)
    assert rs["max_pairwise_mm"] == pytest.approx(20.0, abs=1e-6)
    assert rs["meb_mm"] == pytest.approx(10.0, abs=1e-6)


def test_meb_obtuse_triangle_uses_longest_over_two():
    # obtuse triangle: longest side dominates -> MEB = longest/2
    pts = np.array([[0., 0., 0.], [4., 0., 0.], [5., 1., 0.]])
    r = core_radii(pts, pts.mean(axis=0))
    assert r["meb_mm"] == pytest.approx(np.sqrt(26) / 2.0, abs=1e-6)  # longest side = |(0,0,0)-(5,1,0)| = sqrt(26)


# ---------------------------------------------------------------------------
# Task 5: Perpendicular spread + sampling-geometry classifier
# ---------------------------------------------------------------------------
from src.propagation_skeleton_geometry import perp_spread, classify_sampling_geometry


def test_perp_spread_rms_p75_p90():
    off = np.array([0., 0., 4., 8., np.nan])
    participating = np.array([True, True, True, True, False])
    s = perp_spread(off, participating)
    assert s["rms_mm"] == pytest.approx(np.sqrt((0 + 0 + 16 + 64) / 4.0), abs=1e-9)
    assert s["p75_mm"] == pytest.approx(np.percentile([0, 0, 4, 8], 75), abs=1e-9)
    assert s["p90_mm"] == pytest.approx(np.percentile([0, 0, 4, 8], 90), abs=1e-9)
    assert s["n"] == 4


def test_classify_1d_single_shaft():
    names = ["A1", "A2", "A3", "A4"]
    part = np.ones(4, bool)
    off = np.array([0., 0.1, 0.0, 0.2])     # all near-axis
    g = classify_sampling_geometry(names, part, off, spacing_mm=3.5)
    assert g["geometry"] == "1D"
    assert g["n_shafts"] == 1
    assert g["measurable"] is False


def test_classify_distributed_multi_shaft():
    names = ["A1", "A2", "B1", "C3"]
    part = np.ones(4, bool)
    off = np.array([0., 1., 9., 12.])
    g = classify_sampling_geometry(names, part, off, spacing_mm=3.5)
    assert g["geometry"] == "distributed"
    assert g["n_shafts"] == 3
    assert g["measurable"] is True
    assert g["unparsed_frac"] == pytest.approx(0.0)


def test_classify_shaft_parse_uncertain_when_many_unparsed():
    # >20% of participating names are junk (don't parse to a shaft) -> we cannot
    # trust len(shafts)<=1 to mean 1D; flag it instead of misclassifying.
    names = ["A1", "A2", "EKG", "REF", "X"]   # 3/5 = 60% unparsed
    part = np.ones(5, bool)
    off = np.array([0., 1., 0., 0., 0.])
    g = classify_sampling_geometry(names, part, off, spacing_mm=3.5)
    assert g["geometry"] == "shaft_parse_uncertain"
    assert g["measurable"] is False
    assert g["unparsed_frac"] == pytest.approx(0.6)
    assert g["n_shafts"] == 1   # only "A" parses


# ---------------------------------------------------------------------------
# Task 6: Perpendicular-spread participation sweep (advisor #1 control)
# ---------------------------------------------------------------------------
from src.propagation_skeleton_geometry import perp_spread_participation_sweep


def test_participation_sweep_tightens_with_threshold():
    # far channels (large off) have FEW events; raising the event-count
    # threshold drops them and shrinks the spread.
    off = np.array([0., 1., 2., 30.])
    full_count = np.array([100., 90., 80., 5.])   # far channel = 5 events
    sweep = perp_spread_participation_sweep(off, full_count, thresholds=[1, 10])
    assert sweep[0]["threshold"] == 1 and sweep[0]["n"] == 4
    assert sweep[1]["threshold"] == 10 and sweep[1]["n"] == 3   # 30mm dropped
    assert sweep[1]["rms_mm"] < sweep[0]["rms_mm"]


# ---------------------------------------------------------------------------
# Task 7: Per-channel stereotypy + event-size-matched null
# ---------------------------------------------------------------------------
from src.propagation_skeleton_geometry import (
    channel_stereotypy, channel_stereotypy_excess,
    channel_stereotypy_components,
)


def test_stereotypy_reproducible_high_random_low():
    # masked[c, e] = normalized within-event rank, NaN = not participating.
    # ch0 always at 0.0 (perfectly reproducible); ch1 jumps 0/1 (anti-stereotyped).
    masked = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    s = channel_stereotypy(masked)
    assert s[0] == pytest.approx(1.0, abs=1e-9)     # std 0 -> 1
    assert s[1] < s[0]


def test_stereotypy_excess_z_separates_signal_from_chance():
    # 10 channels all participating -> event size 10, so the integer-position
    # matched null is continuous-enough to match the observed ranks (the null's
    # event sizes MUST match the data or the z is meaningless).
    rng = np.random.default_rng(0)
    n_ev, n_ch = 200, 10
    masked = rng.random((n_ch, n_ev))               # filler channels = uniform
    masked[0] = 0.1 + rng.normal(0, 0.01, n_ev)     # reproducible (front)
    masked[1] = rng.integers(0, n_ch, n_ev) / (n_ch - 1)  # discrete rank grid = matches null
    bools = np.ones((n_ch, n_ev), bool)
    z = channel_stereotypy_excess(masked, bools, rng=np.random.default_rng(1),
                                  n_null=200)
    assert z[0] > 3.0        # clearly above chance
    assert abs(z[1]) < 2.0   # ~chance


def test_stereotypy_components_nan_pattern_mismatch_raises():
    # masked NaN pattern must agree with bools (NaN <=> non-participating).
    # Here masked[0,0] is a finite value but bools[0,0] is False -> mismatch.
    masked = np.array([
        [0.5, np.nan],
        [np.nan, 0.5],
    ])
    bools = np.array([
        [False, False],
        [False, True],
    ])
    with pytest.raises(ValueError):
        channel_stereotypy_components(masked, bools, rng=np.random.default_rng(0))


def test_excess_is_n_invariant_while_z_inflates():
    rng = np.random.default_rng(0)
    def front_channel(n_ev, n_ch=10):
        m = rng.integers(0, n_ch, (n_ch, n_ev)).astype(float) / (n_ch - 1)
        m[0] = 0.0   # pinned to the front every event = fixed reproducibility
        return m, np.ones((n_ch, n_ev), bool)
    comps = {}
    for n_ev in (20, 2000):
        m, b = front_channel(n_ev)
        comps[n_ev] = channel_stereotypy_components(m, b, rng=np.random.default_rng(1), n_null=300)
    # raw excess ~stable across a 100x event-count change; z grows a lot
    assert abs(comps[20]["excess"][0] - comps[2000]["excess"][0]) < 0.05
    assert comps[2000]["z"][0] > 3 * comps[20]["z"][0]


# ---------------------------------------------------------------------------
# Task 8: Along-axis stereotypy profile binning
# ---------------------------------------------------------------------------
from src.propagation_skeleton_geometry import axis_stereotypy_profile


def test_axis_profile_bins_by_along_axis():
    along = np.array([0.0, 2.0, 4.0, 9.0, np.nan])
    excess = np.array([3.0, 2.0, 1.0, -0.5, 5.0])   # last is NaN-along -> dropped
    prof = axis_stereotypy_profile(along, excess, edges=[0, 5, 10])
    assert len(prof) == 2
    assert prof[0]["n"] == 3 and prof[0]["mean_excess"] == pytest.approx(2.0)
    assert prof[1]["n"] == 1 and prof[1]["mean_excess"] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Task 9: Runner integration smoke test (guarded by artifact availability)
# ---------------------------------------------------------------------------
import json as _json
import subprocess
import sys as _sys
import pathlib


def test_runner_smoke_two_subjects(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    if not (root / "results/interictal_propagation_masked/rank_displacement/"
            "per_subject/epilepsiae_253.json").exists():
        pytest.skip("masked rank-displacement artifacts not present")
    out = tmp_path / "skeleton_geometry"
    r = subprocess.run(
        [_sys.executable, str(root / "scripts/run_propagation_skeleton_geometry.py"),
         "--subjects", "epilepsiae:253", "yuquan:chengshuai",
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(root))
    assert r.returncode == 0, r.stderr
    cohort = _json.loads((out / "cohort_summary.json").read_text())
    # phantom-safe cores: zero violations reported
    assert cohort["phantom_core_violations"] == 0
    # an all-error run must fail loudly, not pass vacuously
    assert cohort["n_ok"] >= 1
    # dataset stratification present
    assert "by_dataset" in cohort and set(cohort["by_dataset"]) <= {"yuquan", "epilepsiae"}

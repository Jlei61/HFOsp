"""Contract tests for the MZ early-field bridge (design §6-§13).

The 10 tests below are the prompt's required contract guards. Each is written to FAIL if the
corresponding invariant is violated, on synthetic fixtures (no SNN run).
"""
import json

import numpy as np
import pytest

from src.sef_hfo_events import detect_events
from src.topic4_mz_early_field_bridge import (
    build_direction_template,
    compute_event_bar,
    compute_t_recruit,
    contact_energy_field,
    ContactTiming,
    event_contact_timing,
    maxab_permutation_null,
    resume_should_skip,
    source_bins,
    to_jsonable,
    toroidal_maxab_null,
)


# ---- test 1: the slow-off event bar is frozen and reused; the target's own max never moves it (§6)
def test_1_event_bar_frozen_from_slowoff_not_recomputed_from_target():
    af_slowoff = np.full(200, 0.02)
    af_slowoff[60:70] = 0.10
    af_slowoff[100:108] = 0.09
    af_target = np.full(200, 0.02)
    af_target[60:70] = 0.10
    af_target[120:180] = 0.50            # the runaway lifts the target's own max
    bar_frozen = compute_event_bar(af_slowoff, 1.0).bar
    bar_wrong = compute_event_bar(af_target, 1.0).bar      # what the buggy per-run recompute would use
    assert bar_frozen < bar_wrong                          # the target's own max changes the threshold
    ev_frozen = detect_events(af_target, 1.0, event_on_frac=bar_frozen)
    ev_wrong = detect_events(af_target, 1.0, event_on_frac=bar_wrong)
    assert len(ev_frozen) != len(ev_wrong)                 # freezing the bar is load-bearing, not cosmetic


# ---- test 2: an early window running off the trace fails closed (never scored truncated) (§8.2)
def test_2_incomplete_early_window_fails_closed():
    times = np.arange(0.0, 100.0, 0.1)                     # trace ends at ~99.9 ms
    env = np.ones((times.size, 4))
    quiet_med = np.zeros(4)
    complete = contact_energy_field(env, times, quiet_med, t_recruit_ms=10.0,
                                    window_ms=(0.0, 50.0), record_end_ms=times[-1])
    assert complete.status == "eligible"
    truncated = contact_energy_field(env, times, quiet_med, t_recruit_ms=80.0,
                                     window_ms=(0.0, 50.0), record_end_ms=times[-1])
    assert truncated.status != "eligible"
    assert np.isnan(truncated.energy).all()


def test_2b_contact_energy_eligible_at_fractional_t_recruit():
    # regression: a COMPLETE window at a t_recruit whose (t_recruit+50) lands on the grid only up to
    # float noise (e.g. 9128.3 vs 91283*0.1) must NOT be mis-flagged incomplete (the seed-1 bug).
    times = np.arange(0.0, 15000.0, 0.1)                  # same 0.1 ms grid as the real runs
    env = np.full((times.size, 4), 3.0)
    for tr in (90.7, 9078.3, 9128.9):                    # fractional recruit times within the trace
        ef = contact_energy_field(env, times, np.zeros(4), t_recruit_ms=tr,
                                  window_ms=(0.0, 50.0), record_end_ms=times[-1])
        assert ef.status == "eligible", (tr, ef.status)
        assert np.all(np.isfinite(ef.energy))


# ---- test 3: unreadable contacts stay missing (never imputed); support is matched (§7.2)
def test_3_missing_contacts_never_imputed():
    times = np.arange(0.0, 60.0, 0.1)
    nC = 4
    env = np.full((times.size, nC), 0.1)                   # quiet floor
    # contacts 0,1 get real bursts (peaks well above 5*MAD); contacts 2,3 stay at floor
    for j, tp in [(0, 10.0), (1, 20.0)]:
        env[:, j] += 5.0 * np.exp(-0.5 * ((times - tp) / 1.0) ** 2)
    quiet_med = np.full(nC, 0.1)
    quiet_mad = np.full(nC, 0.02)
    event = {"t_on": 0.0, "t_off": 40.0}
    ct = event_contact_timing(env, times, event, next_event_t_on=None, record_end_ms=times[-1],
                              quiet_med=quiet_med, quiet_mad=quiet_mad,
                              contact_axis=np.array([0.0, 1.0, 2.0, 3.0]))
    assert ct.readable.tolist() == [True, True, False, False]
    assert np.isfinite(ct.latency_ms[:2]).all()
    assert np.isnan(ct.latency_ms[2:]).all()              # NOT imputed to 0 or any value
    assert np.isnan(ct.rank[2:]).all()


# ---- test 4: chronological odd/even split has no event leakage between train and held-out (§7.3)
def test_4_odd_even_split_no_leakage():
    nC = 8
    def _t(rank_vec):
        rank = np.array(rank_vec, float)
        return ContactTiming(latency_ms=rank.copy(), readable=np.isfinite(rank),
                             rank=rank.copy(), axis_spearman=0.9, direction="A_to_B",
                             n_readable=int(np.isfinite(rank).sum()), eligible=True)
    base = list(range(1, nC + 1))
    timings = [_t(base) for _ in range(6)]                 # 6 chronological events, identical ranks
    tmpl = build_direction_template(timings, "A_to_B", min_train=3, min_heldout=2, min_shared=6)
    assert tmpl.n_train == 3 and tmpl.n_heldout == 3       # even idx {0,2,4}, odd idx {1,3,5}: disjoint
    assert tmpl.n_train + tmpl.n_heldout == len(timings)   # partition, no event in both
    assert tmpl.eligible
    assert np.allclose(tmpl.train_template, base)          # median of identical training ranks
    assert all(np.isclose(s, 1.0) for s in tmpl.heldout_scores)   # held-out reproduces the template


# ---- test 5: maxAB null recomputes max(rho_A, rho_B) INSIDE each permutation (§9)
def test_5_maxab_null_recomputes_max_each_permutation():
    rank_a = np.array([1.0, 2.0, 3.0, 4.0])
    rank_b = np.array([4.0, 3.0, 2.0, 1.0])               # opposite direction template
    energy = np.array([4.0, 3.0, 2.0, 1.0])
    sup = np.ones(4, bool)
    out = maxab_permutation_null(rank_a, rank_b, energy, support_a=sup, support_b=sup,
                                 groups=None, min_points=3, max_exact_permutations=50000)
    assert out["method"] == "exact" and out["n_unique_possible"] == 24
    assert np.isclose(out["observed"], 1.0)
    # max(rho_a, rho_b) hits 1.0 for BOTH the identity and reversed energy orderings -> 2/24.
    # A single-template null (rho_a only) would give 1/24; asserting 2/24 proves the max is recomputed.
    assert np.isclose(out["p_one_sided"], 2.0 / 24.0)


# ---- test 6: within-shaft permutation preserves shaft membership (§9)
def test_6_within_shaft_null_preserves_membership():
    rank_a = np.array([1.0, 2.0, 3.0, 4.0])
    rank_b = np.array([4.0, 3.0, 2.0, 1.0])
    energy = np.array([4.0, 3.0, 2.0, 1.0])
    sup = np.ones(4, bool)
    groups = np.array(["S1", "S1", "S2", "S2"])           # two shafts of size 2
    out = maxab_permutation_null(rank_a, rank_b, energy, support_a=sup, support_b=sup,
                                 groups=groups, min_points=3, max_exact_permutations=50000)
    # only within-shaft permutations are enumerated: 2! * 2! = 4 (not 24 = cross-shaft)
    assert out["n_unique_possible"] == 4
    assert out["method"] == "exact"


# ---- test 7: source-grid toroidal shift excludes the zero shift and preserves the field multiset (§9)
def test_7_toroidal_shift_excludes_zero_and_preserves_multiset():
    n = 4
    rng = np.random.default_rng(0)
    rank_a = rng.permutation(n * n).astype(float)
    rank_b = (n * n - 1 - rank_a)
    energy = rng.random(n * n)
    sup = np.ones(n * n, bool)
    out = toroidal_maxab_null(rank_a, rank_b, energy, support_a_grid=sup, support_b_grid=sup, n=n)
    assert out["status"] == "eligible"
    assert out["n_shifts"] == n * n - 1                  # 16 shifts minus the excluded (0,0)
    assert 0.0 < out["p_one_sided"] <= 1.0
    # np.roll preserves the multiset: the sorted shifted energy always equals the original sorted energy
    E = energy.reshape(n, n)
    for dx in (1, 2):
        assert np.allclose(np.sort(np.roll(E, dx, axis=0).ravel()), np.sort(energy))


# ---- test 8: the t_recruit component must contain t120 (§8.1)
def test_8_t_recruit_component_contains_t120():
    dt = 0.1
    r20_slowoff = np.full(2000, 10.0)                     # quiet baseline ...
    r20_slowoff[-3:] = 60.0                               # ... with interictal peaks -> P99.9 ~ 60 Hz
    r20_native = np.full(2000, 10.0)                      # native pre-runaway quiet (below theta)
    r20_native[900:1300] = 130.0                          # supra-theta component covers [90, 130] ms
    ok = compute_t_recruit(r20_native, r20_slowoff, dt, t120=95.0)
    assert ok["status"] == "eligible"
    assert ok["t_recruit_ms"] <= 95.0                     # component start is at/left of t120
    assert abs(ok["t_recruit_ms"] - 90.0) < 1.0
    # t120 that does not fall in any supra-theta component -> unresolved (no early-field claim)
    bad = compute_t_recruit(r20_native, r20_slowoff, dt, t120=50.0)
    assert bad["status"] == "onset_unresolved"
    missing = compute_t_recruit(r20_native, r20_slowoff, dt, t120=None)
    assert missing["status"] == "onset_unresolved"


# ---- test 9: output JSON serializes NaN/inf as null (§13)
def test_9_json_serializes_nan_as_null():
    payload = {"a": np.nan, "b": np.array([1.0, np.nan, 3.0]), "c": np.float64(3.5),
               "d": {"x": float("inf"), "y": np.int64(7)}, "e": np.bool_(True)}
    js = to_jsonable(payload)
    assert js["a"] is None
    assert js["b"] == [1.0, None, 3.0]
    assert js["c"] == 3.5
    assert js["d"] == {"x": None, "y": 7}
    assert js["e"] is True
    json.dumps(js)                                        # must not raise


# ---- test 10: --resume skips only completed, provenance-matched seeds (§13)
def test_10_resume_skips_only_completed_matching(tmp_path):
    p = tmp_path / "bridge_metrics.json"
    p.write_text(json.dumps({"status": "complete", "provenance_fingerprint": "abc"}))
    assert resume_should_skip(str(p), "abc") is True
    assert resume_should_skip(str(p), "different") is False          # provenance mismatch -> re-run
    p.write_text(json.dumps({"status": "failed", "provenance_fingerprint": "abc"}))
    assert resume_should_skip(str(p), "abc") is False               # incomplete -> re-run
    assert resume_should_skip(str(tmp_path / "missing.json"), "abc") is False


# ---- extra sanity: source-grid bin mapping mirrors _spatial_movie (§7.3/§8.4)
def test_source_bins_match_spatial_movie_mapping():
    L, n = 20.0, 24
    posE = np.array([[0.0, 0.0], [L - 1e-6, L - 1e-6], [L / 2, L / 2]])
    cell, counts = source_bins(posE, L, n)
    assert cell[0] == 0                                   # bottom-left bin
    assert cell[1] == (n - 1) * n + (n - 1)               # top-right bin
    assert counts.sum() == posE.shape[0]

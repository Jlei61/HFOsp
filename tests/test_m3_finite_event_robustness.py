"""Tests for scripts/analyze_m3_finite_event_robustness.py.

All inputs are SYNTHETIC per_seed / candidate CSVs written to a tmp dir.
No SNN is ever run. The four required scenarios:

1. silent below / finite event in >=2/3 seeds above -> FINITE_THRESHOLD,
   and the 0.75/1.0 kicks are stable finite_local_returned.
2. graded monotone local-returned ramp from low kicks -> LINEAR_GRADED.
3. every kick core_only_quiet=False -> SELF_IGNITE.
4. finite event in only 1/8 seeds -> NOT a stable candidate.
"""
import csv
import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "..", "scripts", "analyze_m3_finite_event_robustness.py")
_spec = importlib.util.spec_from_file_location("m3_robustness", _SCRIPT)
m3 = importlib.util.module_from_spec(_spec)
sys.modules["m3_robustness"] = m3  # needed so @dataclass can resolve the module
_spec.loader.exec_module(m3)


WIN_LO, WIN_HI = 22.0, 32.0

PER_SEED_FIELDS = [
    "kick_boost", "win_lo", "win_hi", "rep_bin", "seed",
    "downstream_resp", "n_activated_bins", "r95_mm", "far_field_frac",
    "returned", "runaway", "frac_time_on_post", "seed_local_returned",
]


def _write_per_seed(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_SEED_FIELDS)
        w.writeheader()
        for r in rows:
            full = {k: r.get(k, 0) for k in PER_SEED_FIELDS}
            full["win_lo"] = WIN_LO
            full["win_hi"] = WIN_HI
            full["rep_bin"] = 12
            w.writerow(full)


def _write_candidate(path, kicks, quiet_by_kick):
    fields = ["kick_boost", "win_lo", "win_hi", "core_only_quiet"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k in kicks:
            w.writerow({
                "kick_boost": k, "win_lo": WIN_LO, "win_hi": WIN_HI,
                "core_only_quiet": int(quiet_by_kick.get(k, True)),
            })


def _seed_row(kick, seed, local_returned, *, r95=4.0, far=0.1, ds=20.0,
              returned=1, runaway=0, nact=3):
    return {
        "kick_boost": kick, "seed": seed, "downstream_resp": ds,
        "n_activated_bins": nact, "r95_mm": r95, "far_field_frac": far,
        "returned": returned, "runaway": runaway,
        "seed_local_returned": local_returned,
    }


def _analyze(tmp_path, per_seed_rows, kicks, quiet_by_kick):
    d = str(tmp_path)
    _write_per_seed(os.path.join(d, "per_seed_metrics.csv"), per_seed_rows)
    _write_candidate(os.path.join(d, "candidate_table.csv"), kicks, quiet_by_kick)
    res = m3.analyze_substrate(d, WIN_LO, WIN_HI)
    assert res is not None
    return res


# --------------------------------------------------------------------------- #
# Scenario 1: silent below threshold, finite event in >=2/3 seeds above       #
# (>=6 seeds so it can be a STABLE candidate)                                  #
# --------------------------------------------------------------------------- #
def test_finite_threshold_with_stable_candidate(tmp_path):
    n_seeds = 6
    low_kicks = [0.05, 0.1, 0.2, 0.5]   # silent: no local-returned, no scatter
    high_kicks = [0.75, 1.0]            # finite event in >=2/3 (here all) seeds
    kicks = low_kicks + high_kicks
    rows = []
    for k in low_kicks:
        for s in range(n_seeds):
            # silent: nothing fires, small downstream, no far-field scatter
            rows.append(_seed_row(k, s, 0, ds=0.0, far=0.0, nact=0,
                                   returned=1, r95=0.0))
    for k in high_kicks:
        for s in range(n_seeds):
            # local + returned in 5/6 seeds (>=2/3) => P_local_returned ~ 0.83
            lr = 1 if s < 5 else 0
            rows.append(_seed_row(k, s, lr, ds=30.0, far=0.1, r95=4.0,
                                   returned=1))
    quiet = {k: True for k in kicks}   # core stays quiet (trustworthy)

    res = _analyze(tmp_path, rows, kicks, quiet)

    assert res.verdict == "FINITE_THRESHOLD", res.verdict_reason
    by_kick = {a.kick: a for a in res.kicks}
    for k in high_kicks:
        assert by_kick[k].phenotype == "finite_local_returned"
        assert by_kick[k].stable_finite, f"kick {k} should be stable"
        assert by_kick[k].p_local_returned >= m3.ROBUST_FRAC
    # below-threshold kicks must NOT be stable finite events
    for k in low_kicks:
        assert not by_kick[k].stable_finite
        assert by_kick[k].phenotype == "silent"


# --------------------------------------------------------------------------- #
# Scenario 2: graded monotone local-returned ramp from low kicks -> LINEAR    #
# --------------------------------------------------------------------------- #
def test_linear_graded_ramp(tmp_path):
    n_seeds = 6
    kicks = [0.05, 0.1, 0.2, 0.35, 0.5]
    # P_local_returned rises smoothly from the lowest kick: 2/6,3/6,4/6,5/6,6/6
    n_local = {0.05: 2, 0.1: 3, 0.2: 4, 0.35: 5, 0.5: 6}
    rows = []
    for k in kicks:
        for s in range(n_seeds):
            lr = 1 if s < n_local[k] else 0
            rows.append(_seed_row(k, s, lr, ds=15.0, far=0.1, r95=4.0,
                                  returned=1))
    quiet = {k: True for k in kicks}

    res = _analyze(tmp_path, rows, kicks, quiet)

    assert res.verdict == "LINEAR_GRADED", res.verdict_reason
    # the lowest kick already has a non-zero local-returned response (no silent gap)
    lowest = min(res.kicks, key=lambda a: a.kick)
    assert lowest.p_local_returned > 0.0


# --------------------------------------------------------------------------- #
# Scenario 3: every kick core_only_quiet=False -> SELF_IGNITE                  #
# --------------------------------------------------------------------------- #
def test_self_ignite_when_never_quiet(tmp_path):
    n_seeds = 6
    kicks = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    rows = []
    for k in kicks:
        for s in range(n_seeds):
            # even though some seeds look "local returned", core is never quiet
            rows.append(_seed_row(k, s, 1, ds=500.0, far=0.05, r95=5.0,
                                  returned=1))
    quiet = {k: False for k in kicks}   # core fires on its own at every kick

    res = _analyze(tmp_path, rows, kicks, quiet)

    assert res.verdict == "SELF_IGNITE", res.verdict_reason
    for a in res.kicks:
        assert a.phenotype == "confounded"
        assert not a.stable_finite


# --------------------------------------------------------------------------- #
# Scenario 4: finite event in only 1/8 seeds -> NOT a stable candidate         #
# --------------------------------------------------------------------------- #
def test_not_stable_when_only_one_of_eight(tmp_path):
    n_seeds = 8
    low_kicks = [0.05, 0.1, 0.2, 0.5]
    high_kicks = [0.75, 1.0]
    kicks = low_kicks + high_kicks
    rows = []
    for k in low_kicks:
        for s in range(n_seeds):
            rows.append(_seed_row(k, s, 0, ds=0.0, far=0.0, nact=0,
                                  returned=1, r95=0.0))
    for k in high_kicks:
        for s in range(n_seeds):
            lr = 1 if s == 0 else 0     # only 1/8 seeds -> P_local_returned = 0.125
            rows.append(_seed_row(k, s, lr, ds=10.0, far=0.1, r95=4.0,
                                  returned=1))
    quiet = {k: True for k in kicks}

    res = _analyze(tmp_path, rows, kicks, quiet)

    by_kick = {a.kick: a for a in res.kicks}
    for k in high_kicks:
        assert by_kick[k].p_local_returned < m3.ROBUST_FRAC
        assert not by_kick[k].stable_finite, (
            f"1/8 seeds must not be a stable candidate (P="
            f"{by_kick[k].p_local_returned})"
        )
    # no stable finite event anywhere -> not FINITE_THRESHOLD
    assert res.verdict != "FINITE_THRESHOLD"
    assert all(not a.stable_finite for a in res.kicks)


# --------------------------------------------------------------------------- #
# Extra: pure-function spot checks                                             #
# --------------------------------------------------------------------------- #
def test_aggregate_by_kick_counts_seeds(tmp_path):
    kicks = [0.1]
    rows = [_seed_row(0.1, s, 1) for s in range(4)]
    _write_per_seed(os.path.join(str(tmp_path), "per_seed_metrics.csv"), rows)
    quiet_map = {(0.1, WIN_LO, WIN_HI): True}
    per_seed = m3._read_csv(os.path.join(str(tmp_path), "per_seed_metrics.csv"))
    aggs = m3.aggregate_by_kick(per_seed, quiet_map, WIN_LO, WIN_HI)
    assert len(aggs) == 1
    assert aggs[0].n_seeds == 4
    assert aggs[0].p_local_returned == pytest.approx(1.0)


def test_phenotype_escape_on_runaway():
    agg = m3.KickAggregate(
        kick=1.0, n_seeds=6, p_local_returned=0.0, p_returned=0.4,
        p_runaway=0.8, median_downstream=200.0, median_n_activated=6,
        median_r95=8.0, median_far_field=0.05, core_only_quiet=True,
    )
    assert m3.per_kick_phenotype(agg) == "escape"


def test_phenotype_large_returned_when_not_local():
    agg = m3.KickAggregate(
        kick=1.0, n_seeds=6, p_local_returned=0.8, p_returned=1.0,
        p_runaway=0.0, median_downstream=50.0, median_n_activated=5,
        median_r95=11.0, median_far_field=0.3, core_only_quiet=True,
    )
    assert m3.per_kick_phenotype(agg) == "large_returned"


# --------------------------------------------------------------------------- #
# Review P1-b: under-powered (max n_seeds < MIN_SEEDS) must NOT label NO_LOCAL #
# --------------------------------------------------------------------------- #
def test_underpowered_low_n_is_not_no_local(tmp_path):
    n_seeds = 3                        # < MIN_SEEDS (6): cannot call stability
    low_kicks = [0.05, 0.1, 0.2, 0.5]
    high_kicks = [0.75, 1.0]
    kicks = low_kicks + high_kicks
    rows = []
    for k in low_kicks:
        for s in range(n_seeds):
            rows.append(_seed_row(k, s, 0, ds=0.0, far=0.0, nact=0, returned=1, r95=0.0))
    for k in high_kicks:
        for s in range(n_seeds):       # finite event in 3/3 seeds, but only 3 seeds
            rows.append(_seed_row(k, s, 1, ds=30.0, far=0.1, r95=4.0, returned=1))
    quiet = {k: True for k in kicks}

    res = _analyze(tmp_path, rows, kicks, quiet)
    assert res.verdict == "UNDERPOWERED", res.verdict_reason
    assert res.verdict != "NO_LOCAL"
    assert all(not a.stable_finite for a in res.kicks)   # no stable claim at 3 seeds


# --------------------------------------------------------------------------- #
# Review P1-c: missing core_only_quiet source must FAIL CLOSED (raise), not    #
# fail open to "quiet" (a self-igniting core would otherwise be trusted)       #
# --------------------------------------------------------------------------- #
def test_load_core_quiet_fail_closed_when_no_source(tmp_path):
    d = str(tmp_path)
    # a candidate_table WITHOUT a core_only_quiet column, and no per_seed_metrics
    with open(os.path.join(d, "candidate_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kick_boost", "win_lo", "win_hi"])
        w.writeheader()
        w.writerow({"kick_boost": 0.1, "win_lo": WIN_LO, "win_hi": WIN_HI})
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        m3.load_core_quiet_map(d)

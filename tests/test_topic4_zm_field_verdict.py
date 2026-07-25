# tests/test_topic4_zm_field_verdict.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_field_verdict import level_arm_passes, level_is_valid, adjudicate_field_screen

def M(**kw):
    d = dict(occupancy=0.9, P95=0.5, mean_P=0.3, active_area_frac=0.9, osc_frac=0.9, median_R_phase=0.2,
             phase_coverage_frac=0.9, mean_pair_corr=0.1, median_local_period_ms=200.0)
    d.update(kw); return d

def _summary(levels):
    return dict(levels=levels)

def _lvl(local_metrics, lam_local=0.05, lam_global=-0.05, glob=None):
    return dict(arms=dict(dual_local=dict(metrics=local_metrics, lambda_perp_max=lam_local),
                          dual_global=dict(metrics=[glob or M(median_R_phase=0.95)] * 4,
                                           lambda_perp_max=lam_global, period_ms=200.0)))

def test_seed_and_criteria_counting():
    n, _ = level_arm_passes([M(), M(), M(), M(occupancy=0.1)], 200.0)
    assert n == 3
    n2, _ = level_arm_passes([M(), M(), M(median_R_phase=0.95), M(osc_frac=0.1)], 200.0)
    assert n2 == 2

def test_nan_and_missing_fail_closed():
    n, _ = level_arm_passes([M(median_local_period_ms=float("nan")), M(), M(), M()], 200.0)
    assert n == 3
    n2, _ = level_arm_passes([{}, M(), M(), M()], 200.0)
    assert n2 == 3

def test_period_band_enforced():
    n, _ = level_arm_passes([M(median_local_period_ms=2000.0)] * 4, 200.0)     # 10x global -> out of band
    assert n == 0

def test_level_validity_requires_synchronised_global_control():
    assert level_is_valid(dict(median_R_phase=0.95, osc_frac=0.9))
    assert not level_is_valid(dict(median_R_phase=0.2, osc_frac=0.9))          # global desynced -> invalid
    assert not level_is_valid(dict(median_R_phase=0.95, osc_frac=0.05))        # global silent -> invalid

def test_three_consecutive_pass_gives_GO():
    lv = {str(i): _lvl([M()] * 4) for i in range(5)}
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert r["verdict"] == "GO" and len(r["passing_levels"]) >= 3

def test_non_consecutive_levels_do_not_give_GO():
    lv = {str(i): _lvl([M()] * 4) for i in (0, 2, 4)}
    lv.update({str(i): _lvl([M(occupancy=0.1)] * 4) for i in (1, 3)})
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert r["verdict"] != "GO"

def test_two_of_four_seeds_does_not_pass():
    lv = {str(i): _lvl([M(), M(), M(occupancy=0.1), M(occupancy=0.1)]) for i in range(5)}
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert r["verdict"] != "GO"

def test_invalid_global_level_is_excluded_not_counted_as_failure():
    lv = {str(i): _lvl([M()] * 4) for i in range(5)}
    lv["2"] = _lvl([M()] * 4, glob=M(median_R_phase=0.2))          # global desynced -> level excluded
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert 2 not in [int(x) for x in r["passing_levels"]]
    assert "2" in [str(x) for x in r["reasons"].get("excluded_levels", [])]

def test_subcritical_when_nonlinear_passes_but_floquet_stable():
    lv = {str(i): _lvl([M()] * 4, lam_local=-0.05) for i in range(5)}
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert r["verdict"] == "subcritical_finite_amplitude_candidate"

def test_lambda_below_the_noise_floor_is_indeterminate_not_a_verdict():
    """spec §6.2: |lam| under the discretisation error floor cannot resolve a sign, so it must NOT be
    reported as a stable/unstable cell and must never yield GO."""
    lv = {str(i): _lvl([M()] * 4, lam_local=5e-4, lam_global=-5e-4) for i in range(5)}
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert r["taxonomy"] == "indeterminate_below_noise_floor"
    assert r["verdict"] != "GO"


def test_taxonomy_four_cells():
    def tax(ll, lg):   # magnitudes deliberately ABOVE TH["lam_floor"]=2e-3 so signs are resolvable
        lv = {str(i): _lvl([M(occupancy=0.1)] * 4, lam_local=ll, lam_global=lg) for i in range(5)}
        return adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))["taxonomy"]
    assert tax(-0.05, -0.05) == "both_stable"
    assert tax(0.05, 0.05) == "both_unstable"
    assert tax(-0.05, 0.05) == "global_unstable_local_stable"
    assert tax(0.05, -0.05) == "global_stable_local_unstable"

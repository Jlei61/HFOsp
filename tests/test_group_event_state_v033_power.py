"""Power curves and smoke runner (v0.3.3 plan Task 6, clauses P1-P5)."""
from __future__ import annotations

import numpy as np
import pytest

from src.topic5_group_event_state.v033_evaluator import power as P
from tests.test_group_event_state_v033_toyutil import toy_scaffold


def test_assay_runner_presets_are_fixed_and_smoke_has_three_seeds_per_truth():
    from scripts.run_group_event_state_v033_assay import _cells

    sentinel = _cells("sentinel", 99)
    assert [(x.kind, x.replicate) for x in sentinel] == [("D0", 0), ("D3", 0)]
    smoke = _cells("smoke", 3)
    assert len(smoke) == 15
    assert all(sum(x.kind == kind for x in smoke) == 3 for kind in ("D0", "D1", "D2", "D3", "D4"))
    assert len(_cells("power", 3)) == 39


def test_block_snr_and_required_blocks_follow_the_declared_formula():
    block_means = np.array([0.2, 0.4, 0.3, 0.1, 0.5])
    snr = P.block_snr(block_means)
    assert np.isclose(snr, block_means.mean() / block_means.std(ddof=1))
    # two-sided alpha 0.05, power 0.8 -> (1.96 + 0.8416)^2 / snr^2
    assert P.required_blocks_for_power(1.0) == 8
    assert P.required_blocks_for_power(0.5) == 32
    assert P.required_blocks_for_power(0.0) is None and P.required_blocks_for_power(-1.0) is None


def test_run_replicate_returns_both_views_with_provenance_and_resources():
    sc = toy_scaffold(seed=1)
    spec = P.ReplicateSpec(kind="D3", beta_count=0.7, beta_grammar=2.0, replicate=0)
    res = P.run_replicate(sc, spec, horizon=1800.0, views=("count", "grammar"), levels=(0, 1), n_steps=40)
    assert res["spec"]["kind"] == "D3" and res["spec"]["generator_seed"] != res["spec"]["noise_seed"]
    assert set(res["cascades"]) == {"count", "grammar"}
    for view in ("count", "grammar"):
        c = res["cascades"][view]
        assert [lvl["level"] for lvl in c["levels"]] == [0, 1]
        assert "block_snr" in c["levels"][0] and "table" not in c["levels"][0]
    assert res["resources"]["wall_seconds"] > 0 and res["resources"]["peak_rss_mib"] > 0
    assert res["spec"]["seeds_recorded"] is True


def _fake_replicate(kind, beta_c, beta_g, rep, gains, detected, truth):
    levels = [{"level": lvl, "gain": gains[lvl], "detected": detected[lvl], "ci_lower": -0.1 + gains[lvl],
               "block_snr": 2.0 * gains[lvl], "n_blocks": 20} for lvl in (0, 1, 2)]
    return {"spec": {"kind": kind, "beta_count": beta_c, "beta_grammar": beta_g, "replicate": rep},
            "cascades": {"count": {"truth_has_state": truth, "levels": levels,
                                   "failure_location": "none" if truth else "not_applicable_no_state"}}}


def test_power_curve_groups_cells_and_reports_power_false_positive_and_oracle_gain():
    reps = []
    for rep in range(3):
        reps.append(_fake_replicate("D0", 0.0, 0.0, rep, {0: 0.001, 1: 0.0, 2: -0.002}, {0: rep == 0, 1: False, 2: False}, False))
        reps.append(_fake_replicate("D1", 0.2, 0.0, rep, {0: 0.05 + 0.01 * rep, 1: 0.04, 2: 0.03},
                                    {0: True, 1: True, 2: rep < 2}, True))
    curve = P.power_curve(reps, view="count")
    cells = {(c["kind"], c["beta_count"]): c for c in curve["cells"]}
    d0, d1 = cells[("D0", 0.0)], cells[("D1", 0.2)]
    assert d0["n_replicates"] == 3 and d0["truth_has_state"] is False
    assert d0["false_positive_rate_by_level"]["0"] == pytest.approx(1 / 3)
    assert "power_by_level" not in d0
    assert d1["power_by_level"]["2"] == pytest.approx(2 / 3) and d1["power_by_level"]["0"] == 1.0
    assert d1["oracle_gain_level0"]["median"] == pytest.approx(0.06)
    assert d1["block_snr_by_level"]["0"]["median"] == pytest.approx(0.12)
    assert curve["effect_axis"] == "level0_oracle_held_out_deviance_gain_nats_per_anchor"


def test_effect_tiers_pick_the_beta_whose_oracle_gain_is_closest_to_the_tier_target():
    cells = [{"kind": "D1", "beta_count": b, "truth_has_state": True,
              "oracle_gain_level0": {"median": g}, "block_snr_by_level": {"0": {"median": 3 * g}}}
             for b, g in ((0.1, 0.01), (0.2, 0.045), (0.4, 0.16))]
    tiers = P.assign_effect_tiers(cells, beta_key="beta_count")
    assert tiers["medium"]["beta_count"] == 0.2 and tiers["small"]["beta_count"] == 0.1 and tiers["large"]["beta_count"] == 0.4
    assert tiers["medium"]["required_blocks_level0"] == P.required_blocks_for_power(3 * 0.045)
    assert tiers["definition"]["medium_target_gain_nats"] == P.EFFECT_TIER_TARGETS["medium"]

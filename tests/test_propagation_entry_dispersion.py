"""Tests for src.propagation_entry_dispersion.

Two layers:
  * deterministic mechanics (earliest pick, Neff, spatial radius, shape,
    downstream invariance) -- hand-checkable.
  * the load-bearing single-template-noise null -- calibration (one noisy
    template must NOT be flagged as excess) + sensitivity (a planted off-template
    entry mode must break the monotonic-decay shape signature).
"""
import numpy as np
import pytest

from src.propagation_entry_dispersion import (
    earliest_channel_per_event,
    effective_number,
    spatial_radius,
    entry_shape,
    downstream_invariance_to_entry,
    single_template_noise_null,
    earliest_prob_vector,
)


def _ranks_to_masked_norm(latent, part):
    """Helper: turn per-event latent scores + participation into masked
    normalized within-event ranks (n_ch, n_ev), NaN for non-participating.
    Mirrors mask_phantom_ranks(normalize=True)."""
    n_ch, n_ev = latent.shape
    out = np.full((n_ch, n_ev), np.nan)
    for e in range(n_ev):
        idx = np.where(part[:, e])[0]
        if idx.size == 0:
            continue
        if idx.size == 1:
            out[idx[0], e] = 0.5
            continue
        lr = np.argsort(np.argsort(latent[idx, e]))
        out[idx, e] = lr / (idx.size - 1)
    return out


# --------------------------- deterministic mechanics ---------------------------
def test_earliest_pick_and_no_participant():
    m = np.array([[0.0, np.nan], [0.5, np.nan], [1.0, np.nan]])
    e = earliest_channel_per_event(m)
    assert e[0] == 0          # smallest rank wins
    assert e[1] == -1         # no participant


def test_effective_number_bounds():
    # uniform over 4 channels -> Neff == 4
    earl = np.array([0, 1, 2, 3] * 25)
    r = effective_number(earl, 4)
    assert abs(r["neff"] - 4.0) < 1e-9
    assert abs(r["top_share"] - 0.25) < 1e-9
    # all same -> Neff == 1
    r2 = effective_number(np.zeros(50, int), 4)
    assert abs(r2["neff"] - 1.0) < 1e-9


def test_spatial_radius_known():
    coords = np.array([[0.0, 0, 0], [2, 0, 0], [4, 0, 0]])
    mapped = np.ones(3, bool)
    prob = np.array([0.5, 0.0, 0.5])  # mass at x=0 and x=4 -> centroid x=2, dist 2
    r = spatial_radius(prob, coords, mapped)
    assert abs(r["radius_mm"] - 2.0) < 1e-9
    # too few mapped -> NaN
    r2 = spatial_radius(np.array([1.0, 0, 0]), coords, np.array([1, 0, 0], bool))
    assert np.isnan(r2["radius_mm"])


def test_entry_shape_monotonic_vs_offtemplate():
    n_ch = 6
    # monotonic: earliest-prob decreases with template rank
    earl_mono = np.repeat([0, 0, 0, 1, 1, 2], 10)          # ch0 (trank0) dominates
    trank = [0, 1, 2, 3, 4, 5]
    s_mono = entry_shape(earl_mono, trank, n_ch)
    assert s_mono["spearman_prob_vs_trank"] < -0.5          # strong monotonic decay
    # off-template: a template-LATE channel (trank 5 == ch5) often leads
    earl_off = np.repeat([0, 5, 5, 5, 1, 2], 10)
    s_off = entry_shape(earl_off, trank, n_ch)
    assert s_off["spearman_prob_vs_trank"] > s_mono["spearman_prob_vs_trank"]
    # the modal earliest is a template-late channel
    assert s_off["modal_trank"] >= 3


def test_downstream_invariance_identical_groups():
    # two entry channels, but identical downstream order in both groups -> ~1
    n_ch, n_ev = 5, 400
    rng = np.random.default_rng(1)
    masked = np.full((n_ch, n_ev), np.nan)
    earliest = np.empty(n_ev, int)
    for e in range(n_ev):
        if e % 2 == 0:        # group A: ch0 enters, downstream 1<2<3<4
            order = [0, 1, 2, 3, 4]; earliest[e] = 0
        else:                 # group B: ch1 enters, downstream order same on 0,2,3,4
            order = [1, 0, 2, 3, 4]; earliest[e] = 1
        for r, ch in enumerate(order):
            masked[ch, e] = r / (n_ch - 1)
    di = downstream_invariance_to_entry(masked, earliest, n_ch)
    assert di["n_groups"] == 2
    assert di["median_cross_entry_spearman"] > 0.8   # downstream invariant to entry


# --------------------------- the null (load-bearing) ---------------------------
def test_null_calibration_one_noisy_template():
    """One template + symmetric noise must NOT be flagged as excess dispersion."""
    rng = np.random.default_rng(7)
    n_ch, n_ev = 8, 3000
    template = np.arange(n_ch, dtype=float)
    latent = template[:, None] + rng.normal(0, 1.2, size=(n_ch, n_ev))
    part = np.ones((n_ch, n_ev), bool)
    masked = _ranks_to_masked_norm(latent, part)
    res = single_template_noise_null(masked, None, None, n_reps=400,
                                     rng=np.random.default_rng(0))
    # observed should sit inside the null band, NOT exceed it -- BOTH nulls
    assert res["p_neff_excess"] > 0.05, res            # pooled
    assert res["p_neff_excess_gauss"] > 0.05, res      # per-channel Gaussian
    assert res["null_neff"]["lo"] <= res["obs_neff"] <= res["null_neff"]["hi"], res


def test_null_detects_concentration():
    """If one channel leads far more than rank-noise allows, the null should
    say observed is MORE concentrated (p_neff_concentrated small)."""
    rng = np.random.default_rng(3)
    n_ch, n_ev = 8, 3000
    template = np.arange(n_ch, dtype=float)
    latent = template[:, None] + rng.normal(0, 1.2, size=(n_ch, n_ev))
    # force ch0 to be earliest almost always (beyond its noisy-template share)
    latent[0, :] -= 6.0
    part = np.ones((n_ch, n_ev), bool)
    masked = _ranks_to_masked_norm(latent, part)
    res = single_template_noise_null(masked, None, None, n_reps=400,
                                     rng=np.random.default_rng(0))
    assert res["obs_top_share"] > 0.8
    assert res["p_neff_concentrated"] < 0.05, res


def test_null_detects_planted_excess():
    """A 1-D continuum / blend (a moving early-center, so the entry slides across
    many channels) genuinely over-disperses the entry beyond ANY single template.
    The null MUST flag excess (p_neff_excess small) -- without this, the
    robust_excess verdict would be untestable in the positive direction."""
    rng = np.random.default_rng(11)
    n_ch, n_ev = 10, 3000
    phi = rng.uniform(0, n_ch - 1, size=n_ev)             # moving early-center
    cc = np.arange(n_ch)[:, None].astype(float)
    latent = (cc - phi[None, :]) ** 2                     # earliest = channel nearest phi
    part = np.ones((n_ch, n_ev), bool)
    masked = _ranks_to_masked_norm(latent, part)
    res = single_template_noise_null(masked, None, None, n_reps=400,
                                     rng=np.random.default_rng(0))
    # entry is spread far wider than a fixed-template + noise can produce
    assert res["obs_neff"] > res["null_neff"]["hi"], res
    assert res["p_neff_excess"] < 0.05, res               # pooled detects it


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

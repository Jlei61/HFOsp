import numpy as np
from src.topic5_scaffold_ab_contrast import build_D_AB, template_pair_tier


def test_build_D_AB_earlyness_sign():
    rank_a = np.array([0., 1., 2., 3., 4., 5.])   # contact0 earliest = A source
    rank_b = rank_a[::-1].copy()                   # B fully anti-correlated
    out = build_D_AB(rank_a, rank_b)
    assert out["D_AB"][0] > 0 and out["D_AB"][-1] < 0        # A source end D_AB>0
    assert out["rho_AB"] < -0.99                              # anti -> rho approx -1
    zA, zB = -out["eA"], -out["eB"]
    assert abs(out["rho_AB"] - np.corrcoef(zA, zB)[0,1]) < 1e-9


def test_template_pair_tier_boundaries():
    assert template_pair_tier(-0.6) == "reciprocal"
    assert template_pair_tier(-0.5) == "reciprocal"
    assert template_pair_tier(0.0)  == "oblique"
    assert template_pair_tier(0.5)  == "aligned"
    assert template_pair_tier(0.9)  == "hard_degenerate"


from src.topic5_scaffold_ab_contrast import derive_joint_contacts


def _mk_matched(names, ranks):
    return [{"name": n, "typical_rank": r, "x_norm": i*0.1, "y_norm": 0.0, "support": 1.0}
            for i, (n, r) in enumerate(zip(names, ranks))]


def test_joint_requires_finite_in_A_B_and_windows():
    names = [f"A{i}-A{i+1}" for i in range(6)]
    matched = _mk_matched(names, [0,1,2,3,4,5])
    axis_b = {"channels": [{"name": n, "typical_rank": 5-i} for i,n in enumerate(names)]}
    wv = np.random.default_rng(0).normal(size=(10, 6))
    out = derive_joint_contacts(matched, axis_b, wv)
    assert out["status"] == "ok" and out["n_joint"] == 6 and out["tier"] == "reciprocal"


def test_joint_insufficient_when_lt_6():
    names = [f"A{i}-A{i+1}" for i in range(4)]
    matched = _mk_matched(names, [0,1,2,3])
    axis_b = {"channels": [{"name": n, "typical_rank": 3-i} for i,n in enumerate(names)]}
    out = derive_joint_contacts(matched, axis_b, np.zeros((10,4)))
    assert out["status"] == "insufficient_joint"


def test_joint_hard_degenerate_when_templates_identical():
    names = [f"A{i}-A{i+1}" for i in range(6)]
    matched = _mk_matched(names, [0,1,2,3,4,5])
    axis_b = {"channels": [{"name": n, "typical_rank": i} for i,n in enumerate(names)]}  # B==A
    out = derive_joint_contacts(matched, axis_b, np.random.default_rng(1).normal(size=(10,6)))
    assert out["status"] == "hard_degenerate"


from src.topic5_scaffold_ab_contrast import contrast_timecourse


def test_contrast_direct_is_source_of_truth():
    rng = np.random.default_rng(0)
    ranks_a = np.arange(8.0)
    ranks_b = ranks_a[::-1].copy()
    d = build_D_AB(ranks_a, ranks_b)
    E = rng.normal(size=(5, 8))
    out = contrast_timecourse(E, d["D_AB"], d["eA"], d["eB"])
    for w in range(5):
        assert abs(out["C_AB"][w] - np.corrcoef(E[w], d["D_AB"])[0, 1]) < 1e-9


def test_closed_form_only_on_full_finite():
    ranks_a = np.arange(8.0)
    ranks_b = ranks_a[::-1].copy()
    d = build_D_AB(ranks_a, ranks_b)
    E = np.random.default_rng(1).normal(size=(1, 8))
    o = contrast_timecourse(E, d["D_AB"], d["eA"], d["eB"])
    rho = d["rho_AB"]
    closed = (o["r_A"][0] - o["r_B"][0]) / np.sqrt(2 * (1 - rho))
    assert abs(o["C_AB"][0] - closed) < 1e-9


def test_partial_window_uses_direct_not_closed():
    ranks_a = np.arange(8.0)
    ranks_b = ranks_a[::-1].copy()
    d = build_D_AB(ranks_a, ranks_b)
    E = np.random.default_rng(2).normal(size=(1, 8))
    E[0, 3] = np.nan
    o = contrast_timecourse(E, d["D_AB"], d["eA"], d["eB"])
    m = np.isfinite(E[0])
    assert abs(o["C_AB"][0] - np.corrcoef(E[0, m], d["D_AB"][m])[0, 1]) < 1e-9


from src.topic5_scaffold_ab_contrast import axis_present


def test_axis_present_true_when_energy_matches_template():
    # 8 joint contacts on 2 multi-contact shafts (A, B), 4 contacts each, interleaved by rank
    # (idx 0,2,4,6 -> A1..A4; idx 1,3,5,7 -> B1..B4). NOTE: a literal "4 shafts of 2" split
    # (as sketched in the plan doc) was verified NOT to work here -- with only 2 possible
    # within-shaft orderings per shaft, 4 such shafts give just 2**4=16 distinct null
    # realizations, so the "everyone stays put" realization alone ties the observed value
    # with probability 1/16 (~0.0625), which floors the best-achievable p above alpha=0.05
    # regardless of signal strength (verified by exhaustive enumeration). Two shafts of 4
    # contacts (24*24=576 realizations) clears that floor with a wide margin.
    ranks_a = np.arange(8.0)
    ranks_b = ranks_a[::-1].copy()
    d = build_D_AB(ranks_a, ranks_b)
    names = ["A1", "B1", "A2", "B2", "A3", "B3", "A4", "B4"]
    E = np.tile(d["eA"], (6, 1)) + np.random.default_rng(0).normal(scale=0.05, size=(6, 8))
    out = axis_present(E, names, d["eA"], d["eB"], np.random.default_rng(0))
    assert out["testable"] and out["present"].mean() > 0.5


def test_axis_present_low_dof_when_mostly_singletons():
    # 8 joint contacts, each its own shaft (A1,B1,...,H1) -> 0 multi-contact shafts.
    ranks_a = np.arange(8.0)
    ranks_b = ranks_a[::-1].copy()
    d = build_D_AB(ranks_a, ranks_b)
    names = ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"]
    E = np.random.default_rng(1).normal(size=(6, 8))
    out = axis_present(E, names, d["eA"], d["eB"], np.random.default_rng(1))
    assert out["low_dof"] and not out["testable"]

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

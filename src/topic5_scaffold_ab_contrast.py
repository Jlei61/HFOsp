import numpy as np

RHO_RECIPROCAL, RHO_ALIGNED, RHO_DEGEN = -0.5, 0.5, 0.85


def _zscore(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def build_D_AB(rank_a, rank_b):
    eA = -_zscore(rank_a)          # larger = earlier = source-like (rank low=early)
    eB = -_zscore(rank_b)
    D_AB = eA - eB
    rho_AB = float(np.corrcoef(eA, eB)[0, 1]) if eA.std() > 1e-12 and eB.std() > 1e-12 else 1.0
    return {"eA": eA, "eB": eB, "D_AB": D_AB, "rho_AB": rho_AB, "sd_D_AB": float(D_AB.std(ddof=0))}


def template_pair_tier(rho_AB):
    if rho_AB <= RHO_RECIPROCAL: return "reciprocal"
    if rho_AB < RHO_ALIGNED:     return "oblique"
    if rho_AB < RHO_DEGEN:       return "aligned"
    return "hard_degenerate"

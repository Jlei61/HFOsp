"""Path-B cheap-first screen (task §7 Path B): a reduced K-patch rate model of the inhibitory-containment
structure. The full-SNN carrier gate said the Z/M+S_G state is an HFO burst train because a SINGLE GLOBAL
scalar S_G synchronously resets the whole core each cycle. This cheap model asks whether SPATIALLY
RESOLVED inhibition (patchwise local pool, or local + weak global) desynchronizes the microdomains so the
POPULATION signal stays elevated (a sustained-carrier proxy) instead of collapsing between synchronized
bursts.

Each patch i: fast activity a_i (saturating rate) with a slow divisive inhibitory pool.
  tau_a da_i/dt = -a_i + F( (I_i + w_c * neighbours(a)_i) / (1 + g_S * S_i) - theta )
  F(u) = a_max * [u]_+ / (u_half + [u]_+)                 (saturating -> a bounded)
  local pool  : tau_s ds_i/dt = -s_i + a_i
  global pool : tau_s ds_g/dt = -s_g + mean(a)
  S_i = s_g (global) | s_i (patchwise) | (1-eps) s_i + eps s_g (local_global)

This is a SCREEN, not a claim: it only tells us which inhibitory STRUCTURE *could* sustain a population
carrier in a reduced model. A positive result must still be migrated to the full anisotropic SNN and pass
the pre-registered A+B carrier gate. NO E->E change is implied (the pool is inhibition-side only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SYNC_MAX = 0.5     # carrier_proxy requires the patches to be DESYNCHRONIZED (mean pairwise corr below this)


@dataclass
class PatchParams:
    K: int = 8              # number of patches (2-8, task §7)
    tau_a: float = 10.0     # ms, fast activity
    tau_s: float = 80.0     # ms, inhibitory pool (matches the SNN S_G tau_S=80)
    I0: float = 1.0         # mean drive (depleted-z operating point where the burst train lives)
    sigma_I: float = 0.0    # drive heterogeneity across patches (0 = homogeneous)
    w_rec: float = 2.0      # SELF-excitation -> fast bistability (N-shaped nullcline) -> relaxation oscillation
    w_c: float = 0.0        # nearest-neighbour excitatory coupling (ring)
    theta: float = 0.5      # activation threshold
    g_S: float = 16.0       # divisive inhibition strength (matches alpha_G=16)
    a_max: float = 1.0      # activity ceiling (saturating F -> bounded)
    u_half: float = 0.5     # F half-saturation
    eps_global: float = 0.5  # local_global mix weight on the global pool
    pool_sigma: float = 0.0  # spatial smoothing (patch units) of the activity driving the LOCAL pool:
                             # 0 = fully independent patchwise pools; larger = graded shared containment
                             # (task §7 "spatial smoothing / low-rank shared component")
    mode: str = "global"    # global | patchwise | local_global
    seed: int = 0


def _F(u, a_max, u_half):
    x = np.maximum(u, 0.0)
    return a_max * x / (u_half + x)


def _ring_neighbours(a):
    return np.roll(a, 1) + np.roll(a, -1)


def _smooth_ring(x, sigma):
    """Circular Gaussian smoothing of `x` over patch index (sigma in patch units); sigma<=0 -> identity."""
    if sigma <= 0.0:
        return x
    K = len(x)
    off = np.arange(-(K // 2), K - K // 2)
    ker = np.exp(-0.5 * (off / sigma) ** 2)
    ker /= ker.sum()
    return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(np.roll(ker, K // 2))))


def simulate(p: PatchParams, T_ms=4000.0, dt=0.5):
    """Forward-Euler integrate the K-patch model. Returns dict(a (n,K), s_loc (n,K), s_glob (n,), t, dt)."""
    rng = np.random.default_rng(p.seed)
    K = p.K
    I = p.I0 + p.sigma_I * rng.standard_normal(K)
    a = 0.05 * np.abs(rng.standard_normal(K))          # small asymmetric IC (breaks symmetry for desync)
    s_loc = np.zeros(K)
    s_glob = 0.0
    n = int(round(T_ms / dt))
    A = np.zeros((n, K), np.float32)
    Sl = np.zeros((n, K), np.float32)
    Sg = np.zeros(n, np.float32)
    for t in range(n):
        if p.mode == "global":
            S = np.full(K, s_glob)
        elif p.mode == "patchwise":
            S = s_loc
        elif p.mode == "local_global":
            S = (1.0 - p.eps_global) * s_loc + p.eps_global * s_glob
        else:
            raise ValueError(f"unknown mode {p.mode!r}")
        drive = I + p.w_rec * a + (p.w_c * _ring_neighbours(a) if K > 1 else 0.0)
        u = drive / (1.0 + p.g_S * S) - p.theta
        a = a + dt / p.tau_a * (-a + _F(u, p.a_max, p.u_half))
        a = np.maximum(a, 0.0)
        a_pool = _smooth_ring(a, p.pool_sigma) if p.pool_sigma > 0.0 else a   # graded local containment (optional)
        s_loc = s_loc + dt / p.tau_s * (-s_loc + a_pool)   # both pools always advance; mode selects which S is used
        s_glob = s_glob + dt / p.tau_s * (-s_glob + a.mean())
        A[t] = a
        Sl[t] = s_loc
        Sg[t] = s_glob
    return dict(a=A, s_loc=Sl, s_glob=Sg, t=np.arange(n) * dt, dt=dt)


def population_signal(A):
    return np.asarray(A).mean(axis=1)


def population_occupancy(P, settle_frac=0.25, floor_frac=0.20):
    """Fraction of the (post-settling) population signal that stays above a floor set at `floor_frac` of the
    ACTIVE peak, with the OFF state (P=0) as the absolute baseline -- consistent with the SNN gate whose
    pre-onset baseline is ~0. A sustained carrier -> occupancy near 1; a synchronized burst train that
    collapses toward P=0 between bursts -> low. (A dead flat plateau also reads ~1; screen_metrics gates it
    out via patch_osc, so occupancy alone never certifies a carrier.)"""
    P = np.asarray(P, float)
    P = P[int(len(P) * settle_frac):]
    peak = float(np.percentile(P, 95))
    if peak <= 1e-9:
        return 0.0            # fully off
    floor = floor_frac * peak
    return float((P >= floor).mean())


def synchrony(A, settle_frac=0.25):
    """Mean pairwise temporal correlation across patches. High => synchronized (bursts together);
    low/negative => desynchronized microdomains (the mechanism that could fill the troughs)."""
    A = np.asarray(A, float)
    A = A[int(A.shape[0] * settle_frac):]
    K = A.shape[1]
    if K < 2:
        return 1.0
    if np.allclose(A.std(axis=0), 0):
        return 1.0
    C = np.corrcoef(A.T)
    iu = np.triu_indices(K, 1)
    return float(np.nanmean(C[iu]))


def screen_metrics(res, settle_frac=0.25):
    """Carrier-proxy summary for one run: population occupancy, synchrony, activity, oscillation depth."""
    A = res["a"]
    P = population_signal(A)
    ps = P[int(len(P) * settle_frac):]
    occ = population_occupancy(P, settle_frac)
    sync = synchrony(A, settle_frac)
    mean_act = float(ps.mean())
    patch_osc = float(np.mean(A[int(A.shape[0] * settle_frac):].std(axis=0)))   # per-patch temporal std, averaged
    p_depth = float((ps.max() - ps.min()) / (ps.mean() + 1e-9))                  # population oscillation depth
    # sustained population carrier proxy: active, patches oscillate, DESYNCHRONIZED (the narrated mechanism),
    # and P stays up (not collapsing to baseline). Desync is now a PASS CONDITION, not just narration -- a
    # synchronized-but-high-baseline oscillation must NOT count as a carrier.
    carrier_proxy = bool(mean_act > 0.02 and patch_osc > 0.01 and occ >= 0.80 and sync < SYNC_MAX)
    return dict(occupancy=occ, synchrony=sync, mean_activity=mean_act, patch_osc=patch_osc,
                pop_depth=p_depth, carrier_proxy=carrier_proxy)

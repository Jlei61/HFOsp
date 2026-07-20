"""Topic 4 — MZ full-SNN state-aligned finite-time spatial mode tracking: pure science math.

Design contract (BINDING): docs/superpowers/specs/2026-07-21-topic4-mz-m-eigenmode-tracking-design.md

Scientific object: how the empirical finite-time spatial response of the COMPLETE current-based MZ
spiking network (≈40k E/I LIF) — and its identifiable finite-time singular modes — reorganize along
the z+m plateau slow-state trajectory once the adaptation variable m genuinely participates. NOT a
z-only repeat, NOT a rate-field Jacobian, NOT an exact eigenanalysis, NOT a seizure proof.

IMPORT-SAFE and SIDE-EFFECT-FREE: no simulations, no file writes (those live in
scripts/run_topic4_mz_m_eigenmode_tracking.py). It provides:
  1. build_zm_slow_config — the locked z+m plateau slow config (eta_m via eta_m_from_frac, E1).
  2. resting_mask + register_states — 5-state registration from D/a/rate + time ONLY (spec §3, E2).
  3. trajectory_parity — replay <-> upstream NPZ agreement (E3).
  4. transform_m + apply_m_control — the m-mechanism counterfactuals (E6/E7/E8).
  5. principal_angles_deg / subspace_alignment / weighted_centroid / centroid_displacement — the
     cross-state mode-tracking geometry (sign-invariant, degenerate-subspace safe; E16/E17).
  6. state_checkpoint_fingerprint — deterministic full-state hash for parity + manifest (E19).

Reuse (do not reinvent): src.topic4_mz_slowvars.eta_m_from_frac, mz_slow_vars.MZSlowVarsConfig, and
(in the runner) the direct-spatial fork / operator / kick / audit functions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENG = os.path.join(_ROOT, "src", "snn_engine")
if _ENG not in sys.path:
    sys.path.insert(0, _ENG)

from src.topic4_mz_slowvars import eta_m_from_frac  # noqa: E402
from mz_slow_vars import MZSlowVarsConfig  # noqa: E402

SCHEMA_VERSION = "mz-m-eigenmode-tracking-1.0"


# ======================================================================== E1 z+m plateau slow config
def build_zm_slow_config(work_point, I_EE_scale) -> MZSlowVarsConfig:
    """Locked z+m plateau MZSlowVarsConfig. eta_m is computed by eta_m_from_frac(A_target, I_EE_scale,
    peak_m) — NEVER hardcoded (spec §1/E1)."""
    eta_m = eta_m_from_frac(float(work_point["A_target"]), float(I_EE_scale),
                            float(work_point["peak_m_tau2000"]))
    return MZSlowVarsConfig(use_z=bool(work_point["use_z"]), use_m=bool(work_point["use_m"]),
                            I_th_EI=float(work_point["I_th_EI"]), tau_z=float(work_point["tau_z"]),
                            tau_adp=float(work_point["tau_adp_ms"]), eta_m=eta_m)


# ======================================================================== resting mask (mirrors DSM._resting_mask)
def resting_mask(rate_hz, dt_ms, *, win_ms=20.0, k=0.3):
    """Resting (non-event) steps: the `win_ms`-smoothed population E-rate below floor + k*(peak-floor),
    with floor = P20 and peak = P99 of the smoothed rate. Mirrors the direct-spatial runner's
    `_resting_mask` so registration lands checkpoints off fast-event peaks (spec §3)."""
    r = np.asarray(rate_hz, float)
    w = max(1, int(round(float(win_ms) / float(dt_ms))))
    sm = np.convolve(r, np.ones(w) / w, mode="same")
    floor = float(np.percentile(sm, 20))
    peak = float(np.percentile(sm, 99))
    return sm <= floor + float(k) * (peak - floor)


# ======================================================================== E2 state registration (spec §3)
def register_states(D, a, rate_hz, dt_ms, *, baseline_ms, baseline_search_halfwidth_ms, approach_fracs,
                    approach_search_ms, settle_tail_ms, resting_win_ms, resting_k, settled_D_ptp_max,
                    settled_a_ptp_max, settled_min_resting_frac, D_onset_ref):
    """Register baseline / approach_25/50/75 / settled_plateau from the slow-state trajectory ONLY
    (D=1-z̄, a=A_abs/I_EE_scale, population rate) — NEVER a perturbation / mode / figure (E2).

    baseline = resting step nearest baseline_ms; D_base = D there.
    D_plateau = median resting D in the settled tail window.
    approach_f = FIRST time D crosses D_base + f*(D_plateau-D_base); within approach_search_ms of that
                 first crossing pick the lowest-rate resting step (never a fast-event peak).
    settled_plateau = resting step nearest the tail-median D, accepted ONLY if the settled gate passes
                 (tail D/a locally flat, enough resting fraction, D_plateau in the elevated band). A
                 failed gate / unreachable crossing -> that state is unresolved (branch_step=None)."""
    D = np.asarray(D, float); a = np.asarray(a, float); rate = np.asarray(rate_hz, float)
    n = len(D)
    rest = resting_mask(rate, dt_ms, win_ms=resting_win_ms, k=resting_k)

    def _resting_nearest(target, lo, hi):
        idx = np.arange(lo, hi)
        r = idx[rest[lo:hi]]
        pool = r if r.size else idx
        return int(pool[np.argmin(np.abs(pool - target))])

    def _min_rate_resting(lo, hi):
        idx = np.arange(lo, hi)
        r = idx[rest[lo:hi]]
        pool = r if r.size else idx
        return int(pool[np.argmin(rate[pool])])

    b0 = int(round(baseline_ms / dt_ms))
    hw = int(round(baseline_search_halfwidth_ms / dt_ms))
    base_step = _resting_nearest(b0, max(0, b0 - hw), min(n, b0 + hw + 1))
    D_base = float(D[base_step])

    tail_steps = int(round(settle_tail_ms / dt_ms))
    tail_lo = max(0, n - tail_steps)
    tail = np.arange(tail_lo, n)
    tail_rest = rest[tail_lo:n]
    tail_pool = tail[tail_rest] if tail_rest.any() else tail
    D_plateau = float(np.median(D[tail_pool]))
    resting_frac = float(tail_rest.mean()) if tail.size else 0.0
    D_ptp = float(np.ptp(D[tail_pool])); a_ptp = float(np.ptp(a[tail_pool]))
    settled = bool(D_ptp < settled_D_ptp_max and a_ptp < settled_a_ptp_max
                   and resting_frac >= settled_min_resting_frac
                   and D_plateau > 0.5 * D_onset_ref and D_plateau > D_base)
    settled_step = int(tail_pool[np.argmin(np.abs(D[tail_pool] - D_plateau))])

    states = {"baseline": dict(branch_step=base_step, resolved=True, D=D_base,
                               a=float(a[base_step]), rate_hz=float(rate[base_step]))}
    search = int(round(approach_search_ms / dt_ms))
    span = D_plateau - D_base
    for f in approach_fracs:
        name = f"approach_{int(round(f * 100))}"
        target = D_base + f * span
        cross = np.where(D[base_step:] >= target)[0] if span > 0 else np.array([], int)
        if span <= 0 or cross.size == 0:
            states[name] = dict(branch_step=None, resolved=False, target_D=(float(target) if span > 0 else None))
            continue
        c0 = base_step + int(cross[0])
        step = _min_rate_resting(c0, min(n, c0 + search + 1))
        states[name] = dict(branch_step=step, resolved=True, target_D=float(target),
                            D=float(D[step]), a=float(a[step]), rate_hz=float(rate[step]))
    states["settled_plateau"] = dict(
        branch_step=(settled_step if settled else None), resolved=bool(settled), settled=bool(settled),
        D=(float(D[settled_step]) if settled else None), a=(float(a[settled_step]) if settled else None),
        rate_hz=(float(rate[settled_step]) if settled else None),
        settled_D_ptp=D_ptp, settled_a_ptp=a_ptp, resting_frac=resting_frac)
    return dict(states=states, D_base=D_base, D_plateau=D_plateau, settled=bool(settled),
                n_steps=int(n), dt_ms=float(dt_ms))


# ======================================================================== E3 replay <-> NPZ parity
def _parity_field(x, y, rel_tol):
    x = np.asarray(x, float); y = np.asarray(y, float)
    L = min(len(x), len(y))
    x, y = x[:L], y[:L]
    max_abs = float(np.max(np.abs(x - y))) if L else float("nan")
    denom = float(np.max(np.abs(y))) if L else 0.0
    rel = (max_abs / denom) if denom > 0 else (0.0 if max_abs == 0 else float("inf"))
    return {"max_abs": max_abs, "rel": float(rel), "n": int(L), "pass": bool(rel <= rel_tol)}


def trajectory_parity(D_rep, a_rep, rate_rep, D_ref, a_ref, rate_ref, *, rel_tol):
    """Per-field (D, a, rate) max-abs + relative deviation of the replayed trajectory vs the upstream
    NPZ, with a pass gate at rel_tol (spec §3.2/E3). Same code/substrate/seed => expect near-bit
    identical; a relative deviation over rel_tol is a STOP-and-report discrepancy."""
    out = {"D": _parity_field(D_rep, D_ref, rel_tol), "a": _parity_field(a_rep, a_ref, rel_tol),
           "rate": _parity_field(rate_rep, rate_ref, rel_tol), "rel_tol": float(rel_tol)}
    out["pass"] = bool(all(out[k]["pass"] for k in ("D", "a", "rate")))
    return out


# ======================================================================== E6/E7/E8 m counterfactuals
def transform_m(m_E, kind, *, seed=None):
    """m-mechanism counterfactual on the E-cell adaptation vector (spec §4 P4):
      native_zm -> unchanged copy; m_reset -> zeros (immediate m-current removed);
      m_uniform -> mean(m) everywhere (keep the mean brake, flatten the spatial pattern);
      m_shuffle -> a seeded permutation (keep the distribution, scramble location)."""
    m = np.asarray(m_E, float).copy()
    if kind == "native_zm":
        return m
    if kind == "m_reset":
        return np.zeros_like(m)
    if kind == "m_uniform":
        return np.full_like(m, float(m.mean()))
    if kind == "m_shuffle":
        out = m.copy()
        np.random.default_rng(seed).shuffle(out)
        return out
    raise ValueError(f"unknown m-control kind: {kind!r}")


def apply_m_control(ck, kind, NE, *, seed=None):
    """Return a checkpoint whose E-cell m is replaced by transform_m(kind); the fast state (V, ref,
    currents, rings, xi, RNG) and z are SHARED with `ck` (common random numbers + same z), so a fork
    from the returned checkpoint differs from native ONLY in the adaptation current (E6/E9). The
    original checkpoint's slow.m is left intact (private deep copy)."""
    ck_m = copy.copy(ck)                          # shallow: shares fast-state arrays + rng_state
    ck_m.slow = copy.deepcopy(ck.slow)            # private slow copy (mutate its m only)
    ck_m.slow.m[:NE] = transform_m(ck.slow.m[:NE], kind, seed=seed)
    return ck_m


# ======================================================================== E16/E17 mode-tracking geometry
def _orthonormal(M):
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = M[:, None]
    Q, _ = np.linalg.qr(M)
    return Q[:, :M.shape[1]]


def principal_angles_deg(A, B):
    """Principal angles (degrees, ascending) between the column spaces of A and B (n×k each). Uses the
    singular values of Qa^T Qb on orthonormalized bases -> depends only on the subspaces, so it is
    invariant to column sign / in-plane basis rotation (E16/E17). Single columns -> one angle."""
    Qa, Qb = _orthonormal(A), _orthonormal(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.sort(np.arccos(s)))


def subspace_alignment(A, B):
    """Mean cos of the principal angles in [0, 1]: 1 = identical subspace, 0 = orthogonal. Sign- and
    in-plane-rotation invariant (E16)."""
    ang = np.radians(principal_angles_deg(A, B))
    return float(np.mean(np.cos(ang))) if ang.size else float("nan")


def leading_subspace(K, degeneracy_ratio):
    """SVD an ensemble finite-time operator K (n_bins × n_modes) and return the leading empirical
    singular mode + the leading SUBSPACE for cross-state tracking (spec §4 P3). When
    sigma_hat_1/sigma_hat_2 < degeneracy_ratio the leading direction is unstable, so the tracked object
    is the leading subspace (dim ≥ 2) rather than a single vector (E17)."""
    U, s, Vt = np.linalg.svd(np.asarray(K, float), full_matrices=False)
    s1 = float(s[0]) if s.size else 0.0
    s2 = float(s[1]) if s.size > 1 else 0.0
    gap = (s1 / s2) if s2 > 0 else float("inf")
    degenerate = bool(s2 > 0 and gap < degeneracy_ratio)
    r = int(np.sum(s >= s1 / degeneracy_ratio)) if s1 > 0 else 1
    dim = max(r, 2) if degenerate else max(r, 1)
    return dict(u1=U[:, 0], U=U[:, :dim].copy(), sigma1=s1, gap=float(gap),
                degenerate=degenerate, subspace_dim=int(dim),
                singular_values=[float(x) for x in s[:6]])


def weighted_centroid(field, X, Y):
    """|field|²-weighted centroid (cx, cy) over grid coordinates X, Y (sign-invariant)."""
    w = np.abs(np.asarray(field, float)).ravel() ** 2
    x = np.asarray(X, float).ravel(); y = np.asarray(Y, float).ravel()
    if w.sum() <= 0:
        return (float("nan"), float("nan"))
    return (float(np.average(x, weights=w)), float(np.average(y, weights=w)))


def centroid_displacement(field_a, field_b, X, Y):
    """Euclidean distance between the |field|²-weighted centroids of two mode fields (spec §4 P3)."""
    ca = weighted_centroid(field_a, X, Y)
    cb = weighted_centroid(field_b, X, Y)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


# ======================================================================== E19 checkpoint fingerprint
def state_checkpoint_fingerprint(ck):
    """Deterministic sha256 (16 hex) of the FULL recoverable checkpoint: fast state (V, ref, s/I
    currents, rings, xi), slow z/m, and the RNG bit-generator state. Two checkpoints with identical
    recoverable state hash identically; any change (e.g. an m counterfactual) changes the hash (E19)."""
    h = hashlib.sha256()
    for arr in (ck.V, ck.ref, ck.s_E, ck.I_E, ck.s_I, ck.I_I, ck.ring_sE, ck.ring_sI,
                ck.slow.z, ck.slow.m):
        h.update(np.ascontiguousarray(np.asarray(arr)).tobytes())
    h.update(np.asarray([ck.xi], float).tobytes())
    h.update(json.dumps(ck.rng_state, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]

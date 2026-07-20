"""Topic 4 — MZ full-SNN direct spatial-mode dynamics: pure carrier / mapping / operator math.

Design contract (BINDING): docs/superpowers/specs/2026-07-19-topic4-mz-direct-spatial-modes-design.md

Scientific object: the *empirical finite-time spatial response operator* of the COMPLETE
current-based MZ spiking network (≈40 000 E/I LIF neurons), measured by direct per-E-neuron
current perturbation at frozen slow states. This is NOT a rate-field surrogate, NOT an exact
full-SNN eigenanalysis, NOT a seizure-mechanism proof (spec §0/§9).

IMPORT-SAFE and SIDE-EFFECT-FREE: no simulations, no file writes (those live in
scripts/run_topic4_mz_direct_spatial_modes.py). It provides:
  1. MZSpatialProbe(MZOnsetProbe) — an off-by-default per-E-neuron additive-current schedule
     (spec §2.2). With NO schedule it is byte-identical to MZOnsetProbe (parity gate C1). The
     current enters via I_net returned by apply_currents, which the engine consumes AFTER both
     per-step RNG draws -> it CANNOT perturb the draw order -> common random numbers hold (C3).
  2. Coarse 12x12 grid readout (input broadcast + output rate binning + mass conservation, C6),
     reusing the shared cell-assignment convention from src.topic4_state_conditioned_susceptibility
     (NOT its rate-field operator functions — those are the forbidden frozen-q closure, spec §6).
  3. A complete real orthonormal 2-D Fourier basis (144 dims, Q^T Q = I, C5).
  4. Empirical operator assembly (central difference + SVD -> sigma_hat_1 / V1 / U1 / subspace,
     C7-C10) and fixed-kick / kymograph / arrival readouts (spec §4).

Reuse (do not reinvent): src.topic4_mz_onset_dynamics.MZOnsetProbe (freeze/branch/probe hooks +
run_loop checkpoint/resume), src.topic4_state_conditioned_susceptibility.{normalize_subject_
coordinates, coarse_cell_index}, src.topic4_m3b_spectral_phase.Grid.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENG = os.path.join(_ROOT, "src", "snn_engine")
if _ENG not in sys.path:
    sys.path.insert(0, _ENG)

from src.topic4_mz_onset_dynamics import MZOnsetProbe  # noqa: E402
from src.topic4_m3b_spectral_phase import Grid  # noqa: E402
from src.topic4_state_conditioned_susceptibility import (  # noqa: E402
    normalize_subject_coordinates, coarse_cell_index,
)

SCHEMA_VERSION = "mz-direct-spatial-modes-1.0"


# ======================================================================== perturbation carrier (spec §2.2)
class MZSpatialProbe(MZOnsetProbe):
    """MZOnsetProbe + an off-by-default per-E-neuron additive-current schedule.

    Contract clauses (spec §7):
      C1 parity: `_cur is None` -> apply_currents == MZOnsetProbe.apply_currents (byte-identical).
      C4 E-only + window: the pattern is added ONLY to I_net[:NE] (E cells) and ONLY inside the
         half-open step window [lo, hi); I cells never touched, off-window is byte-identical.
      C3 common RNG: apply_currents runs AFTER the engine's per-step RNG draws, so the schedule
         cannot change the draw order — +eps / -eps / no-probe forks share the same noise stream.
    """

    def __init__(self, N, V_th0, cfg=None, *, NE, core_mask_E=None, snapshot_steps=None):
        super().__init__(N, V_th0, cfg, NE=NE, core_mask_E=core_mask_E, snapshot_steps=snapshot_steps)
        self._cur = None                          # off-by-default -> parity (C1)

    def set_current_schedule(self, *, lo, hi, pattern_E):
        """Additive current `pattern_E` (length NE, may be +/-) on E cells over steps [lo, hi)."""
        pat = np.asarray(pattern_E, float)
        if pat.shape != (self.NE,):
            raise ValueError(f"pattern_E must have shape (NE={self.NE},), got {pat.shape}")
        self._cur = dict(lo=int(lo), hi=int(hi), pattern=pat.copy())
        return self

    def clear_current_schedule(self):
        self._cur = None
        return self

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        I_net = super().apply_currents(I_E, I_I, labels, I_E_rec)   # C1: unchanged base path
        cur = self._cur
        if cur is not None and cur["lo"] <= self._step_i < cur["hi"]:   # C4: window gate on engine step
            I_net = np.array(I_net, float, copy=True)                    # do not mutate base's array
            I_net[:self.NE] += cur["pattern"]                           # C4: E cells only
        return I_net


# ======================================================================== coarse-grid readout (spec §2.5)
@dataclass
class GridReadout:
    """Fixed input/output READOUT geometry (never the dynamical model). E-neuron -> coarse cell
    assignment (shared convention with the rate-field binning) + per-cell E occupancy."""
    n: int
    grid: Grid
    pos_norm: np.ndarray
    cell_ii: np.ndarray
    cell_jj: np.ndarray
    cell_flat: np.ndarray          # ii * n + jj  (C-order, matches (n,n).reshape/ravel)
    occupancy: np.ndarray          # (n, n) int: E neurons per cell
    n_bins: int
    empty_mask: np.ndarray         # (n, n) bool: cells with no E neuron


def build_grid_readout(posE, *, grid_n, L_phys, L_norm, center_phys):
    grid = Grid(n=int(grid_n), L=float(L_norm))
    pos_norm, _ = normalize_subject_coordinates(np.asarray(posE, float), L_phys=float(L_phys),
                                                L_norm=float(L_norm), center_phys=center_phys)
    ii, jj = coarse_cell_index(pos_norm, grid)
    n = int(grid_n)
    flat = ii * n + jj
    occ = np.bincount(flat, minlength=n * n).reshape(n, n)
    return GridReadout(n=n, grid=grid, pos_norm=pos_norm, cell_ii=ii, cell_jj=jj, cell_flat=flat,
                       occupancy=occ, n_bins=n * n, empty_mask=(occ == 0))


def grid_pattern_to_current(pattern_2d, readout: GridReadout):
    """Broadcast a coarse (n,n) input pattern to a per-E-neuron current (each neuron gets its
    cell's value). Inverse of the mean-binning readout."""
    p = np.asarray(pattern_2d, float)
    return p[readout.cell_ii, readout.cell_jj]


def rms_normalize(pattern, target_rms=1.0):
    p = np.asarray(pattern, float)
    rms = float(np.sqrt(np.mean(p ** 2)))
    return p.copy() if rms == 0 else p * (float(target_rms) / rms)


def spikes_to_rate_grid(E_spk_bool, readout: GridReadout, *, dt_ms):
    """Mean E firing-rate (Hz) per coarse cell over the whole window (spec §2.5). C6: the binned
    spike total equals the raw spike total (mass conservation); empty cells -> NaN (never 0)."""
    spk = np.asarray(E_spk_bool)
    n_steps = spk.shape[0]
    counts = spk.sum(axis=0).astype(float)                       # per-E-neuron spike counts
    n = readout.n
    binned = np.bincount(readout.cell_flat, weights=counts, minlength=n * n).reshape(n, n)
    total = float(counts.sum())
    mass_ok = bool(np.isclose(binned.sum(), total))
    T_sec = n_steps * dt_ms * 1e-3
    occ = readout.occupancy
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(occ > 0, binned / np.maximum(occ, 1) / max(T_sec, 1e-12), np.nan)
    return dict(rate_hz=rate, spikes_binned=binned, total_spikes=total, mass_ok=mass_ok,
                empty_mask=readout.empty_mask, T_ms=n_steps * dt_ms)


def local_window_maps(E_spk_bool, readout: GridReadout, *, dt_ms, centers_ms, width_ms=5.0):
    """Mean E-rate map in the `width_ms` window ENDING at each center time (spec §2.5 local maps)."""
    spk = np.asarray(E_spk_bool)
    out = {}
    for c in centers_ms:
        lo = max(int(round((c - width_ms) / dt_ms)), 0)
        hi = min(int(round(c / dt_ms)), spk.shape[0])
        if hi > lo:
            out[float(c)] = spikes_to_rate_grid(spk[lo:hi], readout, dt_ms=dt_ms)["rate_hz"]
        else:
            out[float(c)] = np.full((readout.n, readout.n), np.nan)
    return out


def cumulative_rate_grid(E_spk_bool, readout: GridReadout, *, dt_ms, T_ms):
    """Mean E-rate over [0, T_ms] (the operator output Y_T, spec §2.5)."""
    spk = np.asarray(E_spk_bool)
    hi = min(int(round(T_ms / dt_ms)), spk.shape[0])
    return spikes_to_rate_grid(spk[:hi], readout, dt_ms=dt_ms)["rate_hz"]


# ======================================================================== real 2-D Fourier basis (spec §3.2)
def real_fourier_basis_1d(n):
    """Orthonormal real Fourier basis of length n (columns): DC, cos/sin pairs, + Nyquist cosine
    when n is even. B^T B = I."""
    n = int(n)
    j = np.arange(n)
    cols = [np.ones(n) / np.sqrt(n)]                            # DC
    for k in range(1, n // 2 + 1):
        if 2 * k == n:                                          # Nyquist (n even): cosine only
            cols.append(np.cos(np.pi * j) / np.sqrt(n))
        else:
            cols.append(np.sqrt(2.0 / n) * np.cos(2 * np.pi * k * j / n))
            cols.append(np.sqrt(2.0 / n) * np.sin(2 * np.pi * k * j / n))
    return np.column_stack(cols)


def real_fourier_basis_2d(n):
    """Complete real orthonormal 2-D Fourier basis on an n x n grid: n*n columns, each a flattened
    (n,n) mode = outer(1-D mode a, 1-D mode b). Q^T Q = I (C5, spec §3.2 requires the FULL space)."""
    B = real_fourier_basis_1d(n)
    cols = [np.outer(B[:, a], B[:, b]).ravel() for a in range(n) for b in range(n)]
    return np.column_stack(cols)


def balanced_lowk_indices(n, k_max):
    """Column indices of real_fourier_basis_2d(n) that are BALANCED low-k: 2-D modes built from 1-D
    frequencies <= k_max in BOTH axes (symmetric — DC + cos/sin of each low frequency, both x and y).

    This is the corrected low-k audit set: the leading n columns are outer(DC, everything) which
    includes DC x Nyquist (a max-frequency column), so `range(n_sub)` is NOT a low-k selection.
    1-D frequency f -> real_fourier_basis_1d index: f=0 -> 0; 1<=f<n/2 -> {2f-1 (cos), 2f (sin)};
    Nyquist (n even, f=n/2) -> the last index n-1 (included only when k_max >= n/2)."""
    lo1d = [0]
    for f in range(1, int(k_max) + 1):
        if 2 * f == n:
            lo1d.append(n - 1)                     # Nyquist (single cosine)
        else:
            lo1d.extend([2 * f - 1, 2 * f])        # cos + sin of frequency f
    return [a * n + b for a in lo1d for b in lo1d]


# ======================================================================== empirical operator (spec §3.2 / §4)
def central_difference(Y_plus, Y_minus, epsilon):
    """K column = [Y(+eps p) - Y(-eps p)] / (2 eps)  (units: output Hz / input current fraction)."""
    return (np.asarray(Y_plus, float) - np.asarray(Y_minus, float)) / (2.0 * float(epsilon))


def build_empirical_operator(K, basis_P, grid_n, *, degeneracy_ratio=1.05):
    """Assemble + SVD the empirical finite-time response operator (spec §3.2). K is (n_bins x
    n_patterns) central-difference responses to the orthonormal basis columns of `basis_P`
    (n_bins x n_patterns). Re-express the operator in bin<->bin coords (M = K P^T, valid because
    P is a complete orthonormal basis) and SVD.

    Returns sigma_hat_1 (leading finite-time gain, units Hz/fraction — NOT a dimensionless gain,
    spec §3), V1 input field, U1 output field, singular gap s1/s2, and — when s1/s2 is near 1
    (degenerate) — the leading SUBSPACE instead of an unstable single vector (C9/C10)."""
    K = np.asarray(K, float)
    P = np.asarray(basis_P, float)
    M = K @ P.T                                                 # operator in bin<->bin coordinates
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    n = int(grid_n)
    s1 = float(s[0]) if s.size else 0.0
    s2 = float(s[1]) if s.size > 1 else 0.0
    gap = (s1 / s2) if s2 > 0 else float("inf")
    degenerate = bool(s2 > 0 and gap < degeneracy_ratio)
    sub = int(np.sum(s >= s1 / degeneracy_ratio)) if s1 > 0 else 1   # cluster within the gap band
    subspace_dim = max(sub, 2) if degenerate else max(sub, 1)
    return dict(sigma1=s1, singular_values=[float(x) for x in s[:8]], gap=float(gap),
                degenerate=degenerate, subspace_dim=int(subspace_dim),
                u1_field=U[:, 0].reshape(n, n), v1_field=Vt[0, :].reshape(n, n),
                u_subspace=U[:, :subspace_dim].copy(), v_subspace=Vt[:subspace_dim, :].T.copy())


# ---- field geometry (sign-invariant: |field|^2 loading, spec §4) ----
def field_globality(field):
    """Participation-ratio globality in [1/N, 1]: uniform -> 1, single-cell -> 1/N. |field|^2
    loading -> sign-invariant (C9)."""
    f2 = np.abs(np.asarray(field, float).ravel()) ** 2
    s = float(f2.sum())
    denom = float(f2.size * np.sum(f2 ** 2))
    return float(s ** 2 / denom) if denom > 0 else float("nan")


def field_axis_alignment(field, readout: GridReadout, axis_unit):
    """|field|^2-weighted (var_along - var_perp)/(var_along + var_perp) about the source->sink
    axis, in [-1, 1]: +1 elongated along the axis, -1 along perpendicular, 0 isotropic. Sign-
    invariant (C9)."""
    w = np.abs(np.asarray(field, float).ravel()) ** 2
    if w.sum() <= 0:
        return float("nan")
    X, Y = readout.grid.coords()
    pts = np.column_stack([X.ravel(), Y.ravel()])
    c = np.average(pts, axis=0, weights=w)
    d = pts - c
    u = np.asarray(axis_unit, float)
    u = u / np.linalg.norm(u)
    up = np.array([-u[1], u[0]])
    var_par = float(np.average((d @ u) ** 2, weights=w))
    var_perp = float(np.average((d @ up) ** 2, weights=w))
    tot = var_par + var_perp
    return float((var_par - var_perp) / tot) if tot > 0 else 0.0


def normalized_field_overlap(field_a, field_b):
    """Sign-invariant |cos| overlap of two mode fields, in [0, 1] (adjacent-state mode switching)."""
    a = np.asarray(field_a, float).ravel()
    b = np.asarray(field_b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(abs(a @ b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


# ======================================================================== fixed-kick readouts (spec §4)
def gaussian_current_field(readout: GridReadout, *, center_norm, sigma, rms):
    """A source-centered Gaussian input pattern (n,n), RMS-normalized (the FIXED localized kick)."""
    X, Y = readout.grid.coords()
    g = np.exp(-((X - center_norm[0]) ** 2 + (Y - center_norm[1]) ** 2) / (2.0 * float(sigma) ** 2))
    return rms_normalize(g, target_rms=rms)


def response_norm(dY):
    return float(np.sqrt(np.nansum(np.asarray(dY, float) ** 2)))


def region_response(dY, region_masks):
    """Mean |dY| within each named (n,n) boolean region mask."""
    d = np.abs(np.asarray(dY, float))
    out = {}
    for name, m in region_masks.items():
        vals = d[np.asarray(m, bool)]
        vals = vals[np.isfinite(vals)]
        out[name] = float(vals.mean()) if vals.size else float("nan")
    return out


def cumulative_response_ratio(remote_series, source_series):
    """Cumulative remote/source response ratio (spec §4). Uses cumulative sums so it does NOT blow
    up when the instantaneous source response passes through zero."""
    r = np.cumsum(np.abs(np.asarray(remote_series, float)))
    s = np.cumsum(np.abs(np.asarray(source_series, float)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, r / np.maximum(s, 1e-12), 0.0)


def axis_kymograph(stack, readout: GridReadout, *, axis_unit, src_norm, snk_norm, band, n_pos):
    """(kymo[n_t, n_pos], positions, distances, times): |field| along the source->sink axis over
    time. Each axial position bin averages cells within a perpendicular `band`. Shows spatiotemporal
    recruitment ONLY — NOT a proven continuous wavefront (spec §4)."""
    stack = np.asarray(stack, float)
    n_t = stack.shape[0]
    X, Y = readout.grid.coords()
    pts = np.column_stack([X.ravel(), Y.ravel()])
    src = np.asarray(src_norm, float)
    u = np.asarray(axis_unit, float)
    u = u / np.linalg.norm(u)
    up = np.array([-u[1], u[0]])
    rel = pts - src
    proj = rel @ u
    perp = np.abs(rel @ up)
    axis_len = float(np.linalg.norm(np.asarray(snk_norm, float) - src))
    edges = np.linspace(0.0, axis_len, int(n_pos) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    flat_stack = np.abs(stack.reshape(n_t, -1))
    inband = perp <= band
    kymo = np.full((n_t, int(n_pos)), np.nan)
    for pi in range(int(n_pos)):
        hi_incl = (proj <= edges[pi + 1] + 1e-9) if pi == n_pos - 1 else (proj < edges[pi + 1])
        sel = inband & (proj >= edges[pi]) & hi_incl
        if sel.any():
            kymo[:, pi] = flat_stack[:, sel].mean(axis=1)
    return dict(kymo=kymo, positions=centers, distances=centers.copy(),
                times=np.arange(n_t, dtype=float))


def first_arrival_times(kymo, times, *, threshold):
    """First time each axial position crosses `threshold` (NaN if never crossed)."""
    kymo = np.asarray(kymo, float)
    times = np.asarray(times, float)
    arr = np.full(kymo.shape[1], np.nan)
    for pi in range(kymo.shape[1]):
        idx = np.where(np.nan_to_num(kymo[:, pi], nan=-np.inf) >= threshold)[0]
        if idx.size:
            arr[pi] = times[idx[0]]
    return arr


def fit_arrival_distance(distances, arrivals, *, min_points=4, r2_min=0.5):
    """Linear arrival-time-vs-distance fit (spec §4). Eligible ONLY as source-driven axial recruitment:
    >= `min_points` crossed positions, a real (non-constant) front, POSITIVE slope (arrival grows with
    distance from source), and a finite R2 >= `r2_min`. Fails closed otherwise — never a spurious
    0-slope / NaN-R2 / negative-slope / poor-fit 'eligible' (review 2026-07-20)."""
    d = np.asarray(distances, float)
    a = np.asarray(arrivals, float)
    ok = np.isfinite(a) & np.isfinite(d)
    n = int(ok.sum())
    fail = dict(eligible=False, n_points=n, slope=None, velocity_proxy=None, r2=None)
    if n < int(min_points) or np.ptp(a[ok]) == 0:              # too few / degenerate constant-arrival
        return fail
    dd, aa = d[ok], a[ok]
    coef = np.polyfit(dd, aa, 1)
    slope = float(coef[0])                                      # ms per distance-unit
    pred = np.polyval(coef, dd)
    ss_res = float(np.sum((aa - pred) ** 2))
    ss_tot = float(np.sum((aa - aa.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    velocity = float(1.0 / slope) if slope != 0 else float("inf")   # distance-unit per ms
    eligible = bool(slope > 0 and np.isfinite(r2) and r2 >= float(r2_min))
    return dict(eligible=eligible, n_points=n, slope=slope, velocity_proxy=velocity, r2=r2)


def threshold_sensitivity_arrivals(kymo, times, distances, *, fracs, peak=None, r2_min=0.5):
    """Arrival-fit slope/R²/eligibility at several thresholds = `fracs` × peak (spec §4)."""
    kymo = np.asarray(kymo, float)
    pk = float(np.nanmax(np.abs(kymo))) if peak is None else float(peak)
    out = {}
    for fr in fracs:
        arr = first_arrival_times(kymo, times, threshold=fr * pk)
        out[float(fr)] = fit_arrival_distance(distances, arr, r2_min=r2_min)
    return out


# ======================================================================== linearity audit + labels (spec §2.3)
def linearity_discrepancy(K_full, K_half):
    """Normalized discrepancy ||K(eps) - K(eps/2)|| / ||K(eps/2)|| (0 = perfectly linear)."""
    a = np.asarray(K_full, float)
    b = np.asarray(K_half, float)
    denom = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else float("nan")


def select_epsilon(ladder, discrepancies, saturated, *, tol=0.15):
    """Pick the LARGEST ladder amplitude whose linearity discrepancy <= tol and is not saturated
    (spec §2.3). If none qualify -> `nonlinear_response_only` (fixed-kick only; do NOT widen the
    ladder)."""
    qualifying = [i for i, (d, s) in enumerate(zip(discrepancies, saturated))
                  if (d is not None and np.isfinite(d) and d <= tol and not s)]
    if not qualifying:
        return dict(epsilon=None, index=None, qualified=[], mode="nonlinear_response_only")
    idx = max(qualifying)
    return dict(epsilon=float(ladder[idx]), index=int(idx), qualified=qualifying, mode="operator")


def right_censoring_label(noprobe_runaway):
    """`right_censored_native_transition` if the no-probe control ran away inside the window; else
    `resolved` (spec §1: never plot gain=0, never escalate amplitude)."""
    return "right_censored_native_transition" if noprobe_runaway is not None else "resolved"

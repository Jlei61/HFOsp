"""M3B-R2 spectral phase map — finite-Jacobian neural-field spectrum for the SEF-HFO SNN family.

Plan:    docs/superpowers/plans/2026-06-27-sef-hfo-m3b-spectral-phase-map-plan.md
Design:  docs/superpowers/specs/2026-06-27-sef-hfo-m3b-spectral-phase-map-design.md
Interface contract (M3A overlay gate, consumed not reinvented): src/sef_hfo_m3_interface.py

Scientific object
-----------------
A Brunel-style mean-field stability analysis adapted to a FINITE, core-heterogeneous epilepsy
sheet. We coarse-grain the SNN to a spatial E/I rate-synapse field, solve an operating point,
linearize, and read spatial eigenmodes + non-normal finite-time gain. This is a MECHANISM map,
not a seizure proof (see STATUS.md forbidden claims).

State vector (locked, TDD-1)
----------------------------
Per grid cell:  z_i = [rE, rI, sEE, sEI, sIE, sII]   (6 fields).
The four synaptic states are the AMPA/GABA low-pass of the convolved presynaptic rates, exactly
as integrated by ``src.sef_hfo_lif.integrate_lif_field``; the finite Jacobian (TDD-5) is the
linearization of that same 6-field model, and reduces in the homogeneous no-delay limit to
``src.sef_hfo_lif._char_det`` (TDD-4/TDD-5 cross-checks).

This file is built test-first; sections map 1:1 to the plan's TDD-1..N. Only the implemented
sections are present.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.sef_hfo_field import (
    anisotropic_gaussian, isotropic_gaussian, convolve_periodic, _grid,
)
from src.sef_hfo_lif import (
    ELL_PAR, ELL_PERP, L_INH, V_TH, V_RESET,
    TAU_ME, TAU_MI, TREF_E, TREF_I, TAU_AMPA, TAU_GABA,
    C_EE, W_EE, C_EI, W_EI, C_IE, W_IE, C_II, W_II, JX_E, JX_I,
    lif_rate, lif_gains, mean_field, nu_theta_pop,
)

# ---------------------------------------------------------------------------
# Locked conventions
# ---------------------------------------------------------------------------
# Canonical E->E scaffold orientation. The Round-1 virtual-SEEG bridge recovered a 45 degree
# E->E axis (axis error ~3.3 deg); the anisotropic kernel's major axis is pinned there so the
# spectral atlas and the readout bridge share one geometry.
THETA_EE: float = np.pi / 4.0

# 6-field per-cell state, in the order the pack/unpack and the Jacobian blocks assume everywhere.
STATE_FIELDS: tuple[str, ...] = ("rE", "rI", "sEE", "sEI", "sIE", "sII")
N_FIELDS: int = len(STATE_FIELDS)


# ---------------------------------------------------------------------------
# TDD-1: grid / kernel / state-vector contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    """Coarse periodic sheet. ``n`` cells per side, physical extent ``L`` mm.

    Spatial conventions follow ``src.sef_hfo_field._grid`` (``indexing="ij"``): the first axis
    carries the X coordinate, the second carries Y, both centered on 0.
    """
    n: int = 48
    L: float = 12.0

    @property
    def size(self) -> int:
        """Number of grid cells N = n*n."""
        return self.n * self.n

    @property
    def spacing(self) -> float:
        """Cell pitch (mm)."""
        return self.L / self.n

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y) coordinate meshgrids (mm), shape (n, n), indexing='ij'."""
        return _grid(self.n, self.L)


def ee_kernel(grid: Grid, *, ar: float = 2.0, ell_perp: float = ELL_PERP,
              theta: float = THETA_EE) -> np.ndarray:
    """Anisotropic E->E coupling kernel, L1-normalized, periodic.

    ``ar`` is the aspect ratio ell_par / ell_perp (>1 = elongated along ``theta``). ``ar == 1``
    collapses to an isotropic kernel (the TDD-10 AR1 control). The major axis points along
    ``theta`` (default the 45 deg scaffold axis).
    """
    ell_par = ar * ell_perp
    return anisotropic_gaussian(grid.n, grid.L, ell_par, ell_perp, theta)


def inh_kernel(grid: Grid, *, l_inh: float = L_INH) -> np.ndarray:
    """Isotropic inhibitory / E->I coupling kernel, L1-normalized, periodic."""
    return isotropic_gaussian(grid.n, grid.L, l_inh)


def kernel_principal_axis(kernel: np.ndarray, grid: Grid) -> tuple[float, float]:
    """Principal-axis angle (rad, in [-pi/2, pi/2]) and aspect ratio of a (centered) kernel.

    Computes the kernel-weighted second-moment tensor over the grid coordinates and returns
    (angle, sqrt(lambda_major/lambda_minor)). Used by the AR1/AR2 kernel-geometry tests and by
    the off-axis controls.
    """
    X, Y = grid.coords()
    w = kernel / kernel.sum()
    # kernels are symmetric about the origin, so the mean is ~0; center anyway for safety.
    xm = float(np.sum(w * X))
    ym = float(np.sum(w * Y))
    Sxx = float(np.sum(w * (X - xm) ** 2))
    Syy = float(np.sum(w * (Y - ym) ** 2))
    Sxy = float(np.sum(w * (X - xm) * (Y - ym)))
    angle = 0.5 * np.arctan2(2.0 * Sxy, Sxx - Syy)
    evals = np.linalg.eigvalsh(np.array([[Sxx, Sxy], [Sxy, Syy]]))
    evals = np.clip(evals, 1e-30, None)
    aspect = float(np.sqrt(evals.max() / evals.min()))
    return float(angle), aspect


@dataclass(frozen=True)
class CoreMask:
    """Boolean core mask plus the geometry that produced it.

    ``kind`` in {"none", "single", "two", "off_axis"}. ``mask`` is (n, n) bool: True = pathological
    core cell. The remaining fields record the placement so controls (off-axis, two-core) and the
    interface (two_core_reduction) can reason about geometry without re-deriving it.
    """
    kind: str
    mask: np.ndarray
    centers: tuple[tuple[float, float], ...]
    radius: float
    theta: float

    @property
    def area_fraction(self) -> float:
        return float(self.mask.mean())


def make_core_mask(grid: Grid, *, kind: str = "single", radius: float = 1.5,
                   separation: float = 4.0, theta: float = THETA_EE,
                   center: tuple[float, float] = (0.0, 0.0),
                   off_axis_shift: float = 3.0) -> CoreMask:
    """Build a pathological-core mask.

    kind="none"     -> empty mask (homogeneous background / no-core control).
    kind="single"   -> one disk of ``radius`` mm at ``center`` (primary atlas geometry).
    kind="two"      -> two disks separated by ``separation`` mm ALONG the E->E axis ``theta``
                       (validation geometry tied to the Fig4/5 two-core SNN asset).
    kind="off_axis" -> one disk displaced PERPENDICULAR to ``theta`` by ``off_axis_shift`` mm
                       (TDD-10 off-axis control: tests scaffold-core geometry, not any local bump).
    """
    X, Y = grid.coords()
    cx, cy = center

    def disk(dx: float, dy: float) -> np.ndarray:
        return (X - dx) ** 2 + (Y - dy) ** 2 <= radius ** 2

    if kind == "none":
        mask = np.zeros((grid.n, grid.n), dtype=bool)
        return CoreMask("none", mask, (), radius, theta)

    ux, uy = np.cos(theta), np.sin(theta)          # along-axis unit vector
    px, py = -np.sin(theta), np.cos(theta)         # perpendicular unit vector

    if kind == "single":
        centers = ((cx, cy),)
    elif kind == "two":
        h = 0.5 * separation
        centers = ((cx + h * ux, cy + h * uy), (cx - h * ux, cy - h * uy))
    elif kind == "off_axis":
        centers = ((cx + off_axis_shift * px, cy + off_axis_shift * py),)
    else:
        raise ValueError(f"unknown core mask kind {kind!r}")

    mask = np.zeros((grid.n, grid.n), dtype=bool)
    for dx, dy in centers:
        mask |= disk(dx, dy)
    return CoreMask(kind, mask, tuple((float(a), float(b)) for a, b in centers), radius, theta)


# ---------------------------------------------------------------------------
# TDD-2: LIF transfer and local gain (population-aware wrappers over the validated Siegert helpers)
# ---------------------------------------------------------------------------
# There is NO analytic dPhi/dmu in this codebase: src.sef_hfo_lif.lif_gains is itself a symmetric
# finite difference on lif_rate. We reuse that exact convention (h = 1e-3 mV) so the per-cell local
# gain used by the heterogeneous Jacobian (TDD-5) cannot drift from the validated 0-D gain.
_GAIN_H: float = 1e-3


def _pop_consts(pop: str) -> tuple[float, float]:
    if pop == "E":
        return TAU_ME, TREF_E
    if pop == "I":
        return TAU_MI, TREF_I
    raise ValueError(f"pop must be 'E' or 'I', got {pop!r}")


def phi_lif(mu: float, sigma: float, *, pop: str = "E", v_th: float = V_TH) -> float:
    """LIF firing rate Phi(mu, sigma) (kHz) for the E or I population (Siegert via lif_rate)."""
    tau_m, tau_ref = _pop_consts(pop)
    return lif_rate(mu, sigma, tau_m, tau_ref, v_th=v_th)


def dphi_dmu(mu: float, sigma: float, *, pop: str = "E", v_th: float = V_TH,
             h: float = _GAIN_H) -> float:
    """Local transfer slope dPhi/dmu (kHz/mV) by symmetric finite difference."""
    return (phi_lif(mu + h, sigma, pop=pop, v_th=v_th)
            - phi_lif(mu - h, sigma, pop=pop, v_th=v_th)) / (2.0 * h)


def local_gain(mu: float, sigma: float, *, pop: str = "E", v_th: float = V_TH,
               h: float = _GAIN_H) -> float:
    """Dimensionless local gain G_mu = (dPhi/dmu) * tau_m, matching ``lif_gains`` exactly.

    This is the diagonal block entry scale for the heterogeneous Jacobian: each cell carries its
    own gain evaluated at its own operating-point (mu, sigma), so a low-threshold core cell has a
    larger gain than the surround.
    """
    tau_m, _ = _pop_consts(pop)
    return dphi_dmu(mu, sigma, pop=pop, v_th=v_th, h=h) * tau_m


def pack_state(state: dict[str, np.ndarray], grid: Grid) -> np.ndarray:
    """Flatten the 6-field per-cell state dict into a (6*N,) vector in STATE_FIELDS order."""
    return np.concatenate([np.asarray(state[f], dtype=float).ravel() for f in STATE_FIELDS])


def unpack_state(z: np.ndarray, grid: Grid) -> dict[str, np.ndarray]:
    """Inverse of ``pack_state``: (6*N,) vector -> {field: (n, n) array}."""
    z = np.asarray(z, dtype=float)
    expected = N_FIELDS * grid.size
    if z.shape != (expected,):
        raise ValueError(f"state vector has shape {z.shape}, expected ({expected},) for grid n={grid.n}")
    out: dict[str, np.ndarray] = {}
    N = grid.size
    for i, f in enumerate(STATE_FIELDS):
        out[f] = z[i * N:(i + 1) * N].reshape(grid.n, grid.n)
    return out


# ---------------------------------------------------------------------------
# TDD-2 (cont) / TDD-3: heterogeneous fields + operating point
#
# Design contract: docs/archive/topic4/sef_hfo/m3b_jacobian_design_LOCKED_2026-06-27.md (§4, §6).
# Two orthogonal knobs, both touching ONLY the rate rows, never the convolution coupling:
#   * core excitability (x-axis): mu_core additive drive (and/or dVth_core threshold) -> shifts the
#     operating point up in the core -> larger per-cell gain gE_i.
#   * q efficacy (y-axis disinhibition): scales the I->E weight W_EI; q_global field-wide, q_core
#     masked to the core. Lower q -> weaker brake -> more event-prone (contract-correct sign).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Kernels:
    """The two periodic coupling operators (L1-normalized) + their DFT amplitudes for matvec."""
    K_EE: np.ndarray          # (n,n) anisotropic E->E scaffold
    K_I: np.ndarray           # (n,n) isotropic (sEI<-rI, sIE<-rE, sII<-rI all use K_I)
    ghat_EE: np.ndarray       # (n,n) = real(fft2(ifftshift(K_EE))) — discrete DFT amplitudes
    ghat_I: np.ndarray        # (n,n) = real(fft2(ifftshift(K_I)))
    ell_par: float
    ell_perp: float
    l_inh: float
    theta: float


def build_kernels(grid: Grid, *, ar: float = 2.0, ell_perp: float = ELL_PERP,
                  l_inh: float = L_INH, theta: float = THETA_EE) -> Kernels:
    """Build the E->E (anisotropic) and inhibitory (isotropic) coupling kernels + DFT amplitudes."""
    ell_par = ar * ell_perp
    K_EE = anisotropic_gaussian(grid.n, grid.L, ell_par, ell_perp, theta)
    K_I = isotropic_gaussian(grid.n, grid.L, l_inh)
    ghat_EE = np.real(np.fft.fft2(np.fft.ifftshift(K_EE)))
    ghat_I = np.real(np.fft.fft2(np.fft.ifftshift(K_I)))
    return Kernels(K_EE, K_I, ghat_EE, ghat_I, ell_par, ell_perp, l_inh, theta)


@dataclass(frozen=True)
class ExcitabilityField:
    """Core-excitability x-axis knob (operating-point shift, NOT coupling)."""
    v_th: np.ndarray          # (n,n) per-cell E threshold (mV); V_TH off-core, V_TH-dVth_core on core
    mu_core: np.ndarray       # (n,n) per-cell additive drive into muE (mV); 0 off-core
    core: CoreMask
    dVth_core: float
    mu_core_value: float


def build_excitability_field(grid: Grid, core: CoreMask, *, dVth_core: float = 0.0,
                             mu_core: float = 0.0) -> ExcitabilityField:
    """Build the per-cell threshold + core-drive fields. ``mu_core`` is the primary (fast) knob;
    ``dVth_core`` lowers the core threshold (truncated at V_RESET)."""
    v_th = np.full((grid.n, grid.n), V_TH)
    mu = np.zeros((grid.n, grid.n))
    if core.mask.any():
        v_th[core.mask] = max(V_TH - dVth_core, V_RESET)
        mu[core.mask] = mu_core
    return ExcitabilityField(v_th, mu, core, float(dVth_core), float(mu_core))


@dataclass(frozen=True)
class InhibitionField:
    """q inhibition-efficacy y-axis knob (+ core nucleation). q multiplies the I->E weight W_EI."""
    q: np.ndarray             # (n,n) effective I->E efficacy in (0,1]; q_global*(q_core on core)
    q_global: float
    q_core: float
    scale_II: bool
    core: CoreMask


def build_inhibition_field(grid: Grid, core: CoreMask, *, q_global: float = 1.0,
                           q_core: float = 1.0, scale_II: bool = False) -> InhibitionField:
    """Build the per-cell GABA-efficacy field: q = q_global everywhere, * q_core on the core mask."""
    q = np.full((grid.n, grid.n), float(q_global))
    if core.mask.any():
        q[core.mask] = float(q_global) * float(q_core)
    return InhibitionField(q, float(q_global), float(q_core), bool(scale_II), core)


def _field_mu_sigma(rE: np.ndarray, rI: np.ndarray, kernels: Kernels, inh: InhibitionField,
                    *, wee: float, nuext: float, mu_core: np.ndarray):
    """Per-cell synaptic input mean (mu) and s.d. (sigma) — the spatial analog of ``_ms``.

    Reduces EXACTLY to ``src.sef_hfo_lif._ms`` when rE, rI are spatially uniform, q==1, mu_core==0
    (convolving a constant by an L1-normalized kernel returns the constant).
    """
    cEE = convolve_periodic(rE, kernels.K_EE)     # E->E presynaptic (anisotropic scaffold)
    cIE = convolve_periodic(rI, kernels.K_I)      # I->E presynaptic (isotropic)
    cEI = convolve_periodic(rE, kernels.K_I)      # E->I presynaptic (isotropic)
    cII = convolve_periodic(rI, kernels.K_I)      # I->I presynaptic (isotropic)
    wEI = inh.q * W_EI                            # per-cell effective I->E weight
    wII = inh.q * W_II if inh.scale_II else W_II  # I->I weight (scaled only if requested)
    muE = TAU_ME * (C_EE * wee * cEE - C_EI * wEI * cIE) + TAU_ME * JX_E * nuext + mu_core
    muI = TAU_MI * (C_IE * W_IE * cEI - C_II * wII * cII) + TAU_MI * JX_I * nuext
    varE = TAU_ME * (C_EE * wee ** 2 * cEE + C_EI * wEI ** 2 * cIE) + TAU_ME * JX_E ** 2 * nuext
    varI = TAU_MI * (C_IE * W_IE ** 2 * cEI + C_II * wII ** 2 * cII) + TAU_MI * JX_I ** 2 * nuext
    sigmaE = np.sqrt(np.maximum(varE, 1e-9))
    sigmaI = np.sqrt(np.maximum(varI, 1e-9))
    return muE, sigmaE, muI, sigmaI


# Cached 2-D Phi(mu, sigma) lookup tables at v_th = V_TH (built lazily, once per population).
_PHI_LUT_CACHE: dict = {}
_LUT_MU = (-40.0, 120.0, 220)
_LUT_SIG = (0.5, 30.0, 80)


def _phi_lut(pop: str):
    if pop not in _PHI_LUT_CACHE:
        from scipy.interpolate import RegularGridInterpolator
        tau_m, tau_ref = _pop_consts(pop)
        mus = np.linspace(*_LUT_MU)
        sigs = np.linspace(*_LUT_SIG)
        tab = np.array([[lif_rate(float(m), float(s), tau_m, tau_ref) for s in sigs] for m in mus])
        _PHI_LUT_CACHE[pop] = RegularGridInterpolator(
            (mus, sigs), tab, method="linear", bounds_error=False, fill_value=None)
    return _PHI_LUT_CACHE[pop]


def _phi_field(mu: np.ndarray, sigma: np.ndarray, pop: str) -> np.ndarray:
    """Vectorized Phi over a field via the cached LUT (clipped to the LUT support)."""
    rgi = _phi_lut(pop)
    pts = np.column_stack([
        np.clip(mu.ravel(), _LUT_MU[0], _LUT_MU[1]),
        np.clip(sigma.ravel(), _LUT_SIG[0], _LUT_SIG[1]),
    ])
    return np.maximum(rgi(pts).reshape(mu.shape), 0.0)


@dataclass(frozen=True)
class OperatingPoint:
    """Frozen linearization state + per-cell local gains (the diagonal Jacobian scales)."""
    rE: np.ndarray
    rI: np.ndarray
    muE: np.ndarray
    muI: np.ndarray
    sigmaE: np.ndarray
    sigmaI: np.ndarray
    gE: np.ndarray
    gI: np.ndarray
    source: str
    converged: bool
    residual: float
    saturated: bool
    excitability: ExcitabilityField
    inhibition: InhibitionField
    wee_mult: float
    nuext: float

    @property
    def status(self) -> str:
        """One-word op status used by the spectrum. A high-rate runaway is 'saturated' (recognized
        by its rate, even if the integration never settled); a low-rate non-settling point is
        'unresolved'; otherwise 'resolved'. Failed points are NEVER silently stable/axial."""
        if self.saturated:
            return "saturated"
        if not self.converged:
            return "unresolved"
        return "resolved"


# Saturation: an E rate this far above rest flags a runaway / high-rate branch (never 'axial').
_SAT_RATE_KHZ: float = 0.10           # >> any self-limited interictal op (balanced rest ~ 2.5e-4 kHz)


def _init_field(x, shape: tuple[int, int]) -> np.ndarray:
    """Broadcast a warm-start seed to a full ``(n, n)`` field: a bare scalar becomes a constant
    field; an array-like is used as-is (shape-checked). Used by ``solve_operating_point``'s
    ``init=`` (T3a-2, #10 -- branch-protocol warm-start)."""
    if np.isscalar(x):
        return np.full(shape, float(x))
    arr = np.asarray(x, dtype=float)
    assert arr.shape == shape, f"init field has shape {arr.shape}, expected {shape}"
    return arr


def solve_operating_point(grid: Grid, kernels: Kernels, exc: ExcitabilityField,
                          inh: InhibitionField, *, ratio: float = 1.0, w_ee_mult: float = 1.0,
                          source: str = "ratefield_steady", dt: float = 0.5, t_max: float = 2500.0,
                          tol: float = 1e-9, gain_h: float = _GAIN_H,
                          gK_field: np.ndarray | None = None, hG_scalar: float = 0.0,
                          eta_K: float = 1.0, eta_G: float = 1.0,
                          init: dict | None = None) -> OperatingPoint:
    """Solve the heterogeneous operating point by deterministic rate-field integration-to-steady.

    Forward-Euler integration of the 6-field rate model (the same dynamics ``integrate_lif_field``
    runs, with per-cell ``mu_core`` and per-cell GABA efficacy ``q``). This is the plan's primary
    operating-point source: the AMPA/GABA synaptic low-pass damps the inhibition-stabilized E/I
    oscillation that a naive fixed-point iteration cannot, and a genuine runaway grows to the
    refractory ceiling. Reduces to ``mean_field`` in the homogeneous (no-core, q==1) limit.

    Non-settled low-rate points are flagged ``unresolved``; high-rate branches are flagged
    ``saturated`` (=> mode_class 'runaway', never 'axial'). The per-cell gains gE_i, gI_i are the
    diagonal scales of the finite Jacobian (TDD-5).

    ``gK_field``/``hG_scalar`` are the T2.5 ``slow_to_ratefield`` wiring (#P1-1, config-of-record
    ``config/topic4_criticality.yaml``): additive E-only drive shifts, ``muE -= eta_K*gK_field``
    (per-cell ``(n,n)`` K-current field, applied only when ``gK_field is not None``) and
    ``muE -= eta_G*hG_scalar`` (global uniform recovery-current scalar). Both live inside
    ``_moments()`` so the shifted op is self-consistent (steady-state integration AND the final
    gE/gI gain both see it); muI is untouched. Defaults are additive-zero, so existing callers get
    byte-identical output.

    ``init`` (T3a-2, #10) is an optional ``{"rE": ..., "rI": ...}`` warm-start seed -- each value
    scalar or an ``(n, n)`` array (see ``_init_field``) -- used by the branch protocol
    (``src.topic4_criticality.solve_branches``) to solve the SAME operating point from several
    starting conditions (low/high/previous-point/random) and discover which rate branch each one
    lands on. ``init=None`` (the default) is BYTE-PARITY with every existing caller: it reproduces
    the exact ``mean_field``-seeded (with its bare ``1e-3`` fallback) initial condition below,
    unchanged.
    """
    n = grid.n
    wee = w_ee_mult * W_EE
    nuext = ratio * nu_theta_pop()
    muxE = TAU_ME * JX_E * nuext
    muxI = TAU_MI * JX_I * nuext
    wEI = inh.q * W_EI                                       # (n,n) effective I->E weight
    wII = inh.q * W_II if inh.scale_II else np.full((n, n), W_II)

    if init is not None:
        rE = _init_field(init["rE"], (n, n))
        rI = _init_field(init["rI"], (n, n))
    else:
        try:
            base = mean_field(ratio, w_ee_mult, strict=False)
            rE = np.full((n, n), max(base["nuE"], 1e-6))
            rI = np.full((n, n), max(base["nuI"], 1e-6))
        except Exception:
            rE = np.full((n, n), 1e-3)
            rI = np.full((n, n), 1e-3)
    sEE = convolve_periodic(rE, kernels.K_EE)
    sEI = convolve_periodic(rI, kernels.K_I)
    sIE = convolve_periodic(rE, kernels.K_I)
    sII = convolve_periodic(rI, kernels.K_I)

    def _moments():
        muE = TAU_ME * (C_EE * wee * sEE - C_EI * wEI * sEI) + muxE + exc.mu_core
        muE = muE - eta_G * hG_scalar                       # h_G: global uniform recovery current
        if gK_field is not None:
            muE = muE - eta_K * gK_field                     # g_K: per-cell K-current field
        muI = TAU_MI * (C_IE * W_IE * sIE - C_II * wII * sII) + muxI
        varE = TAU_ME * (C_EE * wee ** 2 * sEE + C_EI * wEI ** 2 * sEI) + TAU_ME * JX_E ** 2 * nuext
        varI = TAU_MI * (C_IE * W_IE ** 2 * sIE + C_II * wII ** 2 * sII) + TAU_MI * JX_I ** 2 * nuext
        return muE, muI, np.sqrt(np.maximum(varE, 1e-9)), np.sqrt(np.maximum(varI, 1e-9))

    converged = False
    residual = np.inf
    nsteps = int(t_max / dt)
    for _ in range(nsteps):
        sEE += dt / TAU_AMPA * (convolve_periodic(rE, kernels.K_EE) - sEE)
        sEI += dt / TAU_GABA * (convolve_periodic(rI, kernels.K_I) - sEI)
        sIE += dt / TAU_AMPA * (convolve_periodic(rE, kernels.K_I) - sIE)
        sII += dt / TAU_GABA * (convolve_periodic(rI, kernels.K_I) - sII)
        muE, muI, sigmaE, sigmaI = _moments()
        rhsE = -rE + _phi_field(muE, sigmaE, "E")
        rhsI = -rI + _phi_field(muI, sigmaI, "I")
        rE = rE + dt / TAU_ME * rhsE
        rI = rI + dt / TAU_MI * rhsI
        if not (np.all(np.isfinite(rE)) and np.all(np.isfinite(rI))):
            residual = np.inf
            break
        # fixed-point residual = max rate-of-change magnitude (dt-independent); ->0 at steady state
        residual = float(max(np.max(np.abs(rhsE)), np.max(np.abs(rhsI))))
        if residual < tol:
            converged = True
            break

    muE, muI, sigmaE, sigmaI = _moments()
    saturated = bool(np.all(np.isfinite(rE)) and np.max(rE) > _SAT_RATE_KHZ)

    # per-cell gains gE_i = (dPhi/dmu)*tau_m at each cell's own (mu, sigma) — diagonal Jacobian scales.
    gE = (_phi_field(muE + gain_h, sigmaE, "E") - _phi_field(muE - gain_h, sigmaE, "E")) \
        / (2 * gain_h) * TAU_ME
    gI = (_phi_field(muI + gain_h, sigmaI, "I") - _phi_field(muI - gain_h, sigmaI, "I")) \
        / (2 * gain_h) * TAU_MI
    gE = np.maximum(gE, 0.0)
    gI = np.maximum(gI, 0.0)

    return OperatingPoint(
        rE=rE, rI=rI, muE=muE, muI=muI, sigmaE=sigmaE, sigmaI=sigmaI, gE=gE, gI=gI,
        source=source, converged=converged, residual=residual, saturated=saturated,
        excitability=exc, inhibition=inh, wee_mult=float(w_ee_mult), nuext=float(nuext))


def homogeneous_gains(ratio: float = 1.0, w_ee_mult: float = 1.0) -> tuple[float, float]:
    """Dimensionless homogeneous gains (gE, gI) at the balanced rest op (for the dispersion sanity)."""
    op = mean_field(ratio, w_ee_mult)
    g = lif_gains(op)
    return float(g["E"]), float(g["I"])


# ---------------------------------------------------------------------------
# TDD-4: homogeneous Brunel-style dispersion (continuous k, no delay, rate branch)
#
# The 6-field block A(k) reduces (synaptic Schur elimination) to the delay=0 _char_det. Its rate
# branch is the degree-4 numerator N(lam,k); the two synaptic poles -1/TAU_AMPA, -1/TAU_GABA are NOT
# part of the dispersion. We read lambda(k) off N's rightmost root directly — exact, pole-free.
# ---------------------------------------------------------------------------
def _ghat_aniso(kx: float, ky: float, ell_par: float, ell_perp: float, theta: float) -> float:
    """Continuous Fourier amplitude of the anisotropic E->E Gaussian kernel at (kx, ky)."""
    ku = np.cos(theta) * kx + np.sin(theta) * ky
    kv = -np.sin(theta) * kx + np.cos(theta) * ky
    return float(np.exp(-0.5 * (ell_par ** 2 * ku ** 2 + ell_perp ** 2 * kv ** 2)))


def _ghat_iso(kx: float, ky: float, l_inh: float) -> float:
    return float(np.exp(-0.5 * l_inh ** 2 * (kx ** 2 + ky ** 2)))


def _assemble_a_k(gp: float, gi: float, gE: float, gI: float, *,
                  wee: float, wEI: float, wII: float) -> np.ndarray:
    """The locked per-mode 6x6 block A(k) from (gp, gi) amplitudes. Ordering rE,rI,sEE,sEI,sIE,sII.

    ``wee`` = w_ee_mult*W_EE (recurrent E->E weight), ``wEI`` = q*W_EI (effective I->E),
    ``wII`` = q*W_II or W_II (I->I). Verified against _char_det (delay=0) — see the LOCKED design doc.
    """
    A = np.zeros((6, 6))
    A[0, 0] = -1.0 / TAU_ME
    A[0, 2] = gE * C_EE * wee / TAU_ME
    A[0, 3] = -gE * C_EI * wEI / TAU_ME
    A[1, 1] = -1.0 / TAU_MI
    A[1, 4] = gI * C_IE * W_IE / TAU_MI
    A[1, 5] = -gI * C_II * wII / TAU_MI
    A[2, 0] = gp / TAU_AMPA
    A[2, 2] = -1.0 / TAU_AMPA
    A[3, 1] = gi / TAU_GABA
    A[3, 3] = -1.0 / TAU_GABA
    A[4, 0] = gi / TAU_AMPA
    A[4, 4] = -1.0 / TAU_AMPA
    A[5, 1] = gi / TAU_GABA
    A[5, 5] = -1.0 / TAU_GABA
    return A


def a_k_continuous(kx: float, ky: float, gE: float, gI: float, *, w_ee_mult: float,
                   wEI: float, wII: float, ell_par: float, ell_perp: float, l_inh: float,
                   theta: float) -> np.ndarray:
    """A(k) 6x6 with continuous-Gaussian kernel amplitudes at wavevector (kx, ky)."""
    gp = _ghat_aniso(kx, ky, ell_par, ell_perp, theta)
    gi = _ghat_iso(kx, ky, l_inh)
    return _assemble_a_k(gp, gi, gE, gI, wee=w_ee_mult * W_EE, wEI=wEI, wII=wII)


def _rate_branch_roots(gp: float, gi: float, gE: float, gI: float, *,
                       wee: float, wEI: float, wII: float) -> np.ndarray:
    """The 4 roots of the rate-branch numerator N(lam,k) (the dispersion, poles excluded)."""
    WEE = C_EE * wee * gp
    WEI = C_EI * wEI * gi
    WIE = C_IE * W_IE * gi
    WII = C_II * wII * gi
    t1 = np.array([TAU_ME * TAU_AMPA, TAU_ME + TAU_AMPA, 1.0 - gE * WEE])
    t2 = np.array([TAU_MI * TAU_GABA, TAU_MI + TAU_GABA, 1.0 + gI * WII])
    N = np.convolve(t1, t2)                       # degree-4, highest-order first
    N[-1] += gE * gI * WEI * WIE                  # + constant cross term
    return np.roots(N)


def dispersion_lambda(kx: float, ky: float, gE: float, gI: float, *, w_ee_mult: float = 1.0,
                      wEI: float = W_EI, wII: float = W_II, ell_par: float = ELL_PAR,
                      ell_perp: float = ELL_PERP, l_inh: float = L_INH,
                      theta: float = THETA_EE) -> complex:
    """Leading (rightmost) rate-branch eigenvalue lambda at wavevector (kx, ky) (continuous kernel)."""
    gp = _ghat_aniso(kx, ky, ell_par, ell_perp, theta)
    gi = _ghat_iso(kx, ky, l_inh)
    roots = _rate_branch_roots(gp, gi, gE, gI, wee=w_ee_mult * W_EE, wEI=wEI, wII=wII)
    return complex(roots[int(np.argmax(roots.real))])


def homogeneous_dispersion(gE: float, gI: float, *, w_ee_mult: float = 1.0, wEI: float = W_EI,
                           wII: float = W_II, ell_par: float = ELL_PAR, ell_perp: float = ELL_PERP,
                           l_inh: float = L_INH, theta: float = THETA_EE, kmax: float = 3.0,
                           nk: int = 31, along_axis: bool = True) -> dict:
    """Scan lambda(k) along (or across) the E->E axis. Returns the dispersion curve + summary.

    ``along_axis``: scan k along the E->E major axis (theta); else along the perpendicular axis.
    Mirrors ``src.sef_hfo_lif.closed_loop_leading`` but for the no-delay rate-branch dispersion.
    """
    ks = np.linspace(0.0, kmax, nk)
    ux, uy = (np.cos(theta), np.sin(theta)) if along_axis else (-np.sin(theta), np.cos(theta))
    re = np.empty(nk)
    im = np.empty(nk)
    for j, k in enumerate(ks):
        lam = dispersion_lambda(k * ux, k * uy, gE, gI, w_ee_mult=w_ee_mult, wEI=wEI, wII=wII,
                                ell_par=ell_par, ell_perp=ell_perp, l_inh=l_inh, theta=theta)
        re[j] = lam.real
        im[j] = abs(lam.imag)
    jstar = int(np.argmax(re))
    re_max = float(re[jstar])
    regime = "unstable" if re_max > 1e-3 else ("candidate" if re_max > -0.02 else "stable")
    return dict(
        k=ks, lambda_re=re, lambda_im=im, k_star=float(ks[jstar]), re_max=re_max,
        omega=float(im[jstar]), freq_Hz=float(1000.0 * im[jstar] / (2 * np.pi)),
        is_hopf=bool(ks[jstar] > 1e-3 and im[jstar] > 1e-3 and re[jstar] > re[0] + 1e-4),
        regime=regime, along_axis=bool(along_axis))


def dispersion_to_json(disp: dict) -> dict:
    """JSON-serializable view of a ``homogeneous_dispersion`` result (arrays -> lists)."""
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in disp.items()}


# ---------------------------------------------------------------------------
# TDD-5: finite 6N x 6N Jacobian (dense for tiny grids, matrix-free for real grids) + JVP
#
# Block layout (LOCKED design §3): rate rows carry per-cell diagonal gains; synaptic rows carry the
# periodic convolution operators. Verified: homogeneous no-core J spectrum == union over grid modes
# of A(k) (the dispersion samples).
# ---------------------------------------------------------------------------
def _op_gain_fields(op: OperatingPoint):
    """The four (n,n) rate-row coefficient fields of the Jacobian (all heterogeneity lives here)."""
    wee = op.wee_mult * W_EE
    q = op.inhibition.q
    wEI = q * W_EI
    wII = q * W_II if op.inhibition.scale_II else np.full_like(q, W_II)
    GE_EE = op.gE * C_EE * wee / TAU_ME
    GE_EI = -op.gE * C_EI * wEI / TAU_ME
    GI_IE = op.gI * C_IE * W_IE / TAU_MI
    GI_II = -op.gI * C_II * wII / TAU_MI
    return GE_EE, GE_EI, GI_IE, GI_II


def op_state_vector(op: OperatingPoint, kernels: Kernels, grid: Grid) -> np.ndarray:
    """The packed 6N operating-point state z* (synaptic states at their steady convolved values)."""
    return pack_state({
        "rE": op.rE, "rI": op.rI,
        "sEE": convolve_periodic(op.rE, kernels.K_EE),
        "sEI": convolve_periodic(op.rI, kernels.K_I),
        "sIE": convolve_periodic(op.rE, kernels.K_I),
        "sII": convolve_periodic(op.rI, kernels.K_I),
    }, grid)


def field_rhs(z: np.ndarray, grid: Grid, kernels: Kernels, op: OperatingPoint) -> np.ndarray:
    """Nonlinear 6-field rate RHS, with sigma FROZEN at the op (the linearization holds sigma fixed,
    matching _char_det). The Jacobian of this map at z* is exactly ``build_jacobian_dense``."""
    s = unpack_state(z, grid)
    wee = op.wee_mult * W_EE
    q = op.inhibition.q
    wEI = q * W_EI
    wII = q * W_II if op.inhibition.scale_II else W_II
    muxE = TAU_ME * JX_E * op.nuext
    muxI = TAU_MI * JX_I * op.nuext
    muE = TAU_ME * (C_EE * wee * s["sEE"] - C_EI * wEI * s["sEI"]) + muxE + op.excitability.mu_core
    muI = TAU_MI * (C_IE * W_IE * s["sIE"] - C_II * wII * s["sII"]) + muxI
    drE = (-s["rE"] + _phi_field(muE, op.sigmaE, "E")) / TAU_ME
    drI = (-s["rI"] + _phi_field(muI, op.sigmaI, "I")) / TAU_MI
    dsEE = (convolve_periodic(s["rE"], kernels.K_EE) - s["sEE"]) / TAU_AMPA
    dsEI = (convolve_periodic(s["rI"], kernels.K_I) - s["sEI"]) / TAU_GABA
    dsIE = (convolve_periodic(s["rE"], kernels.K_I) - s["sIE"]) / TAU_AMPA
    dsII = (convolve_periodic(s["rI"], kernels.K_I) - s["sII"]) / TAU_GABA
    return pack_state({"rE": drE, "rI": drI, "sEE": dsEE, "sEI": dsEI, "sIE": dsIE, "sII": dsII}, grid)


def _conv_matrix(K: np.ndarray, n: int) -> np.ndarray:
    """Dense N x N periodic-convolution matrix M with M @ x.ravel() == convolve_periodic(x, K).ravel()."""
    N = n * n
    M = np.empty((N, N))
    e = np.zeros((n, n))
    for b in range(N):
        e.flat[b] = 1.0
        M[:, b] = convolve_periodic(e, K).ravel()
        e.flat[b] = 0.0
    return M


def build_jacobian_dense(grid: Grid, kernels: Kernels, op: OperatingPoint) -> np.ndarray:
    """Assemble the dense 6N x 6N Jacobian (tiny grids / debug)."""
    n, N = grid.n, grid.size
    GE_EE, GE_EI, GI_IE, GI_II = _op_gain_fields(op)
    M_EE = _conv_matrix(kernels.K_EE, n)
    M_I = _conv_matrix(kernels.K_I, n)
    I_N = np.eye(N)
    J = np.zeros((6 * N, 6 * N))

    def put(r, c, block):
        J[r * N:(r + 1) * N, c * N:(c + 1) * N] = block

    put(0, 0, -1.0 / TAU_ME * I_N)
    put(0, 2, np.diag(GE_EE.ravel()))
    put(0, 3, np.diag(GE_EI.ravel()))
    put(1, 1, -1.0 / TAU_MI * I_N)
    put(1, 4, np.diag(GI_IE.ravel()))
    put(1, 5, np.diag(GI_II.ravel()))
    put(2, 0, M_EE / TAU_AMPA)
    put(2, 2, -1.0 / TAU_AMPA * I_N)
    put(3, 1, M_I / TAU_GABA)
    put(3, 3, -1.0 / TAU_GABA * I_N)
    put(4, 0, M_I / TAU_AMPA)
    put(4, 4, -1.0 / TAU_AMPA * I_N)
    put(5, 1, M_I / TAU_GABA)
    put(5, 5, -1.0 / TAU_GABA * I_N)
    return J


def jacobian_linear_operator(grid: Grid, kernels: Kernels, op: OperatingPoint):
    """Matrix-free ``scipy.sparse.linalg.LinearOperator`` (6N x 6N) — never forms an N x N matrix."""
    from scipy.sparse.linalg import LinearOperator
    n, N = grid.n, grid.size
    GE_EE, GE_EI, GI_IE, GI_II = _op_gain_fields(op)
    K_EE, K_I = kernels.K_EE, kernels.K_I

    def matvec(v):
        s = unpack_state(np.asarray(v, dtype=float).ravel(), grid)
        out_rE = -s["rE"] / TAU_ME + GE_EE * s["sEE"] + GE_EI * s["sEI"]
        out_rI = -s["rI"] / TAU_MI + GI_IE * s["sIE"] + GI_II * s["sII"]
        out_sEE = (convolve_periodic(s["rE"], K_EE) - s["sEE"]) / TAU_AMPA
        out_sEI = (convolve_periodic(s["rI"], K_I) - s["sEI"]) / TAU_GABA
        out_sIE = (convolve_periodic(s["rE"], K_I) - s["sIE"]) / TAU_AMPA
        out_sII = (convolve_periodic(s["rI"], K_I) - s["sII"]) / TAU_GABA
        return pack_state({"rE": out_rE, "rI": out_rI, "sEE": out_sEE,
                           "sEI": out_sEI, "sIE": out_sIE, "sII": out_sII}, grid)

    return LinearOperator((6 * N, 6 * N), matvec=matvec, dtype=float)


# Rate-branch leading eigenpair: among eig(J), the max-Re mode whose eigenvector carries real rate
# (rE/rI) power — excluding the boring pure-synaptic relaxation modes near -1/TAU_AMPA, -1/TAU_GABA.
_RATE_PARTICIPATION_MIN: float = 0.02


def _rate_participation(evecs: np.ndarray, N: int) -> np.ndarray:
    rate_pow = np.sum(np.abs(evecs[:2 * N, :]) ** 2, axis=0)
    tot = np.sum(np.abs(evecs) ** 2, axis=0)
    return rate_pow / np.maximum(tot, 1e-300)


def leading_rate_eigenpair(J: np.ndarray, grid: Grid):
    """(lambda, right-eigenvector, rate_participation) of the leading rate-branch mode of dense J."""
    evals, evecs = np.linalg.eig(J)
    part = _rate_participation(evecs, grid.size)
    cand = np.where(part > _RATE_PARTICIPATION_MIN)[0]
    if cand.size == 0:
        cand = np.arange(evals.size)
    j = cand[int(np.argmax(evals[cand].real))]
    return evals[j], evecs[:, j], float(part[j])


def unpack_state_complex(z: np.ndarray, grid: Grid) -> dict[str, np.ndarray]:
    """Like ``unpack_state`` but preserves complex dtype (eigenvectors)."""
    z = np.asarray(z).ravel()
    N = grid.size
    return {f: z[i * N:(i + 1) * N].reshape(grid.n, grid.n) for i, f in enumerate(STATE_FIELDS)}


def mode_e_field(eigvec: np.ndarray, grid: Grid) -> np.ndarray:
    """The (complex) E-rate (rE) field of a 6N eigenvector."""
    return unpack_state_complex(eigvec, grid)["rE"]


def core_overlap(eigvec_or_field, grid: Grid, core: CoreMask) -> float:
    """Fraction of E-power inside the core mask: sum_core|phi_E|^2 / sum_all|phi_E|^2.

    Accepts a full 6N eigenvector (its rE block is used) or a bare (n,n) / N-length E field."""
    arr = np.asarray(eigvec_or_field)
    e = arr.reshape(grid.n, grid.n) if arr.size == grid.size else mode_e_field(arr, grid)
    p = np.abs(e) ** 2
    tot = p.sum()
    return float(p[core.mask].sum() / tot) if tot > 0 else 0.0


# ---------------------------------------------------------------------------
# TDD-6: eigenpair extraction with left/right modes + non-normal diagnostics
#
# The operator is NON-NORMAL: the right eigenvector says what the mode looks like; the LEFT
# eigenvector says whether a core perturbation can excite it (core controllability). Both are
# required. We work on the rate branch (the pure-synaptic pole modes are degenerate and boring).
# ---------------------------------------------------------------------------
EIG_RESIDUAL_TOL: float = 1e-8
BIORTHOGONALITY_TOL: float = 1e-6


@dataclass
class EigResult:
    eigenvalues: np.ndarray       # sorted by Re(lambda) descending
    right: np.ndarray             # (6N, m) right eigenvectors (columns), phi
    left: np.ndarray              # (6N, m) left eigenvectors, psi; biorthonormal: psi_i^H phi_j = delta_ij
    status: str                   # 'resolved' | 'unresolved'
    residual_right: float
    residual_left: float
    biorthogonality_error: float
    synaptic_pole_floor_active: bool


def rate_eigenpairs(J: np.ndarray, grid: Grid, *, n_modes: int = 4,
                    participation_min: float = _RATE_PARTICIPATION_MIN) -> EigResult:
    """Leading rate-branch eigenpairs (right + biorthonormal left) of dense J.

    Failed / non-finite operators are marked 'unresolved', never silently stable. The
    ``synaptic_pole_floor_active`` flag records whether the leading rate mode sits below the
    -1/TAU_GABA synaptic pole (deeply stable; the literal rightmost eigenvalue would be a pole).
    """
    from scipy.linalg import eig as _dense_eig
    empty = np.empty((6 * grid.size, 0))
    if not np.all(np.isfinite(J)):
        return EigResult(np.array([]), empty, empty, "unresolved", np.inf, np.inf, np.inf, False)
    w, vl, vr = _dense_eig(J, left=True, right=True)
    part = _rate_participation(vr, grid.size)
    keep = np.where(part > participation_min)[0]
    if keep.size == 0:
        return EigResult(np.array([]), empty, empty, "unresolved", np.inf, np.inf, np.inf, False)
    order = keep[np.argsort(-w[keep].real)][:n_modes]
    lam = w[order]
    R = vr[:, order].astype(complex)
    L = vl[:, order].astype(complex)
    # biorthonormalize: scale each left vector so psi_i^H phi_i = 1
    for i in range(order.size):
        c = np.vdot(L[:, i], R[:, i])            # conj(L_i) . R_i = psi_i^H phi_i
        if abs(c) < 1e-300:
            return EigResult(lam, R, L, "unresolved", np.inf, np.inf, np.inf, False)
        L[:, i] = L[:, i] / np.conj(c)
    # residuals
    res_r = 0.0
    res_l = 0.0
    for i in range(order.size):
        res_r = max(res_r, np.linalg.norm(J @ R[:, i] - lam[i] * R[:, i]) / np.linalg.norm(R[:, i]))
        res_l = max(res_l, np.linalg.norm(J.conj().T @ L[:, i] - np.conj(lam[i]) * L[:, i])
                    / np.linalg.norm(L[:, i]))
    bio = float(np.max(np.abs(L.conj().T @ R - np.eye(order.size)))) if order.size else 0.0
    status = "resolved" if (res_r < EIG_RESIDUAL_TOL and res_l < EIG_RESIDUAL_TOL) else "unresolved"
    floor = bool(lam[0].real < -1.0 / TAU_GABA)
    return EigResult(lam, R, L, status, float(res_r), float(res_l), bio, floor)


# ---------------------------------------------------------------------------
# TDD-7: synthetic-mode metrics + metric-based classifier
#
# Metric semantics are locked against SYNTHETIC modes before any real eigenmode is read. The two
# axis scores are kept SEPARATE: a propagation ridge (spatial elongation) and its wavevector
# (phase gradient) can differ by 90 degrees (plan §5).
# ---------------------------------------------------------------------------
def mode_growth(lam: complex) -> float:
    return float(np.real(lam))


def mode_frequency_hz(lam: complex) -> float:
    """Oscillation frequency (Hz) of a mode; lambda is in 1/ms, so *1000 to Hz."""
    return float(abs(np.imag(lam)) / (2 * np.pi) * 1000.0)


def spectral_gap(eigenvalues: np.ndarray) -> float:
    """alpha_1 - alpha_2 (leading minus second growth). Small gap = mode competition."""
    e = np.asarray(eigenvalues)
    return float(e[0].real - e[1].real) if e.size >= 2 else float("inf")


def _power_orientation(field: np.ndarray, grid: Grid) -> tuple[float, float]:
    """(principal-axis angle, anisotropy in [0,1]) of the |field|^2 spatial second-moment tensor."""
    p = np.abs(field) ** 2
    X, Y = grid.coords()
    tot = p.sum()
    if tot <= 0:
        return 0.0, 0.0
    xm, ym = (p * X).sum() / tot, (p * Y).sum() / tot
    Sxx = (p * (X - xm) ** 2).sum() / tot
    Syy = (p * (Y - ym) ** 2).sum() / tot
    Sxy = (p * (X - xm) * (Y - ym)).sum() / tot
    ang = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)
    ev = np.clip(np.linalg.eigvalsh(np.array([[Sxx, Sxy], [Sxy, Syy]])), 0.0, None)
    aniso = float((ev.max() - ev.min()) / (ev.max() + ev.min() + 1e-30))
    return float(ang), aniso


def elongation_axis_score(field: np.ndarray, grid: Grid, theta: float = THETA_EE) -> float:
    """Alignment of the mode's spatial ELONGATION (ridge) with the E->E axis, in [-1, 1].

    +1 = elongated along theta; -1 = elongated perpendicular; ~0 = isotropic. (Spatial second
    moment — distinct from the wavevector score below.)"""
    ang, aniso = _power_orientation(field, grid)
    return float(aniso * np.cos(2 * (ang - theta)))


def off_axis_score(field: np.ndarray, grid: Grid, theta: float = THETA_EE) -> float:
    """Elongation power perpendicular to the E->E axis, in [0, 1]. High = ridge off the axis."""
    ang, aniso = _power_orientation(field, grid)
    return float(aniso * np.sin(ang - theta) ** 2)


def phase_gradient_axis_score(field: np.ndarray, grid: Grid, theta: float = THETA_EE) -> float:
    """Alignment of the mode's WAVEVECTOR (phase gradient / Fourier energy) with the E->E axis,
    in [-1, 1]. Distinct from the elongation score: a ridge along theta carries a wavevector ACROSS
    theta (90 deg), so the two must not be collapsed (plan §5)."""
    F = np.fft.fft2(field)
    F[0, 0] = 0.0                                  # drop DC
    p = np.abs(F) ** 2
    kx = 2 * np.pi * np.fft.fftfreq(grid.n, d=grid.spacing)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    tot = p.sum()
    if tot <= 0:
        return 0.0
    Txx = (p * KX ** 2).sum() / tot
    Tyy = (p * KY ** 2).sum() / tot
    Txy = (p * KX * KY).sum() / tot
    beta = 0.5 * np.arctan2(2 * Txy, Txx - Tyy)
    ev = np.clip(np.linalg.eigvalsh(np.array([[Txx, Txy], [Txy, Tyy]])), 0.0, None)
    aniso = float((ev.max() - ev.min()) / (ev.max() + ev.min() + 1e-30))
    return float(aniso * np.cos(2 * (beta - theta)))


def globality(field: np.ndarray, grid: Grid) -> float:
    """Normalized participation ratio in [0, 1]: 1 = uniform/global, ~1/N = delta/localized."""
    p = np.abs(field) ** 2
    s1, s2 = p.sum(), (p ** 2).sum()
    return float(s1 ** 2 / (grid.size * s2)) if s2 > 0 else 0.0


def core_perturbation_vector(grid: Grid, core: CoreMask) -> np.ndarray:
    """Unit 6N perturbation localized to the core E cells (the b_core controllability input)."""
    rE = np.zeros((grid.n, grid.n))
    rE[core.mask] = 1.0
    z = pack_state({f: (rE if f == "rE" else np.zeros((grid.n, grid.n))) for f in STATE_FIELDS}, grid)
    nrm = np.linalg.norm(z)
    return z / nrm if nrm > 0 else z


def core_controllability(left_eigvec: np.ndarray, grid: Grid, core: CoreMask) -> float:
    """|psi_m^H b_core| — how strongly a core perturbation excites the mode. Uses the LEFT
    eigenvector (the right eigenvector would answer a different question)."""
    b = core_perturbation_vector(grid, core)
    return float(abs(np.vdot(left_eigvec, b)))


def transient_gain(matrix, b: np.ndarray, T: float) -> float:
    """||exp(M*T) b|| / ||b|| — finite-time amplification (non-normal transient growth)."""
    from scipy.sparse.linalg import expm_multiply
    b = np.asarray(b, dtype=float)
    yT = expm_multiply(np.asarray(matrix, dtype=float) * T, b)
    return float(np.linalg.norm(yT) / np.linalg.norm(b))


def finite_time_gain(J: np.ndarray, grid: Grid, core: CoreMask, T: float) -> float:
    """Finite-time gain of a core perturbation through the Jacobian over window T (ms)."""
    return transient_gain(J, core_perturbation_vector(grid, core), T)


# Metric-based mode-class thresholds (calibrated in TDD-8/9; conservative defaults here).
_AXIAL_MIN: float = 0.30          # elongation_axis_score for 'axial'
_GLOBAL_MIN: float = 0.60         # globality for 'global'
_CORE_MIN: float = 0.50           # core_overlap for 'local'
_STABLE_GROWTH: float = -0.02     # growth below this (and low finite-time gain) = 'stable'
_FTG_EVENT: float = 2.0           # finite-time gain above this = transiently event-capable


def classify_mode(*, growth: float, core_overlap_: float, globality_: float,
                  elongation_axis: float, off_axis: float, finite_time_gain_: float = 0.0,
                  resolved: bool = True, saturated: bool = False) -> str:
    """Metric-based leading-mode class (plan §7): stable/local/axial/mixed/global/runaway/unresolved.

    Dynamics gate first (a deeply stable point is 'stable' whatever its decaying mode's shape; a
    saturated op is 'runaway'); a point is event-capable if its growth is near/above 0 OR its
    non-normal finite-time gain is large (the interictal self-limited case the plan §5 flags). Only
    event-capable, non-saturated points are classified by SHAPE."""
    if not resolved:
        return "unresolved"
    if saturated:
        return "runaway"
    event_capable = growth > _STABLE_GROWTH or finite_time_gain_ > _FTG_EVENT
    if not event_capable:
        return "stable"
    axial = elongation_axis > _AXIAL_MIN and elongation_axis > off_axis
    glob = globality_ > _GLOBAL_MIN
    if axial and glob:
        return "mixed"
    if glob:
        return "global"
    if axial:
        return "axial"
    if core_overlap_ > _CORE_MIN:
        return "local"
    return "mixed"


# ---------------------------------------------------------------------------
# TDD-8 / TDD-9: single-point spectral analysis + phase-map scan
# ---------------------------------------------------------------------------
_FTG_WINDOW_MS: float = 20.0          # finite-time-gain window (HFO-burst scale)

# mode_metrics.csv schema (the required columns the phase-map artifact must carry).
MODE_METRICS_COLUMNS: tuple[str, ...] = (
    "x_mu_core", "y_q_global", "q_core", "w_ee_mult", "ratio",
    "op_status", "eig_status", "mode_class",
    "alpha_1", "freq_hz", "spectral_gap", "core_overlap", "elongation_axis",
    "phase_gradient_axis", "globality", "off_axis", "core_controllability",
    "finite_time_gain", "synaptic_pole_floor_active",
)


@dataclass
class SpectralPoint:
    x_mu_core: float
    y_q_global: float
    q_core: float
    w_ee_mult: float
    ratio: float
    op_status: str
    eig_status: str
    mode_class: str
    alpha_1: float
    freq_hz: float
    spectral_gap: float
    core_overlap: float
    elongation_axis: float
    phase_gradient_axis: float
    globality: float
    off_axis: float
    core_controllability: float
    finite_time_gain: float
    synaptic_pole_floor_active: bool

    def as_row(self) -> dict:
        return {c: getattr(self, c) for c in MODE_METRICS_COLUMNS}


def _unresolved_point(x, y, q_core, w_ee_mult, ratio, op_status, eig_status) -> SpectralPoint:
    nan = float("nan")
    return SpectralPoint(x, y, q_core, w_ee_mult, ratio, op_status, eig_status, "unresolved",
                         nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, False)


def analyze_spectral_point(grid: Grid, kernels: Kernels, core: CoreMask, *, mu_core: float,
                           q_global: float, q_core: float = 1.0, ratio: float = 1.0,
                           w_ee_mult: float = 1.3, n_modes: int = 4,
                           ftg_window_ms: float = _FTG_WINDOW_MS) -> SpectralPoint:
    """Full spectral readout at one (core-excitability, global-disinhibition) phase-map point."""
    exc = build_excitability_field(grid, core, mu_core=mu_core)
    inh = build_inhibition_field(grid, core, q_global=q_global, q_core=q_core)
    op = solve_operating_point(grid, kernels, exc, inh, ratio=ratio, w_ee_mult=w_ee_mult)
    if op.status == "unresolved":
        return _unresolved_point(mu_core, q_global, q_core, w_ee_mult, ratio, op.status, "n/a")

    J = build_jacobian_dense(grid, kernels, op)
    res = rate_eigenpairs(J, grid, n_modes=n_modes)
    if res.status != "resolved" or res.eigenvalues.size == 0:
        return _unresolved_point(mu_core, q_global, q_core, w_ee_mult, ratio, op.status, res.status)

    lam0 = res.eigenvalues[0]
    eE = mode_e_field(res.right[:, 0], grid)
    co = core_overlap(res.right[:, 0], grid, core)
    elong = elongation_axis_score(eE, grid, kernels.theta)
    pg = phase_gradient_axis_score(eE, grid, kernels.theta)
    glob = globality(eE, grid)
    offax = off_axis_score(eE, grid, kernels.theta)
    ctrl = core_controllability(res.left[:, 0], grid, core)
    ftg = finite_time_gain(J, grid, core, ftg_window_ms)
    cls = classify_mode(growth=lam0.real, core_overlap_=co, globality_=glob,
                        elongation_axis=elong, off_axis=offax, finite_time_gain_=float(ftg),
                        saturated=op.saturated)
    return SpectralPoint(
        x_mu_core=float(mu_core), y_q_global=float(q_global), q_core=float(q_core),
        w_ee_mult=float(w_ee_mult), ratio=float(ratio), op_status=op.status, eig_status=res.status,
        mode_class=cls, alpha_1=mode_growth(lam0), freq_hz=mode_frequency_hz(lam0),
        spectral_gap=spectral_gap(res.eigenvalues), core_overlap=co, elongation_axis=elong,
        phase_gradient_axis=pg, globality=glob, off_axis=offax, core_controllability=float(ctrl),
        finite_time_gain=float(ftg), synaptic_pole_floor_active=res.synaptic_pole_floor_active)


def build_phase_map(grid: Grid, kernels: Kernels, core: CoreMask, *, x_values, y_values,
                    q_core: float = 1.0, ratio: float = 1.0, w_ee_mult: float = 1.3) -> list:
    """Scan core-excitability (x = mu_core) x global-disinhibition (y = q_global) into SpectralPoints."""
    return [analyze_spectral_point(grid, kernels, core, mu_core=float(x), q_global=float(y),
                                   q_core=q_core, ratio=ratio, w_ee_mult=w_ee_mult)
            for y in y_values for x in x_values]


def phase_map_to_rows(points: list) -> list:
    """SpectralPoints -> list of dict rows with the MODE_METRICS_COLUMNS schema."""
    return [p.as_row() for p in points]


def unresolved_fraction(points: list) -> float:
    if not points:
        return 1.0
    return sum(1 for p in points if p.mode_class == "unresolved") / len(points)


# ---------------------------------------------------------------------------
# TDD-10: controls / ablations — prove the structure is core/scaffold-specific
# ---------------------------------------------------------------------------
def dispersion_anisotropy(gE: float, gI: float, *, w_ee_mult: float, ell_par: float,
                          ell_perp: float, l_inh: float, theta: float = THETA_EE,
                          k: float = 1.5) -> float:
    """across-axis growth minus along-axis growth at wavenumber k. ~0 for isotropic (AR1) E->E;
    positive for the elongated AR2 scaffold (ridge-along-axis preference)."""
    along = dispersion_lambda(k * np.cos(theta), k * np.sin(theta), gE, gI, w_ee_mult=w_ee_mult,
                              ell_par=ell_par, ell_perp=ell_perp, l_inh=l_inh, theta=theta).real
    across = dispersion_lambda(-k * np.sin(theta), k * np.cos(theta), gE, gI, w_ee_mult=w_ee_mult,
                               ell_par=ell_par, ell_perp=ell_perp, l_inh=l_inh, theta=theta).real
    return float(across - along)


def _core_localization(grid: Grid, kernels: Kernels, core: CoreMask, *, mu_core: float,
                       q_global: float, ratio: float, w_ee_mult: float) -> float:
    """Enhancement of leading-mode E-power inside the core mask above its area fraction
    (>0 = the core gathers mode power; ~0 = no core-specific localization)."""
    p = analyze_spectral_point(grid, kernels, core, mu_core=mu_core, q_global=q_global,
                               ratio=ratio, w_ee_mult=w_ee_mult)
    if p.core_overlap != p.core_overlap:        # NaN (unresolved)
        return 0.0
    return float(p.core_overlap - core.area_fraction)


def run_controls(grid: Grid, *, ratio: float = 1.0, w_ee_mult: float = 1.3, mu_core: float = 0.8,
                 q_global: float = 1.0, core_radius: float = 0.9, off_axis_shift: float = 1.6,
                 shuffle_seeds=(0, 1, 2, 3)) -> dict:
    """Run the required control ablations and return a control_summary dict."""
    th = THETA_EE
    ker = build_kernels(grid, ell_perp=0.6)
    core = make_core_mask(grid, kind="single", radius=core_radius)
    gE, gI = homogeneous_gains(ratio, w_ee_mult)

    loc_core = _core_localization(grid, ker, core, mu_core=mu_core, q_global=q_global,
                                  ratio=ratio, w_ee_mult=w_ee_mult)
    loc_no_core = _core_localization(grid, ker, core, mu_core=0.0, q_global=q_global,
                                     ratio=ratio, w_ee_mult=w_ee_mult)

    aniso_ar2 = dispersion_anisotropy(gE, gI, w_ee_mult=w_ee_mult, ell_par=ker.ell_par,
                                      ell_perp=ker.ell_perp, l_inh=ker.l_inh)
    aniso_ar1 = dispersion_anisotropy(gE, gI, w_ee_mult=w_ee_mult, ell_par=0.6, ell_perp=0.6,
                                      l_inh=ker.l_inh)

    core_off = make_core_mask(grid, kind="off_axis", radius=core_radius, off_axis_shift=off_axis_shift)
    loc_off_at_offcore = _core_localization(grid, ker, core_off, mu_core=mu_core, q_global=q_global,
                                            ratio=ratio, w_ee_mult=w_ee_mult)

    # shuffled core: the same number of core cells scattered at random (no contiguous scaffold).
    n_core = int(core.mask.sum())
    shuffled_loc = []
    for s in shuffle_seeds:
        rng = np.random.default_rng(s)
        flat = np.zeros(grid.size, dtype=bool)
        flat[rng.choice(grid.size, size=n_core, replace=False)] = True
        sh_mask = flat.reshape(grid.n, grid.n)
        sh_core = CoreMask("shuffled", sh_mask, (), core_radius, th)
        shuffled_loc.append(_core_localization(grid, ker, sh_core, mu_core=mu_core,
                                               q_global=q_global, ratio=ratio, w_ee_mult=w_ee_mult))

    return {
        "no_core": {"core_localization": loc_no_core},
        "core": {"core_localization": loc_core},
        "ar1_isotropic": {"dispersion_anisotropy": aniso_ar1},
        "ar2_anisotropic": {"dispersion_anisotropy": aniso_ar2},
        "off_axis_core": {"core_localization_at_offcore": loc_off_at_offcore,
                          "core_localization_on_axis_ref": loc_core},
        "shuffled_core": {"core_localization_mean": float(np.mean(shuffled_loc)),
                          "core_localization_per_seed": [float(x) for x in shuffled_loc],
                          "contiguous_core_localization": loc_core},
    }


REQUIRED_CONTROLS: tuple[str, ...] = (
    "no_core", "core", "ar1_isotropic", "ar2_anisotropic", "off_axis_core", "shuffled_core",
)


# ---------------------------------------------------------------------------
# TDD-11: rate-field dynamic spot checks — verify the linear predictions against the nonlinear field
#
# integrate_lif_field is HOMOGENEOUS, so M3B carries its own heterogeneous integrator (the SAME
# 6-field dynamics solve_operating_point integrates, plus a transient stimulus). A core kick probes
# the non-normal transient response the linear spectrum predicts.
# ---------------------------------------------------------------------------
from src.sef_hfo_lif import DETECT as _DETECT          # active-pixel threshold above rest
_RETURN_FRAC: float = 0.2                                # final/peak active fraction => returned
_RUNAWAY_FRAC: float = 0.4                               # peak active fraction => high-rate/runaway risk


@dataclass
class RateFieldResponse:
    active_fraction: np.ndarray         # (nsteps,) fraction of cells active above rest
    max_active: float
    final_active: float
    returned: bool
    response_axis_score: float          # elongation of the peak-response E field along the E->E axis
    peak_field: np.ndarray              # (n,n) rE at peak active fraction
    op_status: str


def simulate_ratefield_response(grid: Grid, kernels: Kernels, exc: ExcitabilityField,
                                inh: InhibitionField, *, ratio: float = 1.0, w_ee_mult: float = 1.3,
                                stim_amp: float = 6.0, stim_radius: float = 1.2,
                                stim_center: tuple[float, float] = (0.0, 0.0),
                                stim_window_ms: float = 15.0, dt: float = 0.5, t_max: float = 250.0,
                                noise_amp: float = 0.0, seed: int = 0) -> RateFieldResponse:
    """Integrate the heterogeneous 6-field rate field from its op under a finite core pulse (and/or
    noise), measuring the transient: active fraction over time, return-to-baseline, response axis."""
    op = solve_operating_point(grid, kernels, exc, inh, ratio=ratio, w_ee_mult=w_ee_mult)
    n = grid.n
    wee = w_ee_mult * W_EE
    muxE, muxI = TAU_ME * JX_E * op.nuext, TAU_MI * JX_I * op.nuext
    wEI = inh.q * W_EI
    wII = inh.q * W_II if inh.scale_II else np.full((n, n), W_II)
    X, Y = grid.coords()
    disk = ((X - stim_center[0]) ** 2 + (Y - stim_center[1]) ** 2 <= stim_radius ** 2).astype(float)
    rng = np.random.default_rng(seed)

    rE, rI = op.rE.copy(), op.rI.copy()
    sEE = convolve_periodic(rE, kernels.K_EE)
    sEI = convolve_periodic(rI, kernels.K_I)
    sIE = convolve_periodic(rE, kernels.K_I)
    sII = convolve_periodic(rI, kernels.K_I)
    thr = op.rE + _DETECT
    nsteps = int(t_max / dt)
    active = np.empty(nsteps)
    peak_field = rE.copy()
    peak_active = -1.0
    for t in range(nsteps):
        stim = stim_amp * disk if (t * dt) < stim_window_ms else 0.0
        if noise_amp > 0:
            stim = stim + noise_amp * rng.standard_normal((n, n))
        sEE += dt / TAU_AMPA * (convolve_periodic(rE, kernels.K_EE) - sEE)
        sEI += dt / TAU_GABA * (convolve_periodic(rI, kernels.K_I) - sEI)
        sIE += dt / TAU_AMPA * (convolve_periodic(rE, kernels.K_I) - sIE)
        sII += dt / TAU_GABA * (convolve_periodic(rI, kernels.K_I) - sII)
        muE = TAU_ME * (C_EE * wee * sEE - C_EI * wEI * sEI) + muxE + exc.mu_core + stim
        muI = TAU_MI * (C_IE * W_IE * sIE - C_II * wII * sII) + muxI
        varE = TAU_ME * (C_EE * wee ** 2 * sEE + C_EI * wEI ** 2 * sEI) + TAU_ME * JX_E ** 2 * op.nuext
        varI = TAU_MI * (C_IE * W_IE ** 2 * sIE + C_II * wII ** 2 * sII) + TAU_MI * JX_I ** 2 * op.nuext
        rE = rE + dt / TAU_ME * (-rE + _phi_field(muE, np.sqrt(np.maximum(varE, 1e-9)), "E"))
        rI = rI + dt / TAU_MI * (-rI + _phi_field(muI, np.sqrt(np.maximum(varI, 1e-9)), "I"))
        a = float(np.mean(rE > thr))
        active[t] = a
        if a > peak_active:
            peak_active = a
            peak_field = rE.copy()

    max_active = float(active.max())
    final_active = float(active[-1])
    returned = bool(final_active <= _RETURN_FRAC * max(max_active, 1e-12))
    resp = elongation_axis_score(peak_field - op.rE, grid, kernels.theta)
    return RateFieldResponse(active, max_active, final_active, returned, float(resp),
                             peak_field, op.status)


# ---------------------------------------------------------------------------
# TDD-12: SNN frozen-state spot checks — verify phase-map predictions in actual SNN pilots
#
# Map phase-map params to the SEF-HFO SNN engine (src/snn_engine) with a documented transform, run a
# short frozen-state pilot, classify R0..R4b. The deterministic machinery (mapping/classifier/mass/
# consistency) is tested on synthetic metrics; run_snn_spotcheck does the real (slow) pilot.
# ---------------------------------------------------------------------------
SNN_R_CLASSES: tuple[str, ...] = ("R0", "R1", "R2", "R3", "R4a", "R4b")
SNN_SPOTCHECK_FIELDS: tuple[str, ...] = (
    "x_mu_core", "y_q_global", "w_ee_mult", "baseline", "peak", "tail", "returned",
    "peak_active_frac", "inside", "outside", "active_mass", "R_class",
)
_SNN_BASE_G: float = 3.6          # baseline inhibitory gain (Params.g)
_SNN_BASE_W_EE: float = 0.1575    # baseline E->E weight (Params.w_EE)
_SNN_BASE_V_TH: float = 18.0      # baseline threshold (Params.V_th)


def snn_param_mapping(mu_core: float, q_global: float, w_ee_mult: float = 1.0) -> dict:
    """Documented transform from M3B phase-map params to SEF-HFO SNN controls.

    q_global scales the inhibitory gain g (lower q = disinhibition); w_ee_mult scales the recurrent
    E->E weight; mu_core lowers the CORE E-cell threshold (via sample_core_field), not a global shift.
    """
    return {
        "g": _SNN_BASE_G * q_global,
        "w_EE": _SNN_BASE_W_EE * w_ee_mult,
        "core_v_th": _SNN_BASE_V_TH - mu_core,
        "transforms": {
            "q_global->g": "g = 3.6 * q_global  (scales I->E and I->I; lower q = disinhibition)",
            "w_ee_mult->w_EE": "w_EE = 0.1575 * w_ee_mult  (recurrent E->E gain)",
            "mu_core->core_v_th": "core E threshold = 18.0 - mu_core via sample_core_field (core only)",
        },
    }


def classify_snn_r_class(metrics: dict) -> str:
    """Map SNN ``compute_metrics`` output to an R-class (R0..R4b)."""
    base = max(metrics["baseline"], 1e-9)
    paf = metrics["peak_active_frac"]
    if metrics["peak"] <= 2.0 * base:
        return "R0"                                      # no event / recovery
    if not metrics["returned"]:
        return "R4b"                                     # sustained tonic runaway
    if paf > 0.6:
        return "R4a"                                     # global recruitment, self-terminated
    outside = metrics["outside"] > 1.5 * max(metrics["inside"], 1.0)
    if outside and paf > 0.2:
        return "R3"                                      # larger axial spread
    if outside:
        return "R2"                                      # local axial spread
    return "R1"                                          # local


def snn_active_mass(metrics: dict) -> float:
    """Active mass = peak fraction of E neurons recruited (the spatial recruitment extent)."""
    return float(metrics["peak_active_frac"])


def snn_spotcheck_record(mu_core: float, q_global: float, w_ee_mult: float, metrics: dict) -> dict:
    """Assemble a spot-check record (SNN_SPOTCHECK_FIELDS) including the R-class + active mass."""
    return {
        "x_mu_core": float(mu_core), "y_q_global": float(q_global), "w_ee_mult": float(w_ee_mult),
        "baseline": float(metrics["baseline"]), "peak": float(metrics["peak"]),
        "tail": float(metrics["tail"]), "returned": bool(metrics["returned"]),
        "peak_active_frac": float(metrics["peak_active_frac"]),
        "inside": float(metrics["inside"]), "outside": float(metrics["outside"]),
        "active_mass": snn_active_mass(metrics), "R_class": classify_snn_r_class(metrics),
    }


def snn_spectrum_consistency(pairs: list) -> dict:
    """Stop-status when SNN spot checks systematically contradict the spectral prediction.

    ``pairs`` = list of (predicted spectral mode_class, observed SNN R_class). Triggers a stop if the
    SNN shows ONLY R4b tonic runaway, or if every axial/local/stable prediction comes back global/
    runaway (R4a/R4b) — the substrate-bottleneck signature the plan flags as a stop rule."""
    snn = [s for _, s in pairs]
    if snn and all(s == "R4b" for s in snn):
        return {"status": "stop", "reason": "snn_only_r4b_tonic_runaway"}
    relevant = [(p, s) for p, s in pairs if p in ("axial", "local", "stable")]
    if relevant and all(s in ("R4a", "R4b") for _, s in relevant):
        return {"status": "stop", "reason": "spectral_axial_local_but_snn_global_runaway"}
    return {"status": "ok", "reason": "spectrum_snn_consistent"}


def _snn_recruitment_axis(E_spk_bool: np.ndarray, pos_E: np.ndarray, dt: float,
                          t0_ms: float, t1_ms: float, theta: float = THETA_EE) -> float:
    """Elongation axis (along the E->E scaffold) of the E-spike spatial pattern in [t0, t1] ms.

    >0 = early recruitment elongated along the scaffold axis (axial spread); ~0 = isotropic/global."""
    i0, i1 = int(t0_ms / dt), int(t1_ms / dt)
    counts = np.asarray(E_spk_bool)[i0:i1].sum(axis=0).astype(float)   # (NE,) spikes per E neuron
    if counts.sum() < 10:
        return 0.0
    X, Y = pos_E[:, 0], pos_E[:, 1]
    xm, ym = np.average(X, weights=counts), np.average(Y, weights=counts)
    Sxx = np.average((X - xm) ** 2, weights=counts)
    Syy = np.average((Y - ym) ** 2, weights=counts)
    Sxy = np.average((X - xm) * (Y - ym), weights=counts)
    ang = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)
    ev = np.clip(np.linalg.eigvalsh(np.array([[Sxx, Sxy], [Sxy, Syy]])), 0.0, None)
    aniso = float((ev.max() - ev.min()) / (ev.max() + ev.min() + 1e-30))
    return float(aniso * np.cos(2 * (ang - theta)))


def run_snn_spotcheck(mu_core: float, q_global: float, w_ee_mult: float = 1.0, *, ratio: float = 0.9,
                      core_radius: float = 0.1, L: float = 0.5, density: float = 2000.0,
                      T: float = 450.0, seed: int = 1, core_std: float = 0.5,
                      kick_mult: float = 1.0) -> dict:
    """Run ONE real frozen-state SNN pilot at a phase-map point and classify it (slow ~3s).

    Also measures the early-recruitment spike axis (does the spiking spread AXIALLY, like the §5
    linear transient, or globally?) — the key test of whether the linear axial signal survives into
    spiking."""
    import os
    import sys
    eng = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snn_engine")
    if eng not in sys.path:
        sys.path.insert(0, eng)
    from params import Params, compute_nu_theta            # noqa: E402  (engine on sys.path)
    from model import build_network                        # noqa: E402
    from kick_probe import simulate_kick, compute_metrics  # noqa: E402
    from src.sef_hfo_heterogeneity import sample_core_field

    mp = snn_param_mapping(mu_core, q_global, w_ee_mult)
    p = Params(g=mp["g"], L=L, density=density, T=T, dt=0.1, nu_ext_ratio=ratio, seed=seed)
    p.w_EE = mp["w_EE"]
    net = build_network(p, verbose=False)
    pos = np.asarray(net["pos"])
    vth = None
    if mu_core > 0:
        is_E = np.asarray(net["labels"]) == 0
        cf = sample_core_field(pos, is_E, (L / 2.0, L / 2.0), core_radius, net["rng"],
                               core_mean=_SNN_BASE_V_TH - mu_core, core_std=core_std)
        vth = cf["vth"]
    nut = compute_nu_theta(p)
    nut = float(nut[0]) if np.ndim(nut) else float(nut)
    res = simulate_kick(p, net, KICK_BOOST=kick_mult * nut, verbose=False, V_th_per_neuron=vth)
    rec = snn_spotcheck_record(mu_core, q_global, w_ee_mult, compute_metrics(res, p.dt))
    # early-recruitment axis over the first 30 ms of the event window (150-180 ms)
    rec["recruitment_axis"] = _snn_recruitment_axis(res["E_spk_bool"], pos[:net["NE"]], p.dt, 150.0, 180.0)
    rec["seed"] = int(seed)
    return rec


def run_snn_spotcheck_grid(points: list, *, seeds=(1, 2, 3), kick_mult: float = 1.0,
                           spectral_mode_classes: dict = None) -> dict:
    """Run the SNN spot-check over a grid of (mu_core, q_global) points x seeds; aggregate R-class +
    recruitment axis per point, and form a spectrum<->SNN consistency stop/ok verdict.

    ``points`` = list of (mu_core, q_global[, w_ee_mult]) tuples. ``spectral_mode_classes`` maps a
    point key "mu,q" to the spectral leading mode_class for the consistency check."""
    runs = []
    per_point = {}
    for pt in points:
        mu, q = pt[0], pt[1]
        w = pt[2] if len(pt) > 2 else 1.0
        key = f"{mu},{q}"
        recs = [run_snn_spotcheck(mu, q, w, seed=s, kick_mult=kick_mult) for s in seeds]
        runs.extend(recs)
        rclasses = [r["R_class"] for r in recs]
        axes = [r["recruitment_axis"] for r in recs]
        # modal R-class; self-limited-axial requires a returning event AND an axial recruitment
        modal = max(set(rclasses), key=rclasses.count)
        per_point[key] = {
            "mu_core": mu, "q_global": q, "R_classes": rclasses, "modal_R_class": modal,
            "mean_recruitment_axis": float(np.mean(axes)),
            "self_limited_axial": bool(any(rc in ("R2", "R3") for rc in rclasses)
                                       and np.mean(axes) > 0.15),
            "self_limited_global": bool(any(rc == "R4a" for rc in rclasses)),
            "tonic_runaway": bool(all(rc == "R4b" for rc in rclasses)),
        }
    all_classes = [r["R_class"] for r in runs]
    pairs = []
    if spectral_mode_classes:
        for key, pp in per_point.items():
            pairs.append((spectral_mode_classes.get(key, "unknown"), pp["modal_R_class"]))
    consistency = snn_spectrum_consistency(pairs) if pairs else {"status": "ok", "reason": "no_pairs"}
    n_self_limited_axial = sum(1 for pp in per_point.values() if pp["self_limited_axial"])
    n_self_limited_any = sum(1 for pp in per_point.values()
                             if pp["self_limited_axial"] or pp["self_limited_global"])
    return {
        "per_point": per_point, "n_points": len(per_point),
        "all_R4b": all(rc == "R4b" for rc in all_classes),
        "n_self_limited_axial": n_self_limited_axial, "n_self_limited_any": n_self_limited_any,
        "R_class_counts": {c: all_classes.count(c) for c in sorted(set(all_classes))},
        "consistency": consistency,
        # spontaneous-AXIAL mechanism needs self-limited AXIAL (R2/R3 + axial), not just global R4a
        "snn_grid_pass_axial": n_self_limited_axial > 0,
        "snn_self_limited_present": n_self_limited_any > 0,
    }


# ---------------------------------------------------------------------------
# TDD-13: mode / event projection into the M3B Round-1 virtual-SEEG readout
#
# Push a model field (an eigenmode's E-field or an SNN/rate event) through the SAME virtual-SEEG
# readout Round-1 used (build_record_from_events + masked ranks), so generated modes can be placed
# against the real interictal/ictal cohort with geometry nulls. The new bridge claim requires BOTH
# a spectral mode class AND a readout/null pass (else placement-only).
# ---------------------------------------------------------------------------
def default_cross_montage(*, n_per_shaft: int = 7, pitch_mm: float = 1.0, theta: float = THETA_EE):
    """Two virtual-SEEG shafts crossing at the origin (one along the E->E axis, one across)."""
    from src.sef_hfo_observation import build_shaft, merge_montages
    half = 0.5 * (n_per_shaft - 1) * pitch_mm
    da = (np.cos(theta), np.sin(theta))
    db = (np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2))
    return merge_montages([
        build_shaft(theta, pitch_mm, n_per_shaft, origin=(-half * da[0], -half * da[1]), name_prefix="A"),
        build_shaft(theta + np.pi / 2, pitch_mm, n_per_shaft,
                    origin=(-half * db[0], -half * db[1]), name_prefix="B"),
    ])


def _sample_field_at_contacts(field: np.ndarray, grid: Grid, montage) -> np.ndarray:
    """Sample |field| at each virtual contact (nearest grid cell)."""
    sp, off = grid.spacing, grid.n // 2
    vals = []
    for cx, cy in montage.contacts:
        ix = min(max(int(round(cx / sp + off)), 0), grid.n - 1)
        iy = min(max(int(round(cy / sp + off)), 0), grid.n - 1)
        vals.append(abs(field[ix, iy]))
    return np.asarray(vals, float)


def project_mode_to_record(phi_E_field: np.ndarray, grid: Grid, *, montage=None,
                           model_id: str = "model", template_id: str = "t0",
                           spacing_mm: float = 4.0, participation_frac: float = 0.1,
                           return_intermediates: bool = False):
    """Project a mode/event E-field through the virtual-SEEG readout into a model record.

    Higher field amplitude = earlier in the propagation order. Non-participating contacts (amplitude
    below ``participation_frac`` of max) carry NaN ranks via ``mask_phantom_ranks`` — the SAME masked
    convention the cohort readout uses (no phantom integer ranks)."""
    from scripts.run_contact_plane_readout import build_record_from_events
    from src.lagpat_rank_audit import mask_phantom_ranks
    if montage is None:
        montage = default_cross_montage(pitch_mm=grid.L / 10.0)
    amp = _sample_field_at_contacts(np.asarray(phi_E_field), grid, montage)
    n_ch = amp.size
    thr = participation_frac * amp.max() if amp.max() > 0 else 0.0
    bools = (amp > thr).reshape(-1, 1)
    ranks = np.argsort(np.argsort(-amp)).astype(float).reshape(-1, 1)   # amplitude order (has phantoms)
    masked = mask_phantom_ranks(ranks, bools, normalize=True)           # non-participants -> NaN
    lag_raw = np.where(bools, ranks, np.nan)
    coords = np.column_stack([montage.contacts, np.zeros(n_ch)])
    rec = build_record_from_events(
        dataset="model", subject=model_id, template_id=template_id, names=list(montage.names),
        ranks=masked, bools=bools, lag_raw=lag_raw, coords=coords, mapped=np.ones(n_ch, bool),
        soz_core=set(), montage="single", lag_time_unit="ms", spacing_mm=spacing_mm)
    if return_intermediates:
        return rec, {"masked_ranks": masked, "bools": bools, "amp": amp}
    return rec


GEOMETRY_NULL_STATUSES: tuple[str, ...] = ("not_run", "failed", "passed")


def readout_bridge_verdict(geometry_null_status: str) -> str:
    """Map the geometry-null status to a readout verdict (fail-closed; distinguishes 'not run' from
    'ran and failed'):
      passed  -> 'bridge'          (placement beat the geometry null = model->patient bridge leg)
      failed  -> 'placement_only'  (placement computed but did NOT beat the null, as in Round-1 ictal)
      not_run -> 'projection_only' (only the schema/projection connected; no cohort placement run)."""
    return {"passed": "bridge", "failed": "placement_only", "not_run": "projection_only"}.get(
        geometry_null_status, "projection_only")


# ---------------------------------------------------------------------------
# TDD-15: verdict / claim audit
# ---------------------------------------------------------------------------
ALLOWED_VERDICTS: tuple[str, ...] = (
    "SPM-PASS full bridge", "SPM-PASS spontaneous mechanism", "SPM-PASS frozen map",
    "SPM-BOUNDED negative", "SPM-MODEL mismatch", "SPM-UNRESOLVED",
)


def full_bridge_gate(*, phase_map_coherent: bool, snn_predicts_spotchecks: bool,
                     m3a_trajectory_valid: bool, readout_null_pass: bool) -> str:
    """A FULL bridge requires ALL of: a coherent phase map, SNN spot checks matching the spectrum, a
    valid M3A slow trajectory, AND a readout/geometry-null pass. Anything less is a lower tier."""
    if phase_map_coherent and snn_predicts_spotchecks and m3a_trajectory_valid and readout_null_pass:
        return "SPM-PASS full bridge"
    if phase_map_coherent and snn_predicts_spotchecks:
        return "SPM-PASS spontaneous mechanism"
    if phase_map_coherent:
        return "SPM-PASS frozen map"
    return "SPM-UNRESOLVED"


def m3b_verdict(*, phase_map_resolved: bool, model_matches_dynamics: bool, controls_pass: bool,
                non_normal_axial_pass: bool, snn_grid_pass: bool, m3a_overlay_pass: bool,
                readout_null_pass: bool) -> str:
    """The M3B-R2 verdict from EXPLICIT, FAIL-CLOSED gates (plan §TDD-15 verdict categories).

    Frozen-map PASS requires BOTH specific controls AND the §5 non-normal axial substrate — a control
    failure or a missing axial substrate drops to SPM-BOUNDED negative (no fail-open). The bridge
    tiers each need their own gate: spontaneous mechanism needs a passing SNN grid; full bridge also
    needs a valid M3A overlay AND a readout/geometry-null pass. A single missing gate caps the tier.
    """
    if not phase_map_resolved:
        return "SPM-UNRESOLVED"
    if not model_matches_dynamics:
        return "SPM-MODEL mismatch"
    if not (controls_pass and non_normal_axial_pass):
        return "SPM-BOUNDED negative"
    return full_bridge_gate(phase_map_coherent=True, snn_predicts_spotchecks=snn_grid_pass,
                            m3a_trajectory_valid=m3a_overlay_pass, readout_null_pass=readout_null_pass)


# ---------------------------------------------------------------------------
# Path (a): non-normal transient axial readout (§5 PRIMARY metric)
#
# The leading eigenmode is global (the bounded-negative result), but the operator is non-normal:
# a core perturbation b_core can be TRANSIENTLY amplified along the E->E axis before decaying, even
# when max Re(lambda) < 0. That transient IS the interictal self-limited axial propagation the plan
# §5 names as a primary result — so we read it directly from exp(J*T) b_core, not the leading mode.
# ---------------------------------------------------------------------------
@dataclass
class TransientResponse:
    window_ms: float
    gain: float                 # ||exp(J*T) b_core|| / ||b_core||
    axis_score: float           # elongation of the transient E-field along the E->E axis
    core_overlap: float         # E-power still inside the core
    e_field: np.ndarray         # (n,n) |rE| of the transient response


def core_transient_response(J: np.ndarray, grid: Grid, core: CoreMask, T: float, *,
                            theta: float = THETA_EE, b: np.ndarray = None) -> TransientResponse:
    """Propagate a core perturbation b_core through exp(J*T) and read its gain / axis / localization."""
    from scipy.sparse.linalg import expm_multiply
    if b is None:
        b = core_perturbation_vector(grid, core)
    yT = expm_multiply(np.asarray(J) * float(T), b)
    eT = unpack_state_complex(yT, grid)["rE"]
    return TransientResponse(
        window_ms=float(T), gain=float(np.linalg.norm(yT) / np.linalg.norm(b)),
        axis_score=elongation_axis_score(eT, grid, theta),
        core_overlap=core_overlap(np.abs(eT), grid, core), e_field=np.abs(eT))


# HFO-burst-scale windows (ms) over which the non-normal transient rises and self-limits.
_TRANSIENT_WINDOWS_MS: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 90.0)


def non_normal_axial_readout(J: np.ndarray, grid: Grid, core: CoreMask, *, theta: float = THETA_EE,
                             windows: tuple = _TRANSIENT_WINDOWS_MS) -> dict:
    """§5 primary readout: is a core kick transiently AMPLIFIED, AXIAL, and SELF-LIMITED?

    Returns the gain/axis curves over windows plus the peak summary. ``transient_amplified`` = the
    gain rises above its short-window value and above 1; ``self_limited`` = it decays back below the
    peak by the longest window; ``axial`` flags whether the transient elongates along the E->E axis.
    """
    resp = [core_transient_response(J, grid, core, T, theta=theta) for T in windows]
    gains = [r.gain for r in resp]
    axes = [r.axis_score for r in resp]
    ipeak = int(np.argmax(gains))
    peak = resp[ipeak]
    max_axis = max(axes)
    return {
        "windows": list(windows), "gains": gains, "axes": axes,
        "core_overlaps": [r.core_overlap for r in resp],
        "peak_gain": peak.gain, "T_at_peak_ms": peak.window_ms, "axis_at_peak": peak.axis_score,
        "max_axis": float(max_axis), "axis_at_longest": axes[-1],
        "transient_amplified": bool(peak.gain > gains[0] and peak.gain > 1.0),
        "self_limited": bool(gains[-1] < peak.gain),
        "axial": bool(max_axis > 0.15),
    }


# ---------------------------------------------------------------------------
# Criticality Milestone 1, T3a-3: eigen-metrics on the frozen-Jacobian spectrum
#
# spectral_gap (TDD-7, above) is alpha_1 - alpha_2 by raw array order; it reads 0 when the leading
# mode is a complex-conjugate pair (the "second" eigenvalue is just the pair partner, not a
# competing mode). next_distinct_gap fixes this by skipping same-real-part entries.
# leading_subspace_indices then generalizes "the leading mode" to "the leading invariant SUBSPACE"
# (a conjugate pair, or several near-degenerate real modes), so pair_loading /
# left_mode_input_projection can read the spatial field / core-input coupling of the whole subspace
# instead of an arbitrary single member. Both reuse existing machinery -- mode_e_field (TDD-6 state
# unpacking) and the biorthonormalization rate_eigenpairs applies internally (TDD-6) -- generalized
# from one mode to a subspace of modes, rather than re-deriving either.
# ---------------------------------------------------------------------------
def next_distinct_gap(eigenvalues: np.ndarray, min_sep: float = 1e-3) -> float:
    """alpha_1 minus the first real part more than ``min_sep`` away from alpha_1.

    Unlike ``spectral_gap`` (a1 - a2 by raw array order), this SKIPS a leading complex-conjugate
    partner (identical real part) so the gap reflects genuine mode competition, not the trivial
    pair split. Returns +inf if every eigenvalue is within ``min_sep`` of the leader."""
    re = np.sort(np.real(np.asarray(eigenvalues)))[::-1]
    a1 = re[0]
    for r in re[1:]:
        if abs(a1 - r) > min_sep:
            return float(a1 - r)
    return float("inf")


def leading_subspace_indices(eigenvalues: np.ndarray, min_sep: float = 1e-3,
                             imag_tol: float = 1e-3) -> tuple[int, ...]:
    """Indices spanning the leading invariant subspace of ``eigenvalues``.

    If the leading eigenvalue is complex (``|Im| > imag_tol``), the leading mode is normally a
    conjugate PAIR: returns (leader, conjugate partner). But the partner may have been truncated
    out of ``eigenvalues`` (e.g. ``rate_eigenpairs(n_modes=...)`` split a pair) -- in that case the
    nearest-to-conjugate candidate is not a genuine partner, and pairing with it anyway would
    either self-pair (inflating downstream loadings by sqrt(2)) or fake-pair with an unrelated
    mode. So the candidate is only accepted as the partner if it is a different index AND within
    ``imag_tol`` of the true conjugate; otherwise this returns the lone-mode tuple (i,). If the
    leading eigenvalue is real, returns every index within ``min_sep`` of the leading real part (a
    near-degenerate group of real modes)."""
    ev = np.asarray(eigenvalues)
    i = int(np.argmax(ev.real))
    if abs(ev[i].imag) > imag_tol:
        j = int(np.argmin(np.abs(ev - np.conj(ev[i]))))          # nearest candidate partner
        if j != i and abs(ev[j] - np.conj(ev[i])) <= imag_tol:   # genuine conjugate partner present
            return (i, j)
        return (i,)                                              # partner absent (truncated) -> lone complex mode
    return tuple(int(x) for x in np.where(np.abs(ev.real - ev[i].real) <= min_sep)[0])


def pair_loading(R: np.ndarray, idx: tuple[int, ...], grid: Grid) -> np.ndarray:
    """Subspace-level spatial E-field loading: sqrt(sum_{k in idx} |mode_e_field(R[:,k], grid)|^2).

    Combines a leading invariant subspace (e.g. from ``leading_subspace_indices``) into one
    non-negative (n,n) spatial map. Reuses ``mode_e_field`` for the rE-block unpacking rather than
    re-deriving the state layout."""
    acc = np.zeros((grid.n, grid.n))
    for k in idx:
        acc = acc + np.abs(mode_e_field(R[:, k], grid)) ** 2
    return np.sqrt(acc)


def left_mode_input_projection(L: np.ndarray, R: np.ndarray, idx: tuple[int, ...],
                               b_core: np.ndarray) -> float:
    """Subspace-level generalization of ``core_controllability``: how strongly a core perturbation
    excites the leading invariant subspace, sqrt(sum_{k in idx} |psi_k^H b_core|^2).

    Each mode is biorthonormalized against its own right eigenvector first (``psi_k = L[:,k] /
    conj(L[:,k]^H R[:,k])``, the same normalization ``rate_eigenpairs`` applies internally) so the
    projection does not depend on the arbitrary scale of raw left/right eigenvectors."""
    acc = 0.0
    for k in idx:
        c = np.vdot(L[:, k], R[:, k])
        psi = L[:, k] / np.conj(c) if abs(c) > 1e-300 else L[:, k] / (np.linalg.norm(L[:, k]) + 1e-300)
        acc += abs(np.vdot(psi, b_core)) ** 2
    return float(np.sqrt(acc))

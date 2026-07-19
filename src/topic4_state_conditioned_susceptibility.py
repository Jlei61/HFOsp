"""Topic 4 state-conditioned spatial susceptibility — pure mapping / probe / operator math.

Design: docs/superpowers/specs/2026-07-19-topic4-state-conditioned-spatial-susceptibility-design.md.
IMPORT-SAFE and SIDE-EFFECT-FREE: no simulations, no file writes (those live in the runner).

Scientific object (design §3.5): for a frozen slow state s (a coarse z_bar(x) field on the fixed
E1146 scaffold) we build the M3B rate-field operating point, form the finite heterogeneous Jacobian
J_s, and read the finite-time spatial susceptibility R_s(T) = C exp(J_s T) B via a phase-paired
Gabor/Fourier probe dictionary. Three DISTINCT objects, never conflated (design §11):
  - true finite-system eigenmode of J_s (rate_eigenpairs / leading_subspace_indices) — leading one
    is expected GLOBAL (accepted M3B result);
  - Fourier/Gabor probe (an INPUT, not an eigenmode);
  - the non-normal finite-time RESPONSE C exp(J_s T) b (the axial signal lives here).

Reuse (do not reinvent) from src.topic4_m3b_spectral_phase: Grid, CoreMask, ExcitabilityField,
InhibitionField, build_kernels, build_excitability_field, solve_operating_point, build_jacobian_dense,
rate_eigenpairs, leading_subspace_indices, pair_loading, next_distinct_gap, globality, core_overlap,
elongation_axis_score, off_axis_score, pack_state, STATE_FIELDS.
"""
from __future__ import annotations

import numpy as np

from src.topic4_m3b_spectral_phase import (
    Grid, CoreMask, build_kernels, build_excitability_field, solve_operating_point,
    build_jacobian_dense, rate_eigenpairs, leading_subspace_indices, pair_loading,
    next_distinct_gap, globality, core_overlap, elongation_axis_score, off_axis_score,
    mode_e_field, pack_state, STATE_FIELDS, InhibitionField,
)

SCHEMA_VERSION = "sccs-atlas-1.0"


# ======================================================================== geometry / binning (Gate C)
def affine_to_norm(xy, *, L_phys, L_norm, center_phys):
    """Physical E1146 (mm) -> centered normalized square. Same transform for every seed and state."""
    xy = np.asarray(xy, float)
    return (xy - np.asarray(center_phys, float)) * (float(L_norm) / float(L_phys))


def normalize_subject_coordinates(pos_phys, *, L_phys, L_norm, center_phys):
    """Map E1146 neuron positions into the normalized M3B square; return positions + transform info."""
    pos_norm = affine_to_norm(np.asarray(pos_phys, float), L_phys=L_phys, L_norm=L_norm,
                              center_phys=center_phys)
    info = dict(L_phys=float(L_phys), L_norm=float(L_norm), center_phys=[float(c) for c in center_phys],
                scale=float(L_norm) / float(L_phys))
    return pos_norm, info


def coarse_cell_index(pos_norm, grid: Grid):
    """(ii, jj) cell indices via grid.coords() axes (indexing='ij': axis0=X, axis1=Y). Convention-agnostic:
    each point snaps to the nearest grid-cell CENTER, so it cannot drift from elongation_axis_score."""
    X, Y = grid.coords()
    xs = X[:, 0]                                   # X centers along axis 0 (n,)
    ys = Y[0, :]                                   # Y centers along axis 1 (n,)
    pos = np.asarray(pos_norm, float)
    ii = np.abs(pos[:, 0][:, None] - xs[None, :]).argmin(axis=1)
    jj = np.abs(pos[:, 1][:, None] - ys[None, :]).argmin(axis=1)
    return ii, jj


def bin_neuron_state_to_grid(values, pos_norm, grid: Grid, *, fill="nearest"):
    """Mean-bin per-neuron `values` onto the coarse grid. Returns (field, occupancy, fill_mask).

    Empty bins are invalid for direct averaging (design §5): with fill="nearest" they are filled from
    the nearest OCCUPIED cell and flagged in fill_mask; fill="none" leaves them NaN. Uniform input ->
    uniform field exactly (Gate C: coarse field preserves uniform inputs)."""
    n = grid.n
    ii, jj = coarse_cell_index(pos_norm, grid)
    vals = np.asarray(values, float)
    flat = ii * n + jj
    sums = np.bincount(flat, weights=vals, minlength=n * n).reshape(n, n)
    counts = np.bincount(flat, minlength=n * n).reshape(n, n)
    occupancy = counts.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    fill_mask = (counts == 0)
    if fill == "nearest" and fill_mask.any():
        X, Y = grid.coords()
        occ = ~fill_mask
        occ_xy = np.column_stack([X[occ], Y[occ]])
        occ_v = field[occ]
        for (a, b) in zip(*np.where(fill_mask)):
            d2 = (occ_xy[:, 0] - X[a, b]) ** 2 + (occ_xy[:, 1] - Y[a, b]) ** 2
            field[a, b] = occ_v[int(np.argmin(d2))]
    return field, occupancy, fill_mask


def zbar_to_q(zbar_field, q_floor):
    """q = clip(z_bar, q_floor, 1): the per-cell I->E efficacy field (design §3.2)."""
    return np.clip(np.asarray(zbar_field, float), float(q_floor), 1.0)


# ======================================================================== controls (design §7)
def make_state_controls(zbar_field, grid: Grid, *, shuffle_seed):
    """Five slow-field variants that answer DIFFERENT questions and must not be collapsed (design §7)."""
    z = np.asarray(zbar_field, float)
    mean = float(np.nanmean(z))
    rng = np.random.default_rng(int(shuffle_seed))
    flat = z.ravel().copy()
    rng.shuffle(flat)
    return {
        "real": z.copy(),                                  # observed z_bar(x)
        "uniform_mean": np.full_like(z, mean),             # global disinhibition only (spatial mean)
        "rotated_90": np.rot90(z).copy(),                  # z pattern rotated 90 deg vs fixed scaffold
        "spatial_shuffle": flat.reshape(z.shape),          # z histogram preserved, spatial pattern destroyed
        "z_blocked": np.ones_like(z),                      # pre-depletion operator (q=1 everywhere)
    }


# ======================================================================== two-core scaffold geometry
def two_core_mask_at(grid: Grid, centers, radius, theta):
    """CoreMask of two disks placed EXACTLY at the transformed src/snk (not make_core_mask's symmetric
    placement) so the model core sits on the real E1146 core geometry."""
    X, Y = grid.coords()
    mask = np.zeros((grid.n, grid.n), dtype=bool)
    for (cx, cy) in centers:
        mask |= ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2
    return CoreMask("two", mask, tuple((float(a), float(b)) for a, b in centers), float(radius), float(theta))


def build_fixed_scaffold(grid: Grid, src_norm, snk_norm, *, ell_perp, ar, mu_core, core_radius, theta):
    """FIXED, state-independent scaffold: anisotropic kernels (axis=theta) + two-core excitability.
    Built ONCE and shared across every state/control (only the q field varies)."""
    kernels = build_kernels(grid, ar=float(ar), ell_perp=float(ell_perp), theta=float(theta))
    core = two_core_mask_at(grid, [src_norm, snk_norm], core_radius, theta)
    src_core = two_core_mask_at(grid, [src_norm], core_radius, theta)
    snk_core = two_core_mask_at(grid, [snk_norm], core_radius, theta)
    exc = build_excitability_field(grid, core, mu_core=float(mu_core))
    return dict(kernels=kernels, core=core, src_core=src_core, snk_core=snk_core, exc=exc, theta=float(theta))


# ======================================================================== probe dictionary (design §6.1)
def make_phase_paired_probe_dictionary(grid: Grid, *, p_max=4, sigma=1.0, center=(0.0, 0.0), gabor=True):
    """Phase-paired cos/sin probes on the periodic sheet. k = 2*pi*(p,q)/L (normalized M3B units).

    Half-plane (p,q) up to sign (each spatial frequency/orientation once) with cos AND sin phases;
    plus the single p=q=0 global/uniform probe (design §6.1: exclude (0,0) from directional peak,
    include separately as global). Gabor windows at `center` (source core primary; sink is a
    registered sensitivity)."""
    X, Y = grid.coords()
    cx, cy = center
    win = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * float(sigma) ** 2)) if gabor else np.ones_like(X)
    probes = [dict(p=0, q=0, phase="global", kx=0.0, ky=0.0, k_mag=0.0, orient_deg=None, field=win.copy())]
    for p in range(0, int(p_max) + 1):
        for q in range(-int(p_max), int(p_max) + 1):
            if p == 0 and q <= 0:                          # keep p=0,q>0 only: skip (0,0) + sign-duplicates
                continue
            kx, ky = 2 * np.pi * p / grid.L, 2 * np.pi * q / grid.L
            arg = kx * X + ky * Y
            k_mag = float(np.hypot(kx, ky))
            orient = float(np.degrees(np.arctan2(ky, kx)) % 180.0)
            probes.append(dict(p=p, q=q, phase="cos", kx=kx, ky=ky, k_mag=k_mag, orient_deg=orient,
                               field=win * np.cos(arg)))
            probes.append(dict(p=p, q=q, phase="sin", kx=kx, ky=ky, k_mag=k_mag, orient_deg=orient,
                               field=win * np.sin(arg)))
    return probes


def embed_probe_in_rate_state(field, grid: Grid):
    """Embed a spatial (n,n) probe into the rE row of the 6N state, unit-normalized (mirrors
    core_perturbation_vector's convention: perturb E-rate only)."""
    z = pack_state({f: (np.asarray(field, float) if f == "rE" else np.zeros((grid.n, grid.n)))
                    for f in STATE_FIELDS}, grid)
    nrm = np.linalg.norm(z)
    return z / nrm if nrm > 0 else z


def probe_matrix(probes, grid: Grid):
    """Assemble the 6N x M input matrix B (unit columns) from the probe dictionary."""
    return np.column_stack([embed_probe_in_rate_state(pr["field"], grid) for pr in probes])


def batched_finite_time_response(J, B, T):
    """C-agnostic exp(J*T) @ B for all probe columns at once (== one-at-a-time; Gate D6)."""
    from scipy.sparse.linalg import expm_multiply
    return expm_multiply(np.asarray(J, float) * float(T), np.asarray(B, float))


# ======================================================================== atlas summary (design §6.2)
def _paired_gains(gains, probes):
    """Phase-invariant per-(p,q) gain sqrt(g_cos^2 + g_sin^2); global probe kept separately."""
    by_pq, glob = {}, None
    for g, pr in zip(gains, probes):
        if pr["phase"] == "global":
            glob = float(g)
            continue
        by_pq.setdefault((pr["p"], pr["q"]), {})[pr["phase"]] = (float(g), pr)
    paired = {}
    for pq, d in by_pq.items():
        gc = d.get("cos", (0.0, None))[0]
        gs = d.get("sin", (0.0, None))[0]
        pr = (d.get("cos") or d.get("sin"))[1]
        paired[pq] = dict(gain=float(np.hypot(gc, gs)), k_mag=pr["k_mag"], orient_deg=pr["orient_deg"],
                          p=pr["p"], q=pr["q"])
    return paired, glob


def leading_probe_subspace_svd(response_rE):
    """SVD of the E-rate response matrix R (N x M) = C exp(JT) B (design §3.5). Returns the leading
    singular value (optimal gain), optimal PROBE COMBINATION (right vector), optimal OUTPUT FIELD
    (left vector). NEVER call either an eigenmode."""
    R = np.asarray(response_rE, float)
    U, s, Vh = np.linalg.svd(R, full_matrices=False)
    return dict(s0=float(s[0]) if s.size else 0.0,
                optimal_output_field=U[:, 0].copy() if U.size else np.zeros(R.shape[0]),
                optimal_probe_weights=Vh[0, :].copy() if Vh.size else np.zeros(R.shape[1]),
                singular_values=s[:6].tolist())


def optimal_finite_time_perturbation(J, grid: Grid, T, N):
    """Rigorous non-normal optimal finite-time perturbation: SVD of the E-rate->E-rate propagator block
    M(T) = C_E exp(J T) B_E (N x N), over the FULL E-rate input space (dictionary-INDEPENDENT, unlike
    the Gabor-probe SVD). For a non-normal operator V1/U1 describe short-time transport better than an
    eigenvector. Returns sigma1 (max finite-time E->E gain over all input fields), the optimal INPUT
    field V1 (n,n) and optimal OUTPUT field U1 (n,n) (signed; SVD sign is arbitrary but the +/- lobe
    structure is meaningful)."""
    from scipy.sparse.linalg import expm_multiply
    B_E = np.zeros((6 * grid.size, N))
    B_E[:N, :] = np.eye(N)                                       # embed a unit E-rate field into rE rows
    M = expm_multiply(np.asarray(J, float) * float(T), B_E)[:N, :]   # C_E exp(JT) B_E  (N x N)
    U, s, Vh = np.linalg.svd(M)
    return dict(sigma1=float(s[0]),
                sigma_ratio=float(s[0] / s[1]) if s.size > 1 and s[1] > 0 else float("inf"),
                v1_field=Vh[0, :].reshape(grid.n, grid.n), u1_field=U[:, 0].reshape(grid.n, grid.n),
                singular_values=s[:6].tolist())


def eigen_field(J, grid: Grid):
    """Leading invariant-subspace E-field |phi_E| (n,n) — the TRUE asymptotic eigenmode SHAPE, for
    plotting alongside (never conflated with) V1/U1 and the response. None if J is unresolved."""
    eig = rate_eigenpairs(J, grid, n_modes=4)
    if eig.status != "resolved" or eig.eigenvalues.size == 0:
        return None
    idx = leading_subspace_indices(eig.eigenvalues)
    return pair_loading(eig.right, idx, grid)                    # non-negative (n,n) subspace loading


# ---- fixed-kick time-response (review 2026-07-19: real propagation dynamics, not compressed scores) ----
def sigma1_vs_T(J, grid: Grid, T_list, N):
    """sigma1(T) = max finite-time E->E gain over ALL input fields, vs window T (panel A). sigma1(0)=1
    (identity). Same dictionary-independent propagator-block SVD as V1/U1 -> when it crosses 1, some
    input is net-amplified over that window."""
    return [1.0 if T <= 0 else optimal_finite_time_perturbation(J, grid, float(T), N)["sigma1"] for T in T_list]


def make_localized_kick(grid: Grid, center, sigma):
    """Unit-norm 6N perturbation = a Gaussian bump in the rE field at `center`. Built ONCE and reused
    across states (the SAME kick) so the response comparison isolates the state change (panels B/C)."""
    X, Y = grid.coords()
    g = np.exp(-((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (2.0 * float(sigma) ** 2))
    return embed_probe_in_rate_state(g, grid)


def fixed_kick_evolution(J, grid: Grid, b_fixed, t_list, N):
    """{t: (n,n) rE response} of the FIXED kick under exp(J t), C_E readout (signed). t=0 -> the kick."""
    from scipy.sparse.linalg import expm_multiply
    out = {}
    for t in t_list:
        y = b_fixed if t <= 0 else expm_multiply(np.asarray(J, float) * float(t), b_fixed)
        out[float(t)] = y[:N].reshape(grid.n, grid.n)
    return out


def axial_kymograph(evolution, grid: Grid, y_axis, band):
    """(xs, ts, kymo[n_t, n_x]): |rE| response profile along x at y≈y_axis (averaged over the band
    |y-y_axis|<=band), stacked over time -> propagation along the source->sink axis."""
    X, Y = grid.coords()
    ycol = Y[0, :]
    ymask = np.abs(ycol - y_axis) <= band
    if not ymask.any():
        ymask = np.abs(ycol - y_axis) <= (grid.L / grid.n)          # fall back to the nearest row
    ts = sorted(evolution)
    kymo = np.array([np.abs(evolution[t])[:, ymask].mean(axis=1) for t in ts])   # (n_t, n_x)
    return X[:, 0].copy(), np.array(ts, float), kymo


def summarize_probe_atlas(J, grid: Grid, probes, T_list, *, theta, N, core, T_primary=30.0):
    """Full phase-paired gain atlas over T (design §6.2): axial / perpendicular / global gains, peak
    (k, orientation, gain), persistence G(50)/G(30) & G(75)/G(30), probe-subspace SVD, and the
    T_primary output fields for the figure. Reads only the E-rate block (C readout)."""
    B = probe_matrix(probes, grid)
    per_T = {}
    axial_curve, perp_curve, glob_curve, peak_curve = {}, {}, {}, {}
    fields_primary = {}
    for T in T_list:
        Y = batched_finite_time_response(J, B, T)         # (6N, M)
        R = Y[:N, :]                                       # C: E-rate block
        gains = np.linalg.norm(R, axis=0)                  # ||b||=1 -> gain
        paired, glob = _paired_gains(gains, probes)
        axial = max((v["gain"] for k, v in paired.items() if v["q"] == 0), default=0.0)   # k along axis (theta=0)
        perp = max((v["gain"] for k, v in paired.items() if v["p"] == 0), default=0.0)     # k perpendicular
        peak_pq = max(paired, key=lambda k: paired[k]["gain"]) if paired else None
        peak = paired.get(peak_pq, dict(gain=0.0, k_mag=0.0, orient_deg=None))
        svd = leading_probe_subspace_svd(R)
        per_T[float(T)] = dict(axial_gain=float(axial), perp_gain=float(perp), global_gain=float(glob or 0.0),
                               peak_gain=float(peak["gain"]), peak_k=float(peak["k_mag"]),
                               peak_orient_deg=(None if peak["orient_deg"] is None else float(peak["orient_deg"])),
                               svd_s0=svd["s0"], svd_singular_values=svd["singular_values"],
                               axis_minus_perp=float(axial - perp))
        axial_curve[float(T)] = float(axial); perp_curve[float(T)] = float(perp)
        glob_curve[float(T)] = float(glob or 0.0); peak_curve[float(T)] = float(peak["gain"])
        if abs(T - T_primary) < 1e-9:
            # phase-paired response at the peak (p,q): sqrt(|R_cos|^2 + |R_sin|^2), CONSISTENT with the
            # row-2 paired gain (not a single-phase argmax column).
            peak_cols = [i for i, pr in enumerate(probes) if (pr["p"], pr["q"]) == peak_pq]
            peak_paired = np.sqrt(sum(np.abs(R[:, c]) ** 2 for c in peak_cols)) if peak_cols else np.zeros(N)
            # full (p,q) paired-gain map for the kx-ky susceptibility panel (design §9 row 2); symmetric in k
            pmax = max(pr["p"] for pr in probes)
            gmap = np.zeros((2 * pmax + 1, 2 * pmax + 1))
            for (p, q), v in paired.items():
                gmap[p + pmax, q + pmax] = v["gain"]
                gmap[-p + pmax, -q + pmax] = v["gain"]
            gmap[pmax, pmax] = float(glob or 0.0)
            fields_primary = dict(
                peak_paired_output_rE=peak_paired.reshape(grid.n, grid.n),
                peak_pq=np.array([peak_pq[0], peak_pq[1]] if peak_pq else [0, 0]),
                svd_optimal_output_field=np.abs(svd["optimal_output_field"]).reshape(grid.n, grid.n),
                svd_output_axis=float(elongation_axis_score(np.abs(svd["optimal_output_field"]).reshape(grid.n, grid.n), grid, theta)),
                svd_output_globality=float(globality(svd["optimal_output_field"], grid)),
                gain_kxky=gmap, gain_kxky_pmax=int(pmax),
            )

    def _persist(curve):
        base = curve.get(30.0)
        out = {}
        if base and base > 0:
            for T in (50.0, 75.0):
                if T in curve:
                    out[f"G{int(T)}_over_G30"] = float(curve[T] / base)
        return out

    return dict(per_T=per_T,
                persistence=dict(axial=_persist(axial_curve), perp=_persist(perp_curve),
                                 global_=_persist(glob_curve), peak=_persist(peak_curve)),
                curves=dict(axial=axial_curve, perp=perp_curve, global_=glob_curve, peak=peak_curve),
                fields_primary=fields_primary)


# ======================================================================== true eigenmode summary (design §3.3)
def eigen_summary(J, grid: Grid, core: CoreMask, theta):
    """Leading-eigen-subspace shape of J (the TRUE finite-system modes). Reports globality / core /
    axis of the leading invariant subspace so the report can state whether the leading eigenmode is
    GLOBAL while the transient response is axial (design's accepted M3B layering) — kept DISTINCT."""
    eig = rate_eigenpairs(J, grid, n_modes=4)
    if eig.status != "resolved" or eig.eigenvalues.size == 0:
        return dict(status=eig.status, leading_growth=None, leading_globality=None,
                    leading_core_overlap=None, leading_axis_score=None, next_distinct_gap=None,
                    residual_right=float(eig.residual_right), n_modes=0)
    idx = leading_subspace_indices(eig.eigenvalues)
    field = pair_loading(eig.right, idx, grid)
    return dict(status="resolved",
                leading_growth=float(np.max(eig.eigenvalues.real)),
                leading_globality=float(globality(field, grid)),
                leading_core_overlap=float(core_overlap(field, grid, core)),
                leading_axis_score=float(elongation_axis_score(field, grid, theta)),
                next_distinct_gap=float(next_distinct_gap(eig.eigenvalues)),
                residual_right=float(eig.residual_right), n_modes=len(idx),
                eig_residual_ok=bool(eig.residual_right < 1e-6))


# ======================================================================== per-state operator + top-level
def state_operator(zbar_field, grid: Grid, scaffold, *, w_ee_mult, ratio, q_floor,
                   scale_II=False, op_dt=0.5, op_t_max=2500.0, op_tol=1e-9, init=None):
    """Build the operating point + finite Jacobian for one slow state (the q=clip(z_bar) field on the
    FIXED scaffold). Returns (op, J, q_field). op.status in {resolved, unresolved, saturated}.
    `init` = {"rE","rI"} warm-start (for continuation: track the SAME branch across a slow-state path)."""
    q = zbar_to_q(zbar_field, q_floor)
    inh = InhibitionField(q=q, q_global=float(np.nanmean(q)), q_core=1.0, scale_II=bool(scale_II),
                          core=scaffold["core"])
    op = solve_operating_point(grid, scaffold["kernels"], scaffold["exc"], inh, ratio=float(ratio),
                               w_ee_mult=float(w_ee_mult), dt=float(op_dt), t_max=float(op_t_max),
                               tol=float(op_tol), init=init)
    J = build_jacobian_dense(grid, scaffold["kernels"], op) if op.status == "resolved" else None
    return op, J, q


def leading_eigenvalue(J, grid: Grid):
    """(Re, |Im|, is_complex, freq_hz) of the LEADING rate-branch eigenvalue of J. Re>=0 => linear
    instability; is_complex with Re crossing 0 => Hopf (oscillation birth) at freq_hz = |Im|/(2*pi)*1000
    (eigenvalues are in 1/ms). None if J is unresolved."""
    eig = rate_eigenpairs(J, grid, n_modes=4)
    if eig.status != "resolved" or eig.eigenvalues.size == 0:
        return None
    lam = eig.eigenvalues[0]                                     # max Re (sorted desc)
    return dict(re=float(lam.real), im=float(abs(lam.imag)), is_complex=bool(abs(lam.imag) > 1e-3),
                freq_hz=float(abs(lam.imag) / (2.0 * np.pi) * 1000.0))


def summarize_state_susceptibility(zbar_field, grid: Grid, scaffold, probes, T_list, *,
                                   w_ee_mult, ratio, q_floor, T_primary=30.0, scale_II=False,
                                   op_dt=0.5, op_t_max=2500.0, op_tol=1e-9):
    """Top-level per-(state, control) summary: operating point -> J -> eigen summary + probe atlas.

    FAIL-CLOSED (Gate D7): an unresolved/saturated operating point is reported as such and NEVER
    assigned a stable/axial susceptibility. The atlas is computed only for a resolved operating
    point; otherwise `atlas` is None and `op_status` carries the reason."""
    theta, core = scaffold["theta"], scaffold["core"]
    N = grid.size
    op, J, q = state_operator(zbar_field, grid, scaffold, w_ee_mult=w_ee_mult, ratio=ratio,
                              q_floor=q_floor, scale_II=scale_II, op_dt=op_dt, op_t_max=op_t_max,
                              op_tol=op_tol)
    zb = np.asarray(zbar_field, float)
    corridor = core.mask
    state_field = dict(
        z_mean=float(np.nanmean(zb)), z_min=float(np.nanmin(zb)),
        q_mean=float(np.nanmean(q)), q_min=float(np.nanmin(q)),
        axis_corridor_mean_q=float(np.nanmean(q[corridor])) if corridor.any() else None,
        off_axis_mean_q=float(np.nanmean(q[~corridor])) if (~corridor).any() else None,
    )
    state_field["axis_minus_off_axis_q"] = (
        None if state_field["axis_corridor_mean_q"] is None or state_field["off_axis_mean_q"] is None
        else state_field["axis_corridor_mean_q"] - state_field["off_axis_mean_q"])

    out = dict(schema_version=SCHEMA_VERSION, op_status=op.status, op_source=op.source,
               op_converged=bool(op.converged), op_residual=float(op.residual), op_saturated=bool(op.saturated),
               op_rE_mean=float(np.mean(op.rE)), op_rE_max=float(np.max(op.rE)),
               op_rI_mean=float(np.mean(op.rI)), state_field=state_field)
    if op.status != "resolved" or J is None:
        out.update(eigen=dict(status=op.status), atlas=None,
                   note=f"operating point {op.status}: susceptibility NOT computed (fail-closed, design §8 Gate D).")
        return out, None
    out["eigen"] = eigen_summary(J, grid, core, theta)
    atlas = summarize_probe_atlas(J, grid, probes, T_list, theta=theta, N=N, core=core, T_primary=T_primary)
    out["atlas"] = {k: v for k, v in atlas.items() if k != "fields_primary"}
    # true asymptotic eigenmode field + rigorous non-normal optimal finite-time input/output (V1/U1) at
    # T_primary — three DISTINCT objects (eigenmode / optimal input / optimal output), never conflated.
    ef = eigen_field(J, grid)
    opt = optimal_finite_time_perturbation(J, grid, T_primary, N)
    out["optimal"] = dict(
        sigma1=opt["sigma1"], sigma_ratio=opt["sigma_ratio"],
        v1_wavevector_axis=float(elongation_axis_score(np.abs(opt["v1_field"]), grid, theta)),
        u1_output_axis=float(elongation_axis_score(np.abs(opt["u1_field"]), grid, theta)),   # OUTPUT propagation-direction evidence
        u1_output_globality=float(globality(opt["u1_field"], grid)))
    arrays = dict(q_field=q, zbar_field=zb, **atlas.get("fields_primary", {}),
                  eigen_field=(ef if ef is not None else np.full((grid.n, grid.n), np.nan)),
                  v1_optimal_input=opt["v1_field"], u1_optimal_output=opt["u1_field"])
    return out, arrays

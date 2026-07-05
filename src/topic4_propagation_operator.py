"""Topic 4 M3-final — local-W propagation operator (three objects, never one
row-normalized matrix; spec §5.2 / C6).

朴素版：在当前局部各向异性放电网络里，对每个空间格点(bin)轻轻踢一下、看活动往哪
传、传多远，用这个 small-kick 响应测出一个可观测的局部传播算子 ``W``。**关键**：W 不是
一个矩阵，而是同一组响应派生出的三个对象，各自只能回答一个问题——混用一个 row-normalized
矩阵同时算 ``Λ₀`` 和 ``h`` 会让 ``ρ≈1`` 恒成立、``h`` 抹平到 1（spec §5.2 / C6a 隐藏陷阱）：

  - ``W_resp``  : 未归一的 baseline-减响应（``clip(kick−sham, 0, inf)``，去对角）→ 易感度图 ``h``
  - ``W_step``  : 按源活动质量列归一 → ``Λ₀ = ρ(W_step)``（分支比，可大于/小于 1）
  - ``W_shape`` : 去对角后行归一 → 主轴 + 顺序预测（**不**算 Λ₀）

测量 harness ``build_w_resp`` 循环 bin × seed 调 ``src/snn_engine/kick_probe.py::simulate_kick``
（kick vs sham），只读引擎、不改引擎。``spectral_radius`` 复用
``src/topic4_hub_criticality.py::branching_ratio``，不自己写特征值求解器。
"""
from __future__ import annotations
import numpy as np

from src.topic4_hub_criticality import branching_ratio


# ---------------------------------------------------------------------------
# spatial binning
# ---------------------------------------------------------------------------
def spatial_bins(posE, n_bins_per_axis):
    """Coarse-bin E-cell 2D positions into an ``n_bins_per_axis**2`` regular grid.

    Returns dict(bin_of_cell[NE], bin_centers[n_bins, 2]). ``bin_centers`` are the
    geometric centers of the grid cells (used as ``kick_center`` for each bin q),
    laid out in row-major (x-fast) order so ``bin_of_cell`` indexes into them.
    """
    posE = np.asarray(posE, float)
    nb = int(n_bins_per_axis)
    lo = posE.min(axis=0)
    hi = posE.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    # fractional position in [0, 1), then to integer bin index clipped to [0, nb-1]
    frac = (posE - lo) / span
    ix = np.clip((frac[:, 0] * nb).astype(int), 0, nb - 1)
    iy = np.clip((frac[:, 1] * nb).astype(int), 0, nb - 1)
    bin_of_cell = iy * nb + ix    # row-major over (iy, ix); ix fast

    # grid-cell centers (independent of where cells happen to sit)
    edge = span / nb
    cx = lo[0] + (np.arange(nb) + 0.5) * edge[0]
    cy = lo[1] + (np.arange(nb) + 0.5) * edge[1]
    bin_centers = np.array([[cx[j], cy[i]] for i in range(nb) for j in range(nb)],
                           dtype=float)
    return dict(bin_of_cell=bin_of_cell.astype(int), bin_centers=bin_centers)


# ---------------------------------------------------------------------------
# W_resp measurement harness (reads the engine; no engine change)
# ---------------------------------------------------------------------------
def build_w_resp(p, net, NE, NI, bins, V_th0, *, kick_boost, r_kick, t_kick,
                 win_ms, seeds):
    """Measure the un-normalized small-kick response operator ``W_resp``.

    For each source bin ``q``: kick its center (``simulate_kick(kick_center=
    bin_centers[q], KICK_BOOST=kick_boost, r_kick=r_kick, t_kick=t_kick,
    V_th_per_neuron=V_th0)``) AND a sham run (``KICK_BOOST=0``), each averaged
    over ``seeds`` independent seeds. ``A_p`` = total E spikes in target bin ``p``
    within ``[t_kick+win_ms[0], t_kick+win_ms[1])`` (one propagation generation, C6).

    ``W_resp[p, q] = clip(mean_seed(A_p|kick) - mean_seed(A_p|sham), 0, inf)`` with
    the **diagonal set to 0** and **NO row-normalization**.
    ``src_mass[q]`` = ``W_resp[q, q]`` captured BEFORE zeroing the diagonal (source
    q's own kick-evoked activity; the W_step normalizer).
    ``injected_mass[q]`` = expected spike mass the kick directly injects into q
    (``kick_boost * dur_kick * n_kicked_cells`` — the direct-drive amount; the
    W_step sensitivity denominator).

    Returns dict(W_resp, src_mass, injected_mass).
    """
    import sys, os
    _eng = os.path.join(os.path.dirname(__file__), "snn_engine")
    if _eng not in sys.path:
        sys.path.insert(0, _eng)
    from kick_probe import simulate_kick, DUR_KICK

    bin_of_cell = np.asarray(bins["bin_of_cell"], int)
    bin_centers = np.asarray(bins["bin_centers"], float)
    n_bins = bin_centers.shape[0]
    V_th0 = np.asarray(V_th0, float)

    dt = p.dt
    lo_step = int(round((t_kick + win_ms[0]) / dt))
    hi_step = int(round((t_kick + win_ms[1]) / dt))
    seeds = list(seeds) if np.ndim(seeds) else list(range(int(seeds)))

    def _bin_spike_counts(res):
        """Total E spikes per bin in the response window -> length-n_bins vector."""
        # E_spk_bool: (nsteps, NE) bool; sum spikes per E cell in window, bincount by bin
        per_cell = res["E_spk_bool"][lo_step:hi_step].sum(axis=0).astype(float)
        return np.bincount(bin_of_cell, weights=per_cell, minlength=n_bins)

    W_resp = np.zeros((n_bins, n_bins), dtype=float)
    injected_mass = np.zeros(n_bins, dtype=float)

    for q in range(n_bins):
        center = bin_centers[q]
        A_kick = np.zeros(n_bins)
        A_sham = np.zeros(n_bins)
        n_kicked = 0
        for sd in seeds:
            net["rng"] = np.random.default_rng(int(sd))
            res_on = simulate_kick(p, net, KICK_BOOST=kick_boost, kick_center=center,
                                   r_kick=r_kick, t_kick=t_kick, V_th_per_neuron=V_th0)
            net["rng"] = np.random.default_rng(int(sd))
            res_off = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=center,
                                    r_kick=r_kick, t_kick=t_kick, V_th_per_neuron=V_th0)
            A_kick += _bin_spike_counts(res_on)
            A_sham += _bin_spike_counts(res_off)
            n_kicked = res_on["n_inside"]   # E cells inside the kicked disk (seed-invariant)
        nseed = len(seeds)
        W_resp[:, q] = np.clip(A_kick / nseed - A_sham / nseed, 0.0, np.inf)
        # expected spike mass the kick directly injects into q's cells over its duration
        injected_mass[q] = kick_boost * DUR_KICK * n_kicked

    src_mass = np.diag(W_resp).copy()    # source q's own kick-evoked activity (pre-zeroing)
    np.fill_diagonal(W_resp, 0.0)
    return dict(W_resp=W_resp, src_mass=src_mass, injected_mass=injected_mass)


# ---------------------------------------------------------------------------
# three derived operators
# ---------------------------------------------------------------------------
def make_step_operator(W_resp, src_mass, *, eps=1e-6, src_mass_floor=None,
                       injected_mass=None):
    """``W_step[p, q] = W_resp[p, q] / (normalizer[q] + eps)``, source-activity
    column normalization (NOT row/col-sum == 1).

    ``normalizer = injected_mass if injected_mass is not None else src_mass``.
    Anti-blowup / valid-source: if ``src_mass_floor`` is not None, every source bin
    ``q`` with ``src_mass[q] < src_mass_floor`` has its **entire column zeroed**
    (excluded — a tiny denominator must not inflate a fake gain).

    **The exclusion is ALWAYS keyed on ``src_mass`` (the measured evoked source
    response), even when ``injected_mass`` is the normalizer.** A source bin that
    barely responded to its own kick is an unreliable propagation source regardless
    of which denominator we divide by — this is a data-quality filter on the source,
    not a numerical-blowup guard. Keeping the SAME ``valid_src`` mask for both口径
    makes the main-vs-sensitivity comparison apples-to-apples (only the normalizer
    changes, not which bins are included). The pre-registration records this "same
    valid_src mask" rule; this function does NOT relax the exclusion under
    ``injected_mass``.
    """
    W_resp = np.asarray(W_resp, float)
    normalizer = src_mass if injected_mass is None else injected_mass
    normalizer = np.asarray(normalizer, float)
    W_step = W_resp / (normalizer[None, :] + eps)
    if src_mass_floor is not None:
        low = np.asarray(src_mass, float) < src_mass_floor
        W_step[:, low] = 0.0
    return W_step


def make_shape_operator(W_resp):
    """``W_shape``: zero the diagonal, then row-normalize (each row / its row-sum;
    all-zero rows stay zero). Only for principal axis + ordering, NOT for Λ₀."""
    W = np.asarray(W_resp, float).copy()
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1)
    nz = rs > 0
    W[nz] = W[nz] / rs[nz, None]
    return W


def h_field(W_resp, scheme):
    """Susceptibility / recruitability map from the **un-normalized** ``W_resp``:
    ``post`` = norm(row_sum) (target recruitability; primary), ``out`` =
    norm(col_sum) (source broadcast; sensitivity), ``hybrid`` = 1/2*(post+out).
    ``norm`` = divide by the median (of the positive entries)."""
    W = np.asarray(W_resp, float)

    def _norm(v):
        pos = v[v > 0]
        m = np.median(pos) if pos.size else 1.0
        return v / max(float(m), 1e-9)

    if scheme == "post":
        return _norm(W.sum(axis=1))
    if scheme == "out":
        return _norm(W.sum(axis=0))
    if scheme == "hybrid":
        return 0.5 * (_norm(W.sum(axis=1)) + _norm(W.sum(axis=0)))
    raise ValueError(f"unknown h scheme {scheme!r} (post|out|hybrid)")


def spectral_radius(W_step):
    """``Λ₀ = ρ(W_step)`` — reuse ``branching_ratio`` (no re-implemented eigensolver).

    ``branching_ratio`` calls ``sp.csr_matrix(M)`` on its first line, so a dense
    ndarray is accepted; ``idx=None`` uses the whole matrix. It returns the eigen
    branching ratio (real part of the largest-magnitude eigenvalue) = the asymptotic
    per-generation recruitment multiplier (spec C1/C6b: ρ≈1 = critical). A real
    SNN-measured ``W_step`` is always recurrent, so ρ > 0; the gain-scaling unit test
    therefore uses a recurrent (non-nilpotent) matrix — a strictly feed-forward
    operator is nilpotent (ρ≡0) and is not a meaningful criticality input.
    """
    return branching_ratio(np.asarray(W_step, float), idx=None)


# ---------------------------------------------------------------------------
# geometry / ordering read-outs (from W_shape)
# ---------------------------------------------------------------------------
def principal_axis(W_shape, bin_centers):
    """Unit vector of the response-weighted displacement principal direction.

    Each directed edge q->p (``W_shape[p, q] > 0``) contributes its displacement
    ``bin_centers[p] - bin_centers[q]`` weighted by ``W_shape[p, q]``. The principal
    axis is the dominant eigenvector of the weighted displacement scatter (an axis,
    so sign is arbitrary; returned as a unit vector)."""
    W = np.asarray(W_shape, float)
    centers = np.asarray(bin_centers, float)
    n = W.shape[0]
    C = np.zeros((2, 2))
    for p in range(n):
        for q in range(n):
            w = W[p, q]
            if w > 0:
                d = centers[p] - centers[q]
                C += w * np.outer(d, d)
    vals, vecs = np.linalg.eigh(C)
    axis = vecs[:, int(np.argmax(vals))]
    nrm = np.linalg.norm(axis)
    return axis / nrm if nrm > 0 else axis


def _propagation_depth(W_shape, seed):
    """Number of propagation steps from ``seed`` to each bin along the directed
    reachability graph of ``W_shape``.

    **Direction convention (spec C6; matches ``W_resp[p,q]`` = target p <- source q
    and ``principal_axis``'s q->p edge reading):** from an active SOURCE bin ``a``,
    the targets it recruits next are ``b`` with ``W_shape[b, a] > 0`` — source is the
    COLUMN index, target is the ROW index. The seed (first observed bin) propagates
    outward along its non-zero ``W_shape`` COLUMN, NOT its row. (Walking the row
    would follow edges backward and silently bias / under-estimate ``rho_W`` in the
    Task 6 gate.) Unreached bins get a depth one past the max so they sort last.
    BFS over that directed graph."""
    W = np.asarray(W_shape, float)
    n = W.shape[0]
    depth = np.full(n, np.inf)
    depth[seed] = 0.0
    frontier = [seed]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for a in frontier:
            for b in range(n):
                if W[b, a] > 0 and np.isinf(depth[b]):   # source a -> target b
                    depth[b] = d
                    nxt.append(b)
        frontier = nxt
    if np.isinf(depth).any():
        depth[np.isinf(depth)] = (np.nanmax(depth[np.isfinite(depth)]) + 1
                                  if np.isfinite(depth).any() else 0.0) + 1.0
    return depth


def ordering_predictivity(W_shape, bin_centers, event_bin_order, *, rates):
    """Compare bin-activation order predicted by W_shape vs pure distance vs pure
    rate against the observed order (``event_bin_order``) via Spearman ρ.

    ``event_bin_order`` lists the bins in observed activation order; the seed is its
    first element. For each predictor we score every bin (W: propagation depth from
    the seed; dist: Euclidean distance from the seed center; rate: -rate so high
    rate = early), then Spearman-correlate the predictor scores against the observed
    activation position, over the bins that appear in ``event_bin_order``.
    Returns dict(rho_W, rho_dist, rho_rate).
    """
    from scipy.stats import spearmanr

    centers = np.asarray(bin_centers, float)
    rates = np.asarray(rates, float)
    order = list(event_bin_order)
    seed = order[0]
    obs_pos = {b: i for i, b in enumerate(order)}
    bins_seen = order

    observed = np.array([obs_pos[b] for b in bins_seen], float)

    depth = _propagation_depth(W_shape, seed)
    score_W = np.array([depth[b] for b in bins_seen], float)
    score_dist = np.array([np.linalg.norm(centers[b] - centers[seed]) for b in bins_seen],
                          float)
    score_rate = np.array([-rates[b] for b in bins_seen], float)

    def _rho(score):
        if np.allclose(score, score[0]) or np.allclose(observed, observed[0]):
            return 0.0
        r = spearmanr(score, observed).correlation
        return 0.0 if not np.isfinite(r) else float(r)

    return dict(rho_W=_rho(score_W), rho_dist=_rho(score_dist), rho_rate=_rho(score_rate))

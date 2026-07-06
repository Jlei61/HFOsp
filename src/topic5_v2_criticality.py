"""Topic 5 V2 Phase 2 criticality/state-layer helpers.

This module intentionally stays on pure configuration/math for Tasks 0-1.
Real-data scripts must treat Phase 2 as exploratory peri-ictal susceptibility,
not forecasting and not a stand-alone critical-mode claim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config" / "topic5_v2_phase2.yaml"


def load_phase2_config(path: str | Path | None = None) -> dict:
    """Load the Phase 2 YAML config as a plain dict."""
    cfg_path = Path(path) if path is not None else _DEFAULT_CFG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Phase 2 config must be a mapping: {cfg_path}")
    return dict(cfg)


def _window_pair(idx: Sequence[int | float]) -> tuple[int, int]:
    if len(idx) != 2:
        raise ValueError("window index must be a length-2 pair")
    start, stop = int(idx[0]), int(idx[1])
    if stop < start:
        raise ValueError(f"window stop precedes start: {idx}")
    return start, stop


def _finite(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)[np.isfinite(x)]


def _window_var(x: np.ndarray) -> float:
    xf = _finite(x)
    return float(np.var(xf)) if xf.size >= 3 else float("nan")


def _ar1(x: np.ndarray) -> float:
    xf = _finite(x)
    if xf.size < 3:
        return float("nan")
    if np.std(xf[:-1]) == 0 or np.std(xf[1:]) == 0:
        return float("nan")
    return float(np.corrcoef(xf[:-1], xf[1:])[0, 1])


def _linelength(x: np.ndarray) -> tuple[float, float]:
    xf = _finite(x)
    if xf.size < 3:
        return float("nan"), float("nan")
    total = float(np.sum(np.abs(np.diff(xf))))
    return total, float(total / (xf.size - 1))


def _feature_delta(
    env_2d: np.ndarray,
    early: tuple[int, int],
    late: tuple[int, int],
    fn: Callable[[np.ndarray], float],
) -> np.ndarray:
    out = np.full(env_2d.shape[0], np.nan, dtype=float)
    e0, e1 = early
    l0, l1 = late
    for ch in range(env_2d.shape[0]):
        before = fn(env_2d[ch, e0:e1])
        after = fn(env_2d[ch, l0:l1])
        if np.isfinite(before) and np.isfinite(after):
            out[ch] = after - before
    return out


def contact_susceptibility(
    env_2d: np.ndarray,
    early_idx: Sequence[int | float],
    late_idx: Sequence[int | float],
) -> dict[str, np.ndarray]:
    """Return per-contact late-minus-early susceptibility changes.

    Parameters
    ----------
    env_2d
        Array shaped ``(n_contacts, n_time)``.
    early_idx, late_idx
        ``(start, stop)`` sample-index pairs. Each feature is NaN for a contact
        if either window has fewer than 3 finite samples.

    Returns
    -------
    dict
        Keys are ``variance``, ``lag1_autocorr``, ``line_length_rate`` and
        ``line_length_sum``. ``line_length_rate`` is the length-normalized
        primary feature.
    """
    env = np.asarray(env_2d, dtype=float)
    if env.ndim != 2:
        raise ValueError(f"env_2d must be 2D (n_contacts, n_time), got {env.shape}")
    early = _window_pair(early_idx)
    late = _window_pair(late_idx)

    line_sum = _feature_delta(env, early, late, lambda x: _linelength(x)[0])
    line_rate = _feature_delta(env, early, late, lambda x: _linelength(x)[1])
    return {
        "variance": _feature_delta(env, early, late, _window_var),
        "lag1_autocorr": _feature_delta(env, early, late, _ar1),
        "line_length_rate": line_rate,
        "line_length_sum": line_sum,
    }


def _as_2d_float(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n_ch, n_t)")
    return arr


def _coerce_rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def prepare_var_window(X: np.ndarray, standardize: bool = True) -> np.ndarray:
    """Demean, linearly detrend, and optionally standardize each channel.

    Missing samples are ignored during fitting and replaced by zero after the
    within-window preprocessing, i.e. by the channel's preprocessed baseline.
    """

    arr = _as_2d_float(X)
    n_ch, n_t = arr.shape
    t = np.arange(n_t, dtype=float)
    out = np.zeros((n_ch, n_t), dtype=float)
    finite_rows = np.isfinite(arr).all(axis=1)

    # Fast vectorized path for fully-finite channels (the common envelope case):
    # demean -> linear detrend (lstsq) -> standardize, all channels at once.
    fidx = np.flatnonzero(finite_rows)
    if fidx.size:
        Xf = arr[fidx] - arr[fidx].mean(axis=1, keepdims=True)
        if n_t >= 3:
            design = np.vstack([t, np.ones_like(t)]).T
            coef, *_ = np.linalg.lstsq(design, Xf.T, rcond=None)
            Xf = Xf - (design @ coef).T
        if standardize:
            sd = Xf.std(axis=1, keepdims=True)
            Xf = np.divide(Xf, sd, out=np.zeros_like(Xf), where=sd > 0.0)
        out[fidx] = Xf

    # Per-channel fallback for channels carrying any NaN.
    for ch in np.flatnonzero(~finite_rows):
        x = arr[ch].astype(float, copy=True)
        finite = np.isfinite(x)
        if not np.any(finite):
            continue

        x = x - float(np.nanmean(x))
        if np.count_nonzero(finite) >= 3:
            slope, intercept = np.polyfit(t[finite], x[finite], 1)
            x = x - (slope * t + intercept)

        if standardize:
            sd = float(np.nanstd(x[finite]))
            if np.isfinite(sd) and sd > 0.0:
                x = x / sd
            else:
                x = np.zeros_like(x)

        x[~np.isfinite(x)] = 0.0
        out[ch] = x

    return out


def var_window_ok(n_ch: int, n_t: int, min_t_over_ch: float = 5) -> bool:
    """Return whether a VAR window is sufficiently powered for this plan."""

    n_ch_i = int(n_ch)
    n_t_i = int(n_t)
    if n_ch_i <= 0 or n_t_i <= 0:
        return False
    required = max(int(np.ceil(float(min_t_over_ch) * n_ch_i)), n_ch_i + 10)
    return n_t_i >= required


def _fit_var1_pairs(x0: np.ndarray, x1: np.ndarray, alpha: float) -> np.ndarray:
    n_ch = x0.shape[0]
    gram = x0 @ x0.T + alpha * np.eye(n_ch)
    rhs = x1 @ x0.T
    try:
        return np.linalg.solve(gram.T, rhs.T).T
    except np.linalg.LinAlgError:
        return rhs @ np.linalg.pinv(gram)


def var1_ridge(X: np.ndarray, alpha: float) -> np.ndarray:
    """Fit X[:, t] = A @ X[:, t-1] with ridge regularization."""

    arr = np.nan_to_num(_as_2d_float(X), copy=True)
    n_ch, n_t = arr.shape
    if n_t < 2:
        raise ValueError("X must contain at least two time samples")
    alpha_f = float(alpha)
    if alpha_f < 0.0:
        raise ValueError("alpha must be nonnegative")

    x0 = arr[:, :-1]
    x1 = arr[:, 1:]
    return _fit_var1_pairs(x0, x1, alpha_f)


def spectral_radius(A: np.ndarray) -> float:
    """Largest absolute eigenvalue of a square matrix."""

    mat = np.asarray(A, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("A must be a square matrix")
    if mat.shape[0] == 0:
        return float("nan")
    return float(np.max(np.abs(np.linalg.eigvals(mat))))


def leading_eigvec(A: np.ndarray) -> np.ndarray:
    """Return the nonnegative, L2-normalized magnitude loading of the top mode."""

    mat = np.asarray(A, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("A must be a square matrix")
    if mat.shape[0] == 0:
        return np.array([], dtype=float)

    eigvals, eigvecs = np.linalg.eig(mat)
    idx = int(np.argmax(np.abs(eigvals)))
    if not np.isfinite(eigvals[idx]) or np.abs(eigvals[idx]) <= 0.0:
        return np.zeros(mat.shape[0], dtype=float)

    loading = np.abs(eigvecs[:, idx]).astype(float)
    norm = float(np.linalg.norm(loading))
    if not np.isfinite(norm) or norm == 0.0:
        return np.zeros(mat.shape[0], dtype=float)
    return loading / norm


def recovery_tau(lambda_max: float, dt: float) -> float:
    """Discrete-time recovery constant for a stable lambda."""

    lm = float(lambda_max)
    dt_f = float(dt)
    if dt_f <= 0.0:
        raise ValueError("dt must be positive")
    if not np.isfinite(lm):
        return float("nan")
    if lm >= 1.0:
        return float("inf")
    if lm <= 0.0:
        return 0.0
    return float(-dt_f / np.log(lm))


def cv_one_step_r2(X: np.ndarray, alpha: float, n_folds: int) -> float:
    """Contiguous-fold cross-validated one-step prediction R2."""

    arr = np.nan_to_num(_as_2d_float(X), copy=True)
    n_ch, n_t = arr.shape
    n_folds_i = int(n_folds)
    if n_folds_i < 2:
        raise ValueError("n_folds must be at least 2")
    if n_t < n_folds_i + 2:
        return float("nan")

    edges = np.linspace(1, n_t, n_folds_i + 1, dtype=int)
    sse = 0.0
    sst = 0.0
    used = 0

    for fold in range(n_folds_i):
        test_targets = np.arange(edges[fold], edges[fold + 1], dtype=int)
        if test_targets.size < 2:
            continue

        train_mask = np.ones(n_t, dtype=bool)
        train_mask[test_targets] = False
        train_targets = np.array(
            [t for t in range(1, n_t) if train_mask[t] and train_mask[t - 1]],
            dtype=int,
        )
        if train_targets.size < n_ch + 2:
            continue

        train_x0 = arr[:, train_targets - 1]
        train_x1 = arr[:, train_targets]
        A = _fit_var1_pairs(train_x0, train_x1, float(alpha))

        pred = A @ arr[:, test_targets - 1]
        truth = arr[:, test_targets]
        fold_mean = truth.mean(axis=1, keepdims=True)
        sse += float(np.sum((truth - pred) ** 2))
        sst += float(np.sum((truth - fold_mean) ** 2))
        used += 1

    if used == 0 or sst <= 0.0:
        return float("nan")
    return float(1.0 - sse / sst)


def block_shuffle_surrogate(X: np.ndarray, block_len: int, rng) -> np.ndarray:
    """Shuffle contiguous time blocks with the same block order for all channels."""

    arr = _as_2d_float(X)
    n_t = arr.shape[1]
    block_len_i = int(block_len)
    if block_len_i <= 0:
        raise ValueError("block_len must be positive")
    if n_t <= 1:
        return arr.copy()

    blocks = [
        np.arange(start, min(start + block_len_i, n_t))
        for start in range(0, n_t, block_len_i)
    ]
    order = _coerce_rng(rng).permutation(len(blocks))
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    idx = np.concatenate([blocks[i] for i in order])
    return arr[:, idx].copy()


def phase_randomize_surrogate(X: np.ndarray, rng) -> np.ndarray:
    """Randomize each channel's Fourier phases while preserving its power spectrum."""

    arr = _as_2d_float(X)
    rng_g = _coerce_rng(rng)
    n_ch, n_t = arr.shape
    if n_t <= 1:
        return arr.copy()

    out = np.zeros_like(arr, dtype=float)
    for ch in range(n_ch):
        x = arr[ch].astype(float, copy=True)
        finite = np.isfinite(x)
        if not np.any(finite):
            out[ch] = np.zeros(n_t, dtype=float)
            continue

        mean = float(np.nanmean(x))
        x[~finite] = mean
        centered = x - mean
        freq = np.fft.rfft(centered)
        if freq.size > 2:
            stop = freq.size - 1 if n_t % 2 == 0 else freq.size
            idx = np.arange(1, stop)
            phases = rng_g.uniform(0.0, 2.0 * np.pi, size=idx.size)
            freq[idx] = np.abs(freq[idx]) * np.exp(1j * phases)
        out[ch] = np.fft.irfft(freq, n=n_t) + mean

    return out


def _as_2d_bool(active_bool: np.ndarray) -> np.ndarray:
    active = np.asarray(active_bool, dtype=bool)
    if active.ndim != 2:
        raise ValueError("active_bool must have shape (n_contacts, n_time)")
    return active


def _as_square_atm(atm: np.ndarray) -> np.ndarray:
    mat = np.asarray(atm, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("atm must be a square (n_contacts, n_contacts) array")
    return mat


def _as_rank_vec(rank_vec: np.ndarray, n_ch: int) -> np.ndarray:
    ranks = np.asarray(rank_vec, dtype=float)
    if ranks.ndim != 1 or ranks.shape[0] != n_ch:
        raise ValueError("rank_vec must have shape (n_contacts,)")
    return ranks


def activations_from_z(z_2d: np.ndarray, thr: float) -> np.ndarray:
    """Threshold contact-by-time z traces into avalanche activations."""
    z = np.asarray(z_2d, dtype=float)
    if z.ndim != 2:
        raise ValueError("z_2d must have shape (n_contacts, n_time)")
    return z > float(thr)


def branching_ratio(active_bool: np.ndarray) -> float:
    """Mean active-count ratio n(t+1) / n(t), skipping n(t) == 0 bins."""
    active = _as_2d_bool(active_bool)
    counts = active.sum(axis=0).astype(float)
    if counts.size < 2:
        return float("nan")
    valid = counts[:-1] > 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(counts[1:][valid] / counts[:-1][valid]))


def avalanche_atm(active_bool: np.ndarray) -> np.ndarray:
    """Build a row-normalized avalanche transition matrix.

    M[i, j] counts contact i active at t followed by contact j active at t+1.
    Empty source rows remain all-zero after normalization.
    """
    active = _as_2d_bool(active_bool)
    n_ch, n_t = active.shape
    counts = np.zeros((n_ch, n_ch), dtype=float)
    if n_t < 2:
        return counts

    for t in range(n_t - 1):
        cur = np.flatnonzero(active[:, t])
        nxt = np.flatnonzero(active[:, t + 1])
        if cur.size == 0 or nxt.size == 0:
            continue
        counts[np.ix_(cur, nxt)] += 1.0

    row_sum = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)


def atm_forward_displacement(atm: np.ndarray, rank_vec: np.ndarray) -> float:
    """Primary ATM direction metric: expected next-rank minus current-rank."""
    mat = _as_square_atm(atm)
    ranks = _as_rank_vec(rank_vec, mat.shape[0])
    finite = np.isfinite(ranks)
    if not np.any(finite):
        return float("nan")

    sub = mat[np.ix_(finite, finite)]
    denom = float(sub.sum())
    if denom <= 0:
        return float("nan")

    r = ranks[finite]
    delta = r[None, :] - r[:, None]
    return float(np.sum(sub * delta) / denom)


def atm_direction_index(atm: np.ndarray, rank_vec: np.ndarray) -> float:
    """Forward-vs-backward off-diagonal transition mass index."""
    mat = _as_square_atm(atm)
    ranks = _as_rank_vec(rank_vec, mat.shape[0])
    finite = np.isfinite(ranks)
    if finite.sum() < 2:
        return float("nan")

    sub = mat[np.ix_(finite, finite)]
    r = ranks[finite]
    delta = r[None, :] - r[:, None]
    fwd = float(sub[delta > 0].sum())
    bwd = float(sub[delta < 0].sum())
    denom = fwd + bwd
    if denom <= 0:
        return float("nan")
    return float((fwd - bwd) / denom)


def atm_rank_coupling_spearman(atm: np.ndarray, rank_vec: np.ndarray) -> float:
    """Descriptive only: Spearman(expected next rank, own rank).

    This can be high under pure self-persistence, so it is not the primary
    direction metric.
    """
    mat = _as_square_atm(atm)
    ranks = _as_rank_vec(rank_vec, mat.shape[0])
    finite_rank = np.isfinite(ranks)
    if finite_rank.sum() < 4:
        return float("nan")

    finite_mat = mat[:, finite_rank]
    row_mass = finite_mat.sum(axis=1)
    expected = np.full(mat.shape[0], np.nan, dtype=float)
    has_mass = row_mass > 0
    expected[has_mass] = (finite_mat[has_mass] @ ranks[finite_rank]) / row_mass[has_mass]

    ok = finite_rank & has_mass & np.isfinite(expected)
    if ok.sum() < 4:
        return float("nan")
    if np.std(expected[ok]) < 1e-12 or np.std(ranks[ok]) < 1e-12:
        return float("nan")

    stat = spearmanr(expected[ok], ranks[ok])
    value = getattr(stat, "statistic", getattr(stat, "correlation", np.nan))
    return float(value) if np.isfinite(value) else float("nan")

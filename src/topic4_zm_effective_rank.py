"""Unit-invariant slow-coordinate functional-rank diagnostics (Rev3.1 Task 9A).

The scientific object is a local response matrix from existing slow
coordinates to source/readout observables.  Raw derivatives are not comparable
because Z, M, S_G, rate and energy have different units.  This module therefore
standardizes every derivative by locked trajectory scales before computing a
singular spectrum.

The rank-1 label is intentionally conservative and local.  It means that the
second singular direction remains small under bootstrap resampling; it does
not claim that the full slow manifold is globally one-dimensional.
"""
from __future__ import annotations

import copy

import numpy as np


EFFECTIVE_RANK_VERSION = "zm_effective_rank_v1.1_2026-07-27"
RANK1_S2_RATIO_MAX = 0.20
RANK1_ENERGY_MIN = 0.90


def trajectory_coordinate_directions(early_state, late_state, *, nE):
    """Use the actual early-to-late field displacement as each coordinate axis."""
    nE = int(nE)
    z0 = np.asarray(early_state["slow.z"], float)[:nE]
    z1 = np.asarray(late_state["slow.z"], float)[:nE]
    m0 = np.asarray(early_state["slow.m"], float)[:nE]
    m1 = np.asarray(late_state["slow.m"], float)[:nE]
    sg0 = float(np.asarray(early_state["slow.S_G"]))
    sg1 = float(np.asarray(late_state["slow.S_G"]))
    directions = {
        "z": z1 - z0,
        "m": m1 - m0,
        "sg": float(sg1 - sg0),
    }
    for name, value in directions.items():
        norm = abs(value) if np.ndim(value) == 0 else np.linalg.norm(value)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"{name}: early-to-late trajectory direction is degenerate")
    return directions


def apply_trajectory_coordinate(
    state,
    directions,
    coordinate,
    sign,
    *,
    delta,
    nE,
    min_delta=1e-5,
):
    """Displace one real slow field without clipping a central difference.

    When a requested displacement would leave a physical bound, its magnitude
    is halved until the entire field is valid.  The returned actual delta must
    be paired across plus/minus arms by the caller.
    """
    if coordinate not in {"z", "m", "sg"}:
        raise ValueError(f"unknown coordinate {coordinate!r}")
    if int(sign) not in {-1, 1}:
        raise ValueError("sign must be -1 or +1")
    actual = float(delta)
    if actual <= 0:
        raise ValueError("delta must be positive")
    nE = int(nE)

    while actual >= float(min_delta):
        if coordinate == "z":
            base = np.asarray(state["slow.z"], float)[:nE]
            candidate = base + int(sign) * actual * np.asarray(directions["z"], float)
            valid = np.all((candidate >= 0.0) & (candidate <= 1.0))
        elif coordinate == "m":
            base = np.asarray(state["slow.m"], float)[:nE]
            candidate = base + int(sign) * actual * np.asarray(directions["m"], float)
            valid = np.all(candidate >= 0.0)
        else:
            base = float(np.asarray(state["slow.S_G"]))
            candidate = base + int(sign) * actual * float(directions["sg"])
            valid = bool(candidate >= 0.0)
        if valid:
            break
        actual *= 0.5
    else:
        raise ValueError(f"{coordinate}: no valid central displacement above min_delta")

    out = copy.deepcopy(state)
    if coordinate == "z":
        out["slow.z"] = np.asarray(out["slow.z"], float).copy()
        out["slow.z"][:nE] = candidate
    elif coordinate == "m":
        out["slow.m"] = np.asarray(out["slow.m"], float).copy()
        out["slow.m"][:nE] = candidate
    else:
        out["slow.S_G"] = np.asarray(float(candidate))
    return out, actual


def paired_trajectory_coordinate(state, directions, coordinate, *, delta, nE):
    """Return a symmetric plus/minus pair at the largest jointly valid delta."""
    _, dp = apply_trajectory_coordinate(
        state, directions, coordinate, +1, delta=delta, nE=nE
    )
    _, dm = apply_trajectory_coordinate(
        state, directions, coordinate, -1, delta=delta, nE=nE
    )
    actual = min(dp, dm)
    plus, dp2 = apply_trajectory_coordinate(
        state, directions, coordinate, +1, delta=actual, nE=nE
    )
    minus, dm2 = apply_trajectory_coordinate(
        state, directions, coordinate, -1, delta=actual, nE=nE
    )
    if not np.isclose(dp2, dm2):
        raise RuntimeError("failed to construct a symmetric central pair")
    return plus, minus, float(actual)


def trajectory_coordinate_values(states, origin_state, directions, *, nE):
    """Project visited slow fields onto the three early-to-late coordinate axes."""
    nE = int(nE)
    z0 = np.asarray(origin_state["slow.z"], float)[:nE]
    m0 = np.asarray(origin_state["slow.m"], float)[:nE]
    sg0 = float(np.asarray(origin_state["slow.S_G"]))
    dz = np.asarray(directions["z"], float)
    dm = np.asarray(directions["m"], float)
    dsg = float(directions["sg"])
    out = []
    for state in states:
        z = np.asarray(state["slow.z"], float)[:nE]
        m = np.asarray(state["slow.m"], float)[:nE]
        sg = float(np.asarray(state["slow.S_G"]))
        out.append([
            float(np.dot(z - z0, dz) / np.dot(dz, dz)),
            float(np.dot(m - m0, dm) / np.dot(dm, dm)),
            float((sg - sg0) / dsg),
        ])
    return np.asarray(out, float)


def robust_scales(values):
    """Per-column robust SD; degeneracy is evidence failure, never a silent floor."""
    x = np.asarray(values, float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < 3 or not np.all(np.isfinite(x)):
        raise ValueError("robust scales require at least three finite samples")
    q25, q75 = np.percentile(x, [25, 75], axis=0)
    scale = (q75 - q25) / 1.349
    mad = 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)
    scale = np.where(scale > 1e-12, scale, mad)
    sd = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, sd)
    if np.any(scale <= 1e-12):
        bad = np.flatnonzero(scale <= 1e-12).tolist()
        raise ValueError(f"degenerate trajectory scales at columns {bad}")
    return scale


def response_vectors(series_by_name, observable_order, *, burn_bins, static_tail_bins):
    """Build separate static and time-resolved response vectors from aligned bins."""
    arrays = [np.asarray(series_by_name[name], float) for name in observable_order]
    if not arrays or any(a.ndim != 1 for a in arrays):
        raise ValueError("response series must be non-empty 1D arrays")
    n = arrays[0].size
    if any(a.size != n for a in arrays):
        raise ValueError("response series are not time-aligned")
    burn = int(burn_bins)
    if burn < 0 or burn >= n:
        raise ValueError("invalid burn_bins")
    post = np.column_stack([a[burn:] for a in arrays])
    tail = int(static_tail_bins)
    if tail < 1 or post.shape[0] < tail:
        raise ValueError("not enough post-burn bins for static response")
    return {
        "static": np.mean(post[-tail:], axis=0),
        "impulse": post.reshape(-1),
        "n_time_bins": int(post.shape[0]),
        "n_observables": int(post.shape[1]),
    }


def assemble_paired_sensitivity(rows, coordinate_order):
    """Assemble dy/dq from central-difference rows under matched future noise."""
    columns = []
    for coordinate in coordinate_order:
        pair = [r for r in rows if r.get("coordinate") == coordinate]
        if len(pair) != 2 or {int(r.get("sign", 0)) for r in pair} != {-1, 1}:
            raise ValueError(f"{coordinate}: require exactly one plus and one minus row")
        plus = next(r for r in pair if int(r["sign"]) == 1)
        minus = next(r for r in pair if int(r["sign"]) == -1)
        if plus.get("bank_sha") != minus.get("bank_sha"):
            raise ValueError(f"{coordinate}: central pair uses unmatched future noise")
        dp = float(plus.get("delta", np.nan))
        dm = float(minus.get("delta", np.nan))
        if not np.isfinite(dp) or dp <= 0 or not np.isclose(dp, dm):
            raise ValueError(f"{coordinate}: plus/minus delta mismatch")
        yp = np.asarray(plus["y"], float)
        ym = np.asarray(minus["y"], float)
        if yp.shape != ym.shape or yp.ndim != 1:
            raise ValueError(f"{coordinate}: response vectors must be aligned 1D arrays")
        columns.append((yp - ym) / (2.0 * dp))
    if not columns:
        raise ValueError("no coordinates")
    return np.column_stack(columns)


def standardize_sensitivity(S, q_scales, y_scales):
    """Return S_tilde[i,j] = q_scale[j] / y_scale[i] * dy_i/dq_j."""
    S = np.asarray(S, float)
    q = np.asarray(q_scales, float)
    y = np.asarray(y_scales, float)
    if S.ndim != 2 or q.shape != (S.shape[1],) or y.shape != (S.shape[0],):
        raise ValueError("scale dimensions do not match sensitivity matrix")
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(y)):
        raise ValueError("sensitivity and scales must be finite")
    if np.any(q <= 0) or np.any(y <= 0):
        raise ValueError("trajectory scales must be positive")
    return S * q[None, :] / y[:, None]


def rank_summary(matrix):
    """Continuous singular-spectrum summary plus a descriptive rank-1 flag."""
    A = np.asarray(matrix, float)
    if A.ndim != 2 or min(A.shape) < 1 or not np.all(np.isfinite(A)):
        raise ValueError("matrix must be finite and two-dimensional")
    s = np.linalg.svd(A, compute_uv=False)
    power = s ** 2
    total = float(power.sum())
    if total <= 0:
        energy_first = 0.0
        participation = 0.0
        entropy_rank = 0.0
    else:
        p = power / total
        energy_first = float(p[0])
        participation = float(total ** 2 / np.sum(power ** 2))
        pp = p[p > 0]
        entropy_rank = float(np.exp(-np.sum(pp * np.log(pp))))
    s2_ratio = float(s[1] / s[0]) if s.size >= 2 and s[0] > 0 else 0.0
    return {
        "effective_rank_version": EFFECTIVE_RANK_VERSION,
        "singular_values": s.tolist(),
        "first_direction_energy_fraction": energy_first,
        "s2_over_s1": s2_ratio,
        "effective_rank_participation": participation,
        "effective_rank_entropy": entropy_rank,
        "near_rank1_descriptive": bool(
            s2_ratio < RANK1_S2_RATIO_MAX and energy_first > RANK1_ENERGY_MIN
        ),
        "claim_boundary": (
            "local standardized functional collinearity only; not global "
            "slow-manifold dimensionality"
        ),
    }


def bootstrap_rank(sample_matrices, *, n_boot=2000, seed=0, ci=(2.5, 97.5)):
    """Bootstrap the mean standardized response matrix over seeds/microstates."""
    X = np.asarray(sample_matrices, float)
    if X.ndim != 3 or X.shape[0] < 2 or not np.all(np.isfinite(X)):
        raise ValueError("sample_matrices must be finite sample x output x coordinate")
    if int(n_boot) < 20:
        raise ValueError("n_boot must be at least 20")
    rng = np.random.default_rng(seed)
    ratios, energies, participation = [], [], []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, X.shape[0], size=X.shape[0])
        summary = rank_summary(np.mean(X[idx], axis=0))
        ratios.append(summary["s2_over_s1"])
        energies.append(summary["first_direction_energy_fraction"])
        participation.append(summary["effective_rank_participation"])
    ratio_ci = np.percentile(ratios, ci)
    energy_ci = np.percentile(energies, ci)
    participation_ci = np.percentile(participation, ci)
    point = rank_summary(np.mean(X, axis=0))
    return {
        "effective_rank_version": EFFECTIVE_RANK_VERSION,
        "n_samples": int(X.shape[0]),
        "n_boot": int(n_boot),
        "point": point,
        "s2_over_s1_ci": ratio_ci.tolist(),
        "first_direction_energy_fraction_ci": energy_ci.tolist(),
        "effective_rank_participation_ci": participation_ci.tolist(),
        "rank1_supported": bool(
            ratio_ci[1] < RANK1_S2_RATIO_MAX
            and energy_ci[0] > RANK1_ENERGY_MIN
        ),
        "rank1_thresholds": {
            "s2_over_s1_upper_max": RANK1_S2_RATIO_MAX,
            "first_energy_lower_min": RANK1_ENERGY_MIN,
        },
    }


def hierarchical_bootstrap_rank(
    seed_state_matrices,
    *,
    n_boot=2000,
    seed=0,
    ci=(2.5, 97.5),
):
    """Bootstrap standardized response matrices by seed, then microstate.

    The expected shape is ``seed x microstate x output x coordinate``.
    Network seeds are the independent replication level; carrier microstates
    are resampled only within each selected seed.  Flattening both dimensions
    before resampling would incorrectly count three states from one seed as
    three independent seeds.
    """

    X = np.asarray(seed_state_matrices, float)
    if (
        X.ndim != 4
        or X.shape[0] < 2
        or X.shape[1] < 2
        or not np.all(np.isfinite(X))
    ):
        raise ValueError(
            "seed_state_matrices must be finite "
            "seed x microstate x output x coordinate with >=2 seeds/states"
        )
    if int(n_boot) < 20:
        raise ValueError("n_boot must be at least 20")

    rng = np.random.default_rng(seed)
    ratios, energies, participation = [], [], []
    n_seed, n_state = X.shape[:2]
    for _ in range(int(n_boot)):
        selected_seed = rng.integers(0, n_seed, size=n_seed)
        sampled_seed_means = []
        for seed_index in selected_seed:
            selected_state = rng.integers(0, n_state, size=n_state)
            sampled_seed_means.append(
                np.mean(X[seed_index, selected_state], axis=0)
            )
        summary = rank_summary(np.mean(sampled_seed_means, axis=0))
        ratios.append(summary["s2_over_s1"])
        energies.append(summary["first_direction_energy_fraction"])
        participation.append(summary["effective_rank_participation"])

    ratio_ci = np.percentile(ratios, ci)
    energy_ci = np.percentile(energies, ci)
    participation_ci = np.percentile(participation, ci)
    point = rank_summary(np.mean(X, axis=(0, 1)))
    return {
        "effective_rank_version": EFFECTIVE_RANK_VERSION,
        "bootstrap_structure": "hierarchical_seed_then_microstate",
        "n_seeds": int(n_seed),
        "n_microstates_per_seed": int(n_state),
        "n_samples": int(n_seed * n_state),
        "n_boot": int(n_boot),
        "point": point,
        "s2_over_s1_ci": ratio_ci.tolist(),
        "first_direction_energy_fraction_ci": energy_ci.tolist(),
        "effective_rank_participation_ci": participation_ci.tolist(),
        "rank1_supported": bool(
            ratio_ci[1] < RANK1_S2_RATIO_MAX
            and energy_ci[0] > RANK1_ENERGY_MIN
        ),
        "rank1_thresholds": {
            "s2_over_s1_upper_max": RANK1_S2_RATIO_MAX,
            "first_energy_lower_min": RANK1_ENERGY_MIN,
        },
    }

#!/usr/bin/env python
"""Topic 5 V3a mode-transition — dynamics run (Task 8, H3c co-primary).

Plain-language question (EXPLORATORY): from late-preictal (P3) to early-ictal
(I1), does the DOMINANT DYNAMICAL DIRECTION of the field (the input direction a
short-horizon VAR amplifies most) tilt OFF the fixed interictal HFO axis onto
non-axis contacts? For each seizure phase we slide a short window over the
all-clean envelope, fit a low-rank (DMD-style) VAR, take the dominant right
singular vector of ``A^k*`` (the most-amplified input direction), map it back to
contact space (``u_c``), and measure a per-contact ``density`` mode-shift =
``‖P_N u_c‖²/rank(P_N) − ‖P_A u_c‖²/rank(P_A)`` (non-axis energy minus axis
energy, each per-contact so unequal subspace sizes are comparable). We take the
median over a window, then over a phase's seizures, and ask whether
``mode_shift_I1 − mode_shift_P3`` is larger than chance.

Δ-NULL DISCIPLINE (plan rev2 item 7 — the co-primary trap): the p-value is on
the Δ(I1−P3) permutation distribution, NOT per-phase-null then subtract. Each
permutation applies its null to BOTH P3 and I1, recomputes both phase medians,
and forms ``delta_perm = mode_shift_I1_perm − mode_shift_P3_perm``.

THREE gating nulls (all on the Δ statistic; H3c expects delta > 0):
  - phase  ``p_phase``: per window ``phase_randomize_surrogate`` -> refit
    low-rank VAR -> ``u_c`` -> mode-shift (power-spectrum-matched control).
  - block  ``p_block``: per window ``block_shuffle_surrogate`` -> refit ->
    mode-shift (coarse temporal-order control).
  - label  ``p_label``: does NOT refit — keeps each window's OBSERVED ``u_c`` and
    permutes axis/non-axis labels within shaft (``label_permute``), rebuilding
    ``P_A'/P_N'`` at the SAME contact counts. This is the density
    dimension-bias control: is the positive shift specific to the TRUE non-axis
    contacts, or would any same-count relabeling reproduce it?
``module_null_pass = p_phase < alpha AND p_block < alpha AND p_label < alpha``
(all three gate — the config sets ``h3c_require_label_null: true``).

DIRECTION differs from H3b: the density mode-shift is already ~0-baseline via
the ÷rank normalization, so there is NO surplus column — ``module_direction`` is
read off the RAW observed ``delta_mode_shift_density > 0``.

narrow (primary) and broad (replication) run SEPARATELY, never pooled. A
``geometry_insufficient`` subject is FLAGGED (``status=skipped``) with a full
NaN/False row — never silently dropped. Tier is assigned only in the summary
(Task 10). See docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md
Task 8.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import (  # noqa: E402
    classify_subject_contacts,
    load_subject_phase_envelopes,
)
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from src.topic5_v2_criticality import (  # noqa: E402
    block_shuffle_surrogate,
    cv_one_step_r2,
    phase_randomize_surrogate,
    spectral_radius,
)
from src.topic5_v3_mode_transition import (  # noqa: E402
    axis_nonaxis_vectors,
    continuous_reactivity_approx,
    demean_window,
    direct_2d_var,
    dominant_right_singular_vector,
    label_permute,
    load_v3_config,
    lowrank_var,
    map_lowrank_vector_to_contacts,
    project_2d,
    rank_forward,
    sliding_windows,
    subspace_mode_shift,
    subspace_projectors,
)

CSV_COLS = [
    "subject", "cohort", "status", "skip_reason", "geometry_sufficient",
    "dynamics_primary_model", "dynamics_support_model", "rank_used", "k_star",
    "mode_shift_density_P3", "mode_shift_density_I1", "delta_mode_shift_density",
    "mode_shift_raw_delta", "mode_shift_2D_consistency",
    "p_phase", "p_block", "p_label", "mode_shift_label_z",
    "lambda_surplus_P3", "lambda_surplus_I1", "gain_axis_delta", "gain_nonaxis_delta",
    "reactivity_cont_available", "logm_quality_flag",
    "top_contact_energy_fraction", "single_contact_driven",
    "leave_one_contact_mode_shift_pass", "axis_only_mode_shift_control", "axis_only_control_pass",
    "onset_jitter_pass", "cv_r2", "var_meaningful_flag", "n_ch_fit", "n_seizures",
    "module_support_flag", "module_direction_correct", "module_null_pass",
]

# Every metric column present with NaN floats / False flags — a skipped or
# degenerate subject still carries the full schema (never drop). Model-name /
# rank / k* metadata is filled unconditionally in `_base_row` (config constants).
_METRIC_DEFAULTS = {
    "mode_shift_density_P3": float("nan"), "mode_shift_density_I1": float("nan"),
    "delta_mode_shift_density": float("nan"), "mode_shift_raw_delta": float("nan"),
    "mode_shift_2D_consistency": float("nan"),
    "p_phase": float("nan"), "p_block": float("nan"), "p_label": float("nan"),
    "mode_shift_label_z": float("nan"),
    "lambda_surplus_P3": float("nan"), "lambda_surplus_I1": float("nan"),
    "gain_axis_delta": float("nan"), "gain_nonaxis_delta": float("nan"),
    "reactivity_cont_available": False, "logm_quality_flag": False,
    "top_contact_energy_fraction": float("nan"), "single_contact_driven": False,
    "leave_one_contact_mode_shift_pass": False, "axis_only_mode_shift_control": float("nan"),
    "axis_only_control_pass": False, "onset_jitter_pass": False,
    "cv_r2": float("nan"), "var_meaningful_flag": False, "n_ch_fit": 0, "n_seizures": 0,
    "module_support_flag": False, "module_direction_correct": False, "module_null_pass": False,
}


def _base_row(ds_sid: str, cohort: str, cfg: dict) -> dict:
    return {
        "subject": ds_sid, "cohort": cohort, "status": "skipped", "skip_reason": "",
        "geometry_sufficient": False,
        "dynamics_primary_model": "direct_2D_VAR", "dynamics_support_model": "lowrank_DMD",
        "rank_used": int(cfg["dynamics"]["lowrank"]), "k_star": int(cfg["dynamics"]["finite_horizon_k"]),
        **_METRIC_DEFAULTS,
    }


# ---------------------------------------------------------------------------
# window slicing + aggregation (median over windows -> per seizure; median over
# seizures -> subject). Shared by observed / phase / block passes.
# ---------------------------------------------------------------------------
def _windows_of(env: np.ndarray, hop: float, win_sec: float, step_sec: float) -> list:
    relt_syn = np.arange(env.shape[1]) * hop
    return sliding_windows(relt_syn, 0, env.shape[1], win_sec, step_sec)


def _agg_windows(envs: list, fn, n_out: int, hop: float, win_sec: float, step_sec: float) -> tuple:
    """Median over windows (per seizure) then median over seizures, per component.

    ``fn(Xw)`` returns a length-``n_out`` tuple; each component is finite-filtered
    within a seizure before the window-median, and empty phases propagate NaN
    (never a fake 0) into ``obs_delta``.
    """
    sz_meds: list = [[] for _ in range(n_out)]
    for env in envs:
        outs = [fn(env[:, ws:we]) for ws, we in _windows_of(env, hop, win_sec, step_sec)]
        for k in range(n_out):
            col = [o[k] for o in outs if np.isfinite(o[k])]
            if col:
                sz_meds[k].append(float(np.median(col)))
    return tuple(float(np.median(s)) if s else float("nan") for s in sz_meds)


def _agg_cached(cache_phase: list, fn) -> float:
    """Median over windows then seizures for a cached per-window ``u_c`` list.

    ``cache_phase`` is a list (over contributing seizures) of lists (over that
    seizure's windows) of contact-space ``u_c`` vectors; ``fn(u_c) -> float``.
    Used by the label null / leave-one / axis-only / top-fraction, none of which
    refit the VAR (they reuse the OBSERVED ``u_c``).
    """
    sz_meds: list = []
    for sz_u in cache_phase:
        vals = [fn(u) for u in sz_u]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            sz_meds.append(float(np.median(vals)))
    return float(np.median(sz_meds)) if sz_meds else float("nan")


def _p_upper(obs: float, perm: np.ndarray) -> float:
    """One-sided-upper Δ-null p over FINITE draws only:
    ``(1 + #{finite >= obs}) / (1 + finite.size)``.

    A perm draw can be NaN (e.g. a degenerate phase/block/label refit leaves a
    phase with no valid mode-shift); NaN never satisfies ``>=``, so counting it
    toward the denominator without ever counting it toward the numerator would
    silently deflate p (same class of bug as H3a's ``_p_lower`` / H3b's
    ``_delta_stats`` fixes). If no draw is finite the null is undefined here
    and this returns NaN rather than a fabricated p.
    """
    finite = perm[np.isfinite(perm)]
    if finite.size == 0:
        return float("nan")
    return (1 + int(np.sum(finite >= obs))) / (1 + finite.size)


def _perm_arrays(perm_fn, n_perm: int, seed: int, n_out: int) -> list:
    """Run ``perm_fn(rng)`` (returns length-``n_out`` tuple) over ``n_perm`` seeds.

    Each perm gets a fresh ``np.random.default_rng(seed + p)`` (Task-6 pattern);
    the Δ subtraction happens INSIDE ``perm_fn`` (never per-phase-null then
    subtract), so the returned arrays are the null of the Δ statistic itself.
    """
    cols: list = [[] for _ in range(n_out)]
    for p in range(n_perm):
        out = perm_fn(np.random.default_rng(seed + p))
        for k in range(n_out):
            cols[k].append(out[k])
    return [np.array(c, dtype=float) for c in cols]


def _observed(envs: list, geom: dict, cfg: dict) -> tuple:
    """Observed per-phase aggregates + per-window ``u_c`` cache + flat cv/logm lists.

    Returns ``(agg, cache, cv_flat, logm_flat)`` where ``agg`` holds the
    median-over-windows/seizures of every per-window scalar (density/raw
    mode-shift, 2D singular mode-shift, 2D spectral radius, axis/non-axis 2D
    gains), ``cache`` is the per-window ``u_c`` (for label/leave-one/axis-only/
    top-fraction reuse — no refit), and ``cv_flat``/``logm_flat`` are flat
    per-window lists (full-D cv R² and 2D-VAR ``logm`` quality).
    """
    rank = int(cfg["dynamics"]["lowrank"])
    alpha = float(cfg["dynamics"]["var_ridge_alpha"])
    kstar = int(cfg["dynamics"]["finite_horizon_k"])
    hop = float(cfg["phases"]["hop_sec"])
    win_sec = float(cfg["phases"]["window_sec"])
    step_sec = float(cfg["phases"]["step_sec"])
    P_N, P_A = geom["P_N"], geom["P_A"]
    e_axis, e_nonaxis = geom["e_axis"], geom["e_nonaxis"]

    keys = ("ms_density", "ms_raw", "ms_2d", "lambda2d", "gain_axis", "gain_nonaxis")
    scal: dict = {k: [] for k in keys}
    cache: list = []
    cv_flat: list = []
    logm_flat: list = []
    for env in envs:
        u_list: list = []
        win_scal: dict = {k: [] for k in keys}
        for ws, we in _windows_of(env, hop, win_sec, step_sec):
            Xw = env[:, ws:we]
            Xd = demean_window(Xw)
            # low-rank DMD carrier: dominant amplified input direction -> contacts
            B_r, U_r = lowrank_var(Xw, rank, alpha)  # demeans internally
            u_c = map_lowrank_vector_to_contacts(dominant_right_singular_vector(B_r, kstar), U_r)
            u_list.append(u_c)
            win_scal["ms_density"].append(subspace_mode_shift(u_c, P_N, P_A, "density"))
            win_scal["ms_raw"].append(subspace_mode_shift(u_c, P_N, P_A, "raw"))
            # direct 2D VAR on the (axis-mean, non-axis-mean) projection
            B2 = direct_2d_var(project_2d(Xd, e_axis, e_nonaxis), alpha)
            u2 = dominant_right_singular_vector(B2, kstar)
            win_scal["ms_2d"].append(float(u2[1] ** 2 - u2[0] ** 2))  # nonaxis - axis
            win_scal["lambda2d"].append(spectral_radius(B2))
            win_scal["gain_axis"].append(float(np.linalg.norm(B2[:, 0])))
            win_scal["gain_nonaxis"].append(float(np.linalg.norm(B2[:, 1])))
            cv_flat.append(cv_one_step_r2(Xd, alpha, 5))
            _val, logm_ok = continuous_reactivity_approx(B2, hop)
            logm_flat.append(bool(logm_ok))
        cache.append(u_list)
        for k in keys:
            vv = [x for x in win_scal[k] if np.isfinite(x)]
            if vv:
                scal[k].append(float(np.median(vv)))
    agg = {k: (float(np.median(v)) if v else float("nan")) for k, v in scal.items()}
    return agg, cache, cv_flat, logm_flat


def _run_ok_subject(ds_sid: str, cohort: str, cfg: dict, cc: dict, n_perm: int, row: dict) -> dict:
    """Full H3c metric block for a geometry-sufficient subject."""
    rank = int(cfg["dynamics"]["lowrank"])
    alpha = float(cfg["dynamics"]["var_ridge_alpha"])
    kstar = int(cfg["dynamics"]["finite_horizon_k"])
    hop = float(cfg["phases"]["hop_sec"])
    win_sec = float(cfg["phases"]["window_sec"])
    step_sec = float(cfg["phases"]["step_sec"])
    block_n = int(round(cfg["dynamics"]["block_len_sec"] / hop))
    seed = int(cfg["nulls"]["seed"])
    alpha_stat = float(cfg["nulls"]["alpha"])
    frac_max = float(cfg["statistics"]["single_contact_energy_frac_max"])

    all_clean = cc["all_clean"]
    is_axis_names = cc["is_axis"]
    is_nonaxis_names = cc["is_nonaxis_strict"]
    shaft_by_name = cc["shaft_by_name"]

    # ---- geometry: subspace projectors + axis/non-axis mean vectors ----
    P_A, P_N = subspace_projectors(all_clean, is_axis_names, is_nonaxis_names)
    # typical_rank of axis contacts (ta wins ties) -> signed forward axis. This
    # only feeds `e_axis_grad` (a sensitivity vector NOT used by any Task-8
    # output — the 2D VAR primary is [e_axis_mean, e_nonaxis_mean]); built here
    # because `axis_nonaxis_vectors` requires `rank_forward` as an argument.
    axis_set = set(is_axis_names)
    typical_rank: dict = {}
    for rec in (cc["ctx"]["ta"], cc["ctx"]["tb"]):
        for ch in rec["channels"]:
            nm = ch["name"]
            r = ch.get("typical_rank", np.nan)
            if nm in axis_set and np.isfinite(r):
                typical_rank.setdefault(nm, float(r))
    rf = rank_forward(typical_rank)
    e_axis_mean, _e_axis_grad, e_nonaxis_mean = axis_nonaxis_vectors(
        all_clean, rf, is_axis_names, is_nonaxis_names
    )
    geom = {"P_N": P_N, "P_A": P_A, "e_axis": e_axis_mean, "e_nonaxis": e_nonaxis_mean}

    # ---- envelopes (P3 + I1, rows ordered by all_clean) ----
    env0 = load_subject_phase_envelopes(ds_sid, cohort, cfg, ["P3", "I1"], onset_shift=0.0, cls=cc)
    envs_p3 = [sz["phases"]["P3"] for sz in env0["seizures"] if "P3" in sz["phases"]]
    envs_i1 = [sz["phases"]["I1"] for sz in env0["seizures"] if "I1" in sz["phases"]]
    p3_ids = {sz["idx"] for sz in env0["seizures"] if "P3" in sz["phases"]}
    i1_ids = {sz["idx"] for sz in env0["seizures"] if "I1" in sz["phases"]}
    n_seizures = len(p3_ids | i1_ids)

    agg_p3, cache_p3, cv_p3, logm_p3 = _observed(envs_p3, geom, cfg)
    agg_i1, cache_i1, cv_i1, logm_i1 = _observed(envs_i1, geom, cfg)
    obs_delta = agg_i1["ms_density"] - agg_p3["ms_density"]

    row.update({
        "status": "ok", "skip_reason": "",
        "mode_shift_density_P3": agg_p3["ms_density"], "mode_shift_density_I1": agg_i1["ms_density"],
        "delta_mode_shift_density": obs_delta,
        "n_ch_fit": len(all_clean), "n_seizures": n_seizures,
    })
    if not np.isfinite(obs_delta):
        row.update({"status": "skipped", "skip_reason": "nonfinite_mode_shift"})
        print(f"[warn] {ds_sid} ({cohort}): geometry ok but no paired P3/I1 mode-shift (obs_delta NaN)", flush=True)
        return row

    # ---- per-window helpers reused by the refit nulls / onset jitter ----
    def ms_window(Xw):
        B_r, U_r = lowrank_var(Xw, rank, alpha)
        u_c = map_lowrank_vector_to_contacts(dominant_right_singular_vector(B_r, kstar), U_r)
        return subspace_mode_shift(u_c, P_N, P_A, "density")

    # ---- Δ-null distributions (three nulls; delta formed INSIDE each perm) ----
    def phase_perm(rng):
        # one phase-randomized surrogate per window -> BOTH the low-rank
        # mode-shift and the 2D spectral radius (lambda_surplus reuses these).
        def fn(Xw):
            Xs = phase_randomize_surrogate(Xw, rng)
            B_r, U_r = lowrank_var(Xs, rank, alpha)
            u_c = map_lowrank_vector_to_contacts(dominant_right_singular_vector(B_r, kstar), U_r)
            ms = subspace_mode_shift(u_c, P_N, P_A, "density")
            B2 = direct_2d_var(project_2d(demean_window(Xs), e_axis_mean, e_nonaxis_mean), alpha)
            return (ms, spectral_radius(B2))
        ms_p3, lam_p3 = _agg_windows(envs_p3, fn, 2, hop, win_sec, step_sec)
        ms_i1, lam_i1 = _agg_windows(envs_i1, fn, 2, hop, win_sec, step_sec)
        return (ms_i1 - ms_p3, lam_p3, lam_i1)

    def block_perm(rng):
        def fn(Xw):
            return (ms_window(block_shuffle_surrogate(Xw, block_n, rng)),)
        (ms_p3,) = _agg_windows(envs_p3, fn, 1, hop, win_sec, step_sec)
        (ms_i1,) = _agg_windows(envs_i1, fn, 1, hop, win_sec, step_sec)
        return (ms_i1 - ms_p3,)

    def label_perm(rng):
        # NO refit: keep observed u_c, permute labels within shaft (same A/N
        # counts) -> new projectors -> recompute density mode-shift.
        new_axis, new_nonaxis = label_permute(is_axis_names, is_nonaxis_names, shaft_by_name, rng)
        P_A2, P_N2 = subspace_projectors(all_clean, new_axis, new_nonaxis)
        g = lambda u: subspace_mode_shift(u, P_N2, P_A2, "density")  # noqa: E731
        return (_agg_cached(cache_i1, g) - _agg_cached(cache_p3, g),)

    delta_phase, lam_p3_perm, lam_i1_perm = _perm_arrays(phase_perm, n_perm, seed, 3)
    (delta_block,) = _perm_arrays(block_perm, n_perm, seed, 1)
    (delta_label,) = _perm_arrays(label_perm, n_perm, seed, 1)

    p_phase = _p_upper(obs_delta, delta_phase)
    p_block = _p_upper(obs_delta, delta_block)
    p_label = _p_upper(obs_delta, delta_label)

    med_l = float(np.nanmedian(delta_label))
    mad_l = float(np.nanmedian(np.abs(delta_label - med_l)))
    mode_shift_label_z = (
        float((obs_delta - med_l) / mad_l) if np.isfinite(mad_l) and mad_l > 0 else float("nan")
    )

    # ---- descriptive: raw + 2D-consistency mode-shift deltas ----
    mode_shift_raw_delta = agg_i1["ms_raw"] - agg_p3["ms_raw"]
    mode_shift_2D_consistency = agg_i1["ms_2d"] - agg_p3["ms_2d"]

    # ---- lambda_surplus (NEVER raw lambda) + axis/non-axis 2D gains ----
    lambda_surplus_P3 = agg_p3["lambda2d"] - float(np.nanmedian(lam_p3_perm))
    lambda_surplus_I1 = agg_i1["lambda2d"] - float(np.nanmedian(lam_i1_perm))
    gain_axis_delta = agg_i1["gain_axis"] - agg_p3["gain_axis"]
    gain_nonaxis_delta = agg_i1["gain_nonaxis"] - agg_p3["gain_nonaxis"]

    # ---- continuous-reactivity quality flags (descriptive) ----
    logm_all = logm_p3 + logm_i1
    logm_quality_flag = bool(logm_all and (float(np.mean(logm_all)) >= 0.5))
    reactivity_cont_available = bool(any(logm_all))

    # ---- single-contact energy (median across I1 windows; u_c² sign-invariant) ----
    def top_frac(u):
        e = np.asarray(u, dtype=float) ** 2
        s = float(e.sum())
        return float(e.max() / s) if s > 0 else float("nan")
    top_contact_energy_fraction = _agg_cached(cache_i1, top_frac)
    single_contact_driven = bool(
        np.isfinite(top_contact_energy_fraction) and top_contact_energy_fraction > frac_max
    )

    # ---- leave-one-contact: reuse cached u_c (NO refit), zero one component +
    # renormalize, recompute the FULL aggregate delta; sign must survive all drops.
    def drop_delta(c):
        def g(u):
            v = np.asarray(u, dtype=float).copy()
            v[c] = 0.0
            nv = float(np.linalg.norm(v))
            if nv > 0:
                v = v / nv
            return subspace_mode_shift(v, P_N, P_A, "density")
        return _agg_cached(cache_i1, g) - _agg_cached(cache_p3, g)
    sgn = np.sign(obs_delta)
    drops = [drop_delta(c) for c in range(len(all_clean))]
    leave_one_pass = bool(sgn != 0 and all(np.isfinite(d) and np.sign(d) == sgn for d in drops))

    # ---- axis-only control: relabel all non-axis -> axis (P_N empty, rank 0 ->
    # density N-term -> 0), recompute the delta. NEAR-TRIVIAL-BY-CONSTRUCTION
    # (flagged in report): the mode-shift loses its non-axis half. ----
    axis_all = sorted(set(is_axis_names) | set(is_nonaxis_names))
    P_A_all, P_N_empty = subspace_projectors(all_clean, axis_all, [])
    h = lambda u: subspace_mode_shift(u, P_N_empty, P_A_all, "density")  # noqa: E731
    axis_only_mode_shift_control = _agg_cached(cache_i1, h) - _agg_cached(cache_p3, h)
    axis_only_control_pass = bool(
        np.isfinite(axis_only_mode_shift_control) and axis_only_mode_shift_control < obs_delta
    )

    # ---- onset jitter +-10 s: reload windows at shifted anchors (i1_eligible
    # gate stays at shift 0 inside the loader), recompute obs_delta, require
    # sign stability. cls reused to skip the expensive context/lagPat reload. ----
    def obs_delta_at(shift):
        env = load_subject_phase_envelopes(ds_sid, cohort, cfg, ["P3", "I1"], onset_shift=shift, cls=cc)
        p3s = [sz["phases"]["P3"] for sz in env["seizures"] if "P3" in sz["phases"]]
        i1s = [sz["phases"]["I1"] for sz in env["seizures"] if "I1" in sz["phases"]]
        (m3,) = _agg_windows(p3s, lambda Xw: (ms_window(Xw),), 1, hop, win_sec, step_sec)
        (m1,) = _agg_windows(i1s, lambda Xw: (ms_window(Xw),), 1, hop, win_sec, step_sec)
        return m1 - m3
    d_p10 = obs_delta_at(10.0)
    d_m10 = obs_delta_at(-10.0)
    onset_jitter_pass = bool(np.sign(d_p10) == np.sign(obs_delta) == np.sign(d_m10))

    # ---- full-D VAR meaningfulness (median over all windows, both phases) ----
    cv_vals = [x for x in (cv_p3 + cv_i1) if np.isfinite(x)]
    cv_r2 = float(np.median(cv_vals)) if cv_vals else float("nan")
    var_meaningful_flag = bool(np.isfinite(cv_r2) and cv_r2 > 0)

    module_direction = bool(obs_delta > 0)
    module_null_pass = bool(p_phase < alpha_stat and p_block < alpha_stat and p_label < alpha_stat)
    module_support = bool(module_direction and module_null_pass)

    row.update({
        "mode_shift_raw_delta": mode_shift_raw_delta,
        "mode_shift_2D_consistency": mode_shift_2D_consistency,
        "p_phase": p_phase, "p_block": p_block, "p_label": p_label,
        "mode_shift_label_z": mode_shift_label_z,
        "lambda_surplus_P3": lambda_surplus_P3, "lambda_surplus_I1": lambda_surplus_I1,
        "gain_axis_delta": gain_axis_delta, "gain_nonaxis_delta": gain_nonaxis_delta,
        "reactivity_cont_available": reactivity_cont_available, "logm_quality_flag": logm_quality_flag,
        "top_contact_energy_fraction": top_contact_energy_fraction,
        "single_contact_driven": single_contact_driven,
        "leave_one_contact_mode_shift_pass": leave_one_pass,
        "axis_only_mode_shift_control": float(axis_only_mode_shift_control),
        "axis_only_control_pass": axis_only_control_pass, "onset_jitter_pass": onset_jitter_pass,
        "cv_r2": cv_r2, "var_meaningful_flag": var_meaningful_flag,
        "module_support_flag": module_support, "module_direction_correct": module_direction,
        "module_null_pass": module_null_pass,
    })
    return row


def run_subject(ds_sid: str, cohort: str, cfg: dict, n_perm: int) -> dict:
    """One CSV row. geometry_insufficient / load failure -> skipped (full row)."""
    row = _base_row(ds_sid, cohort, cfg)
    try:
        cc = classify_subject_contacts(ds_sid, cohort, cfg)
    except Exception as exc:  # noqa: BLE001 - external mount; never drop a subject
        row["skip_reason"] = f"error:{type(exc).__name__}:{exc}"
        print(f"[skip] {ds_sid} ({cohort}): {row['skip_reason']}", flush=True)
        return row

    row["geometry_sufficient"] = bool(cc["geometry_sufficient"])
    if not cc["geometry_sufficient"]:
        row["skip_reason"] = cc["geometry_reason"]
        print(f"[skip] {ds_sid} ({cohort}): geometry_insufficient:{cc['geometry_reason']}", flush=True)
        return row

    try:
        return _run_ok_subject(ds_sid, cohort, cfg, cc, n_perm, row)
    except Exception as exc:  # noqa: BLE001 - a compute failure still yields a row
        row["status"] = "skipped"
        row["skip_reason"] = f"compute_error:{type(exc).__name__}:{exc}"
        print(f"[skip] {ds_sid} ({cohort}): {row['skip_reason']}", flush=True)
        return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["narrow", "broad"], required=True)
    ap.add_argument("--n-perm", type=int, default=None, help="default cfg nulls.n_perm_final (1000)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--subjects", nargs="*", default=None, help="optional subset (default = whole cohort)")
    args = ap.parse_args()

    cfg = load_v3_config()
    n_perm = args.n_perm if args.n_perm is not None else int(cfg["nulls"]["n_perm_final"])
    outdir = (
        Path(args.outdir) if args.outdir
        else _ROOT / "results/topic5_ictal_recruitment/v3_mode_transition" / args.cohort
    )
    outdir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or SUBJECTS_BY_SUB[args.cohort]
    rows = [run_subject(ds_sid, args.cohort, cfg, n_perm) for ds_sid in subjects]

    out_csv = outdir / "v3_dynamics_subject.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)

    n_support = sum(1 for r in rows if r["module_support_flag"])
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(
        f"[done] {len(rows)} subjects ({n_ok} ok) -> {out_csv} "
        f"(n_perm={n_perm}; {n_support} module_support_flag=True)",
        flush=True,
    )


if __name__ == "__main__":
    main()

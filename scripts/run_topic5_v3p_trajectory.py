#!/usr/bin/env python
"""Topic 5 V3p — preictal-only non-axial TRAJECTORY runner (Task 7, co-primary H3p-b + H3p-c).

Plain-language question (EXPLORATORY): in the two minutes BEFORE eeg-onset
(P0..P3, -120..-10 s), does the avalanche flow OFF the fixed interictal HFO
axis (H3p-b ``net_offaxis_flux``) and the dominant dynamical direction's tilt
off-axis (H3p-c ``mode_shift_density``) *gradually RISE* toward onset? We slide
short windows across the clean preictal span, compute each metric per window,
fit a per-seizure Theil-Sen SLOPE over (metric, t_center), take the median over
a subject's seizures, and ask whether that rising slope is larger than a
shaft-constrained label-null would give -- and specifically whether the rise
concentrates on the TRUE non-axis contacts (label-null adjudicated) rather than
being a global warm-up.

This ADAPTS the frozen V3a DELTA-null runners (P3-vs-I1 ``delta``) to a
preictal-window SLOPE:
  - V3a's ``_delta_null_distribution`` (delta formed inside each perm) becomes
    Task-6 ``null_slope_distribution`` (a per-seizure trajectory resampled inside
    each perm, then sloped).
  - V3a's per-phase ATM/``u_c`` precompute-once-reuse-in-nulls becomes a
    per-(seizure, window) precompute: ``active``/``atm1``/``atm0``/``u_c`` are
    computed ONCE per window; the cheap nulls (label/rate/spatial) recompute only
    the perturbed part; the expensive mode-shift nulls (phase/block) DO refit the
    low-rank VAR per draw (that is the point of a spectral/temporal surrogate).

CO-PRIMARY (Holm across the two, done in Task 9 -- NOT here):
  - H3p-b flux slope: LABEL null is PRIMARY (surplus/z/p_label); rate + spatial +
    lag1-specific are HARD/descriptive gates. ``module_support_flag_b`` needs
    direction ∧ p_label(full) ∧ p_label(guard) ∧ p_rate ∧ lag1_specific>0.
  - H3p-c mode-shift slope: LABEL null is PRIMARY; phase + block are HARD gates.
    ``h3c_support_grade`` = strong (label full+guard ∧ phase ∧ block) / weak
    (label ∧ one temporal) / none; ``module_support_flag_c`` needs direction ∧
    strong.

TWO SPANS (rev1): every co-primary slope + the LABEL null run on BOTH ``full``
([-120,-10]) and ``guard`` ([-120,-20]); ``_guard`` columns carry the guard
span; ``near_onset_dependent_{b,c}`` flags a signal that survives on full but not
guard (it leans on the last ~10 s before onset). rate/spatial/phase/block run on
``full`` only (the guard gate is on the label null). ``proximal`` ([-60,-10]) is
a sensitivity slope only.

narrow (primary) and broad (replication) run SEPARATELY, never pooled. A
``geometry_insufficient`` subject or one with ``< min_windows_for_slope`` preictal
windows in every seizure is FLAGGED (``status=skipped``) with a full NaN/False
row -- never silently dropped. NO ``tier`` column (that is Task 9). Exploratory;
no forecasting. See docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md
Task 7.
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

from scripts._topic5_v3_io import CACHE, classify_subject_contacts  # noqa: E402
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from scripts.run_topic5_v3p_feasibility import _label_permutations_est  # noqa: E402
from src.topic5_v2_criticality import (  # noqa: E402
    activations_from_z,
    block_shuffle_surrogate,
    cv_one_step_r2,
    phase_randomize_surrogate,
)
from src.topic5_v3_mode_transition import (  # noqa: E402
    atm_offdiag,
    axis_nonaxis_vectors,
    demean_window,
    direct_2d_var,
    dominant_right_singular_vector,
    label_permute,
    load_v3_config,
    lowrank_var,
    map_lowrank_vector_to_contacts,
    net_offaxis_flux,
    phase_bin_range,
    project_2d,
    rank_forward,
    rate_preserving_shuffle,
    shaft_constrained_permute,
    sliding_windows,
    subspace_mode_shift,
    subspace_projectors,
)
from src.topic5_v3p_preictal_trajectory import (  # noqa: E402
    extract_window_metrics,
    load_v3p_config,
    null_slope_distribution,
    residualize_slope,
    slope_over_windows,
    surplus_and_p,
)

PREICTAL_PHASES = ["P0", "P1", "P2", "P3"]

# Subject CSV column contract (brief Task 7; co-primary + rev1 additions; NO
# `tier` -- Task 9 only). Order matches the brief.
CSV_COLS = [
    # --- base / provenance ---
    "subject", "cohort", "in_broad_core", "status", "skip_reason", "geometry_sufficient",
    "n_axis", "n_nonaxis", "n_ambiguous", "n_seizures_used",
    # --- H3p-b flux slope ---
    "net_offaxis_flux_slope_raw", "net_offaxis_flux_surplus_slope",
    "net_offaxis_flux_slope_resid", "net_offaxis_flux_slope_z",
    "p_label_slope_b", "p_rate_slope_b", "p_spatial_slope_b",
    "proximal_flux_slope", "spearman_rho_flux",
    "leave_one_contact_flux_pass", "axis_only_flux_control_pass", "onset_jitter_pass_b",
    "module_support_flag_b", "module_direction_correct_b", "module_null_pass_b",
    # --- H3p-c mode-shift slope ---
    "mode_shift_density_slope_raw", "mode_shift_density_surplus_slope",
    "mode_shift_density_slope_resid", "mode_shift_density_slope_z",
    "p_label_slope_c", "p_phase_slope_c", "p_block_slope_c",
    "mode_shift_2D_consistency_slope", "top_contact_energy_fraction", "single_contact_driven",
    "leave_one_contact_mode_pass", "axis_only_mode_control_pass", "onset_jitter_pass_c",
    "rank_used", "k_star", "spearman_rho_mode",
    "module_support_flag_c", "module_direction_correct_c", "module_null_pass_c",
    # --- rev1: guard-span companions (all LABEL-null based) ---
    "net_offaxis_flux_surplus_slope_guard", "net_offaxis_flux_slope_z_guard", "p_label_slope_b_guard",
    "mode_shift_density_surplus_slope_guard", "mode_shift_density_slope_z_guard", "p_label_slope_c_guard",
    # --- rev1: near-onset dependence + specificity + QC ---
    "near_onset_dependent_b", "near_onset_dependent_c",
    "lag1_specific_slope", "common_drive_sensitive",
    "mode_singular_gap_median", "mode_vector_stable", "cv_r2",
    "n_activation_events_pre", "n_active_windows_pre", "h3b_activation_sufficient",
    "h3c_support_grade", "time_order_p_b", "time_order_p_c",
    "n_label_permutable_shafts", "n_unique_label_permutations_est", "label_null_underpowered",
    "trend_estimator", "slope_span",
]

# Per-window diagnostic CSV (brief Task 7). One row per observed preictal window
# (span="full"; every preictal window is a full-span member -- guard/proximal are
# analytical sub-selections keyed on t_center, so the metric VALUES do not change
# by span and are not duplicated here).
WINDOW_COLS = [
    "subject", "cohort", "seizure_idx", "span", "phase", "t_center",
    "net_offaxis_flux_lag1", "net_offaxis_flux_lag0", "mode_shift_density", "mode_singular_gap",
    "nonaxis_activation_rate", "n_activation_events", "global_energy", "axial_energy",
    "N_self_sustain_lag1", "N_self_sustain_lag0", "gain_axis", "gain_nonaxis",
]

# Every non-base / non-constant column present with a NaN float / False flag /
# 0 count -- a skipped or degenerate subject still carries the full schema.
_METRIC_DEFAULTS = {
    "net_offaxis_flux_slope_raw": float("nan"), "net_offaxis_flux_surplus_slope": float("nan"),
    "net_offaxis_flux_slope_resid": float("nan"), "net_offaxis_flux_slope_z": float("nan"),
    "p_label_slope_b": float("nan"), "p_rate_slope_b": float("nan"), "p_spatial_slope_b": float("nan"),
    "proximal_flux_slope": float("nan"), "spearman_rho_flux": float("nan"),
    "leave_one_contact_flux_pass": False, "axis_only_flux_control_pass": False,
    "onset_jitter_pass_b": False, "module_support_flag_b": False,
    "module_direction_correct_b": False, "module_null_pass_b": False,
    "mode_shift_density_slope_raw": float("nan"), "mode_shift_density_surplus_slope": float("nan"),
    "mode_shift_density_slope_resid": float("nan"), "mode_shift_density_slope_z": float("nan"),
    "p_label_slope_c": float("nan"), "p_phase_slope_c": float("nan"), "p_block_slope_c": float("nan"),
    "mode_shift_2D_consistency_slope": float("nan"), "top_contact_energy_fraction": float("nan"),
    "single_contact_driven": False, "leave_one_contact_mode_pass": False,
    "axis_only_mode_control_pass": False, "onset_jitter_pass_c": False,
    "spearman_rho_mode": float("nan"),
    "module_support_flag_c": False, "module_direction_correct_c": False, "module_null_pass_c": False,
    "net_offaxis_flux_surplus_slope_guard": float("nan"), "net_offaxis_flux_slope_z_guard": float("nan"),
    "p_label_slope_b_guard": float("nan"),
    "mode_shift_density_surplus_slope_guard": float("nan"), "mode_shift_density_slope_z_guard": float("nan"),
    "p_label_slope_c_guard": float("nan"),
    "near_onset_dependent_b": False, "near_onset_dependent_c": False,
    "lag1_specific_slope": float("nan"), "common_drive_sensitive": False,
    "mode_singular_gap_median": float("nan"), "mode_vector_stable": False, "cv_r2": float("nan"),
    "n_activation_events_pre": 0, "n_active_windows_pre": 0, "h3b_activation_sufficient": False,
    "h3c_support_grade": "none", "time_order_p_b": float("nan"), "time_order_p_c": float("nan"),
    "n_label_permutable_shafts": 0, "n_unique_label_permutations_est": float("nan"),
    "label_null_underpowered": True,
}


def _base_row(subj: str, cohort: str, in_broad_core: bool, v3cfg: dict, v3pcfg: dict) -> dict:
    """Full-schema row for a subject; constants (rank/k*/estimator/span) filled unconditionally.

    ``subject`` is the FULL ds_sid (e.g. ``epilepsiae_253``) to key-match the
    feasibility CSV + config ``broad_core``/``candidates_epilepsiae`` (Task 9 join).
    ``in_broad_core`` = membership in config ``cohort_expansion.broad_core`` (the
    curated 9), evaluated identically for both cohorts so Task 9 can report
    ``broad_core`` alongside ``broad_expanded`` (tier-4 needs the direction to hold
    on the core). ``slope_span="full+guard"`` is provenance: BOTH spans are computed
    (headline = full; the ``_guard`` columns carry the onset-guard span).
    """
    return {
        "subject": subj, "cohort": cohort, "in_broad_core": bool(in_broad_core),
        "status": "skipped", "skip_reason": "",
        "geometry_sufficient": False, "n_axis": 0, "n_nonaxis": 0, "n_ambiguous": 0,
        "n_seizures_used": 0,
        "rank_used": int(v3cfg["dynamics"]["lowrank"]),
        "k_star": int(v3cfg["dynamics"]["finite_horizon_k"]),
        "trend_estimator": v3pcfg["trend"]["estimator"],
        "slope_span": "full+guard",
        **_METRIC_DEFAULTS,
    }


# ---------------------------------------------------------------------------
# geometry (once per subject) + window collection (real relt -> t_center axis)
# ---------------------------------------------------------------------------
def _build_geom(cc: dict) -> dict:
    """Subspace projectors + axis/non-axis mean vectors + rank_forward + index maps.

    ``rank_forward`` is read exactly as the feasibility pilot does (finite
    ``typical_rank`` over template-A channels); it only feeds ``e_axis_grad``
    (unused by any Task-7 output) and ``beta_axis_strength`` (H3p-a supportive,
    written to the window detail).
    """
    names = list(cc["all_clean"])
    axis_names = list(cc["is_axis"])
    nonaxis_names = list(cc["is_nonaxis_strict"])
    name_pos = {n: i for i, n in enumerate(names)}
    axis_set, nonaxis_set = set(axis_names), set(nonaxis_names)
    axis_idx = np.array([i for i, n in enumerate(names) if n in axis_set], dtype=int)
    nonaxis_idx = np.array([i for i, n in enumerate(names) if n in nonaxis_set], dtype=int)
    P_A, P_N = subspace_projectors(names, axis_names, nonaxis_names)
    ta_rank = {
        c["name"]: float(c["typical_rank"])
        for c in cc["ctx"]["ta"]["channels"]
        if np.isfinite(c.get("typical_rank", np.nan))
    }
    rf = rank_forward(ta_rank)
    e_axis_mean, _e_axis_grad, e_nonaxis_mean = axis_nonaxis_vectors(names, rf, axis_names, nonaxis_names)
    return {
        "names": names, "axis_names": axis_names, "nonaxis_names": nonaxis_names,
        "shaft_by_name": cc["shaft_by_name"], "name_pos": name_pos,
        "axis_idx": axis_idx, "nonaxis_idx": nonaxis_idx,
        "P_A": P_A, "P_N": P_N,
        "e_axis_mean": e_axis_mean, "e_nonaxis_mean": e_nonaxis_mean, "rank_forward": rf,
    }


def _collect_seizures(ds_sid, cc, geom, v3cfg, v3pcfg, onset_shift=0.0):
    """Per-seizure preictal windows on the REAL relt time axis.

    For each eligible seizure and each preictal phase P0..P3, slide the frozen
    ``sliding_windows`` (real relt, V3a window contract) and record each window's
    ``t_center = mean(relt[ws:we]) - onset`` (the slope's x-axis, anchored on the
    UNSHIFTED onset even under jitter so a shift only moves which windows land in
    the span, not the axis). Each window carries its precomputed per-window atoms
    (``extract_window_metrics`` for the 13 metrics + ``active``/``atm1``/``u_c`` for
    the label/rate/spatial/leave-one nulls + ``ms_2d``/``cv``) -- the observed pass,
    the nulls, and the onset-jitter reloads all consume them. Only seizures with
    >= ``min_windows_for_slope`` FULL-span preictal windows are kept. Returns
    ``(seizures, dt)`` where ``dt`` is the real median relt spacing (drives
    ``block_n`` for the block-shuffle null).
    """
    z_thr = float(v3cfg["avalanche"]["z_threshold"])
    lowrank = int(v3cfg["dynamics"]["lowrank"])
    kstar = int(v3cfg["dynamics"]["finite_horizon_k"])
    alpha = float(v3cfg["dynamics"]["var_ridge_alpha"])
    win_sec = float(v3cfg["phases"]["window_sec"])
    step_sec = float(v3cfg["phases"]["step_sec"])
    min_windows = int(v3pcfg["preictal"]["min_windows_for_slope"])
    full_lo, full_hi = v3pcfg["preictal"]["span_full_rel"]
    names = geom["names"]
    n = len(names)
    P_A, P_N = geom["P_A"], geom["P_N"]
    e_axis_mean, e_nonaxis_mean = geom["e_axis_mean"], geom["e_nonaxis_mean"]

    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    meta = cc["meta"]
    name_to_row = {nm: i for i, nm in enumerate(cc["cache_names"])}
    all_clean_rows = np.array([name_to_row[nm] for nm in names], dtype=int)

    seizures: list = []
    dt = float(v3cfg["phases"]["hop_sec"])  # fallback if no seizure yields a relt
    dt_set = False
    for si in meta.get("eligible_idxs", []):
        zt_key = f"bb_zt__{si}"
        relt_key = f"bb_relt__{si}"
        sz = meta.get("seizure", {}).get(str(si))
        if zt_key not in data.files or relt_key not in data.files or sz is None:
            continue
        onset = float(sz["eeg_onset_rel"])
        offset = float(sz["eeg_offset_rel"])
        dur = float(sz["eeg_duration_sec"])
        relt = np.asarray(data[relt_key], dtype=float)
        if not dt_set and relt.size >= 2:
            d = float(np.median(np.diff(relt)))
            if np.isfinite(d) and d > 0:
                dt, dt_set = d, True
        bb_zt = np.asarray(data[zt_key], dtype=float)[all_clean_rows]  # rows ordered by all_clean

        windows: list = []
        for phase in PREICTAL_PHASES:
            rng_idx = phase_bin_range(relt, onset, offset, dur, phase, v3cfg, onset_shift)
            if rng_idx is None:
                continue
            for (ws, we) in sliding_windows(relt, rng_idx[0], rng_idx[1], win_sec, step_sec):
                env_w = bb_zt[:, ws:we]
                t_center = float(np.mean(relt[ws:we])) - onset
                wd = {"ws": int(ws), "we": int(we), "t_center": t_center, "phase": phase}
                wd["m"] = extract_window_metrics(env_w, geom, v3cfg)
                # per-window null atoms (degenerate-safe: a bad window leaves
                # zeros, contributing 0 to nulls rather than crashing the loop).
                # lag0 flux is read from ``m`` (extract_window_metrics owns its
                # own atm_lag0), so only the lag1 ATM is materialized here.
                try:
                    active = activations_from_z(env_w, z_thr)
                    atm1 = atm_offdiag(active)
                except Exception:
                    active = np.zeros((n, max(1, we - ws)), dtype=bool)
                    atm1 = np.zeros((n, n))
                try:
                    A_lr, U_r = lowrank_var(env_w, lowrank, alpha)
                    u_c = map_lowrank_vector_to_contacts(
                        dominant_right_singular_vector(A_lr, kstar), U_r
                    )
                except Exception:
                    u_c = np.zeros(n)
                try:
                    B2 = direct_2d_var(project_2d(demean_window(env_w), e_axis_mean, e_nonaxis_mean), alpha)
                    u2 = dominant_right_singular_vector(B2, kstar)
                    ms_2d = float(u2[1] ** 2 - u2[0] ** 2)
                except Exception:
                    ms_2d = float("nan")
                try:
                    cv = float(cv_one_step_r2(demean_window(env_w), alpha, 5))
                except Exception:
                    cv = float("nan")
                wd.update({"active": active, "atm1": atm1,
                           "u_c": u_c, "ms_2d": ms_2d, "cv": cv})
                windows.append(wd)

        n_full = sum(1 for w in windows if full_lo <= w["t_center"] <= full_hi)
        if n_full >= min_windows:
            seizures.append({"idx": int(si), "bb_zt": bb_zt, "windows": windows})
    return seizures, dt


# ---------------------------------------------------------------------------
# slope aggregation over seizures (median of per-seizure Theil-Sen slopes)
# ---------------------------------------------------------------------------
def _flux_get(w):
    return w["m"]["net_offaxis_flux_lag1"]


def _mode_get(w):
    return w["m"]["mode_shift_density"]


def _lag1spec_get(w):
    return w["m"]["net_offaxis_flux_lag1"] - w["m"]["net_offaxis_flux_lag0"]


def _ms2d_get(w):
    return w["ms_2d"]


def _slope_from(seizures, getter, span, est="theil_sen"):
    """Median over seizures of each seizure's per-window trend slope in ``span``."""
    lo, hi = span
    per_sz = []
    for sz in seizures:
        vals, centers = [], []
        for w in sz["windows"]:
            if lo <= w["t_center"] <= hi:
                vals.append(getter(w))
                centers.append(w["t_center"])
        s = slope_over_windows(vals, centers, est)
        if np.isfinite(s):
            per_sz.append(s)
    return float(np.median(per_sz)) if per_sz else float("nan")


def _resid_slope_from(seizures, getter, span):
    """Median over seizures of the global+axial-energy-residualized trend slope."""
    lo, hi = span
    per_sz = []
    for sz in seizures:
        vals, centers, glob, axial = [], [], [], []
        for w in sz["windows"]:
            if lo <= w["t_center"] <= hi:
                vals.append(getter(w))
                centers.append(w["t_center"])
                glob.append(w["m"]["global_energy"])
                axial.append(w["m"]["axial_energy"])
        s = residualize_slope(vals, centers, [glob, axial], "theil_sen")
        if np.isfinite(s):
            per_sz.append(s)
    return float(np.median(per_sz)) if per_sz else float("nan")


# ---------------------------------------------------------------------------
# the full metric block for a geometry-sufficient subject
# ---------------------------------------------------------------------------
def _run_ok_subject(ds_sid, cohort, cc, geom, v3cfg, v3pcfg, n_perm, row, window_rows):
    names = geom["names"]
    axis_names, nonaxis_names = geom["axis_names"], geom["nonaxis_names"]
    shaft_by_name, name_pos = geom["shaft_by_name"], geom["name_pos"]
    axis_idx, nonaxis_idx = geom["axis_idx"], geom["nonaxis_idx"]
    P_A, P_N = geom["P_A"], geom["P_N"]
    lowrank = int(v3cfg["dynamics"]["lowrank"])
    kstar = int(v3cfg["dynamics"]["finite_horizon_k"])
    alpha = float(v3cfg["dynamics"]["var_ridge_alpha"])
    seed = int(v3pcfg["nulls"]["seed"])
    alpha_stat = float(v3pcfg["nulls"]["alpha"])
    gap_min = float(v3pcfg["nulls_v3p"]["mode_singular_gap_min"])
    h3b_min_act = int(v3pcfg["nulls_v3p"]["h3b_min_activation_events"])
    label_min_perms = float(v3pcfg["nulls_v3p"]["label_null_min_unique_perms"])
    frac_max = float(v3cfg["statistics"]["single_contact_energy_frac_max"])
    FULL = tuple(v3pcfg["preictal"]["span_full_rel"])
    GUARD = tuple(v3pcfg["preictal"]["span_guard_rel"])
    PROX = tuple(v3pcfg["preictal"]["span_proximal_rel"])

    seizures, dt = _collect_seizures(ds_sid, cc, geom, v3cfg, v3pcfg, onset_shift=0.0)
    n_used = len(seizures)
    row["n_seizures_used"] = n_used
    if n_used == 0:
        row.update({"status": "skipped", "skip_reason": "insufficient_preictal_windows"})
        print(f"[skip] {ds_sid} ({cohort}): no seizure with >= min_windows preictal windows", flush=True)
        return row
    block_n = max(1, int(round(float(v3cfg["dynamics"]["block_len_sec"]) / dt)))

    # ---- window-detail rows (span="full"; every preictal window is full-span) ----
    subj = row["subject"]  # full ds_sid
    for sz in seizures:
        for w in sz["windows"]:
            m = w["m"]
            window_rows.append({
                "subject": subj, "cohort": cohort, "seizure_idx": sz["idx"],
                "span": "full", "phase": w["phase"], "t_center": w["t_center"],
                "net_offaxis_flux_lag1": m["net_offaxis_flux_lag1"],
                "net_offaxis_flux_lag0": m["net_offaxis_flux_lag0"],
                "mode_shift_density": m["mode_shift_density"], "mode_singular_gap": m["mode_singular_gap"],
                "nonaxis_activation_rate": m["nonaxis_activation_rate"],
                "n_activation_events": m["n_activation_events"],
                "global_energy": m["global_energy"], "axial_energy": m["axial_energy"],
                "N_self_sustain_lag1": m["N_self_sustain_lag1"], "N_self_sustain_lag0": m["N_self_sustain_lag0"],
                "gain_axis": m["gain_axis"], "gain_nonaxis": m["gain_nonaxis"],
            })

    # ---- observed co-primary slopes (full/guard/proximal) ----
    obs_flux_full = _slope_from(seizures, _flux_get, FULL)
    obs_flux_guard = _slope_from(seizures, _flux_get, GUARD)
    obs_flux_prox = _slope_from(seizures, _flux_get, PROX)
    obs_mode_full = _slope_from(seizures, _mode_get, FULL)
    obs_mode_guard = _slope_from(seizures, _mode_get, GUARD)

    # ---- null resample closures (Task-6 null_slope_distribution callbacks) ----
    def label_resample(span, metric):
        lo, hi = span

        def resample(rng):
            na, nn = label_permute(axis_names, nonaxis_names, shaft_by_name, rng)
            ai = np.array([name_pos[x] for x in na], dtype=int)
            ni = np.array([name_pos[x] for x in nn], dtype=int)
            PA2, PN2 = subspace_projectors(names, na, nn) if metric == "mode" else (None, None)
            out = []
            for sz in seizures:
                vals, centers = [], []
                for w in sz["windows"]:
                    if lo <= w["t_center"] <= hi:
                        if metric == "flux":
                            v = net_offaxis_flux(w["atm1"], ai, ni, "source_mean")
                        else:
                            v = subspace_mode_shift(w["u_c"], PN2, PA2, "density")
                        vals.append(v)
                        centers.append(w["t_center"])
                out.append((vals, centers))
            return out
        return resample

    def rate_resample(span):
        lo, hi = span

        def resample(rng):
            out = []
            for sz in seizures:
                vals, centers = [], []
                for w in sz["windows"]:
                    if lo <= w["t_center"] <= hi:
                        a = rate_preserving_shuffle(w["active"], rng)  # per-window rate preserved
                        v = net_offaxis_flux(atm_offdiag(a), axis_idx, nonaxis_idx, "source_mean")
                        vals.append(v)
                        centers.append(w["t_center"])
                out.append((vals, centers))
            return out
        return resample

    def spatial_resample(span):
        lo, hi = span

        def resample(rng):
            permuted = shaft_constrained_permute(name_pos, shaft_by_name, rng)
            perm_rows = np.array([permuted[x] for x in names], dtype=int)
            out = []
            for sz in seizures:
                vals, centers = [], []
                for w in sz["windows"]:
                    if lo <= w["t_center"] <= hi:
                        v = net_offaxis_flux(atm_offdiag(w["active"][perm_rows]),
                                             axis_idx, nonaxis_idx, "source_mean")
                        vals.append(v)
                        centers.append(w["t_center"])
                out.append((vals, centers))
            return out
        return resample

    def refit_resample(span, surrogate):
        lo, hi = span

        def resample(rng):
            out = []
            for sz in seizures:
                vals, centers = [], []
                for w in sz["windows"]:
                    if lo <= w["t_center"] <= hi:
                        env_w = sz["bb_zt"][:, w["ws"]:w["we"]]
                        Xs = surrogate(env_w, rng)
                        try:
                            A_lr, U_r = lowrank_var(Xs, lowrank, alpha)
                            uc = map_lowrank_vector_to_contacts(
                                dominant_right_singular_vector(A_lr, kstar), U_r
                            )
                            v = subspace_mode_shift(uc, P_N, P_A, "density")
                        except Exception:
                            v = float("nan")
                        vals.append(v)
                        centers.append(w["t_center"])
                out.append((vals, centers))
            return out
        return resample

    def time_order_resample(getter, span):
        lo, hi = span

        def resample(rng):
            out = []
            for sz in seizures:
                vals, centers = [], []
                for w in sz["windows"]:
                    if lo <= w["t_center"] <= hi:
                        vals.append(getter(w))
                        centers.append(w["t_center"])
                if len(centers) >= 2:  # break the time<->value pairing
                    centers = list(np.asarray(centers, dtype=float)[rng.permutation(len(centers))])
                out.append((vals, centers))
            return out
        return resample

    def run_null(resample, obs, seed_off):
        rng = np.random.default_rng(seed + seed_off)
        null = null_slope_distribution(resample, "theil_sen", n_perm, rng)
        return surplus_and_p(obs, null, "greater")

    # ---- LABEL null (PRIMARY: surplus/z/p) on full + guard for both metrics ----
    res_flux_full = run_null(label_resample(FULL, "flux"), obs_flux_full, 1)
    res_flux_guard = run_null(label_resample(GUARD, "flux"), obs_flux_guard, 2)
    res_mode_full = run_null(label_resample(FULL, "mode"), obs_mode_full, 3)
    res_mode_guard = run_null(label_resample(GUARD, "mode"), obs_mode_guard, 4)
    # ---- HARD gates: rate + spatial (flux, full), phase + block (mode, full) ----
    p_rate_b = run_null(rate_resample(FULL), obs_flux_full, 5)["p"]
    p_spatial_b = run_null(spatial_resample(FULL), obs_flux_full, 6)["p"]
    p_phase_c = run_null(refit_resample(FULL, phase_randomize_surrogate), obs_mode_full, 7)["p"]
    p_block_c = run_null(
        refit_resample(FULL, lambda X, r: block_shuffle_surrogate(X, block_n, r)), obs_mode_full, 8
    )["p"]
    # ---- time-order null (secondary sensitivity, full) ----
    time_order_p_b = run_null(time_order_resample(_flux_get, FULL), obs_flux_full, 9)["p"]
    time_order_p_c = run_null(time_order_resample(_mode_get, FULL), obs_mode_full, 10)["p"]

    surplus_flux = res_flux_full["surplus"]
    surplus_mode = res_mode_full["surplus"]

    # ---- residualization (sensitivity), spearman companion, lag1-specific ----
    resid_flux = _resid_slope_from(seizures, _flux_get, FULL)
    resid_mode = _resid_slope_from(seizures, _mode_get, FULL)
    spearman_flux = _slope_from(seizures, _flux_get, FULL, est="spearman")
    spearman_mode = _slope_from(seizures, _mode_get, FULL, est="spearman")
    lag1_specific_slope = _slope_from(seizures, _lag1spec_get, FULL)
    common_drive_sensitive = bool(lag1_specific_slope <= 0)  # NaN -> False (NaN<=0 is False)
    ms_2d_consistency_slope = _slope_from(seizures, _ms2d_get, FULL)

    # ---- QC: singular gap / cv / activation events / single-contact energy ----
    full_windows = [w for sz in seizures for w in sz["windows"] if FULL[0] <= w["t_center"] <= FULL[1]]
    gaps = [w["m"]["mode_singular_gap"] for w in full_windows if np.isfinite(w["m"]["mode_singular_gap"])]
    mode_singular_gap_median = float(np.median(gaps)) if gaps else float("nan")
    mode_vector_stable = bool(np.isfinite(mode_singular_gap_median) and mode_singular_gap_median >= gap_min)
    cvs = [w["cv"] for w in full_windows if np.isfinite(w["cv"])]
    cv_r2 = float(np.median(cvs)) if cvs else float("nan")
    n_activation_events_pre = int(sum(int(w["m"]["n_activation_events"]) for w in full_windows))
    n_active_windows_pre = int(sum(1 for w in full_windows if int(w["m"]["n_activation_events"]) > 0))
    h3b_activation_sufficient = bool(n_activation_events_pre >= h3b_min_act)

    def _top_frac(u):
        e = np.asarray(u, dtype=float) ** 2
        s = float(e.sum())
        return float(e.max() / s) if s > 0 else float("nan")
    tfs = [_top_frac(w["u_c"]) for w in full_windows]
    tfs = [x for x in tfs if np.isfinite(x)]
    top_contact_energy_fraction = float(np.median(tfs)) if tfs else float("nan")
    single_contact_driven = bool(
        np.isfinite(top_contact_energy_fraction) and top_contact_energy_fraction > frac_max
    )

    # ---- label-null power QC (mirrors feasibility) ----
    n_label_permutable_shafts = int(cc["shafts_with_both"])
    n_unique_label_permutations_est = float(_label_permutations_est(cc))
    label_null_underpowered = bool(n_unique_label_permutations_est < label_min_perms)

    # ---- leave-one-contact: flux (recompute ATM per drop) / mode (zero u_c comp) ----
    def leave_one_flux_pass():
        sgn = np.sign(obs_flux_full)
        if sgn == 0 or not np.isfinite(obs_flux_full):
            return False
        nloc = len(names)
        for d in range(nloc):
            keep = np.ones(nloc, dtype=bool)
            keep[d] = False
            new_of_old = np.full(nloc, -1, dtype=int)
            new_of_old[keep] = np.arange(int(keep.sum()))
            d_axis = new_of_old[axis_idx[axis_idx != d]]
            d_nonaxis = new_of_old[nonaxis_idx[nonaxis_idx != d]]
            per_sz = []
            for sz in seizures:
                vals, centers = [], []
                for w in sz["windows"]:
                    if FULL[0] <= w["t_center"] <= FULL[1]:
                        vals.append(net_offaxis_flux(atm_offdiag(w["active"][keep]),
                                                     d_axis, d_nonaxis, "source_mean"))
                        centers.append(w["t_center"])
                s = slope_over_windows(vals, centers, "theil_sen")
                if np.isfinite(s):
                    per_sz.append(s)
            drop = float(np.median(per_sz)) if per_sz else float("nan")
            if not (np.isfinite(drop) and np.sign(drop) == sgn):
                return False
        return True

    def leave_one_mode_pass():
        sgn = np.sign(obs_mode_full)
        if sgn == 0 or not np.isfinite(obs_mode_full):
            return False
        for c in range(len(names)):
            per_sz = []
            for sz in seizures:
                vals, centers = [], []
                for w in sz["windows"]:
                    if FULL[0] <= w["t_center"] <= FULL[1]:
                        v = np.asarray(w["u_c"], dtype=float).copy()
                        v[c] = 0.0
                        nv = float(np.linalg.norm(v))
                        if nv > 0:
                            v = v / nv
                        vals.append(subspace_mode_shift(v, P_N, P_A, "density"))
                        centers.append(w["t_center"])
                s = slope_over_windows(vals, centers, "theil_sen")
                if np.isfinite(s):
                    per_sz.append(s)
            drop = float(np.median(per_sz)) if per_sz else float("nan")
            if not (np.isfinite(drop) and np.sign(drop) == sgn):
                return False
        return True

    leave_one_contact_flux_pass = leave_one_flux_pass()
    leave_one_contact_mode_pass = leave_one_mode_pass()

    # ---- axis-only controls (relabel all non-axis -> axis; near-trivial by
    # construction: flux net -> 0, mode-shift loses its non-axis half) ----
    axis_all = np.array(sorted(set(axis_idx.tolist()) | set(nonaxis_idx.tolist())), dtype=int)
    empty = np.array([], dtype=int)
    axis_only_flux_slope = _slope_from(
        seizures, lambda w: net_offaxis_flux(w["atm1"], axis_all, empty, "source_mean"), FULL
    )
    axis_only_flux_control_pass = bool(
        np.isfinite(surplus_flux) and axis_only_flux_slope < surplus_flux
    )
    axis_all_names = sorted(set(axis_names) | set(nonaxis_names))
    PA_all, PN_empty = subspace_projectors(names, axis_all_names, [])
    axis_only_mode_slope = _slope_from(
        seizures, lambda w: subspace_mode_shift(w["u_c"], PN_empty, PA_all, "density"), FULL
    )
    axis_only_mode_control_pass = bool(
        np.isfinite(surplus_mode) and axis_only_mode_slope < surplus_mode
    )

    # ---- onset jitter +-10 s: reload windows at shifted anchors, require the
    # observed slope's SIGN to survive (2 reloads shared by flux + mode) ----
    jitter_szs = {
        sh: _collect_seizures(ds_sid, cc, geom, v3cfg, v3pcfg, onset_shift=sh)[0]
        for sh in (10.0, -10.0)
    }

    def jitter_pass(obs_full, getter):
        sgn = np.sign(obs_full)
        if sgn == 0 or not np.isfinite(obs_full):
            return False
        for sh in (10.0, -10.0):
            s = _slope_from(jitter_szs[sh], getter, FULL)
            if not (np.isfinite(s) and np.sign(s) == sgn):
                return False
        return True

    onset_jitter_pass_b = jitter_pass(obs_flux_full, _flux_get)
    onset_jitter_pass_c = jitter_pass(obs_mode_full, _mode_get)

    # ---- module flags (rev1 hardened; unsuffixed p = full/headline span) ----
    def lt(p):
        return bool(np.isfinite(p) and p < alpha_stat)

    p_label_b, p_label_b_guard = res_flux_full["p"], res_flux_guard["p"]
    p_label_c, p_label_c_guard = res_mode_full["p"], res_mode_guard["p"]

    module_direction_correct_b = bool(np.isfinite(surplus_flux) and surplus_flux > 0)
    module_direction_correct_c = bool(np.isfinite(surplus_mode) and surplus_mode > 0)
    module_null_pass_b = bool(lt(p_label_b) and lt(p_label_b_guard) and lt(p_rate_b))
    module_support_flag_b = bool(
        module_direction_correct_b and module_null_pass_b
        and np.isfinite(lag1_specific_slope) and lag1_specific_slope > 0
    )

    if lt(p_label_c) and lt(p_label_c_guard) and lt(p_phase_c) and lt(p_block_c):
        h3c_support_grade = "strong"
    elif lt(p_label_c) and (lt(p_phase_c) or lt(p_block_c)):
        h3c_support_grade = "weak"
    else:
        h3c_support_grade = "none"
    module_null_pass_c = bool(h3c_support_grade == "strong")
    module_support_flag_c = bool(module_direction_correct_c and h3c_support_grade == "strong")

    near_onset_dependent_b = bool(lt(p_label_b) and not lt(p_label_b_guard))
    near_onset_dependent_c = bool(lt(p_label_c) and not lt(p_label_c_guard))

    row.update({
        "status": "ok", "skip_reason": "",
        # H3p-b
        "net_offaxis_flux_slope_raw": obs_flux_full,
        "net_offaxis_flux_surplus_slope": surplus_flux,
        "net_offaxis_flux_slope_resid": resid_flux,
        "net_offaxis_flux_slope_z": res_flux_full["z"],
        "p_label_slope_b": p_label_b, "p_rate_slope_b": p_rate_b, "p_spatial_slope_b": p_spatial_b,
        "proximal_flux_slope": obs_flux_prox, "spearman_rho_flux": spearman_flux,
        "leave_one_contact_flux_pass": leave_one_contact_flux_pass,
        "axis_only_flux_control_pass": axis_only_flux_control_pass,
        "onset_jitter_pass_b": onset_jitter_pass_b,
        "module_support_flag_b": module_support_flag_b,
        "module_direction_correct_b": module_direction_correct_b,
        "module_null_pass_b": module_null_pass_b,
        # H3p-c
        "mode_shift_density_slope_raw": obs_mode_full,
        "mode_shift_density_surplus_slope": surplus_mode,
        "mode_shift_density_slope_resid": resid_mode,
        "mode_shift_density_slope_z": res_mode_full["z"],
        "p_label_slope_c": p_label_c, "p_phase_slope_c": p_phase_c, "p_block_slope_c": p_block_c,
        "mode_shift_2D_consistency_slope": ms_2d_consistency_slope,
        "top_contact_energy_fraction": top_contact_energy_fraction,
        "single_contact_driven": single_contact_driven,
        "leave_one_contact_mode_pass": leave_one_contact_mode_pass,
        "axis_only_mode_control_pass": axis_only_mode_control_pass,
        "onset_jitter_pass_c": onset_jitter_pass_c,
        "spearman_rho_mode": spearman_mode,
        "module_support_flag_c": module_support_flag_c,
        "module_direction_correct_c": module_direction_correct_c,
        "module_null_pass_c": module_null_pass_c,
        # guard companions
        "net_offaxis_flux_surplus_slope_guard": res_flux_guard["surplus"],
        "net_offaxis_flux_slope_z_guard": res_flux_guard["z"],
        "p_label_slope_b_guard": p_label_b_guard,
        "mode_shift_density_surplus_slope_guard": res_mode_guard["surplus"],
        "mode_shift_density_slope_z_guard": res_mode_guard["z"],
        "p_label_slope_c_guard": p_label_c_guard,
        # near-onset dependence + specificity + QC
        "near_onset_dependent_b": near_onset_dependent_b,
        "near_onset_dependent_c": near_onset_dependent_c,
        "lag1_specific_slope": lag1_specific_slope,
        "common_drive_sensitive": common_drive_sensitive,
        "mode_singular_gap_median": mode_singular_gap_median,
        "mode_vector_stable": mode_vector_stable, "cv_r2": cv_r2,
        "n_activation_events_pre": n_activation_events_pre,
        "n_active_windows_pre": n_active_windows_pre,
        "h3b_activation_sufficient": h3b_activation_sufficient,
        "h3c_support_grade": h3c_support_grade,
        "time_order_p_b": time_order_p_b, "time_order_p_c": time_order_p_c,
        "n_label_permutable_shafts": n_label_permutable_shafts,
        "n_unique_label_permutations_est": n_unique_label_permutations_est,
        "label_null_underpowered": label_null_underpowered,
    })
    return row


def run_subject(ds_sid, cohort, in_broad_core, v3cfg, v3pcfg, n_perm, window_rows):
    """One CSV row. geometry_insufficient / load failure / no-windows -> skipped (full row)."""
    row = _base_row(ds_sid, cohort, in_broad_core, v3cfg, v3pcfg)
    try:
        cc = classify_subject_contacts(ds_sid, cohort, v3cfg)
    except Exception as exc:  # noqa: BLE001 - external mount; never drop a subject
        row["skip_reason"] = f"error:{type(exc).__name__}:{exc}"
        print(f"[skip] {ds_sid} ({cohort}): {row['skip_reason']}", flush=True)
        return row

    row.update({
        "geometry_sufficient": bool(cc["geometry_sufficient"]),
        "n_axis": cc["n_axis"], "n_nonaxis": cc["n_nonaxis"], "n_ambiguous": cc["n_ambiguous"],
    })
    if not cc["geometry_sufficient"]:
        row["skip_reason"] = cc["geometry_reason"]
        print(f"[skip] {ds_sid} ({cohort}): geometry_insufficient:{cc['geometry_reason']}", flush=True)
        return row

    try:
        geom = _build_geom(cc)
        return _run_ok_subject(ds_sid, cohort, cc, geom, v3cfg, v3pcfg, n_perm, row, window_rows)
    except Exception as exc:  # noqa: BLE001 - a compute failure still yields a row
        row["status"] = "skipped"
        row["skip_reason"] = f"compute_error:{type(exc).__name__}:{exc}"
        print(f"[skip] {ds_sid} ({cohort}): {row['skip_reason']}", flush=True)
        return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["narrow", "broad"], required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-perm", type=int, default=None, help="default cfg nulls.n_perm_final (1000)")
    ap.add_argument(
        "--subjects", nargs="+", default=None,
        help="subject ids, bare (253) or full (epilepsiae_253); '__none__' writes a header-only CSV",
    )
    args = ap.parse_args(argv)

    v3cfg = load_v3_config()
    v3pcfg = load_v3p_config()
    n_perm = args.n_perm if args.n_perm is not None else int(v3pcfg["nulls"]["n_perm_final"])
    outdir = (
        Path(args.outdir) if args.outdir
        else _ROOT / "results/topic5_ictal_recruitment/v3p_preictal_trajectory" / args.cohort
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # Roster: narrow = the 7 primary; broad = broad_expanded = broad_core (9) +
    # admitted candidates_epilepsiae (4, all admitted at Task-1 gap_min=0.0) = 13,
    # never pooled. `in_broad_core` is config broad_core membership (uniform for
    # both cohorts) so Task 9 can slice broad down to the curated core.
    exp = v3pcfg["cohort_expansion"]
    broad_core_set = set(exp["broad_core"])
    if args.cohort == "broad":
        entries = list(exp["broad_core"]) + list(exp.get("candidates_epilepsiae", []))
    else:
        entries = list(SUBJECTS_BY_SUB["narrow"])
    if args.subjects:
        if args.subjects == ["__none__"]:
            entries = []  # header-only CSV (skipped-path contract test)
        else:
            wanted = set(args.subjects)  # accept full ds_sid OR bare id
            entries = [
                e for e in entries
                if e in wanted or (e.split("_", 1)[1] if "_" in e else e) in wanted
            ]

    window_rows: list = []
    rows = [
        run_subject(ds_sid, args.cohort, ds_sid in broad_core_set, v3cfg, v3pcfg, n_perm, window_rows)
        for ds_sid in entries
    ]

    out_csv = outdir / "v3p_trajectory_subject.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)

    win_csv = outdir / "v3p_window_detail.csv"
    with open(win_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=WINDOW_COLS)
        w.writeheader()
        w.writerows(window_rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_support_b = sum(1 for r in rows if r["module_support_flag_b"])
    n_support_c = sum(1 for r in rows if r["module_support_flag_c"])
    print(
        f"[done] {len(rows)} subjects ({n_ok} ok) -> {out_csv} "
        f"(n_perm={n_perm}; support_b={n_support_b}, support_c={n_support_c}; "
        f"{len(window_rows)} window rows -> {win_csv})",
        flush=True,
    )


if __name__ == "__main__":
    main()

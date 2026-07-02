#!/usr/bin/env python
"""Topic 5 V3a mode-transition — avalanche run (Task 6, H3b co-primary).

Plain-language question (EXPLORATORY): from late-preictal (P3) to early-ictal
(I1), does the avalanche activation FLOW move OFF the fixed interictal HFO
axis onto non-axis contacts? We measure a per-seizure per-phase
``net_offaxis_flux`` (source-normalized axis->non-axis minus non-axis->axis
avalanche transition mass, self-transitions excluded), take the median over a
subject's seizures per phase, and ask whether ``flux_I1 - flux_P3`` is larger
than chance.

Δ-NULL DISCIPLINE (plan rev2 item 7 — the co-primary trap): the p-value is on
the Δ(I1-P3) permutation distribution, NOT per-phase-null then subtract. Each
permutation applies its null to BOTH P3 and I1, recomputes both phase medians,
and forms ``delta_perm = flux_I1_perm - flux_P3_perm`` — see
``_delta_null_distribution``, the single place that owns this invariant.

Three nulls (plan / brief):
  - rate (PRIMARY, gating): ``rate_preserving_shuffle`` per seizure per phase;
    ``delta_surplus`` and ``net_offaxis_flux_z`` use THIS null's median/MAD.
  - label (gating): ``label_permute`` re-draws axis/non-axis labels within
    shaft; activations unchanged.
  - spatial (descriptive, NOT gating): within-shaft activation-row scramble;
    labels fixed.
``module_null_pass = p_rate < alpha AND p_label < alpha`` (spatial descriptive).

narrow (primary) and broad (replication) run SEPARATELY, never pooled. A
``geometry_insufficient`` subject is FLAGGED (``status=skipped``) with a full
NaN/False row — never silently dropped. Tier is assigned only in the summary
(Task 10); this module emits ``module_support_flag`` etc. See
docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md Task 6.
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
from src.topic5_v2_criticality import activations_from_z  # noqa: E402
from src.topic5_v3_mode_transition import (  # noqa: E402
    atm_lag0,
    atm_offdiag,
    label_permute,
    load_v3_config,
    net_offaxis_flux,
    rate_preserving_shuffle,
    shaft_constrained_permute,
)

CSV_COLS = [
    "subject", "cohort", "status", "skip_reason", "geometry_sufficient",
    "n_axis", "n_nonaxis", "n_ambiguous",
    "net_offaxis_flux_P3", "net_offaxis_flux_I1", "delta_net_offaxis_flux_raw",
    "delta_net_offaxis_flux_surplus", "net_offaxis_flux_z",
    "p_rate_delta", "p_spatial_delta", "p_label_delta",
    "lag1_specific_delta", "common_drive_sensitive",
    "max_source_contact_contribution", "leave_one_contact_min_delta",
    "leave_one_contact_pass", "axis_only_flux_delta", "axis_only_control_pass",
    "onset_jitter_pass", "n_seizures",
    "module_support_flag", "module_direction_correct", "module_null_pass",
]

# Every metric column present with NaN floats / False flags — the row for a
# skipped or degenerate subject still carries the full schema (never drop).
_METRIC_DEFAULTS = {
    "net_offaxis_flux_P3": float("nan"), "net_offaxis_flux_I1": float("nan"),
    "delta_net_offaxis_flux_raw": float("nan"), "delta_net_offaxis_flux_surplus": float("nan"),
    "net_offaxis_flux_z": float("nan"),
    "p_rate_delta": float("nan"), "p_spatial_delta": float("nan"), "p_label_delta": float("nan"),
    "lag1_specific_delta": float("nan"), "common_drive_sensitive": False,
    "max_source_contact_contribution": float("nan"), "leave_one_contact_min_delta": float("nan"),
    "leave_one_contact_pass": False, "axis_only_flux_delta": float("nan"),
    "axis_only_control_pass": False, "onset_jitter_pass": False, "n_seizures": 0,
    "module_support_flag": False, "module_direction_correct": False, "module_null_pass": False,
}


def _base_row(ds_sid: str, cohort: str) -> dict:
    return {
        "subject": ds_sid, "cohort": cohort, "status": "skipped", "skip_reason": "",
        "geometry_sufficient": False, "n_axis": 0, "n_nonaxis": 0, "n_ambiguous": 0,
        **_METRIC_DEFAULTS,
    }


# ---------------------------------------------------------------------------
# flux primitives (source-normalized net axis->non-axis, self-transitions out)
# ---------------------------------------------------------------------------
def _atms(acts_list: list, atm_fn) -> list:
    return [atm_fn(a) for a in acts_list]


def _median_net_flux(atms: list, axis_idx: np.ndarray, nonaxis_idx: np.ndarray) -> float:
    """Median over seizures of ``net_offaxis_flux`` from precomputed ATMs.

    Empty phase (no seizures with this phase) -> NaN, so a phase absent for the
    whole subject propagates NaN into ``obs_delta`` rather than a fake 0.
    """
    if not atms:
        return float("nan")
    vals = [net_offaxis_flux(m, axis_idx, nonaxis_idx, "source_mean") for m in atms]
    return float(np.median(vals))


def _delta_stats(obs_delta: float, delta_perm: np.ndarray) -> tuple[float, float, float]:
    """One-sided-upper p, null-corrected surplus, and robust z on a Δ null.

    ``p = (1 + #{delta_perm >= obs_delta}) / (1 + n_perm)`` (H3b expects
    delta > 0). ``surplus = obs_delta - median(delta_perm)``;
    ``z = surplus / MAD(delta_perm)`` (NaN when MAD == 0).
    """
    n_perm = int(delta_perm.size)
    p = (1 + int(np.sum(delta_perm >= obs_delta))) / (1 + n_perm)
    med = float(np.median(delta_perm))
    mad = float(np.median(np.abs(delta_perm - med)))
    surplus = float(obs_delta - med)
    z = float(surplus / mad) if mad > 0 else float("nan")
    return p, surplus, z


def _delta_null_distribution(perm_delta, n_perm: int, seed: int) -> np.ndarray:
    """THE Δ(I1-P3) permutation-null owner.

    For each perm ``p``, a fresh ``np.random.default_rng(seed + p)`` is passed
    to ``perm_delta(rng)``, which must apply the null to BOTH phases and return
    ``flux_I1_perm - flux_P3_perm`` computed inside that call. Because the
    subtraction happens per-perm (never per-phase-null then subtract at the
    end), the returned distribution is the null of the Δ statistic itself —
    which is exactly what ``_delta_stats`` compares ``obs_delta`` against.
    """
    return np.array([perm_delta(np.random.default_rng(seed + p)) for p in range(n_perm)], dtype=float)


def _obs_delta_from_acts(p3_acts, i1_acts, axis_idx, nonaxis_idx) -> float:
    """Raw ``flux_I1 - flux_P3`` (no null) from lag1 off-diagonal ATMs."""
    flux_p3 = _median_net_flux(_atms(p3_acts, atm_offdiag), axis_idx, nonaxis_idx)
    flux_i1 = _median_net_flux(_atms(i1_acts, atm_offdiag), axis_idx, nonaxis_idx)
    return flux_i1 - flux_p3


def _phase_acts(env: dict, phase: str, z_thr: float) -> tuple[list, set]:
    """Activation matrices (rows ordered by all_clean) for one phase + the seizure ids."""
    acts, sz_ids = [], set()
    for sz in env["seizures"]:
        if phase in sz["phases"]:
            acts.append(activations_from_z(sz["phases"][phase], z_thr))
            sz_ids.add(sz["idx"])
    return acts, sz_ids


def _max_source_contribution(i1_atms, axis_idx, nonaxis_idx) -> float:
    """Max over axis sources of that source's share of total A2N mass (I1 phase)."""
    if not i1_atms or axis_idx.size == 0 or nonaxis_idx.size == 0:
        return float("nan")
    mass = np.zeros(axis_idx.size, dtype=float)
    for m in i1_atms:
        mass += np.asarray(m)[np.ix_(axis_idx, nonaxis_idx)].sum(axis=1)
    total = float(mass.sum())
    if total <= 0:
        return 0.0
    return float(mass.max() / total)


def _run_ok_subject(ds_sid, cohort, cfg, cc, n_perm, row) -> dict:
    """Full H3b metric block for a geometry-sufficient subject."""
    z_thr = cfg["avalanche"]["z_threshold"]
    seed = int(cfg["nulls"]["seed"])
    alpha = float(cfg["nulls"]["alpha"])
    all_clean = cc["all_clean"]
    shaft_by_name = cc["shaft_by_name"]
    is_axis_names = cc["is_axis"]
    is_nonaxis_names = cc["is_nonaxis_strict"]

    env0 = load_subject_phase_envelopes(ds_sid, cohort, cfg, ["P3", "I1"], onset_shift=0.0, cls=cc)
    axis_idx, nonaxis_idx = env0["axis_idx"], env0["nonaxis_idx"]
    p3_acts, p3_ids = _phase_acts(env0, "P3", z_thr)
    i1_acts, i1_ids = _phase_acts(env0, "I1", z_thr)
    n_seizures = len(p3_ids | i1_ids)  # union of seizures contributing to either phase median

    # precompute lag1 ATMs once (label/axis-only/max-source/observed reuse them)
    p3_atms = _atms(p3_acts, atm_offdiag)
    i1_atms = _atms(i1_acts, atm_offdiag)
    flux_p3 = _median_net_flux(p3_atms, axis_idx, nonaxis_idx)
    flux_i1 = _median_net_flux(i1_atms, axis_idx, nonaxis_idx)
    obs_delta = flux_i1 - flux_p3

    row.update({
        "status": "ok", "skip_reason": "",
        "net_offaxis_flux_P3": flux_p3, "net_offaxis_flux_I1": flux_i1,
        "delta_net_offaxis_flux_raw": obs_delta, "n_seizures": n_seizures,
    })
    if not np.isfinite(obs_delta):
        print(f"[warn] {ds_sid} ({cohort}): geometry ok but no paired P3/I1 flux (obs_delta NaN)", flush=True)
        return row

    # ---- Δ-null distributions (three nulls; delta formed inside each perm) ----
    name_pos = {n: i for i, n in enumerate(all_clean)}

    def rate_perm(rng):
        def flux(acts):
            if not acts:
                return float("nan")
            atms = [atm_offdiag(rate_preserving_shuffle(a, rng)) for a in acts]
            return _median_net_flux(atms, axis_idx, nonaxis_idx)
        return flux(i1_acts) - flux(p3_acts)

    def label_perm(rng):
        new_axis, new_nonaxis = label_permute(is_axis_names, is_nonaxis_names, shaft_by_name, rng)
        ai = np.array([name_pos[n] for n in new_axis], dtype=int)
        ni = np.array([name_pos[n] for n in new_nonaxis], dtype=int)
        return _median_net_flux(i1_atms, ai, ni) - _median_net_flux(p3_atms, ai, ni)

    values_by_name = {n: i for i, n in enumerate(all_clean)}

    def spatial_perm(rng):
        # ONE within-shaft row scramble per perm, applied to every seizure and
        # both phases (labels fixed). Choice: spatial arrangement is an
        # electrode property, so the scramble is drawn once per perm (like the
        # label null) rather than per-seizure.
        permuted = shaft_constrained_permute(values_by_name, shaft_by_name, rng)
        perm_rows = np.array([permuted[n] for n in all_clean], dtype=int)

        def flux(acts):
            if not acts:
                return float("nan")
            atms = [atm_offdiag(a[perm_rows]) for a in acts]
            return _median_net_flux(atms, axis_idx, nonaxis_idx)
        return flux(i1_acts) - flux(p3_acts)

    delta_rate = _delta_null_distribution(rate_perm, n_perm, seed)
    delta_label = _delta_null_distribution(label_perm, n_perm, seed)
    delta_spatial = _delta_null_distribution(spatial_perm, n_perm, seed)

    p_rate, surplus, z = _delta_stats(obs_delta, delta_rate)   # surplus & z from PRIMARY rate null
    p_label, _, _ = _delta_stats(obs_delta, delta_label)
    p_spatial, _, _ = _delta_stats(obs_delta, delta_spatial)

    # ---- lag1-specific downgrade for common drive (lag1 - lag0) ----
    lag0_p3 = _median_net_flux(_atms(p3_acts, atm_lag0), axis_idx, nonaxis_idx)
    lag0_i1 = _median_net_flux(_atms(i1_acts, atm_lag0), axis_idx, nonaxis_idx)
    lag1_specific_delta = (flux_i1 - lag0_i1) - (flux_p3 - lag0_p3)
    common_drive_sensitive = bool(lag1_specific_delta <= 0)

    # ---- leave-one-contact: recompute obs_delta ONLY per drop (cheaper than a
    # per-drop null; see brief). pass = sign(surplus) survives every drop ----
    n = len(all_clean)
    drop_deltas = []
    for d in range(n):
        keep = np.ones(n, dtype=bool)
        keep[d] = False
        new_of_old = np.full(n, -1, dtype=int)
        new_of_old[keep] = np.arange(int(keep.sum()))
        d_axis = new_of_old[axis_idx[axis_idx != d]]
        d_nonaxis = new_of_old[nonaxis_idx[nonaxis_idx != d]]
        d_p3 = [a[keep] for a in p3_acts]
        d_i1 = [a[keep] for a in i1_acts]
        drop_deltas.append(_obs_delta_from_acts(d_p3, d_i1, d_axis, d_nonaxis))
    leave_one_min = float(np.min(drop_deltas)) if drop_deltas else float("nan")
    sign_surplus = np.sign(surplus)
    leave_one_pass = bool(sign_surplus != 0 and all(np.sign(dd) == sign_surplus for dd in drop_deltas))

    max_source = _max_source_contribution(i1_atms, axis_idx, nonaxis_idx)

    # ---- axis-only control: relabel all non-axis -> axis (non-axis empty).
    # By construction net flux collapses to ~0 (no non-axis target/source), so
    # axis_only_flux_delta ~ 0 and the pass reduces to obs_delta > 0. NEAR-
    # TRIVIAL-BY-CONSTRUCTION (flagged in the report) — implemented literally
    # per the brief so Task 10 can decide whether to strengthen it. ----
    axis_all = np.array(sorted(set(axis_idx.tolist()) | set(nonaxis_idx.tolist())), dtype=int)
    empty_nonaxis = np.array([], dtype=int)
    axis_only_flux_delta = (
        _median_net_flux(i1_atms, axis_all, empty_nonaxis)
        - _median_net_flux(p3_atms, axis_all, empty_nonaxis)
    )
    axis_only_control_pass = bool(axis_only_flux_delta < obs_delta)

    # ---- onset jitter +-10 s: reload windows at shifted anchors (i1_eligible
    # gate stays at shift 0 inside the loader), recompute obs_delta, require
    # sign stability. cls reused to skip the expensive context/lagPat reload. ----
    def obs_delta_at(shift):
        env = load_subject_phase_envelopes(ds_sid, cohort, cfg, ["P3", "I1"], onset_shift=shift, cls=cc)
        p3s, _ = _phase_acts(env, "P3", z_thr)
        i1s, _ = _phase_acts(env, "I1", z_thr)
        return _obs_delta_from_acts(p3s, i1s, env["axis_idx"], env["nonaxis_idx"])

    d_p10 = obs_delta_at(10.0)
    d_m10 = obs_delta_at(-10.0)
    onset_jitter_pass = bool(np.sign(d_p10) == np.sign(obs_delta) == np.sign(d_m10))

    module_direction = bool(surplus > 0)
    module_null_pass = bool(p_rate < alpha and p_label < alpha)
    module_support = bool(module_direction and module_null_pass)

    row.update({
        "delta_net_offaxis_flux_surplus": surplus, "net_offaxis_flux_z": z,
        "p_rate_delta": p_rate, "p_spatial_delta": p_spatial, "p_label_delta": p_label,
        "lag1_specific_delta": float(lag1_specific_delta), "common_drive_sensitive": common_drive_sensitive,
        "max_source_contact_contribution": max_source, "leave_one_contact_min_delta": leave_one_min,
        "leave_one_contact_pass": leave_one_pass, "axis_only_flux_delta": float(axis_only_flux_delta),
        "axis_only_control_pass": axis_only_control_pass, "onset_jitter_pass": onset_jitter_pass,
        "module_support_flag": module_support, "module_direction_correct": module_direction,
        "module_null_pass": module_null_pass,
    })
    return row


def run_subject(ds_sid: str, cohort: str, cfg: dict, n_perm: int) -> dict:
    """One CSV row. geometry_insufficient / load failure -> skipped (full row)."""
    row = _base_row(ds_sid, cohort)
    try:
        cc = classify_subject_contacts(ds_sid, cohort, cfg)
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

    out_csv = outdir / "v3_avalanche_subject.csv"
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

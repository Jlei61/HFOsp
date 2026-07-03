#!/usr/bin/env python
"""Topic 5 V3a mode-transition — avalanche run (Task 6, H3b co-primary).

Plain-language question (EXPLORATORY): from late-preictal (P3) to early-ictal
(I1), does the avalanche activation FLOW move OFF the fixed interictal HFO
axis onto non-axis contacts? We measure a per-seizure per-phase
``net_offaxis_flux`` (source-normalized axis->non-axis minus non-axis->axis
avalanche transition mass, self-transitions excluded). Because P3->I1 is a
WITHIN-SEIZURE change, the contrast is PAIRED BY SEIZURE: only seizures that
carry BOTH phases contribute (``common_ids = P3_ids ∩ I1_ids``),
``delta(sz) = flux_I1(sz) - flux_P3(sz)`` is formed per seizure, and
``obs_delta`` is the subject-median of those per-seizure deltas — NOT
``median(I1-seizures) - median(P3-seizures)``, which would silently mix a
P3-only and an I1-only seizure population when short seizures have P3 but no
eligible I1 window. ``n_seizures`` is ``len(common_ids)``.

Δ-NULL DISCIPLINE (plan rev2 item 7 — the co-primary trap): the p-value is on
the Δ(I1-P3) permutation distribution, NOT per-phase-null then subtract. Each
permutation applies its null to BOTH phases of EACH common seizure, forms the
per-seizure ``delta_perm(sz) = flux_I1_perm(sz) - flux_P3_perm(sz)``, and takes
the subject-median over common seizures — the SAME paired aggregation as
``obs_delta``. See ``_delta_null_distribution``, the single place that owns the
per-perm subtraction invariant.

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
#
# P3->I1 is a WITHIN-SEIZURE change, so every quantity is PAIRED BY SEIZURE: a
# metric is computed per seizure per phase, only seizures present in BOTH
# phases (``common_ids``) contribute, and the subject statistic is the median
# of the per-seizure differences (median of differences, NOT difference of the
# two per-phase medians). Each null forms the same paired delta inside the perm.
# ---------------------------------------------------------------------------
def _flux_sz(acts: np.ndarray, atm_fn, axis_idx: np.ndarray, nonaxis_idx: np.ndarray) -> float:
    """Net off-axis flux for ONE seizure's activation matrix under ``atm_fn``."""
    return net_offaxis_flux(atm_fn(acts), axis_idx, nonaxis_idx, "source_mean")


def _paired_delta_median(m_p3: dict, m_i1: dict, common_ids: list) -> float:
    """Subject-median over common seizures of ``metric_I1(sz) - metric_P3(sz)``.

    A per-seizure delta that is non-finite (a degenerate phase for that
    seizure) is dropped before the median; an empty result -> NaN (never a
    fake 0). This is the paired within-seizure aggregation the P3->I1 contrast
    requires: the median of per-seizure differences, NOT the difference of the
    two per-phase medians (which would mix a P3-only and an I1-only seizure
    population when the two phase sets differ).
    """
    deltas = [m_i1[sz] - m_p3[sz] for sz in common_ids if np.isfinite(m_i1[sz] - m_p3[sz])]
    return float(np.median(deltas)) if deltas else float("nan")


def _phase_median(m_by_sz: dict, common_ids: list) -> float:
    """Subject-median over common seizures of one phase's per-seizure metric.

    Keeps the reported per-phase columns on the SAME paired seizure set as
    ``obs_delta`` (finite-filtered; empty -> NaN).
    """
    vals = [m_by_sz[sz] for sz in common_ids if np.isfinite(m_by_sz[sz])]
    return float(np.median(vals)) if vals else float("nan")


def _delta_stats(obs_delta: float, delta_perm: np.ndarray) -> tuple[float, float, float]:
    """One-sided-upper p, null-corrected surplus, and robust z on a Δ null.

    ``p = (1 + #{finite >= obs_delta}) / (1 + finite.size)`` over the FINITE
    subset of ``delta_perm`` only (H3b expects delta > 0). A perm draw can be
    NaN (e.g. a degenerate rate/label/spatial shuffle leaves a phase with no
    valid flux); NaN never satisfies ``>=``, so counting it toward the
    denominator without ever counting it toward the numerator would silently
    deflate p (same class of bug as H3a's ``_p_lower`` fix). If no draw is
    finite the null is undefined here and this returns ``(nan, nan, nan)``
    rather than a fabricated p/surplus/z.

    ``surplus = obs_delta - median(finite)``; ``z = surplus / MAD(finite)``
    (NaN when MAD == 0).
    """
    finite = delta_perm[np.isfinite(delta_perm)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    p = (1 + int(np.sum(finite >= obs_delta))) / (1 + finite.size)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    surplus = float(obs_delta - med)
    z = float(surplus / mad) if mad > 0 else float("nan")
    return p, surplus, z


def _delta_null_distribution(perm_delta, n_perm: int, seed: int) -> np.ndarray:
    """THE Δ(I1-P3) permutation-null owner.

    For each perm ``p``, a fresh ``np.random.default_rng(seed + p)`` is passed
    to ``perm_delta(rng)``, which must apply the null to BOTH phases of each
    common seizure and return the paired subject-median ``delta_perm`` computed
    inside that call. Because the subtraction happens per-seizure per-perm
    (never per-phase-null then subtract at the end), the returned distribution
    is the null of the Δ statistic itself — which is exactly what
    ``_delta_stats`` compares ``obs_delta`` against.
    """
    return np.array([perm_delta(np.random.default_rng(seed + p)) for p in range(n_perm)], dtype=float)


def _paired_obs_delta(
    p3_by_id: dict, i1_by_id: dict, common_ids: list,
    axis_idx: np.ndarray, nonaxis_idx: np.ndarray,
) -> float:
    """Paired raw obs_delta (lag1 off-diagonal flux) over common seizures."""
    m_p3 = {sz: _flux_sz(p3_by_id[sz], atm_offdiag, axis_idx, nonaxis_idx) for sz in common_ids}
    m_i1 = {sz: _flux_sz(i1_by_id[sz], atm_offdiag, axis_idx, nonaxis_idx) for sz in common_ids}
    return _paired_delta_median(m_p3, m_i1, common_ids)


def _phase_acts(env: dict, phase: str, z_thr: float) -> dict:
    """``{seizure_idx: activation matrix}`` (rows ordered by all_clean) for one phase.

    Keyed by seizure id so the caller can intersect P3 and I1 on common ids —
    short seizures carry P3 but no I1-eligible window, so the two phase sets
    differ and must be PAIRED, not medianed independently.
    """
    out: dict = {}
    for sz in env["seizures"]:
        if phase in sz["phases"]:
            out[sz["idx"]] = activations_from_z(sz["phases"][phase], z_thr)
    return out


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
    p3_by_id = _phase_acts(env0, "P3", z_thr)
    i1_by_id = _phase_acts(env0, "I1", z_thr)
    common_ids = sorted(set(p3_by_id) & set(i1_by_id))  # ONLY seizures carrying BOTH phases
    n_seizures = len(common_ids)
    row["n_seizures"] = n_seizures

    if n_seizures == 0:
        # No seizure has both P3 and I1 -> the paired obs_delta is undefined.
        row.update({"status": "skipped", "skip_reason": "no_paired_seizures"})
        print(f"[skip] {ds_sid} ({cohort}): geometry ok but no seizure carries BOTH P3 and I1", flush=True)
        return row

    # per-seizure lag1 off-diagonal ATMs (observed / label / axis-only / max-source reuse them)
    p3_atm_by_sz = {sz: atm_offdiag(p3_by_id[sz]) for sz in common_ids}
    i1_atm_by_sz = {sz: atm_offdiag(i1_by_id[sz]) for sz in common_ids}

    # per-seizure observed lag1 flux -> paired obs_delta + paired per-phase columns
    f1_p3 = {sz: net_offaxis_flux(p3_atm_by_sz[sz], axis_idx, nonaxis_idx, "source_mean") for sz in common_ids}
    f1_i1 = {sz: net_offaxis_flux(i1_atm_by_sz[sz], axis_idx, nonaxis_idx, "source_mean") for sz in common_ids}
    flux_p3 = _phase_median(f1_p3, common_ids)
    flux_i1 = _phase_median(f1_i1, common_ids)
    obs_delta = _paired_delta_median(f1_p3, f1_i1, common_ids)

    row.update({
        "status": "ok", "skip_reason": "",
        "net_offaxis_flux_P3": flux_p3, "net_offaxis_flux_I1": flux_i1,
        "delta_net_offaxis_flux_raw": obs_delta,
    })
    if not np.isfinite(obs_delta):
        row.update({"status": "skipped", "skip_reason": "nonfinite_flux"})
        print(f"[warn] {ds_sid} ({cohort}): geometry ok but no finite paired P3/I1 flux (obs_delta NaN)", flush=True)
        return row

    # ---- Δ-null distributions (three nulls; each forms the PAIRED delta over
    # common seizures inside the perm — same aggregation as obs_delta) ----
    name_pos = {n: i for i, n in enumerate(all_clean)}

    def rate_perm(rng):
        # rate-preserving shuffle drawn INDEPENDENTLY per seizure per phase.
        m_p3 = {sz: _flux_sz(rate_preserving_shuffle(p3_by_id[sz], rng), atm_offdiag, axis_idx, nonaxis_idx)
                for sz in common_ids}
        m_i1 = {sz: _flux_sz(rate_preserving_shuffle(i1_by_id[sz], rng), atm_offdiag, axis_idx, nonaxis_idx)
                for sz in common_ids}
        return _paired_delta_median(m_p3, m_i1, common_ids)

    def label_perm(rng):
        new_axis, new_nonaxis = label_permute(is_axis_names, is_nonaxis_names, shaft_by_name, rng)
        ai = np.array([name_pos[n] for n in new_axis], dtype=int)
        ni = np.array([name_pos[n] for n in new_nonaxis], dtype=int)
        m_p3 = {sz: net_offaxis_flux(p3_atm_by_sz[sz], ai, ni, "source_mean") for sz in common_ids}
        m_i1 = {sz: net_offaxis_flux(i1_atm_by_sz[sz], ai, ni, "source_mean") for sz in common_ids}
        return _paired_delta_median(m_p3, m_i1, common_ids)

    values_by_name = {n: i for i, n in enumerate(all_clean)}

    def spatial_perm(rng):
        # ONE within-shaft row scramble per perm, applied to every common
        # seizure and both phases (labels fixed). Spatial arrangement is an
        # electrode property, so the scramble is drawn once per perm (like the
        # label null) rather than per-seizure.
        permuted = shaft_constrained_permute(values_by_name, shaft_by_name, rng)
        perm_rows = np.array([permuted[n] for n in all_clean], dtype=int)
        m_p3 = {sz: _flux_sz(p3_by_id[sz][perm_rows], atm_offdiag, axis_idx, nonaxis_idx) for sz in common_ids}
        m_i1 = {sz: _flux_sz(i1_by_id[sz][perm_rows], atm_offdiag, axis_idx, nonaxis_idx) for sz in common_ids}
        return _paired_delta_median(m_p3, m_i1, common_ids)

    delta_rate = _delta_null_distribution(rate_perm, n_perm, seed)
    delta_label = _delta_null_distribution(label_perm, n_perm, seed)
    delta_spatial = _delta_null_distribution(spatial_perm, n_perm, seed)

    p_rate, surplus, z = _delta_stats(obs_delta, delta_rate)   # surplus & z from PRIMARY rate null
    p_label, _, _ = _delta_stats(obs_delta, delta_label)
    p_spatial, _, _ = _delta_stats(obs_delta, delta_spatial)

    if not np.isfinite(surplus):
        row.update({"status": "skipped", "skip_reason": "nonfinite_flux"})
        print(f"[warn] {ds_sid} ({cohort}): geometry ok but surplus is non-finite", flush=True)
        return row

    # ---- lag1-specific downgrade for common drive: paired per-seizure
    # (lag1 - lag0) delta over common seizures. ----
    f0_p3 = {sz: net_offaxis_flux(atm_lag0(p3_by_id[sz]), axis_idx, nonaxis_idx, "source_mean") for sz in common_ids}
    f0_i1 = {sz: net_offaxis_flux(atm_lag0(i1_by_id[sz]), axis_idx, nonaxis_idx, "source_mean") for sz in common_ids}
    ls_p3 = {sz: f1_p3[sz] - f0_p3[sz] for sz in common_ids}
    ls_i1 = {sz: f1_i1[sz] - f0_i1[sz] for sz in common_ids}
    lag1_specific_delta = _paired_delta_median(ls_p3, ls_i1, common_ids)
    common_drive_sensitive = bool(lag1_specific_delta <= 0)

    # ---- leave-one-contact: recompute the PAIRED raw obs_delta per drop
    # (cheaper than a per-drop null; see brief), then convert each to a surplus
    # using the ALREADY-COMPUTED full-data rate-null median (no re-run of the
    # null). pass = sign(surplus) survives every drop, on the surplus basis —
    # raw obs_delta can disagree in sign with the null-corrected surplus. ----
    n = len(all_clean)
    median_rate_null_full = float(np.nanmedian(delta_rate))  # same NaN-dilution class as _delta_stats
    drop_surpluses = []
    for d in range(n):
        keep = np.ones(n, dtype=bool)
        keep[d] = False
        new_of_old = np.full(n, -1, dtype=int)
        new_of_old[keep] = np.arange(int(keep.sum()))
        d_axis = new_of_old[axis_idx[axis_idx != d]]
        d_nonaxis = new_of_old[nonaxis_idx[nonaxis_idx != d]]
        d_p3 = {sz: p3_by_id[sz][keep] for sz in common_ids}
        d_i1 = {sz: i1_by_id[sz][keep] for sz in common_ids}
        dd = _paired_obs_delta(d_p3, d_i1, common_ids, d_axis, d_nonaxis)
        drop_surpluses.append(dd - median_rate_null_full)
    leave_one_min = float(np.min(drop_surpluses)) if drop_surpluses else float("nan")
    sign_surplus = np.sign(surplus)
    leave_one_pass = bool(
        sign_surplus != 0 and all(np.sign(ds) == sign_surplus for ds in drop_surpluses)
    )

    max_source = _max_source_contribution(
        [i1_atm_by_sz[sz] for sz in common_ids], axis_idx, nonaxis_idx
    )

    # ---- axis-only control: relabel all non-axis -> axis (non-axis empty).
    # By construction net flux collapses to ~0 (no non-axis target/source), so
    # axis_only_flux_delta ~ 0 and the pass reduces to surplus > 0 (null-
    # corrected basis, not raw obs_delta). NEAR-TRIVIAL-BY-CONSTRUCTION
    # (flagged in the report) — implemented literally per the brief so Task 10
    # can decide whether to strengthen it. Paired over common seizures. ----
    axis_all = np.array(sorted(set(axis_idx.tolist()) | set(nonaxis_idx.tolist())), dtype=int)
    empty_nonaxis = np.array([], dtype=int)
    aa_p3 = {sz: net_offaxis_flux(p3_atm_by_sz[sz], axis_all, empty_nonaxis, "source_mean") for sz in common_ids}
    aa_i1 = {sz: net_offaxis_flux(i1_atm_by_sz[sz], axis_all, empty_nonaxis, "source_mean") for sz in common_ids}
    axis_only_flux_delta = _paired_delta_median(aa_p3, aa_i1, common_ids)
    axis_only_control_pass = bool(axis_only_flux_delta < surplus)

    # ---- onset jitter +-10 s: reload windows at shifted anchors (i1_eligible
    # gate stays at shift 0 inside the loader), recompute the PAIRED obs_delta
    # (common set RE-DERIVED at the shifted anchor), require sign stability.
    # cls reused to skip the expensive context/lagPat reload. ----
    def obs_delta_at(shift):
        env = load_subject_phase_envelopes(ds_sid, cohort, cfg, ["P3", "I1"], onset_shift=shift, cls=cc)
        p3s = _phase_acts(env, "P3", z_thr)
        i1s = _phase_acts(env, "I1", z_thr)
        common = sorted(set(p3s) & set(i1s))
        return _paired_obs_delta(p3s, i1s, common, env["axis_idx"], env["nonaxis_idx"])

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

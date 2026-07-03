#!/usr/bin/env python
"""Topic 5 V3a mode-transition — susceptibility run (Task 9, H3a SUPPORTIVE-ONLY).

Plain-language question (EXPLORATORY, supportive/contextual only — NOT a
stand-alone endpoint): from late-preictal (P3) to early-ictal (I1), does the
fixed interictal HFO axis's ORDERED STRUCTURE go slack? For every contact and
every (seizure, phase) we compute a bb-envelope LINE-LENGTH RATE — the WHOLE
phase span's per-sample absolute roughness, ``sum(abs(diff(env))) / (n_t -
1)`` (the raw per-phase quantity, NOT
``src.topic5_v2_criticality.contact_susceptibility``'s late-minus-early delta
between two sub-windows within one call — that helper is the wrong shape for
this question and is referenced only for the line-length-rate FORMULA, per
the brief). ``beta_axis`` (signed Spearman between this roughness and the
axis contacts' fixed interictal forward rank) measures how strongly the
roughness still follows the axis's forward order; ``|beta_axis|`` is the
axial-order STRENGTH regardless of sign. Because P3->I1 is a WITHIN-SEIZURE
change, the contrast is PAIRED BY SEIZURE: only seizures carrying BOTH phases
contribute (``common_ids = P3_ids ∩ I1_ids``), the per-seizure
``delta(sz) = |beta_I1|(sz) − |beta_P3|(sz)`` is formed, and ``obs_delta`` is the
subject-median of those per-seizure deltas (``n_seizures = len(common_ids)``) —
we ask whether this axial order SHRINKS (P3 -> I1), NOT whether
``median(I1-seizures) − median(P3-seizures)`` does.

H3a IS SUPPORTIVE-ONLY (plan §2/§7): axial weakening can be explained by
seizure-wide SNR/saturation/coverage effects, not specifically by "axis ->
non-axis" reorganization, so H3a alone can never certify subject-level
support — only H3b (avalanche run, Task 6) or H3c (dynamics run, Task 8) can.
``module_support_flag`` is therefore HARDCODED ``False`` on every row,
regardless of direction or null outcome (see ``_run_ok_subject``).

Δ-NULL DISCIPLINE (plan rev2 item 7 — the co-primary trap, still enforced
here even though H3a is non-gating): the p-value is on the Δ(I1−P3)
permutation distribution, NOT per-phase-null then subtract. Each permutation
applies its null to BOTH phases of EACH common seizure and forms the
per-seizure ``delta_perm(sz) = |beta_I1|_perm(sz) − |beta_P3|_perm(sz)``, then
the subject-median over common seizures — the SAME paired aggregation as
``obs_delta``.

H3a expects ``delta_beta_axis_strength < 0`` (order weakens) — the MIRROR of
H3b/H3c's ``> 0`` convention, so the p-value is one-sided LOWER:
``p = (1 + #{delta_perm <= obs_delta}) / (1 + n_perm)`` — do not reuse an
upper-tail helper here.

Two self-built nulls (plan / brief):
  - spatial: ``shaft_constrained_permute`` of the per-contact line-length-rate
    metric, drawn INDEPENDENTLY for EVERY (seizure, phase) pair inside one
    perm (NOT a single subject-wide draw, unlike the label null below — the
    brief calls this out explicitly as "per phase per seizure").
    ``beta_axis_delta_null_z`` is taken from THIS null's median/MAD.
  - label: ``label_permute`` re-draws axis/non-axis identity within shaft —
    ONE draw per perm (an electrode-identity property, not seizure-resolved,
    matching the H3b/H3c label-null convention); beta_axis is recomputed by
    restricting the (unchanged, observed) metric to the PERMUTED axis contact
    set, against the FIXED (never re-derived) ``rf`` — a relabeled contact
    that was never in the interictal template has no rank to compare against
    and simply drops out of the correlation.
``module_null_pass = p_spatial_delta < alpha AND p_label_delta < alpha``.

``beta_axis_P3_reliable`` (``|beta|_P3 >= beta_axis_reliability_min``) flags
whether there was enough axial structure at P3 to even be capable of
weakening further; if False the row is still written (never skipped for this
reason alone) — H3a is simply not interpretable for that subject.

narrow (primary) and broad (replication) run SEPARATELY, never pooled. A
``geometry_insufficient`` subject is FLAGGED (``status=skipped``) with a full
NaN/False row — never silently dropped. Tier is assigned only in the summary
(Task 10). See docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md
Task 9.
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
from src.topic5_v3_mode_transition import (  # noqa: E402
    beta_axis,
    label_permute,
    load_v3_config,
    rank_forward,
    shaft_constrained_permute,
)

CSV_COLS = [
    "subject", "cohort", "status", "skip_reason", "K_primary_metric",
    "beta_axis_P3", "beta_axis_I1", "beta_axis_P3_reliable",
    "delta_beta_axis_strength", "beta_axis_delta_null_z",
    "p_spatial_delta", "p_label_delta", "onset_jitter_pass", "n_seizures",
    "module_support_flag", "module_direction_correct", "module_null_pass",
]

# Every metric column present with NaN floats / False flags — the row for a
# skipped or degenerate subject still carries the full schema (never drop).
_METRIC_DEFAULTS = {
    "beta_axis_P3": float("nan"), "beta_axis_I1": float("nan"),
    "beta_axis_P3_reliable": False,
    "delta_beta_axis_strength": float("nan"), "beta_axis_delta_null_z": float("nan"),
    "p_spatial_delta": float("nan"), "p_label_delta": float("nan"),
    "onset_jitter_pass": False, "n_seizures": 0,
    "module_support_flag": False, "module_direction_correct": False, "module_null_pass": False,
}


def _base_row(ds_sid: str, cohort: str) -> dict:
    return {
        "subject": ds_sid, "cohort": cohort, "status": "skipped", "skip_reason": "",
        "K_primary_metric": "line_length_rate",
        **_METRIC_DEFAULTS,
    }


# ---------------------------------------------------------------------------
# metric + beta helpers
# ---------------------------------------------------------------------------
def _line_length_rate(env: np.ndarray) -> np.ndarray:
    """Per-contact line-length rate over the FULL phase span (brief step 3).

    ``llr[c] = sum(abs(diff(x))) / (x.size - 1)`` where ``x`` is channel
    ``c``'s samples restricted to finite values — NOT
    ``src.topic5_v2_criticality.contact_susceptibility``'s late-minus-early
    delta (that helper compares two SUB-windows within one call; here we want
    the raw per-phase absolute roughness of the WHOLE phase span, no
    windowing). ``contact_susceptibility`` is referenced only for the
    line-length-rate FORMULA, never called directly. Matches that helper's
    ``_linelength`` finite-sample handling: an interior NaN sample is
    filtered out and the rate is computed over the remaining finite subset,
    rather than propagating to NaN the whole channel; a channel needs at
    least 3 finite samples to form a diff, else it returns NaN for that
    channel alone.
    """
    env = np.asarray(env, dtype=float)
    out = np.full(env.shape[0], float("nan"))
    for c in range(env.shape[0]):
        x = env[c][np.isfinite(env[c])]
        if x.size < 3:
            continue
        out[c] = np.sum(np.abs(np.diff(x))) / (x.size - 1)
    return out


def _abs_beta_sz(llr_by_name: dict, names: list, rf: dict) -> float:
    """``|beta_axis|`` for ONE seizure, restricted to ``names`` (finite pairs).

    ``metric`` is that seizure's observed (or permuted) line-length-rate dict
    restricted to ``names`` — the TRUE axis set for the observed/spatial-null
    path, or the label-null's PERMUTED axis set. Fewer than 4 valid pairs ->
    NaN (never a fake 0), so a degenerate phase propagates NaN.
    """
    metric = {n: llr_by_name[n] for n in names if n in llr_by_name}
    b = beta_axis(metric, rf)
    return abs(b) if np.isfinite(b) else float("nan")


def _paired_delta_median(m_p3: dict, m_i1: dict, common_ids: list) -> float:
    """Subject-median over common seizures of ``metric_I1(sz) - metric_P3(sz)``.

    Median of per-seizure differences (the paired within-seizure change), NOT
    the difference of the two per-phase medians (which would mix a P3-only and
    an I1-only seizure population). Non-finite per-seizure deltas are dropped;
    empty -> NaN.
    """
    deltas = [m_i1[sz] - m_p3[sz] for sz in common_ids if np.isfinite(m_i1[sz] - m_p3[sz])]
    return float(np.median(deltas)) if deltas else float("nan")


def _phase_median(m_by_sz: dict, common_ids: list) -> float:
    """Subject-median over common seizures of one phase's per-seizure metric."""
    vals = [m_by_sz[sz] for sz in common_ids if np.isfinite(m_by_sz[sz])]
    return float(np.median(vals)) if vals else float("nan")


def _p_lower(obs: float, perm: np.ndarray) -> float:
    """One-sided-LOWER Δ-null p over FINITE draws only: ``(1 + #{finite <= obs}) / (1 + finite.size)``.

    H3a expects ``delta_beta_axis_strength < 0`` (axial order weakens into
    I1) — the mirror of H3b/H3c's one-sided-UPPER convention. Do not reuse an
    upper-tail helper here. A perm draw can be NaN (the label null can
    relabel a former non-axis contact into "axis"; that contact was never in
    the interictal template, so it has no ``rank_forward`` entry, and
    ``beta_axis`` can end up with fewer than 4 valid pairs and return NaN).
    NaN never satisfies ``<=``, so counting it toward the denominator without
    ever counting it toward the numerator would silently deflate p. Restrict
    both counts to the finite subset; if no draw is finite the null is
    undefined here and this returns NaN rather than a fabricated p.
    """
    finite = perm[np.isfinite(perm)]
    if finite.size == 0:
        return float("nan")
    return (1 + int(np.sum(finite <= obs))) / (1 + finite.size)


def _phase_llr(env0: dict, all_clean: list, phase: str) -> dict:
    """``{seizure_idx: {name: llr}}`` (rows ordered by ``all_clean``) for one phase.

    Keyed by seizure id so the caller can pair P3 and I1 on common seizures
    (short seizures carry P3 but no I1-eligible window).
    """
    out: dict = {}
    for sz in env0["seizures"]:
        if phase in sz["phases"]:
            out[sz["idx"]] = dict(zip(all_clean, _line_length_rate(sz["phases"][phase])))
    return out


def _run_ok_subject(ds_sid: str, cohort: str, cfg: dict, cc: dict, n_perm: int, row: dict) -> dict:
    """Full H3a metric block for a geometry-sufficient subject."""
    seed = int(cfg["nulls"]["seed"])
    alpha = float(cfg["nulls"]["alpha"])
    rel_min = float(cfg["geometry"]["beta_axis_reliability_min"])

    all_clean = cc["all_clean"]
    is_axis_names = cc["is_axis"]
    is_nonaxis_names = cc["is_nonaxis_strict"]
    shaft_by_name = cc["shaft_by_name"]

    # rf: FIXED rank_forward over the true interictal axis template (Task-8
    # pattern, copied verbatim) — NEVER recomputed under the label null: rf
    # encodes the true interictal ordering, and a relabeled contact that was
    # never in the template simply has no rank to compare against.
    axis_set = set(is_axis_names)
    typical_rank: dict = {}
    for rec in (cc["ctx"]["ta"], cc["ctx"]["tb"]):
        for ch in rec["channels"]:
            nm = ch["name"]
            r = ch.get("typical_rank", np.nan)
            if nm in axis_set and np.isfinite(r):
                typical_rank.setdefault(nm, float(r))
    rf = rank_forward(typical_rank)

    # ---- observed metric: line-length rate over the WHOLE phase span, per
    # contact, per seizure (no sliding windows — unlike Task 8). PAIRED BY
    # SEIZURE: only seizures carrying BOTH phases contribute. ----
    env0 = load_subject_phase_envelopes(ds_sid, cohort, cfg, ["P3", "I1"], onset_shift=0.0, cls=cc)
    p3_llr_by_id = _phase_llr(env0, all_clean, "P3")
    i1_llr_by_id = _phase_llr(env0, all_clean, "I1")
    common_ids = sorted(set(p3_llr_by_id) & set(i1_llr_by_id))  # ONLY seizures with BOTH phases
    n_seizures = len(common_ids)
    row["n_seizures"] = n_seizures

    if n_seizures == 0:
        row.update({"status": "skipped", "skip_reason": "no_paired_seizures"})
        print(f"[skip] {ds_sid} ({cohort}): geometry ok but no seizure carries BOTH P3 and I1", flush=True)
        return row

    beta_p3_sz = {sz: _abs_beta_sz(p3_llr_by_id[sz], is_axis_names, rf) for sz in common_ids}
    beta_i1_sz = {sz: _abs_beta_sz(i1_llr_by_id[sz], is_axis_names, rf) for sz in common_ids}
    beta_p3 = _phase_median(beta_p3_sz, common_ids)
    beta_i1 = _phase_median(beta_i1_sz, common_ids)
    obs_delta = _paired_delta_median(beta_p3_sz, beta_i1_sz, common_ids)
    beta_p3_reliable = bool(beta_p3 >= rel_min)

    row.update({
        "status": "ok", "skip_reason": "",
        "beta_axis_P3": beta_p3, "beta_axis_I1": beta_i1,
        "beta_axis_P3_reliable": beta_p3_reliable,
        "delta_beta_axis_strength": obs_delta,
    })
    if not np.isfinite(obs_delta):
        row.update({"status": "skipped", "skip_reason": "nonfinite_beta_delta"})
        print(f"[warn] {ds_sid} ({cohort}): geometry ok but no finite paired P3/I1 beta_axis (obs_delta NaN)", flush=True)
        return row

    # ---- Δ-null distributions (both nulls form the PAIRED Δ over common
    # seizures INSIDE each perm; H3a expects delta<0 -> LOWER-tail p,
    # mirroring H3b/H3c's upper-tail). ----
    def spatial_perm(rng):
        # "(per phase per seizure)" per the brief: an INDEPENDENT
        # shaft-constrained value-shuffle of the line-length-rate metric is
        # drawn for EVERY (seizure, phase) pair inside this one perm (NOT a
        # single draw applied everywhere, unlike the label null below) — the
        # shared `rng` state is consumed sequentially across these draws.
        def beta_sz(llr_by_name):
            perm_metric = shaft_constrained_permute(llr_by_name, shaft_by_name, rng)
            restricted = {n: perm_metric[n] for n in is_axis_names if n in perm_metric}
            b = beta_axis(restricted, rf)
            return abs(b) if np.isfinite(b) else float("nan")
        m_p3 = {sz: beta_sz(p3_llr_by_id[sz]) for sz in common_ids}
        m_i1 = {sz: beta_sz(i1_llr_by_id[sz]) for sz in common_ids}
        return _paired_delta_median(m_p3, m_i1, common_ids)

    def label_perm(rng):
        # ONE draw per perm — axis/non-axis identity is a per-contact
        # property, not seizure-resolved (matches the Task 6/8 label-null
        # convention). rf stays FIXED; only which contacts get read out
        # against it changes (the PERMUTED axis set).
        new_axis, _new_nonaxis = label_permute(is_axis_names, is_nonaxis_names, shaft_by_name, rng)
        m_p3 = {sz: _abs_beta_sz(p3_llr_by_id[sz], new_axis, rf) for sz in common_ids}
        m_i1 = {sz: _abs_beta_sz(i1_llr_by_id[sz], new_axis, rf) for sz in common_ids}
        return _paired_delta_median(m_p3, m_i1, common_ids)

    delta_spatial = np.array([spatial_perm(np.random.default_rng(seed + p)) for p in range(n_perm)])
    delta_label = np.array([label_perm(np.random.default_rng(seed + p)) for p in range(n_perm)])

    p_spatial = _p_lower(obs_delta, delta_spatial)
    p_label = _p_lower(obs_delta, delta_label)

    med_s = float(np.nanmedian(delta_spatial))
    mad_s = float(np.nanmedian(np.abs(delta_spatial - med_s)))
    beta_axis_delta_null_z = (
        float((obs_delta - med_s) / mad_s) if np.isfinite(mad_s) and mad_s > 0 else float("nan")
    )

    # ---- onset jitter +-10 s: reload windows at shifted anchors (i1_eligible
    # gate stays at shift 0 inside the loader), recompute the PAIRED obs_delta
    # (common set RE-DERIVED at the shifted anchor), require sign stability. cls
    # reused to skip the expensive context/lagPat reload. ----
    def obs_delta_at(shift):
        env = load_subject_phase_envelopes(ds_sid, cohort, cfg, ["P3", "I1"], onset_shift=shift, cls=cc)
        p3s = _phase_llr(env, all_clean, "P3")
        i1s = _phase_llr(env, all_clean, "I1")
        common = sorted(set(p3s) & set(i1s))
        m_p3 = {sz: _abs_beta_sz(p3s[sz], is_axis_names, rf) for sz in common}
        m_i1 = {sz: _abs_beta_sz(i1s[sz], is_axis_names, rf) for sz in common}
        return _paired_delta_median(m_p3, m_i1, common)

    d_p10 = obs_delta_at(10.0)
    d_m10 = obs_delta_at(-10.0)
    onset_jitter_pass = bool(np.sign(d_p10) == np.sign(obs_delta) == np.sign(d_m10))

    module_direction = bool(obs_delta < 0)  # H3a expects WEAKENING — opposite sign of H3b/H3c
    module_null_pass = bool(p_spatial < alpha and p_label < alpha)
    # H3a is SUPPORTIVE-ONLY (plan §2/§7): HARDCODED False on every row —
    # H3a can never define subject-level support, regardless of direction or
    # null outcome. Only H3b/H3c (Tasks 6/8) define module_support_flag=True.
    # Do NOT "fix" this to `module_direction and module_null_pass`.
    module_support_flag = False

    row.update({
        "beta_axis_delta_null_z": beta_axis_delta_null_z,
        "p_spatial_delta": p_spatial, "p_label_delta": p_label,
        "onset_jitter_pass": onset_jitter_pass,
        "module_support_flag": module_support_flag,
        "module_direction_correct": module_direction,
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

    out_csv = outdir / "v3_susceptibility_subject.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_dir = sum(1 for r in rows if r["module_direction_correct"])
    print(
        f"[done] {len(rows)} subjects ({n_ok} ok) -> {out_csv} "
        f"(n_perm={n_perm}; {n_dir} module_direction_correct=True; module_support_flag always False)",
        flush=True,
    )


if __name__ == "__main__":
    main()

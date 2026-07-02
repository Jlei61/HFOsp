#!/usr/bin/env python
"""Topic 5 V2 Phase 2 (2B) — dynamic-mode M_loading + lambda_max trajectory + surrogate nulls.

EXPLORATORY peri-ictal susceptibility (NOT forecasting; NO stand-alone critical-mode claim).
Question: in the <=300 s preictal window, does the leading VAR(1) dynamic mode's loading
concentrate along the FIXED interictal HFO rank axis G_HFO, and does lambda_max drift up
toward onset — beyond phase- and block-surrogate dynamics?

`M_loading` is a loading *concentration* along the rank axis (|leading eigenvector| vs
typical_rank), NOT a propagation direction. Ridge VAR + CV one-step R2 gate: cv_r2<=min_cv_r2
=> var_meaningful_flag=False => descriptive only. Subject is the unit; broad/narrow never pooled.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v2_crit_io import (  # noqa: E402
    load_subject_preictal, window_index_range, get_contact_alignment,
)
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from src.topic5_v2_criticality import (  # noqa: E402
    load_phase2_config, prepare_var_window, var_window_ok, var1_ridge,
    spectral_radius, leading_eigvec, recovery_tau, cv_one_step_r2,
    block_shuffle_surrogate, phase_randomize_surrogate,
)

OUT_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_criticality"
ORIENTED_TEMPLATE = "a"  # pre-fixed orientation (spec §5.2 signed uses a fixed template)

COLUMNS = [
    "subject", "axis_set", "status", "skip_reason", "available_pre_sec", "required_pre_sec",
    "state_band", "M_loading_spearman", "M_loading_abs", "M_chosen_template",
    "cv_r2", "var_meaningful_flag", "lambda_max_late", "tau_late", "lambda_trend_spearman",
    "M_phase_null_z", "M_phase_empirical_p", "M_block_null_z", "M_block_empirical_p",
    "lambda_trend_phase_null_z", "lambda_trend_phase_empirical_p",
    "lambda_trend_block_null_z", "lambda_trend_block_empirical_p",
    "n_ch_fit", "n_t_fit", "nan_fraction", "n_windows_fit", "n_seizures", "n_seizures_total",
    "align_source", "order_null_strength", "tier",
]


def _slide_windows(relt, window_sec, step_sec):
    """List of (w0, w1, center_rel) index ranges tiling the preictal span."""
    span_lo, span_hi = float(relt.min()), float(relt.max())
    out = []
    start = span_lo
    while start + window_sec <= span_hi + 1e-9:
        rng = window_index_range(relt, start, start + window_sec)
        if rng is not None and (rng[1] - rng[0]) >= 3:
            out.append((rng[0], rng[1], float(np.mean(relt[rng[0]:rng[1]]))))
        start += step_sec
    return out


def _seizure_dynamics(E, relt, cfg, align_fn, ta_rank, tb_rank, names, with_cv=False):
    """Per-seizure VAR trajectory + latest-window M_loading. NaN-safe.

    ``with_cv`` gates the expensive full-span CV R2 (needed only for the observed
    ``var_meaningful_flag``, never inside the per-perm null loop).
    """
    dyn = cfg["dynamics"]
    alpha = float(dyn["var_ridge_alpha"])
    min_t_over_ch = float(dyn.get("min_t_over_ch", 5))
    hop = float(cfg["preictal"].get("hop_sec", 0.1))
    n_ch = E.shape[0]

    wins = _slide_windows(relt, float(cfg["preictal"]["window_sec"]),
                          float(cfg["preictal"]["step_sec"]))
    centers, lambdas = [], []
    last_loading = None
    n_t_fit = 0
    for (w0, w1, ctr) in wins:
        Xw = E[:, w0:w1]
        if not var_window_ok(n_ch, Xw.shape[1], min_t_over_ch):
            continue
        Xp = prepare_var_window(Xw)
        A = var1_ridge(Xp, alpha)
        lam = spectral_radius(A)
        if not np.isfinite(lam):
            continue
        centers.append(ctr)
        lambdas.append(lam)
        last_loading = leading_eigvec(A)  # latest powered window wins
        n_t_fit = Xw.shape[1]

    out = {"M_spearman": np.nan, "M_abs": np.nan, "lambda_trend": np.nan,
           "lambda_max_late": np.nan, "cv_r2": np.nan,
           "n_ch_fit": n_ch, "n_t_fit": n_t_fit, "nan_fraction": float(np.mean(~np.isfinite(E))),
           "n_windows_fit": len(lambdas)}
    if not lambdas:
        return out

    out["lambda_max_late"] = float(lambdas[-1])
    if len(lambdas) >= 4 and np.std(centers) > 0 and np.std(lambdas) > 0:
        out["lambda_trend"] = float(spearmanr(lambdas, centers).statistic)

    if last_loading is not None and last_loading.size == n_ch:
        loading_by_name = {names[i]: float(last_loading[i]) for i in range(n_ch)
                           if np.isfinite(last_loading[i])}
        al = align_fn(loading_by_name, ta_rank, tb_rank, ORIENTED_TEMPLATE)
        out["M_spearman"] = al["align_signed_oriented"]
        out["M_abs"] = al["align_abs_maxab_contact"]

    # cv_r2 on the whole prepared preictal span (VAR meaningfulness gate); observed only
    if with_cv:
        out["cv_r2"] = cv_one_step_r2(prepare_var_window(E), alpha, int(dyn["cv_folds"]))
    return out


def _subject_point(sub, cfg, align_fn):
    """Median-over-seizures observed dynamics summary (subject unit)."""
    per = [_seizure_dynamics(s["E"], s["relt"], cfg, align_fn,
                            sub["ta_rank"], sub["tb_rank"], sub["mapped"], with_cv=True)
           for s in sub["seizures"]]

    def med(key):
        vals = [p[key] for p in per if np.isfinite(p[key])]
        return float(np.median(vals)) if vals else np.nan

    return {
        "M_spearman": med("M_spearman"), "M_abs": med("M_abs"),
        "lambda_trend": med("lambda_trend"), "lambda_max_late": med("lambda_max_late"),
        "cv_r2": med("cv_r2"),
        "n_ch_fit": int(per[0]["n_ch_fit"]) if per else 0,
        "n_t_fit": int(np.median([p["n_t_fit"] for p in per])) if per else 0,
        "nan_fraction": med("nan_fraction"),
        "n_windows_fit": int(np.sum([p["n_windows_fit"] for p in per])),
    }


def _null_dist(sub, cfg, align_fn, surrogate_fn, n_perm, rng):
    """Null M_loading_spearman + lambda_trend arrays (surrogate the envelope, refit)."""
    block_len = max(1, int(round(float(cfg["dynamics"]["block_len_sec"])
                                 / float(cfg["preictal"].get("hop_sec", 0.1)))))
    mvals, tvals = [], []
    for _ in range(int(n_perm)):
        ms, ts = [], []
        for s in sub["seizures"]:
            if surrogate_fn is phase_randomize_surrogate:
                Es = surrogate_fn(s["E"], rng)
            else:
                Es = surrogate_fn(s["E"], block_len, rng)
            d = _seizure_dynamics(Es, s["relt"], cfg, align_fn,
                                 sub["ta_rank"], sub["tb_rank"], sub["mapped"])
            if np.isfinite(d["M_spearman"]):
                ms.append(d["M_spearman"])
            if np.isfinite(d["lambda_trend"]):
                ts.append(d["lambda_trend"])
        mvals.append(np.median(ms) if ms else np.nan)
        tvals.append(np.median(ts) if ts else np.nan)
    return np.array(mvals, float), np.array(tvals, float)


def _z_p_abs(obs, null):
    """Two-sided (magnitude) z + empirical p for an alignment-type statistic."""
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan, np.nan
    mu, sd = float(np.mean(np.abs(null))), float(np.std(np.abs(null)))
    z = (abs(obs) - mu) / sd if sd > 0 else np.nan
    p = (1 + int(np.sum(np.abs(null) >= abs(obs)))) / (1 + null.size)
    return z, p


def _z_p_upper(obs, null):
    """One-sided upper z + empirical p (lambda_max rising toward onset)."""
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan, np.nan
    mu, sd = float(np.mean(null)), float(np.std(null))
    z = (obs - mu) / sd if sd > 0 else np.nan
    p = (1 + int(np.sum(null >= obs))) / (1 + null.size)
    return z, p


def run_subject(ds_sid, substrate, cfg, n_perm, seed):
    align_fn, align_source = get_contact_alignment()
    subj = ds_sid.split("_", 1)[1] if "_" in ds_sid else ds_sid
    base = {c: "" for c in COLUMNS}
    base.update(subject=subj, axis_set=substrate, state_band=cfg["state_band"],
                required_pre_sec=cfg["preictal"]["min_required_pre_sec"],
                align_source=align_source, order_null_strength="pending_phase1",
                tier=cfg.get("tier", "exploratory"))

    sub = load_subject_preictal(ds_sid, substrate, cfg)
    base.update(available_pre_sec=round(sub["available_pre_sec"], 2),
                n_seizures=sub["n_seizures"], n_seizures_total=sub.get("n_seizures_total", sub["n_seizures"]))
    if sub["status"] != "ok":
        base.update(status="skipped", skip_reason=sub["skip_reason"])
        return base

    obs = _subject_point(sub, cfg, align_fn)
    rng = np.random.default_rng(seed)
    mp, tp = _null_dist(sub, cfg, align_fn, phase_randomize_surrogate, n_perm, rng)
    mb, tb = _null_dist(sub, cfg, align_fn, block_shuffle_surrogate, n_perm, rng)

    mpz, mpp = _z_p_abs(obs["M_spearman"], mp)
    mbz, mbp = _z_p_abs(obs["M_spearman"], mb)
    tpz, tpp = _z_p_upper(obs["lambda_trend"], tp)
    tbz, tbp = _z_p_upper(obs["lambda_trend"], tb)

    cv_r2 = obs["cv_r2"]
    base.update(
        status="ok", skip_reason="",
        M_loading_spearman=_r(obs["M_spearman"]), M_loading_abs=_r(obs["M_abs"]),
        M_chosen_template=ORIENTED_TEMPLATE,
        cv_r2=_r(cv_r2),
        var_meaningful_flag=bool(np.isfinite(cv_r2) and cv_r2 > float(cfg["dynamics"]["min_cv_r2"])),
        lambda_max_late=_r(obs["lambda_max_late"]), tau_late=_r(recovery_tau(obs["lambda_max_late"], float(cfg["preictal"].get("hop_sec", 0.1)))),
        lambda_trend_spearman=_r(obs["lambda_trend"]),
        M_phase_null_z=_r(mpz), M_phase_empirical_p=_r(mpp),
        M_block_null_z=_r(mbz), M_block_empirical_p=_r(mbp),
        lambda_trend_phase_null_z=_r(tpz), lambda_trend_phase_empirical_p=_r(tpp),
        lambda_trend_block_null_z=_r(tbz), lambda_trend_block_empirical_p=_r(tbp),
        n_ch_fit=obs["n_ch_fit"], n_t_fit=obs["n_t_fit"],
        nan_fraction=_r(obs["nan_fraction"]), n_windows_fit=obs["n_windows_fit"],
    )
    return base


def _r(x, nd=4):
    return round(float(x), nd) if x is not None and np.isfinite(x) else ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", choices=["broad", "narrow"], default="broad")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    cfg = load_phase2_config()
    n_perm = args.n_perm if args.n_perm is not None else int(cfg["nulls"]["n_perm_smoke"])
    seed = int(cfg["nulls"]["seed"])
    subjects = args.subjects or SUBJECTS_BY_SUB[args.substrate]
    outdir = Path(args.outdir) if args.outdir else (OUT_ROOT / args.substrate)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds_sid in subjects:
        try:
            rows.append(run_subject(ds_sid, args.substrate, cfg, n_perm, seed))
        except Exception as exc:  # never silently drop a subject
            subj = ds_sid.split("_", 1)[1] if "_" in ds_sid else ds_sid
            r = {c: "" for c in COLUMNS}
            r.update(subject=subj, axis_set=args.substrate, status="skipped",
                     skip_reason=f"error:{type(exc).__name__}:{exc}",
                     state_band=cfg["state_band"], tier=cfg.get("tier", "exploratory"))
            rows.append(r)
            print(f"[WARN] {ds_sid}: {type(exc).__name__}: {exc}", file=sys.stderr)

    out_csv = outdir / "phase2_dynamics_subject.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"[dynamics] {args.substrate}: {n_ok}/{len(rows)} ok, n_perm={n_perm} -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    main()

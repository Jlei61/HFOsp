"""V3c-2 (gated mechanistic secondary): ictal recruitment timing of axis-surplus.
Primary contrast A∩S vs A∖S; sensitivity S vs A∖S. Gated on set-thresholds AND
label-blind assay_valid (from latency_qc). Subject-first, within-shaft AUC null.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import classify_subject_contacts
from scripts._topic5_v3c_io import V3C_SUBJECTS, extract_latency_matrix, axis_soz_join, load_soz
from src.topic5_v3_mode_transition import load_v3_config
from src.topic5_v3c_latency import (auc_late, auc_null_distribution, delta_t,
                                    encode_latency_for_rank, latency_seconds)

OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"
COLS = ["subject", "cohort", "n_covered", "n_surplus", "n_sz_used", "auc_primary",
        "auc_drop_censored", "auc_exclude_t0", "auc_sensitivity_allsoz",
        "delta_t_sec", "auc_null_p", "auc_null_q05", "auc_null_q50", "auc_null_q95",
        "eligible"]   # per-subject null quantiles for the AUC forest figure


def latency_eligible(join: dict, qc_valid: bool, cfg: dict) -> bool:
    lat = cfg["v3c"]["latency"]
    return bool(qc_valid and join["n_surplus"] >= lat["min_surplus"]
                and join["n_covered"] >= lat["min_covered_soz"])


def _assay_valid_map(cohort: str) -> dict:
    p = OUT / cohort / "latency_qc" / "qc_subject.csv"
    out = {}
    if p.exists():
        import csv as _csv
        for r in _csv.DictReader(p.open()):
            out[r["subject"]] = (r["assay_valid"] == "True")
    return out


def _auc_variant(by, group_soz, group_surplus, window_sec, mode):
    """AUC_late(surplus,soz) under a censoring-sensitivity mode (spec §5.4):
    'primary' (censored->last, t0->first), 'drop_censored' (exclude censored from
    both groups), 'exclude_t0' (exclude t0 from both groups). nan if a group empties.
    """
    def keep(n):
        k = by[n][0]
        if mode == "drop_censored":
            return k != "censored"
        if mode == "exclude_t0":
            return k != "t0"
        return True
    su = [n for n in group_surplus if keep(n)]
    so = [n for n in group_soz if keep(n)]
    sv = np.array([encode_latency_for_rank(*by[n], window_sec=window_sec) for n in su])
    zv = np.array([encode_latency_for_rank(*by[n], window_sec=window_sec) for n in so])
    return auc_late(sv, zv)


def latency_subject_row(ds_sid, cohort, cfg, qc_map) -> dict:
    dataset, subj = ds_sid.split("_", 1)
    vc = cfg["v3c"]; row = {c: float("nan") for c in COLS}
    row.update({"subject": ds_sid, "cohort": cohort, "eligible": False})
    try:
        cls = classify_subject_contacts(ds_sid, cohort, cfg)
        j = axis_soz_join(cls, load_soz(dataset, subj))
        eligible = latency_eligible(j, qc_map.get(ds_sid, False), cfg)
        covered, surplus, soz_all = j["covered"], j["surplus"], j["soz_in_pool"]
        names = covered + surplus + [n for n in soz_all if n not in set(covered)]
        mats = extract_latency_matrix(ds_sid, cfg, names, thresholds=[vc["z_cross"]])
        aucs_p, aucs_dc, aucs_xt0, aucs_s, dts, nulls = [], [], [], [], [], []
        for m in mats:
            kinds = m["kinds"][vc["z_cross"]]; secs = m["secs"][vc["z_cross"]]
            by = {n: (kinds[i], secs[i]) for i, n in enumerate(names)}
            aucs_p.append(_auc_variant(by, covered, surplus, vc["window_sec"], "primary"))
            aucs_dc.append(_auc_variant(by, covered, surplus, vc["window_sec"], "drop_censored"))
            aucs_xt0.append(_auc_variant(by, covered, surplus, vc["window_sec"], "exclude_t0"))
            aucs_s.append(_auc_variant(by, soz_all, surplus, vc["window_sec"], "primary"))  # clinical sensitivity: S vs A∖S
            sv = np.array([latency_seconds(*by[n]) for n in surplus])   # signed Δt: surplus − covered (>0 = surplus later = H-B)
            zv = np.array([latency_seconds(*by[n]) for n in covered])
            dts.append(delta_t(sv, zv))
            if eligible:
                svr = np.array([encode_latency_for_rank(*by[n], window_sec=vc["window_sec"]) for n in surplus])
                zvr = np.array([encode_latency_for_rank(*by[n], window_sec=vc["window_sec"]) for n in covered])
                nulls.append(auc_null_distribution(svr, zvr, cls["shaft_by_name"], surplus, covered,
                                                   n_perm=vc["nulls"]["n_perm"], rng=vc["nulls"]["seed"]))
        auc_p = float(np.nanmedian(aucs_p)) if aucs_p else float("nan")
        dt_med = float(np.nanmedian(dts)) if dts else float("nan")
        null_med = np.median(np.vstack(nulls), axis=0) if nulls else np.array([])
        p = float((np.sum(null_med >= auc_p) + 1) / (null_med.size + 1)) if null_med.size else float("nan")
        q05, q50, q95 = (np.percentile(null_med, [5, 50, 95]) if null_med.size
                         else (np.nan, np.nan, np.nan))
        row.update({"n_covered": j["n_covered"], "n_surplus": j["n_surplus"], "n_sz_used": len(mats),
                    "auc_primary": auc_p,
                    "auc_drop_censored": float(np.nanmedian(aucs_dc)) if aucs_dc else float("nan"),
                    "auc_exclude_t0": float(np.nanmedian(aucs_xt0)) if aucs_xt0 else float("nan"),
                    "auc_sensitivity_allsoz": float(np.nanmedian(aucs_s)) if aucs_s else float("nan"),
                    "delta_t_sec": dt_med, "auc_null_p": p,
                    "auc_null_q05": float(q05), "auc_null_q50": float(q50), "auc_null_q95": float(q95),
                    "eligible": bool(eligible),
                    "_auc": auc_p, "_dt": dt_med, "_null_med": null_med,
                    "_auc_dc": float(np.nanmedian(aucs_dc)) if aucs_dc else float("nan"),
                    "_auc_xt0": float(np.nanmedian(aucs_xt0)) if aucs_xt0 else float("nan")})
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {ds_sid} ({cohort}): {type(exc).__name__}: {exc}", flush=True)
    return row


def _same_side(a, b, band):
    lo, hi = band
    side = lambda x: (1 if x > hi else (-1 if x < lo else 0)) if np.isfinite(x) else 99
    return side(a) == side(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    cfg = load_v3_config(); qc_map = _assay_valid_map(args.cohort)
    band = cfg["v3c"]["interpretation"]["auc_ha_band"]
    outdir = OUT / args.cohort / "latency"; outdir.mkdir(parents=True, exist_ok=True)
    rows = [latency_subject_row(s, args.cohort, cfg, qc_map) for s in V3C_SUBJECTS[args.cohort]]
    elig = [r for r in rows if r.get("eligible") and np.isfinite(r.get("_auc", float("nan")))]
    cohort = {}
    if elig:
        n_perm = min(r["_null_med"].size for r in elig)
        perm_med = np.median(np.vstack([r["_null_med"][:n_perm] for r in elig]), axis=0)
        obs = float(np.median([r["_auc"] for r in elig]))
        dc_med = float(np.median([r["_auc_dc"] for r in elig]))
        xt0_med = float(np.median([r["_auc_xt0"] for r in elig]))
        cohort = {
            "obs_cohort_median_auc": obs, "n_perm": int(n_perm), "n_subjects": len(elig),
            "p_value": float((np.sum(perm_med >= obs) + 1) / (n_perm + 1)),
            "auc_null_q05": float(np.percentile(perm_med, 5)),
            "auc_null_q50": float(np.percentile(perm_med, 50)),
            "auc_null_q95": float(np.percentile(perm_med, 95)),
            "subject_aucs": {r["subject"]: r["_auc"] for r in elig},
            # P1-2: Δt aggregation (H-B needs the SIGNED cohort median; missing -> summary fails closed)
            "delta_t_med": float(np.median([r["_dt"] for r in elig])),
            "subject_delta_t": {r["subject"]: r["_dt"] for r in elig},
            # P1-5: censor/t0 sensitivity AUC cohort medians + sign concordance vs primary
            "auc_drop_censored_med": dc_med, "auc_exclude_t0_med": xt0_med,
            "sensitivity_concordant": bool(_same_side(obs, dc_med, band) and _same_side(obs, xt0_med, band)),
        }
    with open(outdir / "latency_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in COLS})
    (outdir / "latency_cohort.json").write_text(json.dumps(cohort, indent=2))
    print(f"[done] {args.cohort} latency: {len(elig)} eligible; cohort_auc={cohort.get('obs_cohort_median_auc')} "
          f"delta_t_med={cohort.get('delta_t_med')} concordant={cohort.get('sensitivity_concordant')}")


if __name__ == "__main__":
    main()

"""V3c-2 pre-check: label-blind latency assay-QC on axis contacts (spec §5.2).
Emits QC metrics that gate whether latency is a mechanistic endpoint. Does NOT
compute any SOZ-vs-surplus contrast (endpoint-blind).
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
from src.topic5_v3c_latency import (assay_valid, censoring_tallies, rank_diagnostics,
                                    threshold_stability)

OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"
COLS = ["subject", "cohort", "n_axis", "n_covered", "n_surplus", "n_sz_used",
        "finite_frac", "t0_frac", "cens_frac", "uniq_ranks_med", "max_tie_med",
        "thr_spearman", "n_informative", "sz_stability_std", "assay_valid", "cens_flag"]


def qc_subject(ds_sid: str, cohort: str, cfg: dict) -> dict:
    dataset, subj = ds_sid.split("_", 1)
    vc = cfg["v3c"]
    row = {c: float("nan") for c in COLS}; row.update({"subject": ds_sid, "cohort": cohort})
    try:
        cls = classify_subject_contacts(ds_sid, cohort, cfg)
        A = cls["is_axis"]
        j = axis_soz_join(cls, load_soz(dataset, subj))
        thrs = [vc["z_cross"]] + list(vc["z_cross_sensitivity"])
        mats = extract_latency_matrix(ds_sid, cfg, A, thresholds=thrs)
        all_kinds, uniq_l, tie_l, rho_l, szmed_l, n_info = [], [], [], [], [], 0
        for m in mats:
            kinds = m["kinds"][vc["z_cross"]]; secs = np.array(m["secs"][vc["z_cross"]], float)
            all_kinds += kinds
            rd = rank_diagnostics(secs); uniq_l.append(rd["uniq_ranks"]); tie_l.append(rd["max_tie_block"])
            fin = secs[np.isfinite(secs)]
            if fin.size:
                szmed_l.append(float(np.median(fin)))
            if rd["uniq_ranks"] >= vc["assay_qc"]["informative_min_unique_ranks"] and \
               any(k == "finite" for k in kinds):
                n_info += 1
            for alt in vc["z_cross_sensitivity"]:
                rho = threshold_stability(secs, np.array(m["secs"][alt], float))
                if np.isfinite(rho):
                    rho_l.append(rho)
        tal = censoring_tallies(all_kinds)
        qc = {"finite_frac": tal["finite_frac"], "t0_frac": tal["t0_frac"],
              "uniq_ranks_med": float(np.median(uniq_l)) if uniq_l else 0.0,
              "thr_spearman": float(np.median(rho_l)) if rho_l else float("nan"),
              "n_informative": n_info}
        row.update({
            "n_axis": len(A), "n_covered": j["n_covered"], "n_surplus": j["n_surplus"],
            "n_sz_used": len(mats), "finite_frac": tal["finite_frac"], "t0_frac": tal["t0_frac"],
            "cens_frac": tal["cens_frac"], "uniq_ranks_med": qc["uniq_ranks_med"],
            "max_tie_med": float(np.median(tie_l)) if tie_l else float("nan"),
            "thr_spearman": qc["thr_spearman"], "n_informative": n_info,
            "sz_stability_std": float(np.std(szmed_l)) if len(szmed_l) >= 2 else float("nan"),
            "assay_valid": bool(assay_valid(qc, cfg)),
            "cens_flag": bool(tal["cens_frac"] > vc["assay_qc"]["cens_frac_flag"]),
        })
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {ds_sid} ({cohort}): {type(exc).__name__}: {exc}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    cfg = load_v3_config()
    outdir = OUT / args.cohort / "latency_qc"; outdir.mkdir(parents=True, exist_ok=True)
    rows = [qc_subject(s, args.cohort, cfg) for s in V3C_SUBJECTS[args.cohort]]
    with open(outdir / "qc_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in COLS})
    print(f"[done] {args.cohort} assay-QC; valid={[r['subject'] for r in rows if r.get('assay_valid')]}")


if __name__ == "__main__":
    main()

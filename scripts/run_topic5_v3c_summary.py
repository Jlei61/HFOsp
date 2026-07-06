"""V3c summary: coverage double-condition verdict + gated latency interpretation
(H-B_supported / H-A_compatible / indeterminate, spec §5.6, R3 descriptive H-A)
+ claim-language string. broad primary / narrow sensitivity (separate files).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.topic5_v3_mode_transition import load_v3_config

OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"


def interpret_latency(cohort_auc, subject_aucs, delta_t_med, null_p, sensitivity_concordant, cfg) -> str:
    """4-way (spec §5.6 + review P1-3/P1-5). H-B needs SIGNED delta_t>=+thr (surplus
    later) AND censor/t0 sensitivity concordant. surplus_earlier_unverified is the
    low-tail (surplus earlier) case — kept distinct from indeterminate. H-A is a
    DESCRIPTIVE compatibility statement (R3), never proven equivalence.
    """
    it = cfg["v3c"]["interpretation"]; alpha = 0.05
    subj = np.asarray(subject_aucs, dtype=float)
    n = subj.size
    maj = int(np.floor(n / 2) + 1)
    lo, hi = it["auc_ha_band"]
    n_late = int(np.sum(subj > it["subject_hb_auc_min"]))
    # H-B: surplus LATER — signed delta_t must be positive (P1-3) AND sensitivity concordant (P1-5)
    if (cohort_auc >= it["auc_hb_min"] and n_late >= maj
            and delta_t_med >= it["delta_t_hb_min_sec"] and null_p < alpha
            and bool(sensitivity_concordant)):
        return "H-B_supported"
    # surplus EARLIER than SOZ core (low tail) — distinct from indeterminate; needs artifact check
    if cohort_auc <= (1.0 - it["auc_hb_min"]) and delta_t_med <= -it["delta_t_hb_min_sec"]:
        return "surplus_earlier_unverified"
    # H-A compatible (descriptive): AUC in band, no consistent late bias, small |delta_t|
    consistent = np.all(subj <= it["subject_hb_auc_min"]) if n else False
    if lo <= cohort_auc <= hi and consistent and abs(delta_t_med) < 2.0:
        return "H-A_compatible"
    return "indeterminate"


CLAIM = {
    "H-B_supported": "Axis-surplus contacts were recruited after clinical SOZ contacts, "
                     "supporting the interpretation that the interictal axis captures a broader "
                     "propagation scaffold rather than only the seizure onset core.",
    "H-A_compatible": "Axis-surplus recruitment latency was compatible with onset-synchronous "
                      "recruitment relative to the axis-covered SOZ core (descriptive; n too small "
                      "for a formal equivalence test).",
    "surplus_earlier_unverified": "Axis-surplus contacts appeared to be recruited BEFORE the "
                      "axis-covered SOZ core; this low-latency-tail pattern requires a t0 "
                      "left-censoring artifact check before interpretation and is reported unverified.",
    "indeterminate": "First-threshold recruitment latency was not sufficiently resolved to "
                     "distinguish onset-synchronous from downstream surplus recruitment.",
}

# R2: primary same-shaft null only licenses 'beyond implantation geometry', NOT 'beyond HFO-rich'
# (that needs the follow-up HFO-rate-matched null). Wording is pinned accordingly.
COVERAGE_CLAIM = {
    "specific_axis_soz_organization": "The interictal propagation axis covered clinical SOZ beyond "
        "implantation geometry AND its non-SOZ surplus was spatially structured (closer to SOZ than a "
        "same-shaft random axis), indicating specific axis-SOZ spatial organization.",
    "beyond_implantation_geometry": "The interictal propagation axis covered clinical SOZ beyond "
        "implantation geometry; surplus spatial structure was NOT established, so specificity beyond a "
        "geometric coincidence is not claimed.",
    "none": "SOZ coverage by the interictal axis did not exceed the same-shaft geometric null.",
}


def _require(d: dict, keys: list, where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{where} missing required fields {missing} (fail-closed, review P1-2)")


def _spatial_primary_ok(spa: dict, cfg: dict) -> bool:
    """Surplus spatial structure counts toward the coverage double-condition only when
    ENOUGH coord-eligible subjects support it (review P1: a 1-subject spatial claim is
    too fragile to license 'specific axis-SOZ organization') AND the cohort-median
    distance null passes."""
    return bool(spa.get("n_spatial_eligible", 0) >= cfg["v3c"]["spatial"]["min_subjects_for_primary"]
                and spa.get("p_value", 1.0) < 0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    cfg = load_v3_config(); base = OUT / args.cohort

    def _load(p):
        return json.loads(p.read_text()) if p.exists() else {}
    cov = _load(base / "coverage_cohort.json")
    lat = _load(base / "latency/latency_cohort.json")
    spa = _load(base / "surplus_spatial/surplus_spatial_cohort.json")

    # P1-1: coverage DOUBLE-condition (spec §4.4) — significant coverage AND structured surplus.
    # spatial leg also gated on n_spatial_eligible (review P1: >=min_subjects_for_primary).
    coverage_sig = bool(cov.get("p_value", 1.0) < 0.05)
    spatial_sig = _spatial_primary_ok(spa, cfg)
    coverage_primary_pass = coverage_sig and spatial_sig
    coverage_claim_level = ("specific_axis_soz_organization" if coverage_primary_pass
                            else ("beyond_implantation_geometry" if coverage_sig else "none"))

    interp = "not_run"
    if lat.get("subject_aucs"):
        # P1-2 fail-closed: these fields are the Task-11 contract; missing => raise, don't default
        _require(lat, ["obs_cohort_median_auc", "subject_aucs", "delta_t_med", "p_value",
                       "sensitivity_concordant"], "latency_cohort.json")
        interp = interpret_latency(lat["obs_cohort_median_auc"], list(lat["subject_aucs"].values()),
                                   lat["delta_t_med"], lat["p_value"], lat["sensitivity_concordant"], cfg)

    summary = {
        "cohort": args.cohort,
        "coverage_significant": coverage_sig, "coverage_p": cov.get("p_value"),
        "surplus_spatial_significant": spatial_sig, "surplus_spatial_p": spa.get("p_value"),
        "coverage_primary_pass": coverage_primary_pass,
        "coverage_claim_level": coverage_claim_level,
        "coverage_claim": COVERAGE_CLAIM[coverage_claim_level],
        "latency_interpretation": interp, "latency_claim": CLAIM.get(interp, ""),
        "latency_delta_t_med": lat.get("delta_t_med"),
        "latency_sensitivity_concordant": lat.get("sensitivity_concordant"),
    }
    (base / "v3c_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stage 2A — Fig3-B R3 score provenance for the locked E1146 seizure 2 exemplar.

The exemplar is NOT reselected: subject and seizure are fixed to the paper-ready
locked case (docs/fig3b_interictal_ictal_shared_field_spec.md; E1146 seizure 2).
This recomputes the *statistical* R3 dense-grid maxAB concordance with the Stage 1
engine (shared grid, one sigma_common, [0,10]s BB150 activation) and the paired
R2 contact-evaluated sensitivity on identical inputs, keeps the 6 mm *display*
smoothing explicitly separate, and writes a provenance JSON into the staging root
without touching the locked production figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.topic5_gradient_grid_field as gg
from scripts.run_topic5_figure3_ictal_grid_rebuild import (
    ActivationCache, SubjectField, build_common_mask, load_primary_bands,
)
from src.topic5_template_axis_field import score_field

SUBJECT = "epilepsiae_1146"
LOCKED_SEIZURE_IDX = 2                        # paper-locked; NOT reselected
DISPLAY_SIGMA_MM = 6.0                        # display-only kernel, never statistical
STAGE = REPO / "results/paper-ready-figure/fig3_ictal_field_concordance_grid_rebuild"


def r2_maxab_at_contacts(sf: SubjectField, activation_masked: np.ndarray):
    sa, sb = sf.build_r2_scorers()
    a = score_field(sa, activation_masked)
    b = score_field(sb, activation_masked)
    cands = [(t, v["abs_r"], v["signed_r"], v["mirror_choice"])
             for t, v in (("A", a), ("B", b)) if np.isfinite(v["abs_r"])]
    best = max(cands, key=lambda z: z[1]) if cands else (None, np.nan, np.nan, None)
    return {"abs_a": a["abs_r"], "abs_b": b["abs_r"],
            "signed_a": a["signed_r"], "signed_b": b["signed_r"],
            "maxab": float(best[1]), "best_template": best[0]}


def main():
    bands = load_primary_bands()
    sf = SubjectField(SUBJECT)
    ac = ActivationCache(SUBJECT, bands)
    if sf.route != "shared":
        raise SystemExit(f"expected shared route for {SUBJECT}, got {sf.route}")
    finite, band_acts, anchor, drop = build_common_mask(sf, ac, bands, LOCKED_SEIZURE_IDX)
    if drop is not None:
        raise SystemExit(f"cannot build mask for {SUBJECT} sz{LOCKED_SEIZURE_IDX}: {drop}")
    n_common = int(finite.sum())
    activation_masked = np.where(finite, anchor, np.nan)      # BB150 [0,10]s, common mask

    # statistical R3 (Stage 1 engine, shared grid, sigma_common)
    ev = sf.build_event_scorers(finite, gg.GRID_N)
    r3 = gg.score_event_detail_single(ev, activation_masked)
    r2 = r2_maxab_at_contacts(sf, activation_masked)

    prov = {
        "figure": "fig3b_interictal_ictal_shared_field",
        "subject": SUBJECT, "seizure_idx": LOCKED_SEIZURE_IDX,
        "exemplar_reselected": False,
        "exemplar_lock_source": "docs/fig3b_interictal_ictal_shared_field_spec.md",
        "route": sf.route, "sigma_common": sf.sigma_common,
        "n_common_contacts": n_common,
        "activation": "bb150_auc (mean baseline-robust-z 1-150 Hz over clinical [0,10] s)",
        "statistical_r3": {
            "estimand": "R3 dense-grid support-gated maxAB field concordance (N=81)",
            "maxab": r3["maxab"], "best_template": r3["best_template"],
            "abs_a": r3["abs_a"], "abs_b": r3["abs_b"],
            "signed_a": r3["signed_a"], "signed_b": r3["signed_b"],
            "mirror_a": r3["mirror_a"], "mirror_b": r3["mirror_b"],
            "overlap_a": r3["overlap_a"], "overlap_b": r3["overlap_b"],
            "sigma": sf.sigma_common, "grid_a_sha256": ev["grid_a"]["sha256"]},
        "paired_r2_sensitivity": {
            "estimand": "R2 contact-evaluated smoothed maxAB (same inputs, same sigma_common)",
            "maxab": r2["maxab"], "best_template": r2["best_template"],
            "abs_a": r2["abs_a"], "abs_b": r2["abs_b"],
            "signed_a": r2["signed_a"], "signed_b": r2["signed_b"]},
        "display_field_6mm": {
            "role": "rendered heatmap smoothing ONLY; not a statistical kernel",
            "display_sigma_mm": DISPLAY_SIGMA_MM,
            "note": "the 6 mm display sigma must never be reported as the frozen scoring sigma"},
        "claim_boundary": "single locked exemplar of the cohort-level field concordance; "
                          "an intentionally selected representative, not independent validation.",
        "locked_production_figure_overwritten": False,
    }
    STAGE.mkdir(parents=True, exist_ok=True)
    out = STAGE / "fig3b_r3_score_provenance.json"
    out.write_text(json.dumps(prov, indent=2, default=float))
    print(f"[fig3b] R3 maxab={r3['maxab']:.4f} (best {r3['best_template']}, "
          f"overlap A={r3['overlap_a']}/B={r3['overlap_b']}) | "
          f"R2 maxab={r2['maxab']:.4f} | n_common={n_common} | sigma={sf.sigma_common:.4f}")
    print(f"[fig3b] provenance -> {out}")


if __name__ == "__main__":
    main()

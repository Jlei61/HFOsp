#!/usr/bin/env python3
"""SEF-HFO SOZ-localization runner — Comparison A (static localization, §4).

Consumes results/topic4_sef_hfo/soz_localization/cohort.json (frozen by
build_soz_localization_cohort.py). Per subject: propagation-geometry scores
(endpoint primary; source/sink/swap sensitivities) vs firing-rate, ROC-AUC + top-k
overlap against held-out clinical SOZ within the channel universe U. Cohort: paired
Wilcoxon (geom AUC >= rate AUC), reported for all / epilepsiae(primary) / yuquan.

Reliability curve (Comparison B) is a separate task (run with --reliability later).
Contract: plan v3 §4; diagnostic channel_universe_montage_diagnostic_2026-06-06.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sef_hfo_soz_localization import (
    geom_scores, align_rate_and_soz, comparison_a_subject, aggregate_comparison_a,
)
from src.rank_displacement import derive_swap_endpoint

OUT_DIR = _ROOT / "results" / "topic4_sef_hfo" / "soz_localization"
GEOM_DIR = _ROOT / "results" / "interictal_propagation_masked" / "rank_displacement" / "per_subject"


def _load_geom_pair(dataset: str, subject: str):
    d = json.loads((GEOM_DIR / f"{dataset}_{subject}.json").read_text())
    pairs = d.get("pairs") or []
    return pairs[0] if pairs else None


def _swap_score(universe, pair):
    """Binary swap-endpoint indicator over the universe (PR-6 supplement, reused helper).

    Uses decision_k swap source∪sink on the VALID channel subset, then marks universe
    channels in that set. Returns None if swap is undefined for this subject.
    """
    sweep = pair.get("swap_sweep") or {}
    k = sweep.get("decision_k")
    if not k or k < 1:
        return None
    valid = [(c, ra) for c, v, ra in zip(pair["channel_names"], pair["joint_valid"],
                                         pair["rank_a_dense_full"]) if v]
    if 2 * k > len(valid):
        return None
    valid_ch = [c for c, _ in valid]
    valid_ra = np.array([ra for _, ra in valid], dtype=float)
    try:
        swap_set = set(derive_swap_endpoint(valid_ch, valid_ra, int(k)))
    except ValueError:
        return None
    return np.array([1.0 if c in swap_set else 0.0 for c in universe], dtype=float)


def _subject_result(rec: dict) -> dict:
    ds, subj = rec["dataset"], rec["subject"]
    universe = rec["universe"]
    pair = _load_geom_pair(ds, subj)
    g = geom_scores(universe, pair)
    rate_vec, y = align_rate_and_soz(universe, rec["rate_in_universe"], rec["soz_core"])

    scores = {"rate": rate_vec, "endpoint": g["endpoint"],
              "source": g["source"], "sink": g["sink"]}
    swap = _swap_score(universe, pair)
    if swap is not None:
        scores["swap"] = swap

    res = comparison_a_subject(scores, y)
    res.update({"dataset": ds, "subject": subj, "montage": rec["montage"],
                "soz_coverage": rec["soz_coverage"], "n_soz_core": rec["n_soz_core"],
                "has_swap": swap is not None})
    return res


def main() -> int:
    cohort = json.loads((OUT_DIR / "cohort.json").read_text())
    per_subject = [_subject_result(r) for r in cohort["kept"]]
    (OUT_DIR / "per_subject").mkdir(parents=True, exist_ok=True)
    for r in per_subject:
        (OUT_DIR / "per_subject" / f"comparison_a_{r['dataset']}_{r['subject']}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False))

    agg_all = aggregate_comparison_a(per_subject)
    agg_epi = aggregate_comparison_a([r for r in per_subject if r["dataset"] == "epilepsiae"])
    agg_yq = aggregate_comparison_a([r for r in per_subject if r["dataset"] == "yuquan"])
    out = {"meta": {"primary_cohort": "epilepsiae", "claim": cohort["meta"]["claim"],
                    "n_kept": len(per_subject)},
           "cohort_all": agg_all, "cohort_epilepsiae_primary": agg_epi, "cohort_yuquan": agg_yq,
           "per_subject": per_subject}
    (OUT_DIR / "comparison_a.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ----- report -----
    def line(name, agg):
        ma = agg["median_auc"]
        return (f"  {name:<22} n_elig={agg['n_eligible']:>2}  "
                f"medAUC rate={ma.get('rate', float('nan')):.3f} endpoint={ma.get('endpoint', float('nan')):.3f}  "
                f"Δ(endp-rate)={agg['median_delta_auc'].get('endpoint', float('nan')):+.3f}  "
                f"endp≥rate={agg['n_geom_ge_rate'].get('endpoint', 0)}/{agg['n_eligible']}  "
                f"Wilcoxon p(endp≥rate)={agg['wilcoxon_p_endpoint_ge_rate']:.3f}")

    print("\n=== Comparison A: geometry-endpoint vs firing-rate SOZ localization (within U) ===")
    print(line("ALL", agg_all))
    print(line("EPILEPSIAE (primary)", agg_epi))
    print(line("YUQUAN (supplement)", agg_yq))
    print(f"\n  median SOZ coverage: {agg_all['median_soz_coverage']}")
    print(f"  insufficient (excluded from paired test): "
          f"{[r['subject'] for r in per_subject if r['insufficient']]}")
    print("\n  sensitivity median AUC (epilepsiae primary): " +
          ", ".join(f"{k}={v:.3f}" for k, v in agg_epi["median_auc"].items()))
    print(f"\nwrote {OUT_DIR / 'comparison_a.json'} (+ per_subject/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
